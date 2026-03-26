import os

import math
import torch
import torch.nn as nn
from einops import rearrange
from basicsr.archs.arch_util import to_2tuple, trunc_normal_
import torch.nn.functional as F
from torch.nn import Module, Sequential, Conv2d, ReLU,AdaptiveMaxPool2d, AdaptiveAvgPool2d, \
    NLLLoss, BCELoss, CrossEntropyLoss, AvgPool2d, MaxPool2d, Parameter, Linear, Sigmoid, Softmax, Dropout, Embedding
import matplotlib.pyplot as plt
def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    From: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).

    From: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class ChannelAttention(nn.Module):
    """Channel attention used in RCAN.
    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
    """

    def  __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
        self.num_feat = num_feat  # 正确设置 num_feat 属性
        self.squeeze_factor = squeeze_factor
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid())

    def forward(self, x):
        y = self.attention(x)
        return x * y
    def flops(self, H, W):
        # FLOPs for 1x1 Conv layers
        conv1_flops = (self.num_feat * (self.num_feat // self.squeeze_factor) * 1 * 1 * H * W)
        conv2_flops = ((self.num_feat // self.squeeze_factor) * self.num_feat * 1 * 1 * 1 * 1)  # output of pool is 1x1
        # Total FLOPs
        total_flops = conv1_flops + conv2_flops
        return total_flops

def window_partition(x, window_size):
    """
    Args:
        x: (b, h, w, c)
        window_size (int): window size

    Returns:
        windows: (num_windows*b, window_size, window_size, c)
    """
    b, h, w, c = x.shape
    x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, c)
    return windows


def window_reverse(windows, window_size, h, w):
    """
    Args:
        windows: (num_windows*b, window_size, window_size, c)
        window_size (int): Window size
        h (int): Height of image
        w (int): Width of image

    Returns:
        x: (b, h, w, c)
    """
    b = int(windows.shape[0] / (h * w / window_size / window_size))
    x = windows.view(b, h // window_size, w // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer('relative_position_index', relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)

        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*b, n, c)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        b_, n, c = x.shape
        qkv = self.qkv(x).reshape(b_, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(b_ // nw, nw, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(b_, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, n):
        flops = 0
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * n * (self.dim // self.num_heads) * n
        #  x = (attn @ v)
        flops += self.num_heads * n * n * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += n * self.dim * self.dim
        return flops

class CAB(nn.Module):

    def __init__(self, num_feat, compress_ratio=3, squeeze_factor=30):
        super(CAB, self).__init__()

        self.cab = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // compress_ratio, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(num_feat // compress_ratio, num_feat, 3, 1, 1),
            ChannelAttention(num_feat, squeeze_factor)
        )

    def forward(self, x):
        return self.cab(x)


class WindowAttentionHATFIR(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)

        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, rpi, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*b, n, c)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        b_, n, c = x.shape
        qkv = self.qkv(x).reshape(b_, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[rpi.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(b_ // nw, nw, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(b_, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class HAB(nn.Module):
    r""" Hybrid Attention Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 num_heads,
                 window_size=7,
                 shift_size=0,
                 compress_ratio=3,
                 squeeze_factor=30,
                 conv_scale=0.01,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, 'shift_size must in 0-window_size'

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttentionHATFIR(
            dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop)

        self.conv_scale = conv_scale
        self.conv_block = CAB(num_feat=dim, compress_ratio=compress_ratio, squeeze_factor=squeeze_factor)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, x_size, rpi_sa, attn_mask):
        h, w = x_size
        b, _, c = x.shape
        # assert seq_len == h * w, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(b, h, w, c)

        # Conv_X
        conv_x = self.conv_block(x.permute(0, 3, 1, 2))
        conv_x = conv_x.permute(0, 2, 3, 1).contiguous().view(b, h * w, c)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = attn_mask
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nw*b, window_size, window_size, c
        x_windows = x_windows.view(-1, self.window_size * self.window_size, c)  # nw*b, window_size*window_size, c

        # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size
        attn_windows = self.attn(x_windows, rpi=rpi_sa, mask=attn_mask)

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, c)
        shifted_x = window_reverse(attn_windows, self.window_size, h, w)  # b h' w' c

        # reverse cyclic shift
        if self.shift_size > 0:
            attn_x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            attn_x = shifted_x
        attn_x = attn_x.view(b, h * w, c)

        # FFN
        x = shortcut + self.drop_path(attn_x) + conv_x * self.conv_scale
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: b, h*w, c
        """
        h, w = self.input_resolution
        b, seq_len, c = x.shape
        assert seq_len == h * w, 'input feature has wrong size'
        assert h % 2 == 0 and w % 2 == 0, f'x size ({h}*{w}) are not even.'

        x = x.view(b, h, w, c)

        x0 = x[:, 0::2, 0::2, :]  # b h/2 w/2 c
        x1 = x[:, 1::2, 0::2, :]  # b h/2 w/2 c
        x2 = x[:, 0::2, 1::2, :]  # b h/2 w/2 c
        x3 = x[:, 1::2, 1::2, :]  # b h/2 w/2 c
        x = torch.cat([x0, x1, x2, x3], -1)  # b h/2 w/2 4*c
        x = x.view(b, -1, 4 * c)  # b h/2*w/2 4*c

        x = self.norm(x)
        x = self.reduction(x)

        return x
    def flops(self, input_resolution=None):
        h, w = self.input_resolution if input_resolution is None else input_resolution
        total_flops= h * w * self.dim
        total_flops += (h // 2) * (w // 2) * 4 * self.dim * 2 * self.dim
        return total_flops

class OCAB(nn.Module):
    # overlapping cross-attention block

    def __init__(self, dim,
                 input_resolution,
                 window_size,
                 overlap_ratio,
                 num_heads,
                 qkv_bias=True,
                 qk_scale=None,
                 mlp_ratio=2,
                 norm_layer=nn.LayerNorm
                 ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.overlap_win_size = int(window_size * overlap_ratio) + window_size

        self.norm1 = norm_layer(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.unfold = nn.Unfold(kernel_size=(self.overlap_win_size, self.overlap_win_size), stride=window_size,
                                padding=(self.overlap_win_size - window_size) // 2)

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((window_size + self.overlap_win_size - 1) * (window_size + self.overlap_win_size - 1),
                        num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

        self.proj = nn.Linear(dim, dim)

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU)

    def forward(self, x, x_size, rpi):
        h, w = x_size
        b, _, c = x.shape

        shortcut = x
        x = self.norm1(x)
        x = x.view(b, h, w, c)

        qkv = self.qkv(x).reshape(b, h, w, 3, c).permute(3, 0, 4, 1, 2)  # 3, b, c, h, w
        q = qkv[0].permute(0, 2, 3, 1)  # b, h, w, c
        kv = torch.cat((qkv[1], qkv[2]), dim=1)  # b, 2*c, h, w

        # partition windows
        q_windows = window_partition(q, self.window_size)  # nw*b, window_size, window_size, c
        q_windows = q_windows.view(-1, self.window_size * self.window_size, c)  # nw*b, window_size*window_size, c

        kv_windows = self.unfold(kv)  # b, c*w*w, nw
        kv_windows = rearrange(kv_windows, 'b (nc ch owh oww) nw -> nc (b nw) (owh oww) ch', nc=2, ch=c,
                               owh=self.overlap_win_size, oww=self.overlap_win_size).contiguous()  # 2, nw*b, ow*ow, c
        k_windows, v_windows = kv_windows[0], kv_windows[1]  # nw*b, ow*ow, c

        b_, nq, _ = q_windows.shape
        _, n, _ = k_windows.shape
        d = self.dim // self.num_heads
        q = q_windows.reshape(b_, nq, self.num_heads, d).permute(0, 2, 1, 3)  # nw*b, nH, nq, d
        k = k_windows.reshape(b_, n, self.num_heads, d).permute(0, 2, 1, 3)  # nw*b, nH, n, d
        v = v_windows.reshape(b_, n, self.num_heads, d).permute(0, 2, 1, 3)  # nw*b, nH, n, d

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[rpi.view(-1)].view(
            self.window_size * self.window_size, self.overlap_win_size * self.overlap_win_size,
            -1)  # ws*ws, wse*wse, nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, ws*ws, wse*wse
        attn = attn + relative_position_bias.unsqueeze(0)

        attn = self.softmax(attn)
        attn_windows = (attn @ v).transpose(1, 2).reshape(b_, nq, self.dim)

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, self.dim)
        x = window_reverse(attn_windows, self.window_size, h, w)  # b h w c
        x = x.view(b, h * w, self.dim)

        x = self.proj(x) + shortcut

        x = x + self.mlp(self.norm2(x))
        return x


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  # b Ph*Pw c
        if self.norm is not None:
            x = self.norm(x)
        return x
    def flops(self,H,W):
        total_flops= 0

        if self.norm is not None:
            total_flops += H * W * self.embed_dim
        return total_flops

class PatchUnEmbed(nn.Module):
    r""" Image to Patch Unembedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        x = x.transpose(1, 2).view(x.shape[0], self.embed_dim, x_size[0], x_size[1])  # b Ph*Pw c
        return x


class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        self.scale=scale
        self.num_feat=num_feat
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)

    def flops(self, H,W):
        flops = 0

        if (self.scale & (self.scale - 1)) == 0:
            flops += self.num_feat * 4 * self.num_feat * 9 * H * W * int(math.log(self.scale, 2))
        else:
            flops += self.num_feat * 9 * self.num_feat * 9 * H * W
        return flops

class UpsampleOneStep(nn.Sequential):
    """UpsampleOneStep module (the difference with Upsample is that it always only has 1conv + 1pixelshuffle)
       Used in lightweight SR to save parameters.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.

    """

    def __init__(self, scale, num_feat, num_out_ch, input_resolution=None):
        self.num_feat = num_feat
        self.input_resolution = input_resolution
        m = []
        m.append(nn.Conv2d(num_feat, (scale ** 2) * num_out_ch, 3, 1, 1))
        m.append(nn.PixelShuffle(scale))
        super(UpsampleOneStep, self).__init__(*m)
class GAM(Module):
    def __init__(self, in_dim):
        super(GAM, self).__init__()
        self.Conv1=nn.Conv2d(in_dim,in_dim,1,1)
        self.softmax = Softmax(dim=-1)
        self.fusion=nn.Conv2d(2*in_dim,in_dim,1,1)
        self.fusion2 = nn.Conv2d(2, 1, 1, 1)
        # self.avg=nn.AdaptiveAvgPool2d(1)
        # self.max=nn.AdaptiveMaxPool2d(1)
    def forward(self, x):
        x=x.permute(0,3,1,2).contiguous()
        b,c,h,w=x.size()
        x1=self.Conv1(x)
        x2 = self.Conv1(x)
        x3 = self.Conv1(x)
        x4 = self.Conv1(x)
        x1 = x1.view(b,c,-1)
        x2 = x2.view(b, c, -1)
        x3 = x3.view(b, c, -1)
        x4 = x4.view(b, c, -1)
        x2=x2.permute(0,2,1).contiguous()
        fmap2=torch.bmm(x2,x3)
        fmap2=self.softmax(fmap2)
        x1=torch.mean(x1,dim=1,keepdim=True)
        fmap1=torch.bmm(x1,fmap2)
        fmap1=fmap1.view(b,1,h,w)
        x4,_=torch.max(x4,dim=1,keepdim=True)
        fmap3 = torch.bmm(x4, fmap2)
        fmap3=fmap3.view(b,1,h,w)
        fmap=torch.cat([fmap1,fmap3],dim=1)
        fmap=self.fusion2(fmap)
        x=x*fmap
        x=x.permute(0,2,3,1).contiguous()
        return x
class GCAM(Module):
    """ Channel attention module"""
    def __init__(self, in_dim,reduction_ratio=16):
        super(GCAM, self).__init__()
        self.chanel_in = in_dim
        self.reduction_ratio = reduction_ratio

        # 第一层和第二层卷积，用于通道压缩和扩展
        self.conv1 = nn.Conv2d(in_dim, in_dim // reduction_ratio, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(in_dim // reduction_ratio, in_dim, kernel_size=1, padding=0)
        # self.conv3=nn.Conv2d(in_dim,in_dim,1,1)
        self.fusion=nn.Conv2d(2*in_dim,in_dim,1,1)
        self.gamma = Parameter(torch.zeros(1))
        self.softmax  = Softmax(dim=-1)
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        shortcut=x
        #x=x.permute(0,3,1,2).contiguous()
        x = F.gelu(self.conv1(x))
        x = self.conv2(x)
        # x=x.permute(0,2,3,1).contiguous()
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        #energy_new =energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)
        out = self.gamma*out + x
        out=torch.cat([out,shortcut],dim=1)
        # out = out.permute(0, 3, 1, 2).contiguous()
        out=self.fusion(out)
        # out = out.permute(0, 2, 3, 1).contiguous()
        return out

    def flops(self, H, W):
        # Initial FLOPs from the convolutions
        flops = 0
        flops += self.chanel_in * (self.chanel_in // self.reduction_ratio) * H * W  # conv1
        flops += (self.chanel_in // self.reduction_ratio) * self.chanel_in * H * W  # conv2
        flops += 2 * self.chanel_in * self.chanel_in * H * W  # fusion conv

        # Compute the FLOPs for the attention mechanism
        flops += self.chanel_in * H * W * (self.chanel_in * H * W)  # energy calculation: bmm
        flops += self.chanel_in * H * W * (self.chanel_in * H * W)  # attention application: bmm

        return flops

class ChannelAttention2(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention2, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)  # 7,3     3,1
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention2(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.ca(x)
        result = out * self.sa(out)
        return result


# class GCAM(nn.Module):
#     def __init__(self, channels, reduction_ratio=16):
#         super(GCAM, self).__init__()
#         self.channels = channels
#         self.reduction_ratio = reduction_ratio
#
#         # 第一层和第二层卷积，用于通道压缩和扩展
#         self.conv1 = nn.Conv2d(channels, channels // reduction_ratio, kernel_size=1, padding=0)
#         self.conv2 = nn.Conv2d(channels // reduction_ratio, channels, kernel_size=1, padding=0)
#         self.g = nn.Conv3d(in_channels=channels, out_channels=channels,
#                          kernel_size=1, stride=1, padding=0)
#         self.theta = nn.Conv3d(in_channels=channels, out_channels=channels,
#                              kernel_size=1, stride=1, padding=0)
#         self.phi = nn.Conv3d(in_channels=channels, out_channels=channels,
#                            kernel_size=1, stride=1, padding=0)
#         # 用于学习的尺度参数
#         self.beta = nn.Parameter(torch.zeros(1))
#
#     def forward(self, x):
#         # 输入特征图 x 的维度为 [batch_size, channels, height, width]
#         x=x.permute(0,3,1,2).contiguous()
#         # 通道注意力机制的实现
#         # Step 1: 通过卷积层处理
#         x = F.gelu(self.conv1(x))
#         x = self.conv2(x)
#         # Step 2: 计算注意力图
#         b, c, h, w = x.size()
#         self.g = self.g.to(x.device)
#         self.theta = self.theta.to(x.device)
#         self.phi = self.phi.to(x.device)
#         #g_x:CxN,theta:NxC,phi:CxN
#         g_x = self.g(x).view(b, c, -1)
#         theta = self.theta(x).view(b, c, -1)
#         theta = theta.permute(0, 2, 1).contiguous()
#         phi = self.phi(x).view(b, c, -1)
#         f=torch.matmul(phi,theta)
#         fmap=F.softmax(f,dim=-1)
#         fmap=torch.matmul(fmap,g_x)
#         f=fmap.view(b,c,h,w)
#         x=f+x
#         x=x.permute(0,2,3,1).contiguous()
#
#         return x

# class FourierUnit(nn.Module):
#     def __init__(self, embed_dim, fft_norm='ortho'):
#         # bn_layer not used
#         super(FourierUnit, self).__init__()
#         self.conv_layer = torch.nn.Conv2d(embed_dim * 2, embed_dim * 2, 1, 1, 0)
#         self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
#         self.depthwise=nn.Conv2d(2*embed_dim, 2*embed_dim, groups=2*embed_dim, kernel_size=3, stride=1, padding=1)
#         self.pointwise=nn.Conv2d(2*embed_dim, 2*embed_dim,1,1)
#         self.fusion = nn.Conv2d(4*embed_dim, 2*embed_dim, 1, 1)
#         self.fft_norm = fft_norm
#
#     def forward(self, x):
#         batch = x.shape[0]
#
#         r_size = x.size()
#         # (batch, c, h, w/2+1, 2)
#         fft_dim = (-2, -1)
#         ffted = torch.fft.rfftn(x, dim=fft_dim, norm=self.fft_norm)
#         ffted = torch.stack((ffted.real, ffted.imag), dim=-1)
#         ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()  # (batch, c, 2, h, w/2+1)
#         ffted = ffted.view((batch, -1,) + ffted.size()[3:])
#
#         ffted = self.conv_layer(ffted)  # (batch, c*2, h, w/2+1)
#         ffted = self.relu(ffted)
#         ftfed1=self.depthwise(ffted)
#         ftfed2=self.pointwise(ffted)
#         ffted=torch.cat([ftfed1,ftfed2],dim=1)
#         ffted=self.fusion(ffted)
#         ffted=self.relu(ffted)
#
#         ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(0, 1, 3, 4,
#                                                                        2).contiguous()  # (batch,c, t, h, w/2+1, 2)
#         ffted = torch.complex(ffted[..., 0], ffted[..., 1])
#
#         ifft_shape_slice = x.shape[-2:]
#         output = torch.fft.irfftn(ffted, s=ifft_shape_slice, dim=fft_dim, norm=self.fft_norm)
#
#         return output
class FourierUnit(nn.Module):
    def __init__(self, embed_dim, fft_norm='ortho'):
        # bn_layer not used
        super(FourierUnit, self).__init__()
        self.conv_layer = torch.nn.Conv2d(embed_dim * 2, embed_dim * 2, 1, 1, 0)
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.depthwise=nn.Conv2d(2*embed_dim, 2*embed_dim, groups=2*embed_dim, kernel_size=3, stride=1, padding=1)
        self.depthwise2 = nn.Conv2d(2, 2, groups=2, kernel_size=3, stride=1,
                                   padding=1)
        self.pointwise=nn.Conv2d(2*embed_dim, 2*embed_dim,1,1)
        self.fusion = nn.Conv2d(4*embed_dim, 2*embed_dim, 1, 1)
        self.fusion2 = nn.Conv2d(2, 1, 1, 1)
        self.fft_norm = fft_norm
        self.sig=nn.Sigmoid()
    def forward(self, x):
        batch= x.shape[0]

        r_size = x.size()
        # (batch, c, h, w/2+1, 2)
        fft_dim = (-2, -1)
        ffted = torch.fft.rfftn(x, dim=fft_dim, norm=self.fft_norm)
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()  # (batch, c, 2, h, w/2+1)
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])

        ffted = self.conv_layer(ffted)  # (batch, c*2, h, w/2+1)
        ffted = self.relu(ffted)
        ftfed1=self.depthwise(ffted)
        ftfed2=self.pointwise(ffted)
        ffted=torch.cat([ftfed1,ftfed2],dim=1)
        ffted=self.fusion(ffted)
        avg_attn = torch.mean(ffted, dim=1, keepdim=True)
        max_attn, _ = torch.max(ffted, dim=1, keepdim=True)
        agg = torch.cat([avg_attn, max_attn], dim=1)
        agg=self.depthwise2(agg)
        agg=self.relu(agg)
        agg=self.fusion2(agg)
        # agg=self.relu(agg)
        agg=self.sig(agg)
        ffted=ffted*agg

        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(0, 1, 3, 4,
                                                                       2).contiguous()  # (batch,c, t, h, w/2+1, 2)
        ffted = torch.complex(ffted[..., 0], ffted[..., 1])

        ifft_shape_slice = x.shape[-2:]
        output = torch.fft.irfftn(ffted, s=ifft_shape_slice, dim=fft_dim, norm=self.fft_norm)
        return output

    def flops(self, H, W):
        # H and W are the height and width of the input feature maps
        total_flops = 0
        # Compute FLOPs for each Conv2d layer
        # 1x1 convolutions
        total_flops += (self.conv_layer.in_channels * self.conv_layer.out_channels * H * (W // 2 + 1))
        total_flops += (self.depthwise.in_channels * 3 * 3 * H * (W // 2 + 1))
        total_flops += (self.pointwise.in_channels * self.pointwise.out_channels * H * (W // 2 + 1))
        total_flops += (self.fusion.in_channels * self.fusion.out_channels * H * (W // 2 + 1))
        total_flops += (self.fusion2.in_channels * self.fusion2.out_channels * H * (W // 2 + 1))

        # Assuming the FFT and IFFT are performed on the last two dimensions
        # FFT and IFFT FLOPs: Typically 5 * N * log(N) for each FFT or IFFT operation
        fft_flops = 5 * (H * (W // 2 + 1)) * (math.log(H) + math.log(W // 2 + 1))
        # FLOPs for FFT and IFFT each
        total_flops += fft_flops  # FFT
        total_flops += fft_flops  # IFFT

        return total_flops

class SpectralTransform(nn.Module):
    def __init__(self, embed_dim, last_conv=False):
        # bn_layer not used
        super(SpectralTransform, self).__init__()
        self.last_conv = last_conv

        self.conv1 = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // 2, 1, 1, 0),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.fu = FourierUnit(embed_dim // 2)

        self.conv2 = torch.nn.Conv2d(embed_dim // 2, embed_dim, 1, 1, 0)

        if self.last_conv:
            self.last_conv = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        output = self.fu(x)
        output = self.conv2(x + output)
        if self.last_conv:
            output = self.last_conv(output)
        return output

    def flops(self, H, W):
        total_flops = 0
        # conv1 FLOPs
        total_flops += self.conv1[0].in_channels * self.conv1[0].out_channels * H * W

        # FourierUnit FLOPs
        total_flops += self.fu.flops(H, W)

        # conv2 FLOPs
        total_flops += self.conv2.in_channels * self.conv2.out_channels * H * W

        # last_conv FLOPs
        if self.last_conv is not None:
            total_flops += self.conv2.out_channels * self.conv2.out_channels * 3 * 3 * H * W

        return total_flops
# class ConvGroup(nn.Module):
#     def  __init__(self, embed_dim):
#         super(ConvGroup, self).__init__()
#         self.conv11=nn.Conv2d(embed_dim,embed_dim,1,1)
#         self.conv12=nn.Conv2d(2*embed_dim,embed_dim,1,1)
#         self.conv13 = nn.Conv2d(3 * embed_dim, embed_dim, 1, 1)
#         self.conv3 = nn.Conv2d( embed_dim, embed_dim, 3, 1, 1)
#         self.conv5 = nn.Conv2d( embed_dim, embed_dim, 5, 1, 2)
#         self.act=nn.ReLU(inplace=True)
#     def __call__(self, x):
#         x11=self.conv11(x)
#         x12=self.conv11(x11)
#         x13=self.conv3(x11)
#         x1=x12+x13
#         x1=self.act(x1)
#         x21 = self.conv11(x)
#         x22 = self.conv3(x21)
#         x23 = self.conv5(x21)
#         x2 = x22+x23
#         x2 = self.act(x2)
#         x31 = self.conv11(x)
#         x32 = self.conv11(x31)
#         x33 = self.conv5(x31)
#         x3 = x32+x33
#         x3 = self.act(x3)
#         x3=torch.cat([x1,x2,x3],dim=1)
#         x3=self.conv13(x3)
#         return x3
#
# class ResBN(nn.Module):
#     def  __init__(self, embed_dim):
#         super(ResBN, self).__init__()
#         self.conv12 = nn.Conv2d(2 * embed_dim, embed_dim, 1, 1)
#         self.conv3 = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
#         self.act = nn.ReLU(inplace=True)
#
#     def __call__(self, x):
#         shortcut=x
#         x=self.conv3(x)
#         x=self.act(x)
#         x=self.conv3(x)
#         x=torch.cat([x,shortcut],dim=1)
#         x=self.conv12(x)
#         x = self.act(x)
#         shortcut2 = x
#         x = self.conv3(x)
#         x = self.act(x)
#         x = self.conv3(x)
#         x = torch.cat([x, shortcut2], dim=1)
#         x = self.conv12(x)
#         x = self.act(x)
#         x=self.conv3(x)
#         x=torch.cat([x,shortcut],dim=1)
#         x = self.conv12(x)
#         return x

# class Freq_block(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.dim = dim
#         self.dw_amp_conv = nn.Sequential(
#             nn.Conv2d(dim, dim, groups=dim, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0),
#             nn.ReLU()
#         )
#         self.df1 = nn.Sequential(
#             nn.Conv2d(2, 2, groups=2, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(2, 1, kernel_size=1, stride=1, padding=0),
#             nn.Sigmoid()
#         )
#         self.df2 = nn.Sequential(
#             nn.Conv2d(2, 2, groups=2, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(2, 1, kernel_size=1, stride=1, padding=0),
#             nn.Sigmoid()
#         )
#         self.dw_pha_conv = nn.Sequential(
#             nn.Conv2d(dim*2, dim*2, groups=dim*2, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(dim*2, dim, kernel_size=1, stride=1, padding=0),
#             nn.Sigmoid()
#             )
#
#     def forward(self, x):
#         b,c,h,w = x.shape
#         msF = torch.fft.rfft2(x+1e-8, dim=(-2, -1))
#         msF = torch.cat([
#             msF[:, :, msF.size(2) // 2 + 1:, :],
#             msF[:, :, :msF.size(2) // 2 + 1, :]], dim=2)
#         # msF = torch.fft.fftshift(msF, dim=(-2, -1))
#         msF_amp = torch.abs(msF)
#         msF_pha = torch.angle(msF)
#
#         amp_fuse = self.dw_amp_conv(msF_amp)
#         avg_attn = torch.mean(amp_fuse, dim=1, keepdim=True)
#         max_attn, _ = torch.max(amp_fuse, dim=1, keepdim=True)
#         agg = torch.cat([avg_attn, max_attn], dim=1)
#         agg=self.df1(agg)
#         amp_fuse=amp_fuse*agg
#         amp_res = amp_fuse - msF_amp
#         pha_guide=torch.cat((msF_pha,amp_res),dim=1)
#         pha_fuse = self.dw_pha_conv(pha_guide)
#         avg_attn = torch.mean(pha_fuse, dim=1, keepdim=True)
#         max_attn, _ = torch.max(pha_fuse, dim=1, keepdim=True)
#         agg = torch.cat([avg_attn, max_attn], dim=1)
#         agg = self.df2(agg)
#         pha_fuse = pha_fuse * agg
#         #这一行将处理过的相位信息pha_fuse的范围调整到了[−π,π]。
#         # 这是因为相位值通常在这个范围内，而处理后的相位值可能超出这个范围，
#         # 所以需要通过这种方式进行调整以确保其有效性
#         pha_fuse=pha_fuse*(2.*math.pi)-math.pi
#         # pha_fuse = torch.clamp(pha_fuse, -math.pi, math.pi)
#         ## amp_fuse = amp_fuse + msF_amp
#         # pha_fuse = pha_fuse + msF_pha
#         #使用调整后的振幅amp_fuse和相位pha_fuse来构建复数表示的频率信息。实部由amp_fuse * torch.cos(pha_fuse)给出，
#         #虚部由amp_fuse * torch.sin(pha_fuse)给出，这两部分结合生成了复数形式的频率域表示out
#         real = amp_fuse * torch.cos(pha_fuse)
#         imag = amp_fuse * torch.sin(pha_fuse)
#         out = torch.complex(real, imag)
#         # out=torch.fft.ifftshift(out, dim=(-2, -1))
#         #对逆变换后的结果进行调整，以确保它与原始图像的空间尺寸一致
#         out = torch.cat([
#             out[:, :, out.size(2) // 2 - 1:, :],
#             out[:, :, :out.size(2) // 2 - 1, :]], dim=2)
#         #用于获取复数结果的模，因为最终的图像应该是实值的
#         out = torch.abs(torch.fft.irfft2(out+1e-8, s=(h, w)))
#         if torch.isnan(out).sum()>0:
#             print('freq feature include NAN!!!!')
#             assert torch.isnan(out).sum() == 0
#             out = torch.nan_to_num(out, nan=1e-5, posinf=1e-5, neginf=1e-5)
#         out = out + x
#         return F.relu(out)
## Residual Block (RB)
class ResB(nn.Module):
    def __init__(self, embed_dim, red=1):
        super(ResB, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // red, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(embed_dim // red, embed_dim, 3, 1, 1),
        )

        self.CA=ChannelAttention(num_feat=embed_dim,squeeze_factor=16)
        self.fusion=nn.Conv2d(2*embed_dim,embed_dim,1,1)
    def __call__(self, x):
        shortcut=x
        x=self.body(x)
        x=self.CA(x)
        x=torch.cat([x,shortcut],dim=1)
        x=self.fusion(x)
        # out=self.body(x)
        return x

    def flops(self, H, W):
        total_flops = 0
        # Calculate FLOPs for each conv layer in the body
        total_flops += self.body[0].in_channels * self.body[0].out_channels * 3 * 3 * H * W  # first conv
        total_flops += self.body[2].in_channels * self.body[2].out_channels * 3 * 3 * H * W  # second conv

        # Calculate FLOPs for the ChannelAttention
        total_flops += self.CA.flops(H, W)

        # Calculate FLOPs for the fusion conv layer
        total_flops += self.fusion.in_channels * self.fusion.out_channels * H * W  # 1x1 conv

        return total_flops
def save_feature_maps(feature_maps, filename, folder_path='/tmp'):
    print("Running save_feature_maps with gray cmap")  # 打印确认信息

    # 构建完整的文件保存路径
    full_path = os.path.join(folder_path, filename)
    # 确保目标文件夹存在
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # 转换特征映射到 CPU 并且从 Tensor 转换为 NumPy 数组
    feature_maps = feature_maps.detach().cpu()
    img = feature_maps[0, 0].numpy()  # 假设查看批次中第一个图像的第一个通道

    # 创建图像并保存
    plt.figure()
    plt.imshow(img, cmap='viridis')
    plt.colorbar()
    plt.title("Feature Map Visualization")
    plt.axis('off')
    print(f"Saving image to {full_path} with gray cmap")  # 打印保存信息
    plt.savefig(full_path)  # 保存图像到指定路径
    plt.close()  # 关闭图像以释放资源

class SFB(nn.Module):
    def __init__(self, embed_dim, red=1):
        super(SFB, self).__init__()
        self.S = ResB(embed_dim, red)
        self.F = SpectralTransform(embed_dim)
        self.fusion = nn.Conv2d(embed_dim * 2, embed_dim, 1, 1, 0)

    def __call__(self, x):
        s = self.S(x)
        f = self.F(x)
        out = torch.cat([s, f], dim=1)
        out = self.fusion(out)
        # out=f
        return out

    def flops(self, H, W):
        total_flops = 0
        # Calculate FLOPs for ResB
        total_flops += self.S.flops(H, W)
        # Calculate FLOPs for SpectralTransform
        total_flops += self.F.flops(H, W)
        # Calculate FLOPs for the fusion conv layer
        total_flops += self.fusion.in_channels * self.fusion.out_channels * H * W  # 1x1 conv

        return total_flops