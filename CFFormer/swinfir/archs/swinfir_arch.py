import os
import time

import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from basicsr.archs.arch_util import to_2tuple, trunc_normal_
from basicsr.utils.registry import ARCH_REGISTRY

from swinfir.archs.swinfir_utils import PatchEmbed, PatchUnEmbed, Upsample, UpsampleOneStep
from swinfir.archs.swinfir_utils import WindowAttention, DropPath, Mlp, SFB, GCAM,save_feature_maps
from swinfir.archs.swinfir_utils import window_partition, window_reverse


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

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
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 conv_scale=0.01,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.conv_scale=conv_scale
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
        self.attn = WindowAttention(
            dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            attn_mask = self.calculate_mask(self.input_resolution)
        else:
            attn_mask = None

        self.register_buffer('attn_mask', attn_mask)
        # self.GCAM = GCAM(in_dim=dim)
        # self.GA=GAM(in_dim=dim)
    def calculate_mask(self, x_size):
        # calculate attention mask for SW-MSA
        h, w = x_size
        img_mask = torch.zeros((1, h, w, 1))  # 1 h w 1
        h_slices = (slice(0, -self.window_size), slice(-self.window_size,
                                                       -self.shift_size), slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size), slice(-self.window_size,
                                                       -self.shift_size), slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nw, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def forward(self, x, x_size):
        h, w = x_size
        b, _, c = x.shape
        # assert seq_len == h * w, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(b, h, w, c)
        # x_GCA = self.GCAM(x)
        # x_GCA = x_GCA.permute(0, 2, 3, 1).contiguous().view(b, h * w, c)
        # x_GA=self.GA(x)
        # x_GA = x_GA.permute(0, 2, 3, 1).contiguous().view(b, h * w, c)
        # # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x


        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nw*b, window_size, window_size, c
        x_windows = x_windows.view(-1, self.window_size * self.window_size, c)  # nw*b, window_size*window_size, c

        # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size
        if self.input_resolution == x_size:
            attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nw*b, window_size*window_size, c
        else:
            attn_windows = self.attn(x_windows, mask=self.calculate_mask(x_size).to(x.device))

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, c)
        shifted_x = window_reverse(attn_windows, self.window_size, h, w)  # b h' w' c

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(b, h * w, c)

        # FFN

        #x = shortcut + self.drop_path(x)+x_GCA*self.conv_scale
        x = shortcut + self.drop_path(x)

        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x
    def flops(self, H, W):
        total_flops = 0

        # Norm layers
        total_flops += 2 * self.dim * H * W * 2  # norm1 and norm2

        # Window attention module
        total_flops += self.attn.flops(H*W)

        # MLP module
        mlp_hidden_dim = int(self.dim * self.mlp_ratio)
        total_flops += self.dim * mlp_hidden_dim * H * W  # first dense layer in MLP
        total_flops += mlp_hidden_dim * self.dim * H * W  # second dense layer in MLP

        return total_flops
    def extra_repr(self) -> str:
        return (f'dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, '
                f'window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}')


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 depth,
                 num_heads,
                 window_size,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        # self.shift_size_outputs = []  # 用于存储 shift_size=0 的层的输出
        # self.shift_size_outputs_withshift=[]
        # build blocks
        self.blocks = nn.ModuleList()
        for i in range(depth):
            shift_size = 0 if (i % 2 == 0) else window_size // 2
            block = SwinTransformerBlock(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=shift_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer)
            self.blocks.append(block)

        # downsample layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None
        # self.CA=ChannelAttention2(in_planes=dim,ratio=16)
        # self.SA=SpatialAttention()
        self.y=None
        self.z=None
        self.w=None
        self.fusion=nn.Conv2d(2*dim,dim,1,1)
    def forward(self, x, x_size):
        b,_,c=x.size()
        h,w=x_size

        for i, blk in enumerate(self.blocks):
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x, x_size)
                if i == 0:  # shift_size==0 的层
                    self.y=x  # 存储当前层的输出以用于下一个 shift_size=0 的层
                if i == 1:
                    self.z=x
                if i == 2:
                    self.w=x
                if i == 3:
                    x =torch.cat([x,self.y],dim=2)
                    x=x.view(b,h,w,2*c)
                    x=x.permute(0,3,1,2).contiguous()
                    x=self.fusion(x)
                    x = x.permute(0, 2,3,1).contiguous()
                    x=x.view(b, h*w, c)
                if i ==4:
                    x = torch.cat([x, self.z], dim=2)
                    x=x.view(b, h, w, 2*c)
                    x = x.permute(0, 3, 1, 2).contiguous()
                    x = self.fusion(x)
                    x = x.permute(0, 2, 3, 1).contiguous()
                    x=x.view(b, h * w,c)
                if i ==5:
                    x = torch.cat([x, self.w], dim=2)
                    x=x.view(b, h, w, 2*c)
                    x = x.permute(0, 3, 1, 2).contiguous()
                    x = self.fusion(x)
                    x = x.permute(0, 2, 3, 1).contiguous()
                    x=x.view(b, h * w,c)

        if self.downsample is not None:
            x = self.downsample(x)
        return x
    # def forward(self, x, x_size):
    #     for blk in self.blocks:
    #         if self.use_checkpoint:
    #             x = checkpoint.checkpoint(blk, x)
    #         else:
    #             x = blk(x, x_size)
    #     if self.downsample is not None:
    #         x = self.downsample(x)
    #     return x
    def flops(self, H, W):
        total_flops = 0

        # Sum flops for all SwinTransformerBlocks
        for block in self.blocks:
            total_flops += block.flops(H, W)

        # If downsample is used, calculate its flops
        if self.downsample is not None:
            if isinstance(self.downsample, nn.Sequential):
                # Assuming the downsample layer is a conv layer followed by a norm layer
                for layer in self.downsample:
                    if isinstance(layer, nn.Conv2d):
                        total_flops += layer.in_channels * layer.out_channels * layer.kernel_size[0] * layer.kernel_size[1] * (H // 2) * (W // 2)
            else:
                # Direct method call if flops calculation is defined
                total_flops += self.downsample.flops(H, W)

        # Fusion convolutions
        # Assuming the number of these fusion steps is defined by some structural logic of the network
        fusion_steps = 3  # Example: 3 fusion steps as seen in the forward method
        for _ in range(fusion_steps):
            total_flops += 180 * 2 * 180 * H * W  # 2*dim because of cat operation, and then reduced by 1x1 conv

        return total_flops
    def extra_repr(self) -> str:
        return f'dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}'


class RSTB(nn.Module):
    """Residual Swin Transformer Block (RSTB).
x
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        img_size: Input image size.
        patch_size: Patch size.
        resi_connection: The convolutional block before residual connection.
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 depth,
                 num_heads,
                 window_size,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 img_size=224,
                 patch_size=4,
                 resi_connection='1conv'):
        super(RSTB, self).__init__()

        self.dim = dim
        self.input_resolution = input_resolution

        self.residual_group = BasicLayer(
            dim=dim,
            input_resolution=input_resolution,
            depth=depth,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=drop_path,
            norm_layer=norm_layer,
            downsample=downsample,
            use_checkpoint=use_checkpoint)

        if resi_connection == '1conv':
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == 'SFB':
            self.conv = SFB(dim)
        elif resi_connection == 'HSFB':
            self.conv = SFB(dim, 2)
        elif resi_connection == 'identity':
            self.conv = nn.Identity()

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim, norm_layer=None)

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim, norm_layer=None)

    def forward(self, x, x_size):
        return self.patch_embed(self.conv(self.patch_unembed(self.residual_group(x, x_size), x_size))) + x
    def flops(self, H, W):
        total_flops = 0

        # Calculate FLOPs for the residual group
        total_flops += self.residual_group.flops(H, W)

        # Calculate FLOPs for the convolution in residual connection
        if isinstance(self.conv, nn.Conv2d):
            total_flops += (self.conv.in_channels * self.conv.out_channels *
                            self.conv.kernel_size[0] * self.conv.kernel_size[1] * H * W)
        elif hasattr(self.conv, 'flops'):
            total_flops += self.conv.flops(H, W)  # Assuming a complex block like SFB

        # Calculate FLOPs for patch embedding and unembedding
        if hasattr(self.patch_embed, 'flops'):
            total_flops += self.patch_embed.flops(H,W)
        if hasattr(self.patch_unembed, 'flops'):
            total_flops += self.patch_unembed.flops(H, W)

        return total_flops

@ARCH_REGISTRY.register()
class SwinFIR(nn.Module):
    r""" SwinFIR
        A PyTorch impl of : `SwinFIR: Revisiting the SwinIR with Fast Fourier Convolution and
        Improved Training for Image Super-Resolution`, based on Swin Transformer and Fast Fourier Convolution.

    Args:
        img_size (int | tuple(int)): Input image size. Default 64
        patch_size (int | tuple(int)): Patch size. Default: 1
        in_chans (int): Number of input image channels. Default: 3
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        upscale: Upscale factor. 2/3/4/8 for image SR, 1 for denoising and compress artifact reduction
        img_range: Image range. 1. or 255.
        upsampler: The reconstruction reconstruction module. 'pixelshuffle'/'pixelshuffledirect'/'nearest+conv'/None
        resi_connection: The convolutional block before residual connection. '1conv'/'3conv'
    """

    def __init__(self,
                 img_size=64,
                 patch_size=1,
                 in_chans=3,
                 embed_dim=96,
                 depths=(6, 6, 6, 6),
                 num_heads=(6, 6, 6, 6),
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 patch_norm=True,
                 use_checkpoint=False,
                 upscale=2,
                 img_range=1.,
                 upsampler='',
                 resi_connection='SFB',
                 **kwargs):
        super(SwinFIR, self).__init__()
        num_in_ch = in_chans
        num_out_ch = in_chans
        num_feat = 64
        self.img_range = img_range
        if in_chans == 3:
            rgb_mean = (0.3014, 0.3152, 0.3094)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        self.upscale = upscale
        self.upsampler = upsampler

        # ------------------------- 1, shallow feature extraction ------------------------- #
        self.conv_first = nn.Sequential(
            nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1),
            GCAM(in_dim=embed_dim, reduction_ratio=8))

        # ------------------------- 2, deep feature extraction ------------------------- #
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # merge non-overlapping patches into image
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build Residual Swin Transformer blocks (RSTB)
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = RSTB(
                dim=embed_dim,
                input_resolution=(patches_resolution[0], patches_resolution[1]),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],  # no impact on SR results
                norm_layer=norm_layer,
                downsample=None,
                use_checkpoint=use_checkpoint,
                img_size=img_size,
                patch_size=patch_size,
                resi_connection=resi_connection)
            self.layers.append(layer)
        self.norm = norm_layer(self.num_features)

        # build the last conv layer in deep feature extraction
        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)

        # ------------------------- 3, high quality image reconstruction ------------------------- #
        if self.upsampler == 'pixelshuffle':
            # for classical SR
            self.conv_before_upsample = nn.Sequential(
                GCAM(in_dim=embed_dim, reduction_ratio=8),
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True))
            self.upsample = Upsample(upscale, num_feat)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        elif self.upsampler == 'pixelshuffledirect':
            # for lightweight SR (to save parameters)
            self.upsample = UpsampleOneStep(upscale, embed_dim, num_out_ch,
                                            (patches_resolution[0], patches_resolution[1]))
        elif self.upsampler == 'nearest+conv':
            # for real-world SR (less artifacts)
            assert self.upscale == 4, 'only support x4 now.'
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True))
            self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        else:
            # for image denoising and JPEG compression artifact reduction
            self.conv_last = nn.Conv2d(embed_dim, num_out_ch, 3, 1, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x):
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x, x_size)

        x = self.norm(x)  # b seq_len c
        x = self.patch_unembed(x, x_size)

        return x

    def forward(self, x):
        input = x.clone()  # 保存输入的副本
        self.mean = self.mean.type_as(x)  # 确保mean的数据类型与输入相同
        x = (x - self.mean) * self.img_range  # 标准化输入

        # GCAM特征提取
        x = self.conv_first(x)  # 经过GCAM层
        # 保存GCAM输出的特征映射
        # save_feature_maps(x, "完整的buildings64.png", '/tmp/pycharm_project_swinfirlib3/assets')

        if self.upsampler == 'pixelshuffle':
            # for classical SR
            # x = self.conv_first(x)
            s=self.forward_features(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            # x = self.forward_features(x)
            # save_feature_maps(s, "完整的buildings64尾部.png", '/tmp/pycharm_project_swinfirlib3/assets')
            # x = self.conv_after_body(x) + x
            x = self.conv_before_upsample(x)

            x = self.conv_last(self.upsample(x))
        elif self.upsampler == 'pixelshuffledirect':
            # for lightweight SR
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.upsample(x)
        elif self.upsampler == 'nearest+conv':
            # for real-world SR
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.conv_before_upsample(x)
            x = self.lrelu(self.conv_up1(torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')))
            x = self.lrelu(self.conv_up2(torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')))
            x = self.conv_last(self.lrelu(self.conv_hr(x)))
        else:
            # for image denoising and JPEG compression artifact reduction
            x_first = self.conv_first(x)
            res = self.conv_after_body(self.forward_features(x_first)) + x_first
            x = x + self.conv_last(res)

        x = x / self.img_range + self.mean

        return x

    def flops(self, H,W):

        flops = 0
        flops += H * W * 3 * self.embed_dim * 9
        flops += self.patch_embed.flops(H,W)
        for layer in self.layers:
            flops += layer.flops(H,W)
        flops += H * W * 3 * self.embed_dim * self.embed_dim
        if self.upsampler == 'pixelshuffle':
            flops += self.upsample.flops(H,W)
        else:
            flops += self.upsample.flops(H,W)

        return flops

def print_memory_usage():
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    print(f'Current memory usage: {memory_info.rss / (1024 ** 2):.2f} MB')  # RSS: Resident Set Size

if __name__ == '__main__':

    model = SwinFIR(
        upscale=2,
        in_chans=3,
        img_size=60,
        embed_dim=180,
        depths=[6, 6, 6, 6, 6, 6],
        num_heads=[6, 6, 6, 6, 6, 6],
        window_size=12,
        img_range=1.,
        mlp_ratio=2,
        upsampler='pixelshuffle',
        resi_connection="SFB")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    dummy_input = torch.randn(1, 3, 60, 60, device=device)

    # Warm-up
    for _ in range(10):
        output = model(dummy_input)

    # Measure execution time
    start_time = time.time()
    iterations = 100
    for _ in range(iterations):
        output = model(dummy_input)
    torch.cuda.synchronize()  # Wait for all CUDA kernels to finish
    end_time = time.time()

    # Calculate and print FPS
    duration = (end_time - start_time) / iterations
    fps = 1 / duration
    print(f'Average execution time per image: {duration:.3f} seconds')
    print(f'Average FPS: {fps:.2f}')

    # Model Size
    total = sum([param.nelement() for param in model.parameters()])
    print(f"Number of parameter: {total / 1e6:.3f}M")

    # Memory usage
    print(f"Memory allocated: {torch.cuda.memory_allocated(device) / 1e6:.2f} MB")
    print(f"Max memory allocated: {torch.cuda.max_memory_allocated(device) / 1e6:.2f} MB")

    # Calculate FLOPs
    print(128, 128, model.flops(128,128) / 1e9, 'G')
    print(256, 256, model.flops(256,256) / 1e9, 'G')
    print(64, 64, model.flops(60,60) / 1e9, 'G')
    # # Test
    # _input = torch.randn([2, 3, 64, 64])
    # output = model(_input)
    # print(output.shape)
