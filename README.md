

# CFFormer: Channel Fourier Transformer for Remote Sensing Super Resolution

Zexin Xie , Jian Wang , Wei Song , Yanling Du , Huifang Xu , and Qinhan Yang

> The objective of super-resolution in remote sensing imagery is to enhance low-resolution images to recover high-quality details. With the rapid progress of deep learning technology, the deep learning-based super-resolution technology for remote sensing images has also made remarkable achievements. However, these methods encounter several challenges. They often struggle with processing long-range spatial information that encompasses complex scene changes, adversely affecting the image’s coherence and accuracy. Furthermore, the lack of connectivity in feature extraction blocks hinders effective feature utilization in deeper network layers, leading to issues such as gradient vanishing and exploding. Additionally, constraints in the spatial domain of previous methods frequently result in severe shape distortion and blurring. To address these issues, this study proposes the CFFormer, a new super-resolution framework that employs the Swin Transformer as its core architecture and incorporates the Channel Fourier Block (CFB) to refine features in the frequency domain. The Global Attention Block (GAB) is also integrated to enhance global information capture, thereby improving the extraction of spatial features. To increase model stability and feature utilization efficiency, a Jump Joint Fusion Mechanism is designed, culminating in a Residual Fusion Swin Transformer Block (RFSTB) that alleviates the gradient vanishing issue and optimizes feature reuse. Experimental results confirm the CFFormer’s superior performance in remote sensing image reconstruction, demonstrating outstanding perceptual quality and reliability. Notably, the CFFormer achieves a Peak Signal-to-Noise Ratio (PSNR) of 29.83 dB on the UcMercedx4 dataset, surpassing the SwinIR method by approximately 0.5 dB,indicating a substantial enhancement

## Pretrained Models
We provide the pre-trained model for HAMD-RSISR. You can download it from the following link:
- **Baidu Drive **: [link](https://pan.baidu.com/s/1vWqgQFWYuCkgvS03_-hTkQ?pwd=1544) 
Download the `.pth` file and place it in your designated model directory...


## Environment
- [PyTorch >= 1.7](https://pytorch.org/)
- [BasicSR == 1.3.5](https://github.com/XPixelGroup/BasicSR/blob/master/INSTALL.md) 








## Citation
If you find this project useful for your research, please consider citing:
~~~
@article{xie2024hamd,
  title={CFFormer: Channel Fourier Transformer for Remote Sensing Super Resolution},
  author={Zexin Xie , Jian Wang , Wei Song , Yanling Du , Huifang Xu , and Qinhan Yang},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing},
  year={2024},
  publisher={IEEE}
}
~~~
## Acknowledgement
This project is mainly based on [BasicSR](https://github.com/XPixelGroup/BasicSR) and [SwinFIR](https://arxiv.org/abs/2208.11247).
