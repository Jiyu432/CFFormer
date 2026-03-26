

# CFFormer: Channel Fourier Transformer for Remote Sensing Super Resolution

Zexin Xie , Jian Wang , Wei Song , Yanling Du , Huifang Xu , and Qinhan Yang

This job opportunity arises from the paper "HAMD-RSISR: Hybrid Attention and Multi-Dictionary for Remote Sensing Super-Resolution",which has been published on IEEE JSTARS  [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10742930)].
The improvement is primarily based on "SwinFIR: Revisiting the SwinIR with Fast Fourier Convolution and Improved Training for Image Super-Resolution" .  [[arXiv](https://arxiv.org/abs/2208.11247)]
Given my limited coding proficiency, I am submitting this as a record. It should be noted that this code cannot be used for professional learning or work
## Abstract
> The objective of super-resolution in remote sensing imagery is to enhance low-resolution images to recover high-quality details. With the rapid progress of deep learning technology, the deep learning-based super-resolution technology for remote sensing images has also made remarkable achievements. However, these methods encounter several challenges. They often struggle with processing long-range spatial information that encompasses complex scene changes, adversely affecting the image’s coherence and accuracy. Furthermore, the lack of connectivity in feature extraction blocks hinders effective feature utilization in deeper network layers, leading to issues such as gradient vanishing and exploding. Additionally, constraints in the spatial domain of previous methods frequently result in severe shape distortion and blurring. To address these issues, this study proposes the CFFormer, a new super-resolution framework that employs the Swin Transformer as its core architecture and incorporates the Channel Fourier Block (CFB) to refine features in the frequency domain. The Global Attention Block (GAB) is also integrated to enhance global information capture, thereby improving the extraction of spatial features. To increase model stability and feature utilization efficiency, a Jump Joint Fusion Mechanism is designed, culminating in a Residual Fusion Swin Transformer Block (RFSTB) that alleviates the gradient vanishing issue and optimizes feature reuse. Experimental results confirm the CFFormer’s superior performance in remote sensing image reconstruction, demonstrating outstanding perceptual quality and reliability. Notably, the CFFormer achieves a Peak Signal-to-Noise Ratio (PSNR) of 29.83 dB on the UcMercedx4 dataset, surpassing the SwinIR method by approximately 0.5 dB,indicating a substantial enhancement

## Pretrained Models
We provide the pre-trained model for HAMD-RSISR. You can download it from the following link:
- **Baidu Drive **: [link](https://pan.baidu.com/s/1vWqgQFWYuCkgvS03_-hTkQ?pwd=1544) 
Download the `.pth` file and place it in your designated model directory...


## Environment
- [PyTorch >= 1.7](https://pytorch.org/)
- [BasicSR == 1.3.5](https://github.com/XPixelGroup/BasicSR/blob/master/INSTALL.md) 

## database
The dataset can be downloaded from here 
- **Baidu Drive **: [link](https://pan.baidu.com/s/17kDP6AGSBh6SL8EasCgtpA?pwd=1544) 

The dataset is sourced from Uc Merced:
Yi Yang and Shawn Newsam, "Bag-Of-Visual-Words and Spatial Extensions for Land-Use Classification," ACM SIGSPATIAL International Conference on Advances in Geographic Information Systems (ACM GIS), 2010.
Shawn D. Newsam
Assistant Professor and Founding Faculty
Electrical Engineering & Computer Science
University of California, Merced
Email: snewsam@ucmerced.edu
Web: http://faculty.ucmerced.edu/snewsam/






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
