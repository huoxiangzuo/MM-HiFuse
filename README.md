# MM-HiFuse
![](https://img.shields.io/github/license/huoxiangzuo/MM-HiFuse)  
This repo. is the official implementation of [MM-HiFuse: Multi-modal Multi-task Hierarchical Feature Fusion for Esophageal Cancer Staging and Differentiation Classification](https://link.springer.com/article/10.1007/s40747-024-01708-5)    
Authors: Xiangzuo Huo, Shengwei Tian, Long Yu, Wendong Zhang, Aolun Li, Qimeng Yang & Jinmiao Song.  
Enjoy the code and find its convenience to produce more awesome works!

## Overview
<img width="1395" alt="figure1" src="https://github.com/user-attachments/assets/0cb21fa1-5f41-4855-b6b9-fa7ba41c90b7" width="80%">

## Multi-modal Stem
<img src="https://github.com/user-attachments/assets/9cec9f4b-ee09-43b9-8f4c-031e9c8755a4" width="50%">

## MHF Block
<img src="https://github.com/user-attachments/assets/ba9efb30-3e9b-43a2-9f0a-5d7cdf5c67c9" width="70%">

## Task of MM-HiFuse
<img src="https://github.com/user-attachments/assets/a4a2a983-82be-4c62-a1ea-20e9a5582930" width="45%">

## Run
0. Requirements:
* python3
* pytorch 1.10
* torchvision 0.11.1
1. Training:
* Prepare the required images and store them in categories, set up training image folders and validation image folders respectively
* Run `python train.py`
2. Resume training:
* Modify `parser.add_argument('--RESUME', type=bool, default=True)` in `python train.py`
* Run `python train.py`

## TensorBoard
Run `tensorboard --logdir runs --port 6006` to view training progress

## Reference
Some of the codes in this repo are borrowed from:  
* [HiFuse](https://github.com/huoxiangzuo/HiFuse) 
* [Swin Transformer](https://github.com/microsoft/Swin-Transformer)  
* [ConvNeXt](https://github.com/facebookresearch/ConvNeXt)  

## Citation

If you find our paper/code is helpful, please consider citing:

```bibtex
@article{huo2025mm,
  title={MM-HiFuse: multi-modal multi-task hierarchical feature fusion for esophagus cancer staging and differentiation classification},
  author={Huo, Xiangzuo and Tian, Shengwei and Yu, Long and Zhang, Wendong and Li, Aolun and Yang, Qimeng and Song, Jinmiao},
  journal={Complex \& Intelligent Systems},
  volume={11},
  number={1},
  pages={1--12},
  year={2025},
  publisher={Springer}
}
@article{huo2024hifuse,
  title={HiFuse: Hierarchical multi-scale feature fusion network for medical image classification},
  author={Huo, Xiangzuo and Sun, Gang and Tian, Shengwei and Wang, Yan and Yu, Long and Long, Jun and Zhang, Wendong and Li, Aolun},
  journal={Biomedical Signal Processing and Control},
  volume={87},
  pages={105534},
  year={2024},
  publisher={Elsevier}
}
```
