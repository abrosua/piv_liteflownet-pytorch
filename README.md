# PIV LiteFlowNet with PyTorch 
This is my PyTorch reimplementation of the PIV-LiteFlowNet [1] model. It's an upgraded LiteFlowNet [2] model with additional layers and trained specifically on Particle Image Velocimetry (PIV) images (please refer to the original paper [1] for the complete dataset details and sources). Also thanks to [Sniklaus](https://github.com/sniklaus/pytorch-liteflownet) [3] for the PyTorch reimplementation of the original LiteFlowNet model. If you would like to use this particular implementation, please acknowledge it appropriately [4].


## Model
The trained models for both the original LiteFlowNet and PIV-LiteFlowNet are already available in the `models/pretrain_torch/`. These originate from the original authors, I just converted them to PyTorch.

The correlation layer is implemented in CUDA using CuPy, which is why CuPy is a required dependency. It can be installed using `pip install cupy` or alternatively using one of the provided binary packages as outlined in the CuPy repository.

## Usage
To run it sequentially on 1000 images in your directory, starting from the first image, use the following command. There are 2 model options the are avaialble to run with, `piv` for the PIV-LiteFlowNet-en and `hui` for the original LiteFlowNet model. Please check on the CLI `--help` for more info.
```
python run.py --model piv -s 0 -n 1000 --input ./images/test --output ./test-output
```
If you want to visualize the flow results in sequence, or even compile it as video format, you can use the [piv-viz](https://github.com/abrosua/piv-viz) package.

## References
```
[1]  @article{piv-liteflownet,
         author = {Shengze Cai and Jiaming Liang and Qi Gao and Chao Xu and Runjie Wei},
         journal = {IEEE Transactions on Instrumentation and Measurement},
         title = {Particle Image Velocimetry Based on a Deep Learning Motion Estimator},
         year = {2020},
         volume = {69},
         number = {6},
         pages = {3538-3554},
         doi = {10.1109/TIM.2019.2932649}
}
```
```
[2]  @inproceedings{Hui_CVPR_2018,
         author = {Tak-Wai Hui and Xiaoou Tang and Chen Change Loy},
         title = {{LiteFlowNet}: A Lightweight Convolutional Neural Network for Optical Flow Estimation},
         booktitle = {IEEE Conference on Computer Vision and Pattern Recognition},
         year = {2018}
     }
```

```
[3]  @misc{pytorch-liteflownet,
         author = {Simon Niklaus},
         title = {A Reimplementation of {LiteFlowNet} Using {PyTorch}},
         year = {2019},
         howpublished = {\url{https://github.com/sniklaus/pytorch-liteflownet}}
    }
```

```
[4]  @misc{piv-liteflownet-pytorch,
         author = {Faber Silitonga},
         title = {A Reimplementation of {PIV-LiteFlowNet-en} Using {PyTorch}},
         year = {2021},
         howpublished = {\url{https://github.com/abrosua/piv_liteflownet-pytorch}}
    }
```
