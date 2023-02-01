# UNeXt

Official Pytorch Code base for [RMMLP:Rolling MLP and Matrix Decomposition for Skin Lesion Segmentation]





## Using the code:

The code is stable while using Python 3.6.13, CUDA >=10.1


## Datasets

1) ISIC 2018 - [Link](https://challenge.isic-archive.com/data/)


## Data Format

Make sure to put the files as the following structure (e.g. the number of classes is 2):

```
inputs
└── <dataset name>
    ├── images
    |   ├── 001.png
    │   ├── 002.png
    │   ├── 003.png
    │   ├── ...
    |
    └── masks
        ├── 0
        |   ├── 001.png
        |   ├── 002.png
        |   ├── 003.png
        |   ├── ...
        |
        └── 1
            ├── 001.png
            ├── 002.png
            ├── 003.png
            ├── ...
```

For binary segmentation problems, just use folder 0.

## Training and Validation

1. Train the model.
```
CUDA_VISIBLE_DEVICES=0 python train.py --dataset ISIC2016 --arch RMMLP  --name RMMLP --img_ext .jpg --mask_ext _Segmentation.png --lr 0.0001 --epochs 200 --input_w 256 --input_h 256 --batch_size 8

2. Evaluate.
```
CUDA_VISIBLE_DEVICES=0 python val.py --name <exp name>

``` 

### Acknowledgements:

This code-base uses certain code-blocks and helper functions from [UNet++](https://github.com/4uiiurz1/pytorch-nested-unet) [Segformer](https://github.com/NVlabs/SegFormer), and [AS-MLP](https://github.com/svip-lab/AS-MLP).


