# GAN-based-Visible-Thermal-Person-ReID
GAN based Image to Image translation model to disentangle identity specific and identity independent components from both Visible and Thermal images. Repository is based on [DG-Net](https://github.com/NVlabs/DG-Net) implementation.

This project was done as a part of Deep Learning for Computer Vision Course ([DLCV](https://val.cds.iisc.ac.in/DLCV/)): DS-265 at Indian Institute of Science ([IISc](https://iisc.ac.in/))

We proposed a Structure based encoder to encode structural information from Visible and Thermal Images. For more details about the project work, please refer to the [Project report](https://github.com/chaitrasj/GAN-based-Visible-Thermal-Person-ReID/blob/main/Project%20Report/DLCV_RGB%20Infrared%20Cross%20Modal%20Person%20Re-identification_V1.pdf)

### Datasets
- [SYSU-MM01](https://github.com/wuancong/SYSU-MM01)
- [RegDB](http://dm.dongguk.edu/link.html) (Dataset can be downloaded from this [website](http://dm.dongguk.edu/link.html) by submitting a copyright form.)


### Training 
1. Setup the yaml file. Check out `configs/latest.yaml`. Change the data_root field to the path of your prepared folder-based dataset, e.g. `../Market-1501/pytorch`.


2. Start training
```
python train.py --config configs/latest.yaml
```
Or train with low precision (fp16)
```
python train.py --config configs/latest-fp16.yaml
```
Intermediate image outputs and model binary files are saved in `outputs/latest`.

3. Check the loss log
```
 tensorboard --logdir logs/latest
```
For more details on Training and Testing procedures, please refer to the [DG-Net](https://github.com/NVlabs/DG-Net) implementation.
