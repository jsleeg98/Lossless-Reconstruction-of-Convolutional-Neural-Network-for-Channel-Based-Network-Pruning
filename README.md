# [Lossless-Reconstruction-of-Convolutional-Neural-Network-for-Channel-Based-Network-Pruning](https://www.mdpi.com/1424-8220/23/4/2102)
This is the official implementation of "Lossless Reconstruction of Convolutional Neural Network for Channel-Based Network Pruning".

## Contents
1. [Abstract](#Abstract)
2. [Methods](#Methods)
3. [Experiments](#Experiments)
4. [Reference](#Reference)

## Abstract<a id='Abstract'></a>
Network pruning reduces the number of parameters and computational costs of convolutional neural networks while maintaining high performance. Although existing pruning methods have achieved excellent results, they do not consider reconstruction after pruning in order to apply the network to actual devices. This study proposes a reconstruction process for channel-based network pruning. For lossless reconstruction, we focus on three components of the network: the residual block, skip connection, and convolution layer. Union operation and index alignment are applied to the residual block and skip connection, respectively. Furthermore, we reconstruct a compressed convolution layer by considering batch normalization. We apply our method to existing channel-based pruning methods for downstream tasks such as image classification, object detection, and semantic segmentation. Experimental results show that compressing a large model has a 1.93% higher accuracy in image classification, 2.2 higher mean Intersection over Union (mIoU) in semantic segmentation, and 0.054 higher mean Average Precision (mAP) in object detection than well-designed small models. Moreover, we demonstrate that our method can reduce the actual latency by 8.15× and 5.29× on Raspberry Pi and Jetson Nano, respectively.

## Methods<a id='Methods'></a>
### Pruning process with reconstruction
<img src = "https://github.com/jsleeg98/Lossless-Reconstruction-of-Convolutional-Neural-Network-for-Channel-Based-Network-Pruning/blob/main/Figures/Figure_1_Pruning%20process.png" width="800" height="450">

### Residual Block
<img src = "https://github.com/jsleeg98/Lossless-Reconstruction-of-Convolutional-Neural-Network-for-Channel-Based-Network-Pruning/blob/main/Figures/Figure_2_Residual%20Block.png" width="800" height="450">

### Skip Connection
<img src = "https://github.com/jsleeg98/Lossless-Reconstruction-of-Convolutional-Neural-Network-for-Channel-Based-Network-Pruning/blob/main/Figures/Figure_3_Skip%20Connection.png" width="800" height="450">

### Batch Normalization
<img src = "https://github.com/jsleeg98/Lossless-Reconstruction-of-Convolutional-Neural-Network-for-Channel-Based-Network-Pruning/blob/main/Figures/Figure_4_Batch%20Normalization.png" width="800" height="600">

## Experiments<a id='Experiments'></a>
<img src = "https://github.com/jsleeg98/Lossless-Reconstruction-of-Convolutional-Neural-Network-for-Channel-Based-Network-Pruning/blob/main/Figures/Table_1_Hyperparameters.png" width="800" height="250">

<img src = "https://github.com/jsleeg98/Lossless-Reconstruction-of-Convolutional-Neural-Network-for-Channel-Based-Network-Pruning/blob/main/Figures/Table_2_Pruning%20ResNet.png" width="800" height="1000">

<img src = "https://github.com/jsleeg98/Lossless-Reconstruction-of-Convolutional-Neural-Network-for-Channel-Based-Network-Pruning/blob/main/Figures/Table_3_Pruning%20FCN.png" width="800" height="250">

<img src = "https://github.com/jsleeg98/Lossless-Reconstruction-of-Convolutional-Neural-Network-for-Channel-Based-Network-Pruning/blob/main/Figures/Table_4_Pruning%20YOLOv3.png" width="800" height="250">

<img src = "https://github.com/jsleeg98/Lossless-Reconstruction-of-Convolutional-Neural-Network-for-Channel-Based-Network-Pruning/blob/main/Figures/Table_5_Latency.png" width="800" height="400">

<img src = "https://github.com/jsleeg98/Lossless-Reconstruction-of-Convolutional-Neural-Network-for-Channel-Based-Network-Pruning/blob/main/Figures/Table_6_DHP.png" width="800" height="300">

<img src = "https://github.com/jsleeg98/Lossless-Reconstruction-of-Convolutional-Neural-Network-for-Channel-Based-Network-Pruning/blob/main/Figures/Table_7_Ablation_Operation.png" width="800" height="300">

<img src = "https://github.com/jsleeg98/Lossless-Reconstruction-of-Convolutional-Neural-Network-for-Channel-Based-Network-Pruning/blob/main/Figures/Table_8_Ablation_Batch_Normalization.png" width="800" height="300">

## Reference<a id='Reference'></a>
```
@article{lee2023lossless,
  title={Lossless Reconstruction of Convolutional Neural Network for Channel-Based Network Pruning},
  author={Lee, Donghyeon and Lee, Eunho and Hwang, Youngbae},
  journal={Sensors},
  volume={23},
  number={4},
  pages={2102},
  year={2023},
  publisher={MDPI}
}
```
