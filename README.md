# [Lossless-Reconstruction-of-Convolutional-Neural-Network-for-Channel-Based-Network-Pruning](https://www.mdpi.com/1424-8220/23/4/2102)
This is the official implementation of "Lossless Reconstruction of Convolutional Neural Network for Channel-Based Network Pruning".

## Contents
1. [Abstract](#Abstract)
2. [Methods](#Methods)
3. [Experiments](#Experiments)

## Abstract<a id='Abstract'></a>
Network pruning reduces the number of parameters and computational costs of convolutional neural networks while maintaining high performance. Although existing pruning methods have achieved excellent results, they do not consider reconstruction after pruning in order to apply the network to actual devices. This study proposes a reconstruction process for channel-based network pruning. For lossless reconstruction, we focus on three components of the network: the residual block, skip connection, and convolution layer. Union operation and index alignment are applied to the residual block and skip connection, respectively. Furthermore, we reconstruct a compressed convolution layer by considering batch normalization. We apply our method to existing channel-based pruning methods for downstream tasks such as image classification, object detection, and semantic segmentation. Experimental results show that compressing a large model has a 1.93% higher accuracy in image classification, 2.2 higher mean Intersection over Union (mIoU) in semantic segmentation, and 0.054 higher mean Average Precision (mAP) in object detection than well-designed small models. Moreover, we demonstrate that our method can reduce the actual latency by 8.15× and 5.29× on Raspberry Pi and Jetson Nano, respectively.

## Methods<a id='Methods'></a>
### Pruning process with reconstruction
![Figure 1](https://github.com/jsleeg98/Lossless-Reconstruction-of-Convolutional-Neural-Network-for-Channel-Based-Network-Pruning/blob/main/Figures/Figure_1_Pruning%20process.png){: width="400" height="300"}

### Residual Block
![Figure 2](https://github.com/jsleeg98/Lossless-Reconstruction-of-Convolutional-Neural-Network-for-Channel-Based-Network-Pruning/blob/main/Figures/Figure_2_Residual%20Block.png)

### Skip Connection
![Figure 3](https://github.com/jsleeg98/Lossless-Reconstruction-of-Convolutional-Neural-Network-for-Channel-Based-Network-Pruning/blob/main/Figures/Figure_3_Skip%20Connection.png)

### Batch Normalization
![Figure 4](https://github.com/jsleeg98/Lossless-Reconstruction-of-Convolutional-Neural-Network-for-Channel-Based-Network-Pruning/blob/main/Figures/Figure_4_Batch%20Normalization.png)

## Experiments<a id='Experiments'></a>
