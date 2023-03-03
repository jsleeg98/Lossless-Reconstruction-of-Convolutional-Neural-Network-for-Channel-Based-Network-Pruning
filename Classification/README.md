# Train original
```python ./Classification/resnet_train.py -tb resnet50 -m resnet50 -result_dir train_result -dataset_dir datasets```
1. ```-tb```: name of log
2. ```-m```: model name (```resnet18``` or ```resnet34``` or ```resnet50``` or ```resnet101```)
3. ```-result_dir```: where the result model is saved. Default is ```train_result```
4. ```-dataset_dir```: where the CIFAR10 dataset is saved. Default is ```datasets```
## Result
result model is saved in ```./train_result/resnet50/resnet50_original_best.pth```

# Prune-Finetune-Reconstuction
```python ./Classification/prune_finetune_recon.py -tb resnet50 -m resnet50 -ln 2 -c 0.5 -result_dir train_result -dataset_dir datasets```
1. ```-tb```: name of log
2. ```-m```: model name (```resnet18``` or ```resnet34``` or ```resnet50``` or ```resnet101```)
3. ```-ln```: pruning method (```1```: L1 norm or ```2```: L2 norm or ```-1```: Geometric Median)
4. ```-c```: compression ratio. Default is ```0.5```
5. ```-result_dir```: where the result model is saved. Default is ```train_result```
6. ```-dataset_dir```: where the CIFAR10 dataset is saved. Default is ```datasets```
## Result
result model is saved in ```./train_result/resnet50/resnet50_0.5_2_fine_recon_best.pth```

# Prune-Reconstruction-Finetune
```python ./Classification/prune_recon_fine.py -tb resnet50 -m resnet50 ln 2 -result_dir train_result -dataset_dir datasets```
1. ```-tb```: name of log
2. ```-m```: model name (```resnet18``` or ```resnet34``` or ```resnet50``` or ```resnet101```)
3. ```-ln```: pruning method (```1```: L1 norm or ```2```: L2 norm or ```-1```: Geometric Median)
4. ```-c```: compression ratio. Default is ```0.5```
5. ```-result_dir```: where the result model is saved. Default is ```train_result```
6. ```-dataset_dir```: where the CIFAR10 dataset is saved. Default is ```datasets```
## Result
result model is saved in ```./train_result/resnet50/resnet50_0.5_2_recon_fine_best.pth```
