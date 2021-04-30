# A PyTorch Deployment Implementation For PointPillars

This repository is a modified version of [OpenPCDet](https://github.com/open-mmlab/OpenPCDet). We added a larger scale head to detect pedestrians and cyclists, only cars are detected in the original scale. To make deployment on a GPU more convenient and more coherent, we implemented the OPs Voxel Generator, Dense and NMS to get a complete end-to-end model. All OPs are consistent with the original OpenPCDet.

## Docker

Prepare a docker container:

```bash
$ docker pull scrin/dev-spconv:f22dd9aee04e2fe8a9fe35866e52620d8d8b3779
$ pip install Scikit-Image
```

## Install

The installation method is the same as OpenPCDet:

```bash
$ cd tools
$ python setup.py develop
```

## Dataset

We use the Kitti dataset to train and validate our model.

The dataset preparation is the same as in [GETTING_STARTED@OpenPCDet](https://github.com/open-mmlab/OpenPCDet/blob/master/docs/GETTING_STARTED.md).

Make sure to modify the root path in [kitti_dataset.yaml](tools/cfgs/dataset_configs/kitti_dataset.yaml).

## Pretrained

* We provide two pretrained models: the original multi-scale model and a pruned version, get them from [pretrained](https://pan.baidu.com/s/1sWRhbeEZRYADN0YnE10Csw), the code is "0t1h".
* 3D mAP
    
|  | Car@R11 | Pedestrian@R11 | Cyclist@R11 | mAP|
| :------: | :------: | :------: |:----:|:----:|
| OpenPCDet | 77.28 | 52.29 | 62.68 | 64.08 |
| MultiScale | 75.18 | 54.86 | 67.98 | 66 |
| MultiScale-Pruned | 67.81 | 53.15 | 64.35 | 61.77 |

 * BEV mAP
    
|  | Car@R11 | Pedestrian@R11 | Cyclist@R11 | mAP|
| :------: | :------: | :------: |:----:|:----:|
| MultiScale | 85.24 | 62.3 | 72.16 | 73.23 |
| MultiScale-Pruned | 83.83 | 58.37 | 70.41 | 70.87 |

After pruning, the AP values for each of the three classes will be a little lower than before. 
However, FLOPs and inference time are significantly reduced (tested on a Tesla T4).

|  | FLOPs | Params | FP32(PyTorch) | FP32(TRT)
| :------: | :------: | :------: |:----:|:----:|
| MultiScale | 274.9G | 20.01M | 75.8ms | 58.95ms
| MultiScale-Pruned | 40.0G | 944K | N/A | 20.62ms

## ONNX export

To export the original model, run:

```bash
$ cd tools
$ python export.py --cfg_file ./cfgs/kitti_models/pointpillar_multiscale.yaml --pretrained_model ../baseline/checkpoint_epoch_50.pth --eval_onnx_model
```

To export the pruned model, run:

```bash
$ python export.py --cfg_file ./cfgs/kitti_models/pointpillar_multiscale.yaml --pruned_model ../baseline/checkpoint_epoch_75_model.pth --eval_onnx_model
```

## Contributors

Zhongyuan Qiu, Bo Wen. 

[Novauto 超星未来](https://www.novauto.com.cn/)

![Novauto.png](docs/novauto.png)

## Acknowledgement

The code is developed based on [OpenPCDet](https://github.com/open-mmlab/OpenPCDet).

## License

This project is licensed under the Apache license 2.0 License - see the [LICENSE](LICENSE) file for details.