# A Deployment Implementation For PointPillars

This repository is a deployment implementation for pointpillars. It has two parts: a PyTorch implementation for ONNX export and
a TensorRT implementation for deployment. 

## PyTorch

We train and export the model based on [OpenPCDet](https://github.com/open-mmlab/OpenPCDet). 

Go to [pytorch](pytorch) for more details.

## TensorRT

The deployment implementation is a modified version of [TensorRT](https://github.com/NVIDIA/TensorRT) and  [tiny-tensorrt](https://github.com/zerollzeng/tiny-tensorrt).

Go to [trt](trt) for more details.

## Contributors

Zhongyuan Qiu, Hao Liu, Chenchen Zhang, Bo Wen. 

[Novauto 超星未来](https://www.novauto.com.cn/)

![Novauto.png](pytorch/docs/novauto.png)

## Acknowledgement

The code is developed based on [OpenPCDet](https://github.com/open-mmlab/OpenPCDet) , [TensorRT](https://github.com/NVIDIA/TensorRT) and  [tiny-tensorrt](https://github.com/zerollzeng/tiny-tensorrt).

## License

This project is licensed under the Apache license 2.0 License - see the [LICENSE](LICENSE) file for details.