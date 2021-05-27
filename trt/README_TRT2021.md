# A TRT Deployment Implementation For PointPillars

This repository is a modified version of [TensorRT](https://github.com/NVIDIA/TensorRT) and  [tiny-tensorrt](https://github.com/zerollzeng/tiny-tensorrt). For deployment of the pointpillars model, we added three plugins (Voxelgenerate, Dense, NMS) to the TensorRT source code. To cater to different precision requirements, we provide three inference modes: FP32, FP16, INT8.


## Docker

Prepare a docker container:

```bash
$ docker pull nvcr.io/nvidia/tensorrt:21.04-py3
```

Inside the docker container, run:

```bash
$ apt update
$ apt install libopencv-dev
```

## TensorRT

```make tensorrt(add plugin)
$ cd pointpillars_pytorch_trt/trt/TensorRT7.2
$ mkdir build
$ cmake -DUSE_FP16=OFF ..
$ make -j12
$ ./cp.sh
```

Set DUSE_FP16=ON for FP32 or INT8 inference mode.

## Tiny-TensorRT

```make tiny-tensorrt7.2
$ cd pointpillars_pytorch_trt/trt/tiny-tensorrt7.2
$ mkdir build
$ cmake ..
$ make -j8
$ ./unit_test --onnx_path model.onnx --data_path reduce_bin_dir --calibrate_path calibrate_data_dir
```

NOTE:

For the Kitti dataset, use the reduced bins to replicate our results.

Refer to the parameter "MODE" in [test.xml](tiny-tensorrt7.2/test.xml) to select a certain inference mode.

```model(test.xml)
$ <MODE>0</MODE>    :: fp32
$ <MODE>1</MODE>    :: fp16
$ <MODE>2</MODE>    :: int8
```

```bash
$ ./unit_test --onnx_path ../model/pointpillars.onnx --data_path ../bindata --calibrate_path ../testbin
```


## Evaluation

We evaluated the performance on a Tesla T4. Inference times (in ms) for the original MultiScale-Head pointpillars model and the pruned version are provided below.

|  | fp32 | fp16 | int8 |
| :------: | :------: | :------: |:----:|
| MultiScale model| 58.95 | 25.06 | 29.08 |
| MultiScale-Pruned model| 20.62 | 14.42 | 15.88 |

Furthermore, we tested the average precision in the BEV perspective (BEV AP) for the categories car, pedestrian, cyclist.

| CAR | fp32 | fp16 | int8 |
| :------: | :------: | :------: |:----:|
| MultiScale model| 85.24 | 85.51 | 75.78 |
| MultiScale-Pruned model| 83.83 | 83.23 | 75.03 |

| PEDESTRIAN | fp32 | fp16 | int8 |
| :------: | :------: | :------: |:----:|
| MultiScale model| 61.51 | 60.88 | 60.16 |
| MultiScale-Pruned model| 58.77 | 56.27 | 58.05 |

| CYCLIST | fp32 | fp16 | int8 |
| :------: | :------: | :------: |:----:|
| MultiScale model| 72.13 | 71.89 | 69.36 |
| MultiScale-Pruned model| 70.40 | 68.75 | 65.81 |

## Contributors

Hao Liu, Chenchen Zhang, Zhongyuan Qiu, Bo Wen. 

[Novauto 超星未来](https://www.novauto.com.cn/)

![Novauto.png](novauto.png)

## Acknowledgement

The code is based on [TensorRT](https://github.com/NVIDIA/TensorRT) and  [tiny-tensorrt](https://github.com/zerollzeng/tiny-tensorrt).

## License

This project is licensed under the Apache license 2.0 License - see the [LICENSE](LICENSE) file for details.