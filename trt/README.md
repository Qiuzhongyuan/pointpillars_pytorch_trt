# A TRT Deployment Implementation For PointPillars

This repository is a modified version of [TensorRT](https://github.com/NVIDIA/TensorRT) and  [tiny-tensorrt](https://github.com/zerollzeng/tiny-tensorrt). For deployment of the pointpillars model, we added three plugins (Voxelgenerate, Dense, NMS) to the TensorRT source code. To cater to different precision requirements, we provide three inference modes: FP32, FP16.


## Docker

Prepare a docker container:

```bash
$ docker pull nvcr.io/nvidia/tensorrt:20.07-py3
```

Inside the docker container, run:

```bash
$ apt update
$ apt install libopencv-dev
$ apt install libxml2
$ apt install libxml2-dev
$ cp -r /usr/include/libxml2/libxml /usr/include/
```

## TensorRT
fp16 mode now is under the inference code`s contral
```make tensorrt(add plugin)
$ cd pointpillars_pytorch_trt/trt/TensorRT7.2
$ mkdir build
$ cmake -DPARAM_PATH="{the absolute path of tiny-tensorrt7.2}/test.xml" ..
$ make -j12
$ echo "export LD_LIBRARY_PATH={the absolute path of TensorRT7.2}/build/:$LD_LIBRARY_PATH" >> ~/.bashrc
$ source ~/.bashrc
```


## Tiny-TensorRT

```make tiny-tensorrt7.2
$ cd pointpillars_pytorch_trt/trt/tiny-tensorrt7.2
$ mkdir build
$ cmake ..
$ make -j8
$ ./unit_test --onnx_path model.onnx --data_path reduce_bin_dir
```

NOTE:

For the Kitti dataset, use the reduced bins to replicate our results.

Refer to the parameter "MODE" in [test.xml](tiny-tensorrt7.2/test.xml) to select a certain inference mode.

```model(test.xml)
$ <MODE>0</MODE>    :: fp32
$ <MODE>1</MODE>    :: fp16
```

```bash
$ ./unit_test --onnx_path ../model/pointpillars.onnx --data_path ../bindata
```
Result is saved in "pointpillars_pytorch_trt/trt/tiny-tensorrt7.2/output/".


## Evaluation

We evaluated the performance on a **2080 TI**. Inference times (in ms) for the original MultiScale-Head pointpillars model and the pruned version are provided below.

|  | fp32 | fp16 |
| :------: | :------: | :------: |
| MultiScale model| 24.84 | 11.71 |
| MultiScale-Pruned model| 8.97 | 7.62 |

We also evaluated the performance on **Xavier AGX**.

|  | fp32 | fp16 |
| :------: | :------: | :------: |
| MultiScale model| 190.20 | 94.69 |
| MultiScale-Pruned model| 62.96 | 43.92 |

and the performance on **Xavier NX**.

|  | fp32 | fp16 |
| :------: | :------: | :------: |
| MultiScale model| 320.64 | 148.36 |
| MultiScale-Pruned model| 89.58 | 66.23 |

Furthermore, we tested the average precision in the BEV perspective (BEV AP) for the categories car, pedestrian, cyclist.

The test environment is the same as [pytorch](../pytorch) .
```
$ cd pointpillars_pytorch_trt/trt/tiny-tensorrt7.2/eval_tools/
$ python eval_results.py --trt_outputs ../output/ 
```

| CAR | fp32 | fp16 |
| :------: | :------: | :------: |
| MultiScale model| 85.24 | 84.84 |
| MultiScale-Pruned model| 83.83 | 83.79 |

| PEDESTRIAN | fp32 | fp16 |
| :------: | :------: | :------: |
| MultiScale model| 61.45 | 61.41 |
| MultiScale-Pruned model| 58.71 | 57.24 |

| CYCLIST | fp32 | fp16 |
| :------: | :------: | :------: |
| MultiScale model| 72.48 | 72.02 |
| MultiScale-Pruned model| 70.35 | 69.54 |

## Contributors

Hao Liu, Zhongyuan Qiu, Chenchen Zhang, Bo Wen. 

[Novauto 超星未来](https://www.novauto.com.cn/)

![Novauto.png](novauto.png)

## Acknowledgement

The code is based on [TensorRT](https://github.com/NVIDIA/TensorRT) and  [tiny-tensorrt](https://github.com/zerollzeng/tiny-tensorrt).

## License

This project is licensed under the Apache license 2.0 License - see the [LICENSE](LICENSE) file for details.