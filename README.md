# custom-op-FT

## 介绍

本开源库是在[Custom-op](https://github.com/tensorflow/custom-op)的基础上，加入自行开发的深度学习加速算子，并汇入[FasterTransformer](https://github.com/NVIDIA/FasterTransformer)中的部分算子，构建custom-op-FT pip包，用户可以以较低的门槛，方便地使用这些加速深度学习模型。

## 当前支持的算子(持续更新中)
BERT
SBERT
OPENNMT

### 后续更新计划：

gemm bazel编译
DEMO
poly-encoder的attention算子

## 如何使用
### 安装pip包
```bash
pip install xxx
```
暂时还未上传pypi库，正在加急中
### 使用
使用方式和Nvidia的FT类似，但是不需要自己去构建`.so`文件，再去载入它了，具体的示例代码请看`tensorflow_faster_transformer/sample/tensorflow/bert`

持续更新中

## 速度测试
### BERT
#### nvidia-2070s
```bash
# transformer config: seq_len=32, layers=12, heads=12, size_per_head=64
# data type = fp32 
# Device: nvidia-2070-Super
[INFO] Encoder Cross check True
[INFO] Max diff 4.05311584473e-06
[INFO] min diff 0.0
[INFO] TF decoder time costs: 26.06256 ms
[INFO] OP decoder time costs: 14.17237 ms
```
#### nvidia-p100
```bash
# transformer config: seq_len=32, layers=12, heads=12, size_per_head=64

# data type = fp32 
# Device: nvidia-p100
[INFO] Encoder Cross check True
[INFO] Max diff 4.29153442383e-06
[INFO] min diff 0.0
[INFO] TF decoder time costs: 6.07602 ms
[INFO] OP decoder time costs: 4.69454 ms
```
可见，推理时间缩短了几乎50%，这是非常高效的。

### OPENNMT-encoder-tf
#### nvidia-p100
```bash
# opennmt config: seq_len=128, beam_width=4, encoder_layers=6, encoder_heads=8, size_per_head=64, batch_size=1
# data type = fp32 
[INFO] Encoder Cross check True
[INFO] Max diff 1.66893005371e-06
[INFO] min diff 0.0
[INFO] tf_opennmt_time: 5.88176 ms
[INFO] tf_opennmt_time: 3.50101 ms
```
### Sentence BERT
#### nvidia-2070s

```bash
# transformer config: seq_len=32, layers=12, heads=12, size_per_head=64
# data type = fp32 
# Device: nvidia-2070-Super
# py36+tf1.14
[INFO] Query Cross check True
[INFO] Max diff 4.887580871582031e-06
[INFO] min diff 0.0
[INFO] Question Cross check True
[INFO] Max diff 4.76837158203125e-06
[INFO] min diff 0.0
[INFO] TF encoder time costs: 51.90491 ms
[INFO] OP encoder time costs: 27.00781 ms
```
#### nvidia-p100
```bash
# transformer config: seq_len=32, layers=12, heads=12, size_per_head=64
# data type = fp32 
# Device: nvidia-tesla-P100
# py36+tf1.14
[INFO] Query Cross check True
[INFO] Max diff 4.887580871582031e-06
[INFO] min diff 0.0
[INFO] Question Cross check True
[INFO] Max diff 4.76837158203125e-06
[INFO] min diff 0.0
[INFO] TF encoder time costs: 9.94811 ms
[INFO] OP encoder time costs: 8.93003 ms
```
## Reference：
[tensorflow/Custom-op](https://github.com/tensorflow/custom-op)

[NVIDIA/FasterTransformer](https://github.com/NVIDIA/FasterTransformer)



---------------------



# for 高级用户

## 代码总览

整体代码目录树如下：

```bash
├── gpu  # Set up crosstool and CUDA libraries for Nvidia GPU, only needed for GPU ops
│   ├── crosstool/
│   ├── cuda/
│   ├── BUILD
│   └── cuda_configure.bzl
├── tensorflow_faster_transformer # FT部分
│   ├── BUILD
│   ├── __init__.py
│   ├── cc
│   │   ├── kernels/ 
│   │   ├── ops/ # 一些算子的接口定义，例如名称，输入输出，形状等
│   │   ├── cuda/ # cuda计算实现
│   │   ├── allocator.h
│   │   ├── beamsearch_opennmt.h
│   │   ├── bert_encoder_transformer.h
│   │   ├── common.h
│   │   ├── common_structure.h
│   │   ├── ...
│   └── python
│       ├── __init__.py
│       └── ops
│           ├── __init__.py
│           └── faster_transformer_ops.py  #python 加载算子
├── tensorflow_zero_out  # 官方 CPU 示例
│   ...
├── tensorflow_time_two  # GPU 示例
│   ...
├── tf  # Set up TensorFlow pip package as external dependency for Bazel
│   ├── BUILD
│   ├── BUILD.tpl
│   └── tf_configure.bzl
|
├── BUILD  # top level Bazel BUILD file that contains pip package build target
├── build_pip_pkg.sh  # script to build pip package for Bazel and Makefile
├── configure.sh  # script to install TensorFlow and setup action_env for Bazel
├── LICENSE
├── Makefile  # Makefile for building shared library and pip package
├── setup.py  # file for creating pip package
├── MANIFEST.in  # files for creating pip package
├── README.md
└── WORKSPACE  # Used by Bazel to specify tensorflow pip package as an external dependency
```

最主要的部分就是添加的`tensoflow_faster_transformer`文件夹，里面包含了bert、sbert、opennmt等gpu加速的代码实现，以及相应的bazel构建代码。

## 前期准备

### 镜像和代码

本pip包构建的环境是py36+tf1.14，请用下面的镜像

```bash
docker pull opeceipeno/custom-op-ft:19.07-py3
rm -r custom-op-FasterTransformer    #旧的删除掉
git clone https://github.com/Qksidmx/custom-op-FasterTransformer.git
cd custom-op-FasterTransformer
```

### 环境配置

#### 运行配置脚本

```bash
./configure.sh
```

这个可以一键配置编译环境，避免自己手动去配置了，依次选N、Y即可大概如下

```bash
Do you want to build ops again TensorFlow CPU pip package? Y or enter for CPU, N for GPU. [Y/n] N
Does the pip package have tag manylinux2010 (usually the case for nightly release after Aug 1, 2019, or official releases past 1.14.0)?. Y or enter for manylinux2010, N for manylinux1. [Y/n] Y
...
```

*注意：*这个官方的`configure.sh`有些问题，在选择配置gpu环境的时候，他并不能正确识别环境中的tensorflow-gpu，而是去识别cpu版本的，导致每次都会重新下载，并且下载的是cpu版本，这样就会出问题。

因此做了一定的修改

### 利用Bazel编译

```bash
bazel build build_pip_pkg
```

因为体量不大，编译速度比较快，前面配置都成功的话，这步应该不会出问题。

编译得到一个二进制文件，`build_pip_pkg`，用下列命令来制作pip包。

### 生成及安装pip包

```bash
bazel-bin/build_pip_pkg artifacts
```

正确执行完毕之后，`artifacts`文件夹下就会有个`.whl`文件

```bash
pip install artifacts/*whl
```

## 使用

在编写代码的时候，import包之后，调用方法使用即可，其中参数需要一一对应

```python
import tensorflow_faster_transformer as ft

bert_result = ft.bert_transformer(from_tensor, to_tensor,
                                  attr_kernel_q, attr_kernel_k, attr_kernel_v,
                                  attr_bias_q, attr_bias_k, attr_bias_v,
                                  attr_mask,
                                  attr_output_kernel, attr_output_bias,
                                  attr_output_layernorm_beta, attr_output_layernorm_gamma,
                                  inter_kernel, inter_bias, output_kernel,
                                  output_bias, output_layernorm_beta, output_layernorm_gamma,
                                  from_tensor_question, to_tensor_question)
opennmt_result = ft.open_nmt_transformer(...)
sbert_result = ft.sentence_bert_transformer(...)

```

