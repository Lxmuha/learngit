# 1 概要

**简单介绍一下AI编译是什么，做了哪些事情。**

**传统深度学习框架执行流如下：**

算法模型(resnet/bert..) -> 框架前端实现(Python) -> 运行时(算子库) -> device `

**基于AI编译的执行流如下:**

算法模型(resnet/bert...) -> 框架前端实现(Python) -> AI compiler(parsing，lowering, codegen，runtime) -> device

**Note. 这里做parsing可能不太准确，更明确是Graph capture**

**区别显而易见，在执行框架的代码时，与传统不同的AI编译拿到框架层建立的静态Graph IR信息，经过lowering，到最后codegen生成可执行代码，在device上执行。**

**目前比较活跃的AI编译技术路线，如XLA/TVM/MLIR，各家对于如何获取Graph IR，如何lowering都有各自的实践，下面也会详细展开去探讨。**

**这里重点关注以下几个问题：**

1. **如何从深度学习框架(如PyTorch，Tensorflow)获取静态Graph IR表达？**
2. **如何设计中间IR？**
3. **做了哪些优化Pass？(lower down/shape infer/layerfusion/memory alloc/codegen……)**
4. **新接入一个device(如swai) 工作量和挑战在哪里？**

**接下来会分别介绍MLIR，XLA，TVM具体的实现思路。**

**典型芯片硬件比较：**

| **太初**                                                                                                                                                                                                                    | **nvidia**                                                                                                                                                                                                                                       | **寒武纪**                                                                                                                                                                                                                                                                                                         | **biren**                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **众核架构*** **主核64位RISC结构通用处理单元，从核64为RISC多用处理单元。*** **4个核组，每个有32个从核*** **从核有256K LDM，可通过DMA与主存通信**底层软件栈：SDAA/compiler**算子库：swdnn**多卡:TCCL | **A100*** **8个GPC，128个SM*** **6912个fp32 CUDA core,432个tensor core*** **显存：**80GB HBM2e* **带宽：**2039 GB/s(**A100 80GB SXM**)底层软件栈:driver/nvcc/CUDA**算子库：cudnn**多卡：NCCL(NVLink or PCIe 4.0) | **MLU370-X8为例**MLUarch03架构* **内存支持LPDDR5*** **单芯片4个cluster，每个cluster有4个core*** **支持到fp32浮点运算*** **支持虚拟化 vMLU实例 4个*** **显存：48 GB*** **带宽：614.4 GB/s**底层软件栈：CNRT/CNTOOLKIT**算子库：CNNL**通信库：CNCL**多卡：MLU-Link** | **大体参考nvidia硬件设计**硬件：* **16或32个SPC,每个spc内有tcore和vector单元*** **特殊单元：reduce buffer，用来归约运算*** **CSR：存放learning rate等*** **近存计算:显存分为UMA和NUMA,分别用于存放weight和activation,多个SPC数据并行*** **GMB：global memory buffer，大约4MB，做极致性能优化时可按照GMB容量放置tensor**底层软件栈:UMD/KMD/compiler**编程接口：类CUDA**算子库：类cuDNN**多卡:SCCL** |

# 2 [MLIR](https://github.com/llvm/llvm-project/tree/main/mlir) 概述

**准确地说，mlir并不是AI编译的方案，而是一个framework。与xla，tvm提供完整的端到端解决方案不同，mlir提供的是一套framework，具体表现为提供了如何定义Dialect(IR), 如何实现lowering pass等一些接口，而** **怎么设计IR，进行哪些优化Pass，以及compile backend如何实现都需要使用者自己去定义与实现。** **当然社区提供了一系列推的Dialect和对应的Pass，供开发者复用。mlir这一套设计哲学与llvm及其相似，极大的开放包容性给开发者很大的操作空间去发挥。**

**业内已有不少公司follow mlir这个项目，并开源了他们的代码，以以下几个项目为例，我们深入去分析基于mlir怎么去实现端到端的AI 编译flow。**

## 2.1 [tpu-mlir](about:blank) 技术方案分析

* **比特大陆开源的，基于mlir的推理引擎。**
* **后端device为自研tpu**

**架构图如下：**

![](https://cdn.nlark.com/yuque/0/2022/png/29482156/1672203393213-afb032c4-559a-4b19-b3d8-d4b0805bc8a5.png)

**可以看出以下几个关键信息**

1. **通过onnx与Tensorflow, PyTorch等深度学习框架进行桥接。**
2. **具体Compile过程在****NN Toolchain **这个阶段，包括Graph Capture，优化pass等
3. **没有codegen，还有一层runtime api。**

**更近一步，我们具体去看NN Toolchain这个过程，具体是切分了多少dialect，做了哪些优化pass。**

**这一部分的架构如下：**

![](https://cdn.nlark.com/yuque/0/2022/png/29482156/1672204717215-23f9c7e8-18ef-4dfb-8bc1-21a9518393da.png)

**同样，结合源码阅读提炼出以下关键信息。**

* **1. 按与device是否有关切分成Top层和Tpu层，共2层**
* **2. Top层优化Pass主要包括以下：**

  * **常规图优化，诸如算子融合，shape合并等(canonicalize）**
  * **量化相关优化(calibration）**
* **3. Tpu层优化Pass主要包括以下：**

  * *Op层图优化，比如连续Requant的合并(canonicalize)*
  * **权重重排优化，如卷积的filter和bias(weight reorder)**
  * **子网络切分优化，按tpu/cpu切分不同的子网络(subnet)**
  * **子图切分优化，对网络进行切分，使尽可能多的OP在local mem中连续计算(layer group)**
  * **mem优化，给需要global mem的op分配地址(mem asign)**
* **codegen生成bmodel这个格式的文件给PyRuntime去在设备上执行**

**bmodel是比特大陆自己定义的一种模型描述格式，对应的他们runtime(**[BMRuntime](about:blank))可以读取该文件并在tpu上运行。

**所以这里的codegen与传统编译生成二进制的那个codegen存在差异。**

**详细的各个部分的实现原理比特大陆也发布了以下一些博客供参考(当然阅读源码更直接)**

* [TPU-MLIR之前端转换](https://tpumlir.org/zh-cn/2022/12/07/tpu-mlir-zhi-qian-duan-zhuan-huan.html)
* [TPUMLIR中的量化概述](https://tpumlir.org/zh-cn/2022/12/07/tpumlir-zhong-de-liang-hua-gai-shu.html)
* [TPU-MLIR之量化感知训练（上）](https://tpumlir.org/zh-cn/2022/12/14/tpu-mlir-zhi-liang-hua-gan-zhi-xun-lian-shang.html)
* [TPU-MLIR之量化感知训练（下）](https://tpumlir.org/zh-cn/2022/12/14/tpu-mlir-zhi-liang-hua-gan-zhi-xun-lian-xia.html)
* [TPU-MLIR之精度验证](https://tpumlir.org/zh-cn/2022/12/27/tpu-mlir-zhi-jing-du-yan-zheng.html)

## 2.2 [MagicMind](https://www.cambricon.com/index.php?m=content&c=index&a=lists&catid=378) 技术方案分析

* **寒武纪开发的基于MLIR的图编译推理引擎，未开源**
* **后端device为MLU推理芯片**

**这篇文章我们重点关注MagicMind中图编译相关技术路线。**

**整个MagicMind架构如下：**

![](https://cdn.nlark.com/yuque/0/2022/png/29482156/1672214433870-01dd616c-7130-43aa-b25b-601690e16ae0.png)

**提取其中的关键信息，MagicMind分成三大部分，分别是** **前端表示层，图编译引擎和运行时** **。**

1. **前端表示层可通过****2**种方式构建Graph(MagicMind IR)

* **parsing PyTorch/Tensorflow等开源框架模型文件**
* **使用MagicMind提供的C++/Python接口搭建网络，形如下方所示。**

![](https://cdn.nlark.com/yuque/0/2022/png/29482156/1672215126972-247bce69-9836-4604-ac90-3752c9f21147.png)

2. **图编译引擎层对前段表示层得到的IR进行编译优化，优化按是否和device相关分为两层。**

* **HLO(High Level Optimization)，device无关优化Pass，比如op融合，常量折叠，冗余分支消除等常规图优化。**
* **LLO(Low Level Optimization), device相关优化Pass，比如Layout转换，前后算子融合等。**

3. **Codegen这部分支持****自动生成代码和手写算子库匹配**两种方式，也支持fall back cpu。

## 2.3 iree 技术方案分析

* **开源项目**

**项目架构**

![](https://cdn.nlark.com/yuque/0/2022/png/29482156/1672279274042-64c8fa41-3555-40b5-b265-f576b0a187d7.png)

**提取关键信息**

1. **前端接入深度学习框架通过****Importer**组件导入(模型 --> mlir表达)
2. **flow dialect(device 无关), hal dialect(device 相关)**
3. **多device后端支持**

## 2.4 mlir方向技术方案总结

**总结下来以下几点**

1. **前段对框架的支持通过parsing组件转换(model -> graph)**
2. **ir 设计大体可以分为high level(device 无关）和low level(device 相关)，对应优化优化pass也相应不同，其中high level的优化基本可以借鉴社区工作**
3. **codegen这部分可以直接生成或者使用手写算子(高性能算子库）**

**真正在落地的时候有很多dirty work和难点**

1. **parsing组件开发，各家框架接口统一**
2. **算子支持**
3. **控制流，变长输入(这个问题编译路线都会遇到)**

# 3 Torch Dynamo

**前段时间刚刚预发布的PyTorch 2.0的新feature，看起来像PyTorch在不断摸索编译这条路上(torch.jit.trace, torch.jit.script, fx, functorch) 最后选择了dynamo。**

**dynamo可以分成两部分**

* **graph capture: ****字节码l**evel构建图，对算法同学改动基本无(这也是PyTorch团队强推的一点特性）
* **compile backend: 编译后端，这部分工作2.0其实没有大的变动，更多是前面探索的compile 后端，如nvfuser增加了对dynamo这一前端的支持。**

**具体的dynamo分析文档见下。**

[浅谈Torch Dynamo](https://taichu-platform.yuque.com/xy3e23/kmihv5/fkx5bz1pugg73fpr)

# 4 TVM

**TODO**

# 5 XLA

[浅谈 xla](https://taichu-platform.yuque.com/efdn1t/vkr4s6/zgy6te7q1970s51f)

[为 xla 开发新后端](https://www.tensorflow.org/xla/developing_new_backend)

[openxla 项目](https://github.com/openxla/xla)

**The OpenXLA compiler is a community-driven and modular ML compiler. It will enable efficient optimization and deployment of ML models from most major frameworks to any hardware backend notably CPUs, GPUs, and ML ASICs.**

# 以往遇到的一些实际问题(gzz)

* **IR设计：pytorch/tensorflow一些op难以对齐，不容易抽象出一套通用的IR，不同框架对input和weight的存储和处理方式差异较大，需要设计通用的allocator。**
* **硬件支持：硬件产品对bfloat16类型和batchnorm有特殊优化(bncache)，但是需要存储一些中间计算结果，导致需要添加一些特殊的图优化pass，并对框架一些地方进行修改。**
* **多个子图的支持和处理，比如pytorch中经过jit script可能会形成多个子图，而tensorflow中dataset op、前向、反向、optimizer全在一张图上，如何统一处理。**
* **一些图规模很大，节点数目可能数千个，后端codegen之后硬件是否支持。**
* **出于性能考虑，是否需要做fuse optimizer，混合精度情况下不同数据类型需要分开处理，对DDP的影响。**
* **静态图中一些动态shape算子的支持(如boolean_mask)**
* **一些规模较大的图进行kernel gen速度很慢，需要考虑kernel cache功能**
* **control flow和TensorArray等特性的支持**

# 参考

**[1] **[AI编译器的概览、挑战和实践](https://zhuanlan.zhihu.com/p/508345356).

**[2] **[NNToolChain](https://doc.sophgo.com/docs/3.0.0/docs_latest_release/nntc/html/index.html)

**[3] **[寒武纪MagicMind用户手册](https://www.cambricon.com/docs/sdk_1.9.0/magicmind_1.0.1/user_guide/index.html)

**[4] **[iree](https://iree-org.github.io/iree/)
