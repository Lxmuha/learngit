**如未做特殊声明，tvm代码均为**[v0.10.0](https://github.com/apache/tvm/tree/v0.10.0)版本。

# TVM架构以及代码结构分析

# 1 项目架构

## 1.1 [用户视角](https://tvm.hyper.ai/docs/tutorial/intro)

![](https://cdn.nlark.com/yuque/0/2023/png/29482156/1675146646916-ab522482-7abf-43c7-8da9-44a625f52da9.png)

1. **从Tensorflow/PyTorch/ONNX等深度学习框架导入模型**
2. **转换成Relay IR，device无关，这一环节会进行一些图级别的优化Pass**
3. **Relay IR切分成许多**小子图**，子图lowering成TE(Tensor Expression)表达**
4. **自动调优，两种方式。AutoTVM(基于模版)和AutoScheduler(无模板，通过分析计算定义自动生成搜索空间)**
5. **经过调优后，为每个子图选择最佳schedule**
6. **lowering TIR(Tensor IR)，这一环节会进行比较底层，硬件相关的一些优化Pass。并且优化后的TIR会经过CodeGen进行代码生成，TVM支持LLVM/NVCC/**特定target后端(通过BYOC框架实现)
7. **编译生成机器码，TVM将模型编译为**可链接对象模块**，后续TVM runtime提供用C/Python/Rust语言API来动态加载模型**

## 1.2 [开发者视角](https://tvm.hyper.ai/docs/arch/)

1. **IRModule: Internediate representation module, 核心数据结构，描述IR**

**逻辑架构组件**

![](https://cdn.nlark.com/yuque/0/2023/png/29482156/1675317563467-c5bdc64d-24df-49e2-8dd0-5e3c62f793c3.png)

## 1.3 新硬件接入示例

[TVM接入ARM NPU](https://github.com/apache/tvm-rfcs/blob/main/rfcs/0011_Arm_Ethos-U_Integration.md)

![](https://cdn.nlark.com/yuque/0/2023/png/29482156/1675318205751-03c49181-1309-4937-b38f-90607c9e7398.png)

# 2 代码结构分析

**TVM代码结构目录如下，需要重点关注的是加粗标红的文件/文件夹。**

**.**

**|--**3rdparty: 三方库

**|-- **apps`: tvm使用示例

**|-- **ci`: github ci相关文件

**|-- **cmake`: cmake文件

**|-- **conda`: build相关脚本文件

**|-- **configs`: 配置文件

**|-- **docker`: docker脚本

**|-- **docs`: 文档，官网的docs就是由此生成的

**|-- **gallery: 搭配文档使用的一些示例代码

**|-- **golang: tvm runtime的golang接口实现

**|-- **include: 核心实现的头文件

**|-- **jvm: tvm runtime的java接口实现

**|-- **licenses: license文件

**|-- ** python: tvm提供的python接口的实现代码，也是后续作为代码分析的切入口

**|-- **tvm

**|-- **_ffi: FFI(**Foreign Function Interface**), C++和python接口相互调用，可参考[【TVM系列九】FFI注册机制](https://www.jianshu.com/p/669f0194bdc5)

**|-- **arith`: 与TIR相关，该模块提供了一组进行分析(主要是整数)的工具，**TIR pass 可以用这些分析来简化和优化代码。**

**|-- **auto_scheduler`: 自动调优模块( **无模版方式** **）**

**|-- **autotvm`: 自动调优模块( **基于模版方式** **)**

**|-- **contrib`: 一些非核心接口，诸如提供与三方库(例如cublas)交互，调用系统编译器等，可参考[tvm.contrib](https://tvm.apache.org/docs/reference/api/python/contrib.html#)

**|-- **driver: tvmc实现

**|-- **exec: 一些命令行接口，例如起rpc服务

**|-- **ir: IR**基本**数据结构和接口，主**要包括IRModule/Type/PassContext and Pass/Op, 这部分基本数据结构** **供relay ir和tir共同使用** **。**

**|-- **meta_schedule: 调优模块的核心组件实现，供auto_scheduler/autotvm使用

**|-- **micro`: [mirco tvm模块](https://tvm.hyper.ai/docs/topic/microtvm/)

**|-- **parser: IR转换( *under development* **)**

**|-- **relay: 上层relay ir，包括IR定义，OP，与前端深度学习框架转换，优化Pass，**量化**等等，是重点关注的组件

**|-- **rpc: rpc 服务

**|-- **runtime: runtime组件实现

**|-- **script: [TVMScript](https://zhuanlan.zhihu.com/p/433540150)组件, 提供一组PyThon接口描述TIR

**|-- **`<strong><span class="ne-text">target</span></strong>`: ** IRModule 转换为 target runtime.Module 的Codegen模块实现，可参考**[设备/Target 交互](https://tvm.hyper.ai/docs/arch/arch/device_target_interactions)

**|-- **`<strong><span class="ne-text">te</span></strong>`: TE(Tensor Expression)实现，可参考[TVM 自底向上（三）：TE 的概念和编译原理](https://zhuanlan.zhihu.com/p/534313816)

**|-- **`<span class="ne-text">testing</span>`: 测试用的一些Utilty函数

**|-- **`<strong><span class="ne-text">tir</span></strong>`: TIR(Tensor IR)实现，可参考[TVM 自底向上（二）：TIR 的概念和编译原理](https://zhuanlan.zhihu.com/p/533161438)

**|-- **`<strong><span class="ne-text">topi</span></strong>`: TOPI(TVM Operator Inventory)算子清单，可参考[tvm.topi](https://tvm.apache.org/docs/reference/api/python/topi.html)

**|-- **`<span class="ne-text">utils</span>`: utils函数

**|-- **`<span class="ne-text">rust</span>`:  rust语言前端

**|--  **`<strong><span class="ne-text">src</span></strong>`: tvm C++实现源码

**|-- **`<span class="ne-text">arith</span>`: 同上

**|-- **`<strong><span class="ne-text">auto_scheduler</span></strong>`: 同上

**|-- **`<strong><span class="ne-text">autotvm</span></strong>`: 同上

**|-- **`<span class="ne-text">contrib</span>`: 同上

**|-- **`<strong><span class="ne-text">driver</span></strong>`: 同上

**|-- **`<strong><span class="ne-text">ir</span></strong>`: 同上

**|-- **`<strong><span class="ne-text">meta_schedule</span></strong>`: 同上

**|-- **`<span class="ne-text">node</span>`: IR/AST node数据结构

**|-- **`<span class="ne-text">parser</span>`: 同上

**|-- **`<span class="ne-text">printer</span>`: 打印IR

**|-- **`<strong><span class="ne-text">relay</span></strong>`: 同上

**|-- **`<strong><span class="ne-text">runtime</span></strong>`: 同上

**|-- **`<span class="ne-text">script</span>`: 同上

**|-- **`<span class="ne-text">support</span>`: TVM框架内部使用的一些函数接口

**|-- **`<strong><span class="ne-text">target</span></strong>`: 同上

**|-- **`<strong><span class="ne-text">te</span></strong>`: 同上

**|-- **`<strong><span class="ne-text">tir</span></strong>`: 同上

**|-- **`<strong><span class="ne-text">topi</span></strong>`: 同上

**|-- **`<span class="ne-text">tests</span>`: 测试代码

**|-- **`<span class="ne-text">vta</span>`: [多功能张量加速器](https://tvm.hyper.ai/docs/topic/vta)

**|-- **`<span class="ne-text">web</span>`: TVM WebAssembly Runtime
