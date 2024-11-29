# 1、传统编译器（异构）编译器的流程介绍

## 1.1  传统编译器流程

![](https://cdn.nlark.com/yuque/0/2023/png/34921621/1701142198515-728e0f41-6a6d-431b-8dc8-292d5884aa8c.png)

**（参考：comp412）**[📎L01Intro.pdf](https://taichu-platform.yuque.com/attachments/yuque/0/2023/pdf/34921621/1701142351539-46124930-89a8-46fc-a0c3-fec1eab9661a.pdf)

## 1.2 mesa OpenGL编译流程介绍

[📎nir_vec4_i965_fosdem_2016_rc1.pdf](https://taichu-platform.yuque.com/attachments/yuque/0/2023/pdf/34921621/1701140095402-e1c9f32e-91c7-4357-a850-e1a03a0a6a2f.pdf)

## 1.3 intel DPC++编译流程介绍

[📎CompilerAndRuntimeDesign.pdf](https://taichu-platform.yuque.com/attachments/yuque/0/2023/pdf/34921621/1701139446474-274df391-08a7-4fb2-b51a-021073e0adb4.pdf)

# 2、Clang+LLVM软件栈介绍

## 2.1 LLVM项目发起者chris对项目架构的介绍

[https://aosabook.org/en/v1/llvm.html](https://aosabook.org/en/v1/llvm.html)

## 2.2  Clang软件栈介绍

![](https://cdn.nlark.com/yuque/0/2023/png/34921621/1701154169340-20aa21df-8bac-4405-8ba6-cba38a51d1b5.png)

**标准的clang前端流程**

![](https://cdn.nlark.com/yuque/0/2023/png/34921621/1701140661085-5417afc3-b064-4225-b3be-a82250638f35.png)

** 在clang前端增加Loop metadata信息来做LoopVectorize**

![](https://cdn.nlark.com/yuque/0/2023/png/34921621/1701154344277-4f4cb371-8967-424a-aa09-94ce8fa83c4f.png)

**SVF：做value-flow如指针分析等**

## 2.3  LLVM中端+后端软件栈介绍

**如图所示：在前端生成合适的IR之后，llvm需要lowering到选择的target。在这整个过程中，后端一共有三种IR的相关转换：LLVM IR -> Machine Instr -> MI。**

![](https://cdn.nlark.com/yuque/0/2023/png/34921621/1701154876335-c5287869-5070-45c1-87ef-9b75ca05d4b2.png)							LLVM中端+后端基本流程

### 2.3.1 指令选择

**基本的指令选择，LLVM通过peephole实现，大体分几步LLVM IR -> SelectionDAG -> MachineInstr DAG -> MCInst**

**1.xxxISelLowering定义IR到DAG的转换。能legal就legal，不行就expand，实在不行才考虑custom。	        2.xxxISelDAGToDAG定义SelectionDAG到MachineInstr DAG的转换，也就是指令选择。在xxxInstrInfo.td里用pattern表示或者在Select函数里switch case吧。不考虑优化的话，其实就是个体力活。**

**      3.MCInst是汇编器内部对指令的一种表示结构，MachineInstr DAG 到MCInst按规则实现printer**

**4.附：xxxFrameLowering和xxxCallingConv，负责调用约定相关。处理参数传递，prelogue和epilogue，FrameIndex、dynamic realign、alloca对sp、fp的对应关系。**

### 2.3.2 指令调度

**指令调度的目标：最大化IPC（最大化利用硬件各部件，如st/ld,alu,fpu,vector fpu单元等），减少数据依赖、控制依赖等。**

**pre-ra **Instruction scheduling

[📎Larin-Trick-Scheduling.pdf](https://taichu-platform.yuque.com/attachments/yuque/0/2023/pdf/34921621/1701184510532-8ffcd66d-a753-403d-95d1-5e75e823f950.pdf)

**MI层指令调度优化：**

[📎Estes-MISchedulerTutorial.pdf](https://taichu-platform.yuque.com/attachments/yuque/0/2023/pdf/34921621/1701184049873-4bdfdd6f-4e12-412c-95bc-e9193879988c.pdf)

### 2.3.3 寄存器分配

**减少register spill，最大程度复用reg或者register file。LLVM实现了BASIC和Greedy、Linear分配器。**

**在寄存器分配之前，需要做很多准备工作，例如指令序号标记、活跃分析等。LLVM中的活跃分析主要分为两个部分：**活跃变量分析**（live variable analysis，在LiveVariables类中实现）和生命期分析（live interval analysis，在LiveInterval类中实现）。这些都是在分配器之外的其它pass中完成。如果将分配器选择进一步细分，可以分成切分权重计算（Split weight calculation）、生命期入队列（Enqueue）、**指定物理寄存器**（Assignment）、剔除（Eviction）、切分（Splitting）和溢出（Spilling）等阶段。**

## 2.4  LLVM project与异构计算社区khronos

![](https://cdn.nlark.com/yuque/0/2023/png/34921621/1701223678697-ce33e06d-b424-4e9f-b00f-37da4ecaf490.png)

# 3、IR与程序分析

## 3.1 IR设计是整个编译器内部设计的重心

1. **编译器和中间表示（IR）：**

* **编译是复杂的过程，而编译器是将高级源语言程序转化为计算机可执行形式的软件程序。**
* **中间表示（IR）是在编译过程中管理复杂性的数据结构，允许编译器分为多个阶段和组件，从而实现模块化。**

2. **IR对编译器的重要性：**

* **IR是编译器内部程序的通用接口，用于表示程序而不丢失信息，以确保准确执行。**
* **IR的设计要通用，能够表示从多种语言翻译而来的程序。**

3. **IR的形式和层次：**

* **IR应该能够转化为不同形式，以在多个平台上执行。**
* **使用IR使得编译器能够支持多个前端（翻译不同编程语言）和多个后端（为不同处理器目标生成代码）。**

4. **编译过程的模块化：**

* **编译过程分为前端、中端和后端。前端处理编程语言方面，后端处理目标机器方面，而中端进行与目标无关的优化。**

5. **IR的设计形式** **：***决定了IR在内容中的数据结构设计*

* **IR应提供足够信息以正确执行原始程序，但比大多数编程语言要简化。**
* **IR可以是分层次的（树状结构）或扁平的（抽象或虚拟机的指令序列），或者介于两者之间的抽象栈机的语言形式。**

![](https://cdn.nlark.com/yuque/0/2023/png/34921621/1701155189065-b45d54cf-eefe-4513-8b4d-4f35877b0c35.png)

**             结构化IR**

![](https://cdn.nlark.com/yuque/0/2023/png/34921621/1701155275193-2f5f1704-d9d3-464d-a5e2-f9a37711fe7e.png)

**     三地址码格式的Linear IR与内存实现方式**

6. **IR用于程序分发和执行：**

* **IR可以作为程序分发的标准媒介，实现“一次编写，随处运行”。**
* **IR的使用可以通过JIT（即时编译）提高解释执行的性能。**
* **针对特殊硬件的定制编译需要在安装或程序加载时提前进行（AOT），并强调了IR在此过程中的作用。**

7. **标准化IR的好处：**

* **标准IR解决了软件兼容性和编译器互操作性的问题，为不同类型处理器的制造商创造了一个公平竞争的平台。**
* **一个标准的IR有助于编译器之间的合作，使它们能够结合各自的优势，降低进入编译器写作的门槛，促进行业创新。**

8. **IR设计属性** **：***决定了程序分析算法与程序转换算法实现的方式*

* **完整性、语义差距、硬件中立性、手动可编程性、可扩展性是共享的设计属性。**
* **对于编译器内部使用的IR，简单性、**程序信息、分析信息等属性也是重要的考虑因素**。**

9. **制定IR标准的挑战：**

* **制定一个通用IR标准对整个计算机行业是理想的，但需要时间、共识和不断更新来适应技术趋势。**

**参考：fred chow对IR设计的文章（**[https://queue.acm.org/detail.cfm?id=2544374](https://queue.acm.org/detail.cfm?id=2544374)）

# 4、程序分析（变换）program analyis（transform）

![](https://cdn.nlark.com/yuque/0/2023/png/34921621/1701225391172-79810819-bbcd-428c-804f-4852867a3a68.png)

**程序分析在整个编译体系中的位置**

**静态分析是对程序 P 进行分析，以推断其行为并确定在运行 P 之前是否满足某些属性。****程序分析给我们提供了程序需要获取的信息，让我们有能力对程序进行对应的变换。**

![](https://cdn.nlark.com/yuque/0/2023/png/34921621/1701227320334-5353ef27-841e-407e-a120-9603d4edbb42.png)

## 4.1 must may分析

![](https://cdn.nlark.com/yuque/0/2023/png/34921621/1701241610350-ca468fde-e188-4054-ae48-ed14096c0b53.png)

![](https://cdn.nlark.com/yuque/0/2023/png/34921621/1701241590265-8dc83830-5626-4b77-b1eb-1558cf5d769f.png)

![](https://cdn.nlark.com/yuque/0/2023/png/34921621/1701241653415-977a21fc-2122-4d6e-b03e-94fe72f55b16.png)

![](https://cdn.nlark.com/yuque/0/2023/png/34921621/1701241674399-499c2a60-0989-4526-9f01-15b34e399989.png)

**暂时看：**[https://sumsec.me/PL/Data-Analysis-Foundation.html](https://sumsec.me/PL/Data-Analysis-Foundation.html)

# 5、案例：基于程序依赖图（CFG+DFG）对含控制流的程序自动向量化（纯理论示例）

## 5.1 仅有数据依赖的自动向量化：

**本节旨于介绍了一种通用的，可以较好的对串行程序（多层循环下）进行自动向量化变换的算法，与平台无关。**

**对于一个只包含数据依赖的多层循环块程序** **向量化执行的本质上是寻求内层循环的并行化机会（inner parallelization），** **进而可以对内层循环进行循环分布变换改变语句实例的执行顺序而不影响程序的依赖关系，并且将变换后的程序映射至向量处理器上，借助执行模型的设定以提高程序执行的性能。在这里我们借助ISL库，对一个简单的多重循环程序进行分析，分别通过依赖分析获取各类语句实例冲突访问的依赖分类（RAW/WAW/WAR）以及循环嵌套层级（loop carried level），再根据数据依赖图（Data Dependence Graph）重新构建调度树（scedule tree）。再由调度树生成标记可并行的loop部分。**

### 5.1.1 使用循环分布变换

 **机器无关的重排序变换但为机器相关的向量化变换创造了条件。** **循环分布变换是保持原程序所有嵌套循环中**语句迭代空间不变**情况下的一种**有效**的**重排序变换**。变换后的程序的某些语句从某层开始，直至循环最内层，都是可向量化的循环（当然也可以并行化）。**

## 5.2 控制流的引入与GPU条件化简方案（TODO）

**转换求解控制依赖的condition数组，将其转化为数据依赖（去除PDG图中分支的回边）**

# 6、问题：DSA compiler与AI compiler的low level IR抽象

**附录：**

[📎Keith_Cooper_Linda_Torczon-Engineering_a_Compiler-EN.pdf](https://taichu-platform.yuque.com/attachments/yuque/0/2023/pdf/34921621/1701153399007-05afef54-15a2-4926-bd84-098b2bbae502.pdf)
