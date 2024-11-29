# 背景

**  Teco-Inference 是基于 TVM 开发的推理引擎，在使用时主要分为两个阶段，编译模型和执行推理。在编译模型阶段，Teco-Inference 会将模型中的算子分别编译成不同的函数，在这些函数中会调用对应的函数指针并执行，这些函数指针指向了对应的算子实现函数。下面的代码是 Less 算子编译之后得到的，**

```cpp
static void* tvm_contrib_swdnn_less_unary_forward_packed = NULL;
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t tvmgen_default_fused_less_unary(void* args, int32_t* arg_type_ids, int32_t num_args, void* out_ret_value, int32_t* out_ret_tcode, void* resource_handle) {
  // TVM set device

  // do some check

  if (tvm_contrib_swdnn_less_unary_forward_packed == NULL) {
    if (TVMBackendGetFuncFromEnv(__tvm_module_ctx, "tvm.contrib.swdnn.less_unary.forward", &tvm_contrib_swdnn_less_unary_forward_packed) != 0) {
      return -1;
    }
  }
  TVMValue ret_val_1;
  int ret_type_code_1;
  if (TVMFuncCall(tvm_contrib_swdnn_less_unary_forward_packed, (TVMValue*) stack_value, (int*) stack_tcode, 3, &ret_val_1, &ret_type_code_1) != 0) {
    return -1;
  }
  return 0;
}

// CodegenC: NOTE: Auto-generated entry function
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t __tvm_main__(void* args, int* arg_type_ids, int num_args, void* out_ret_value, int* out_ret_tcode, void* resource_handle) {
  return tvmgen_default_fused_less_unary(args, arg_type_ids, num_args, out_ret_value, out_ret_tcode, resource_handle);
}
```

**  经过编译之后，Less 算子变成了函数 **`<span class="ne-text">tvmgen_default_fused_less_unary</span>`。在 `<span class="ne-text">tvmgen_default_fused_less_unary</span>` 中，会调用 `<span class="ne-text">TVMBackendGetFuncFromEnv</span>` 去查找是否存在名为 `<span class="ne-text">tvm.contrib.swdnn.less_unary.forward</span>` 的函数，如果不存在这个函数立即返回；如果存在，那么执行这个函数。这里的 `<span class="ne-text">tvm.contrib.swdnn.less_unary.forward</span>` 函数就是我们算子接入是实现的 C++ 代码。在原有的接入方式下，我们的接入代码是这么实现的，

```cpp
void swdnn_less_unary_impl(TVMArgs args, TVMRetValue* ret) {
	// impementation
}

TVM_REGISTER_GLOBAL("tvm.contrib.swdnn.less_unary.forward")
    .set_body([](TVMArgs args, TVMRetValue* ret) { swdnn_less_unary_impl(args, ret); });
```

**  我们通过宏 **`<span class="ne-text">TVM_REGISTER_GLOBAL</span>` 定义了名为 `<span class="ne-text">tvm.contrib.swdnn.less_unary.forward</span>` 的函数，并通过 `<span class="ne-text">set_body</span>` 设置了 lambda 函数，这个 lambda 函数包含了具体的算子实现。

**  这样的接入方式，所有算子实现都是以函数的形式存在。同时，为了能够实现资源复用，TecoDNN 相关的资源和工具都放到了一个统一的全局变量中。这样带来了一些问题，**

1. **在算子的实现中需要用到一些临时资源时，只能及时申请及时释放，无法在推理前申请推理时复用**
2. **无法单独为每一个核组进行单独配置，如，Stream**
3. **当一个算子有多种不同实现时，需要人工选择不同实现，随着实现增多，代码冗余，不利于维护；同时无法自动选择最优的算子实现**

**. . . . . . .**

# 整体方案

## Runtime 重构

**  为了能够实现资源复用和多核组异步推理，我们对 Runtime 进行了重构，具体可参考文档**[【Runtime 重构】](https://taichu-platform.yuque.com/xy3e23/kmihv5/fqzx7twnfcoiqyh7)。

![](https://cdn.nlark.com/yuque/0/2023/png/29482156/1690942605465-a7fc3489-2b37-484d-84f0-8301c95c4c21.png)

**  在 TVM 原有的 GraphExecutor 基础上，我们增加了 RuntimeContext 和 InferenceEngine 两层抽象。在最新的代码中，RuntimeContext 已经被更名为 Engine，InferenceEngine 被更名为 Executor，在本文的后续部分，都会使用 Engine 和 Executor 进行讨论。**

**  Runtime 重构之后，Engine 是直接暴露给用户的对象，用户创建 Engine 之后，进行相应的设置就可以完成推理。Engine 不仅提供了推理相关的 API，也维护了内部推理的状态。Engine 执行流程分为 Init, SetInput, Run, GetOutput 四个阶段。Init 和 Run 都和算子相关。我们对原有的算子实现进行拆分，将部分资源的配置都放到了 Init 中处理。**

**  在 Executor 中，维护了每个核组推理所需要的资源。每个 Executor 对应一个核组。Executor 对于用户是不可见的，一个 Engine 可以持有多个 Executor。**

## Op - Tactic

**  为了能更好的解决一个算子对应多个计算实现的问题，我们在 Op 这个维度下面增加了一层 Tactic 的抽象，用来表现计算实现，由 Op 持有，这样就可以实现一个 Op 对应多个 Tactic，同时把不同的实现都解耦开来。下图是 Op 和 Tactic 关系的示意，**
问题： 1. 算子库或推理框架对同一个API有多种实现（比如：**一种shape和format的多个性能版本 --皓诚；一个功能由多个分支完成--如resize 立志**）。 2. 在框架层代替算子库对一个API下的多个 kernel做选择 。 // 完善

![](https://cdn.nlark.com/yuque/0/2023/png/812668/1690858864892-06407977-19c0-4ba6-b61d-6eadf2771b9a.png)

# 代码结构设计

**  在算子重构接入的过程中，我们不难发现，不同的算子内部的计算逻辑是不同的，但其外部表现是基本类似的，因此，选择抽象工厂模式和单例模式相结合，来实现算子的创建和管理。下面将以 Tactic 为例，介绍如何使用抽象工厂模式和单例模式来管理 Tactic。**

1. **创建 Tactic 抽象接口类**

```cpp
class AbstractOpTactic {
 public:
  virtual ~AbstractOpTactic() {}

  virtual void init(std::shared_ptr<OpAttrs> attrs, runtime::TVMArgs args,
                    runtime::TVMRetValue* ret) = 0;
  virtual void enqueue(runtime::TVMArgs args, runtime::TVMRetValue* ret) = 0;
};

class TecoOpTactic : public AbstractOpTactic {
 public:
  virtual ~TecoOpTactic() {}

  virtual void init(std::shared_ptr<OpAttrs> attrs, TVMArgs args, TVMRetValue* ret);
  virtual void enqueue(TVMArgs args, TVMRetValue* ret) = 0;
};
```

2. **实现 Tactic 的实体类（以 Clip 为例）**

```cpp
class TecoClipOpTactic : public TecoOpTactic {
 public:
  virtual ~TecoClipOpTactic(){};
  virtual void init(std::shared_ptr<OpAttrs> attrs, TVMArgs args, TVMRetValue* ret);
  virtual void enqueue(TVMArgs args, TVMRetValue* ret);
};
```

3. **创建 Tactic creator 的抽象接口类**

```cpp
class OpTacticCreator {
 public:
  virtual ~OpTacticCreator() {}
  virtual std::shared_ptr<AbstractOpTactic> create_op_tactic(OpType op_type) = 0;
};
```

4. **使用模版实现 Tactic creator 的实体类**

```cpp
template <typename T>
class TypeOpTacticCreator : public OpTacticCreator {
  std::shared_ptr<AbstractOpTactic> create_op_tactic(OpType op_type) {
    return std::make_shared<T>();
  }
};
```

5. **使用单例模式构建 Tactic 管理类，管理所有 Tactic creator 实现**

```cpp
class TecoOpTacticManager {
 public:
  static TecoOpTacticManager* get();

  void insert(const OpType& op_type, const std::shared_ptr<OpTacticCreator>& creator);
  std::unordered_set<std::shared_ptr<OpTacticCreator>> search(const OpType& op_type);

  TecoOpTacticManager(const TecoOpTacticManager&) = delete;
  TecoOpTacticManager(const TecoOpTacticManager&&) = delete;
  TecoOpTacticManager& operator=(const TecoOpTacticManager&) = delete;

  ~TecoOpTacticManager() = default;

 private:
  TecoOpTacticManager() {}

  static TecoOpTacticManager* teco_op_tactic_manager_;
  std::unordered_map<OpType, std::unordered_set<std::shared_ptr<OpTacticCreator>>>
      teco_op_tactic_map_;
};
```

6. **创建 Tactic 注册类**

```cpp
template <typename T>
class TecoTypeOpTacticRegister {
 public:
  explicit TecoTypeOpTacticRegister(OpType op_type) {
    auto op_tactic = std::make_shared<T>();
    auto* teco_op_tactic_manager = TecoOpTacticManager::get();
    teco_op_tactic_manager->insert(op_type, op_tactic);
  }
};
```

7. **向 Tactic 管理类添加 creator （以 Clip 为例）**

```cpp
TecoTypeOpTacticRegister<TypeOpTacticCreator<TecoClipOpTactic>> g_teco_Clip_tactic_register(CLIP);
```

**  这样，我们就搭建好了管理 Tactic 的工厂。当我们要增加新的算子，只需要参考步骤 2 创建 Tactic 的具体实现和步骤 7 将 Tactic 实现添加到管理类里即可。如果我们需要创建 Clip 算子的 Tactic 可以这样，**

```cpp
auto op_type_ = CLIP;
auto creator = TecoOpTacticManager::get()->search(op_type_).begin();
auto tactic = creator->create_op_tactic(op_type_);
```
