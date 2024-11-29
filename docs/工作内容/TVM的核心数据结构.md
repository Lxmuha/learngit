**本文旨在介绍TVM的核心数据结构。读者应在了解TVM各模块分工之后阅读本文。**

![](https://cdn.nlark.com/yuque/0/2024/png/36108506/1716364675599-4bb6ca96-847c-498e-98f9-22a43e9ff269.png)

# 准备部分

**TVM的核心数据类型一般会分成 XXX，与XXXNode两个类，其中XXX类是对XXXNode的引用，真正的成员变量存储在XXXNode中。**

**常见工具类：**

```plain
class PrimExpr : public BaseExpr {
  TVM_DLL PrimExpr operator+(PrimExpr a, PrimExpr b);

  TVM_DLL PrimExpr operator-(PrimExpr a, PrimExpr b);
}
```

```plain
class PackedFunc(PackedFuncBase):
  - Automatic exposure of C++ API into python
  - To call PackedFunc from python side
  - To call python callbacks to inspect results in generated code
  - Bring python hook into C++ backend
```

# Relay

**V0 组网**

**Expr: Relay层的核心抽象，通过递归的方式，可以从output Expr拿到整个网络的结构。**

**最常见的Expr: CallNode**

```plain
class RelayExprNode : public BaseExprNode {
  mutable Type checked_type_ = Type(nullptr);
}

class CallNode : public ExprNode {

 public:
  /*!
   * \brief The operator(function) being invoked
   *
   *  - It can be tvm::Op which corresponds to the primitive operators.
   *  - It can also be user defined functions (Function, GlobalVar, Var).
   */
  Expr op;

  /*! \brief The arguments(inputs) of the call */
  tvm::Array<relay::Expr> args;

  /*! \brief The additional attributes */
  Attrs attrs;
```

**V1 与OP部分的连接**

**OpNode: 通过字符串作为Key和OP部分建立连接**

```plain
class OpNode : public RelayExprNode {
  public:
    String name;
    Array<AttrFieldInfo> arguments;
}
```

```plain
reg.register_strategy("nn.softmax", strategy.softmax_strategy)
```

**V2 Shape的推导与记录**

**在Relay中会维护一套Type系统，TensorType会包含Tensor的Shape。**

```plain
class TensorTypeNode : public BaseTensorTypeNode {
  public:
    Array<PrimExpr> shape;
    DataType dtype;
}
```

**每个Expr的type都会在InferType后确定。**

```plain
class RelayExprNode : public BaseExprNode {
  mutable Type checked_type_ = Type(nullptr);
}
```

**TypeFunctor用来存储op的shape之间的计算关系**

```plain
template <typename R, typename... Args>
class TypeFunctor<R(const Type& n, Args...)> {
  R operator()(const Type& n, Args... args) { return VisitType(n, std::forward<Args>(args)...); }
}
```

**例子：**

```plain
RELAY_REGISTER_OP("nn.softmax")
    .describe(R"code(Softmax layer.

.. math:: \text{softmax}(x)_i = \frac{exp(x_i)}{\sum_j exp(x_j)}

.. note::
    This operator can be optimized away for inference.

- **data**: The input data
)code" TVM_ADD_FILELINE)
    .set_attrs_type<SoftmaxAttrs>()
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The input tensor.")
    .set_support_level(1)
    .add_type_rel("Softmax", SoftmaxRel)
    .set_attr<TOpPattern>("TOpPattern", kOutEWiseFusable);
```

**V3 更复杂的图表示（函数，分支，循环）**

```plain
// Functions that can be overriden by subclass
  virtual R VisitExpr_(const ConstantNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const TupleNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const VarNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const GlobalVarNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const FunctionNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const CallNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const LetNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const IfNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const OpNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const TupleGetItemNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const RefCreateNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const RefReadNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const RefWriteNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const ConstructorNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
  virtual R VisitExpr_(const MatchNode* op, Args... args) EXPR_FUNCTOR_DEFAULT;
```

**需要特别关注一下Function，Function对于子图划分，Lower作用都比较大**

```plain
class FunctionNode : public BaseFuncNode {
  public:
    tvm::Array<Var> params;
    Expr body;
    Type ret_type;
    tvm::Array<TypeVar> type_params;
}
```

# OP(TOPI/TE/TIR)

**TVM将OP的逻辑解耦为计算和调度**

**V0 计算**

```plain
using FTVMCompute = runtime::TypedPackedFunc<Array<te::Tensor>(
    const Attrs& attrs, const Array<te::Tensor>& inputs, const Type& out_type)>;
```

**te::Tensor**

```plain
class TensorNode : public DataProducerNode {
  public:
    Array<PrimExpr> shape;
    DataType dtype;
    Operation op;
}
```

**ComputeOpNode(pointwise): 只记录循环体内的计算关系**

```plain
class TVM_DLL ComputeOpNode : public BaseComputeOpNode {
  public:
    Array<PrimExpr> body;
}
```

**V1 调度**

```plain
using FTVMSchedule = runtime::TypedPackedFunc<te::Schedule(
    const Attrs& attrs, const Array<te::Tensor>& outs, const Target& target)>;
```

**记录调度的结构体：Schedule，核心是一个从Operation到Stage的映射，提供和记录一系列schedule操作。**

```plain
class ScheduleNode : public Object {
  public:
    Map<Operation, Stage> stage_map;
}
```

```plain
class StageNode : public Object {
  Operation op;
  Array<IterVar> all_iter_vars;
}
```

```plain
 # Schedule for A's(B's) shared memory load
        def shared_schedule(stage, strides):
            s[stage].compute_at(s[CF], ko)
            bs, xo, yo = stage.op.axis
            s[stage].storage_align(xo, strides - 1, strides)
            t = s[stage].fuse(xo, yo)
            t, vi = s[stage].split(t, factor=vec)
            t, tx = s[stage].split(t, factor=warp_size)
            t, ty = s[stage].split(t, factor=block_row_warps)
            _, tz = s[stage].split(t, factor=block_col_warps)
            s[stage].bind(ty, thread_y)
            s[stage].bind(tz, thread_z)
            s[stage].bind(tx, thread_x)
            s[stage].vectorize(vi)
```

**V2 Lower后的底层IR（TIR），和RelayExpr一样也可以继承出多种子类。**

```plain
class StmtNode : public Object {
}
```

**V3 OP的封装与组织**

**将op的字符串和stragy作为绑定。**

```plain
reg.register_strategy("nn.softmax", strategy.softmax_strategy)
```

**strategy可以理解为计算与调度的总和。把算子对应的FTVMCompute的结果，调用FTVMSchedule**

```plain
@softmax_strategy.register(["cuda", "gpu"])
def softmax_strategy_cuda(attrs, inputs, out_type, target):
    """softmax cuda strategy"""
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_compute_softmax(topi.nn.softmax),
        wrap_topi_schedule(topi.cuda.schedule_softmax),
        name="softmax.cuda",
    )
    if target.kind.name == "cuda" and "cudnn" in target.libs:
        strategy.add_implementation(
            wrap_compute_softmax(topi.cuda.softmax_cudnn),
            wrap_topi_schedule(topi.cuda.schedule_softmax_cudnn),
            name="softmax.cudnn",
            plevel=15,
        )
    return strategy
```

# Runtime(Graph Executor)

**V0 封装编译结果**

**Binary的封装：Module**

```plain
class TVM_DLL ModuleNode : public Object {
  virtual PackedFunc GetFunction(const String& name, const ObjectPtr<Object>& sptr_to_self) = 0;

  virtual void SaveToFile(const String& file_name, const String& format);

  void Import(Module other);
}
```

**IRModule：将IR（Relay IR或TIR）封装，为编译成可执行的Module做准备**

```plain
class IRModuleNode : public Object {
  public:
    Map<GlobalVar, BaseFunc> functions;

    Map<GlobalTypeVar, TypeData> type_definitions;
}
```

**V1 运行网络所需信息**

**GraphExecutor = Module + Json(记录网络节点拓扑序，attrs等) + 权重**

```plain
class TVM_DLL GraphExecutor : public ModuleNode {
  void Run();

  std::vector<std::function<void()>> op_execs_;

  struct NodeEntry {
    uint32_t node_id;

    // JSON Loader
    void Load(dmlc::JSONReader* reader) {

    }
  };
}
```

**V2 显存资源管理**

```plain
std::vector<NDArray> storage_pool_;

std::vector<NDArray> data_entry_;

storage_pool_.push_back(NDArray::Empty(shape, pit.dtype, dev, mem_scope));
data_entry_[i] = storage_pool_[storage_id].CreateView(attrs_.shape[i], vtype[i]);
```
