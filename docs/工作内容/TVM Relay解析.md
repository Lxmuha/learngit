# 【源码学习】 TVM Relay解析

# Relay Frontend

![](https://cdn.nlark.com/yuque/0/2024/png/38586013/1717488474490-34b66e4e-a812-4856-8924-0ef76a22e35f.png)

* **onnx**
* **tensorflow**
* **pytorch**
* **paddlepaddle**
* **oneflow**

## from_onnx

**加载onnx模型并将其转换为Realy IR。**

**核心逻辑实现在函数** **relay.frontend.from_onnx()** **中，位于tvm/python/tvm/relay/frontend/onnx.py。**

```python
def from_onnx(
    model, shape=None, dtype="float32", opset=None, freeze_params=True, convert_config=None
):
    """Convert a ONNX model into an equivalent Relay Function.

    Parameters
    ----------
    model : protobuf object
        ONNX ModelProto after ONNX v1.1.0

    shape : dict of str to tuple, optional
        The input shape to the graph

    dtype : str or dict of str to str
        The input types to the graph

    opset : int, optional
        Override to autodetected opset.
        This can be helpful for some testing.

    freeze_params: bool
        If this parameter is true, the importer will take any provided
        onnx input values (weights, shapes, etc) and embed them into the relay model
        as Constants instead of variables. This allows more aggressive optimizations
        at compile time and helps in making models static if certain inputs represent
        attributes relay would traditionally consider compile-time constants.

    convert_config : Optional[Dict[str, Any]]
        Default config:
            use_nt_batch_matmul : bool = True
                True to convert qualified onnx `matmul` to `nn.batch_matmul` strict to NT format
                (transpose_a=False, transpose_b=True).

    Returns
    -------
    mod : tvm.IRModule
        The relay module for compilation

    params : dict of str to tvm.nd.NDArray
        The parameter dict to be used by relay
    """
    # 新建GraphProto类来管理ONNX模型的OP转换以及生成Relay IR
    g = GraphProto(shape, dtype, freeze_params)
    # onnx模型的GraphProto
    graph = model.graph
    if opset is None:
        try:
            opset = model.opset_import[0].version if model.opset_import else 1
        except AttributeError:
            opset = 1
    # Use the graph proto as a scope so that ops can access other nodes if needed.
    with g:
	    # 从onnx graph转换成relay ir
        mod, params = g.from_onnx(graph, opset)

    if freeze_params:
        # 如果freeze_params为True，则会固定输入shape，接收非该shape的时候就会报错
        mod = relay.transform.DynamicToStatic()(mod)

    return mod, params
```

**从onnx graph转为relay ir的核心逻辑实现在** **g.from_onnx(graph, opset)** **。**

```python
def from_onnx(self, graph, opset, get_output_expr=False):
	"""Construct Relay expression from ONNX graph.

	Parameters
	----------
	graph : onnx protobuf object
		The loaded onnx graph

	opset : opset version

	get_output_expr: bool
		If set to true, this conversion will return each output expression rather
		than a packaged module. This can be useful when converting subgraphs to
		relay.

	Returns
	-------
	mod : tvm.IRModule
		The returned relay module

	params : dict
		A dict of name: tvm.nd.array pairs, used as pretrained weights
	"""
        self.opset = opset
        '''--------------------- Step1：解析权重 ----------------------'''
        for init_tensor in graph.initializer:
            if not init_tensor.name.strip():
                raise ValueError("Tensor's name is required.")
            # 先把这个TensorProto使用get_numpy函数获得值，再reshape到特定形状，
            # 再基于这个numpy构造tvm.nd.array
            array = self._parse_array(init_tensor)
            if self._freeze_params:
                self._nodes[init_tensor.name] = _expr.const(array)
            else:
                self._params[init_tensor.name] = array
                self._nodes[init_tensor.name] = new_var(
                    init_tensor.name,
                    shape=self._params[init_tensor.name].shape,
                    dtype=self._params[init_tensor.name].dtype,
                )

        '''--------------------- Step2：解析输入 ----------------------'''
        for i in graph.input:
            i_name, i_shape, d_type, i_shape_name = get_info(i)
            if i_name in self._params:
                # i is a param instead of input
                self._num_param += 1
                self._params[i_name] = self._params.pop(i_name)
                self._nodes[i_name] = new_var(
                    i_name, shape=self._params[i_name].shape, dtype=self._params[i_name].dtype
                )
            # 输入节点已经在Relay IR中，跳过不处理
            elif i_name in self._nodes:
                continue
            else:
                # 真正的输入节点，依赖用户进行指定
                self._num_input += 1
                self._input_names.append(i_name)
                if i_name in self._shape:
                    i_shape = self._shape[i_name]
                else:
                    if "?" in str(i_shape):
                        warning_msg = (
                            "Input %s has unknown dimension shapes: %s. "
                            "Specifying static values may improve performance"
                            % (i_name, str(i_shape_name))
                        )
                        warnings.warn(warning_msg)
                if isinstance(self._dtype, dict):
                    dtype = self._dtype[i_name] if i_name in self._dtype else d_type
                else:
                    dtype = d_type
                self._nodes[i_name] = new_var(i_name, shape=i_shape, dtype=dtype)
            self._inputs[i_name] = self._nodes[i_name]

        # 获取不支持的算子列表
        convert_map = _get_convert_map(opset)
        unsupported_ops = set()
        for node in graph.node:
            op_name = node.op_type
            if (
                op_name not in convert_map
                and op_name != "Constant"
                and op_name not in _identity_list
            ):
                unsupported_ops.add(op_name)
        # 输出不支持的算子集合并抛出异常
        if unsupported_ops:
            msg = "The following operators are not supported for frontend ONNX: "
            msg += ", ".join(unsupported_ops)
            raise tvm.error.OpNotImplemented(msg)

        ## 执行到这里说明ONNX模型的所有算子都被Relay支持，可以进行转换
        '''--------------------- Step3：解析每个op ----------------------'''
        for node in graph.node:
            op_name = node.op_type
            # 解析attribute
            attr = self._parse_attr(node.attribute)
            inputs = onnx_input()
            for i in node.input:
                if i != "":
                    inputs[i] = self._nodes[self._renames.get(i, i)]
                else:
                    inputs[i] = None
            i_name = self._parse_value_proto(node)
            node_output = self._fix_outputs(op_name, node.output)
            attr["tvm_custom"] = {}
            attr["tvm_custom"]["name"] = i_name
            attr["tvm_custom"]["num_outputs"] = len(node_output)
            # 执行每个op的转换操作
            op = self._convert_operator(op_name, inputs, attr, opset)
            if not isinstance(op, _expr.TupleWrapper):
                outputs_num = 1
            else:
                outputs_num = len(op)
            if outputs_num > 1:
                valid_outputs = [False] * outputs_num
                for i, output in enumerate(node_output):
                    if output != "":
                        valid_outputs[i] = True
                if not all(valid_outputs):
                    tup = op.astuple()
                    if isinstance(tup, _expr.Tuple):
                        outputs = [tup.fields[i] for i, valid in enumerate(valid_outputs) if valid]
                    else:
                        outputs = [op[i] for i, valid in enumerate(valid_outputs) if valid]
                    if len(outputs) == 1:
                        op = outputs[0]
                    else:
                        op = _expr.TupleWrapper(outputs, len(outputs))
                    outputs_num = len(outputs)
                    node_output = [output for output in node_output if output != ""]
            assert (
                len(node_output) == outputs_num
            ), "Number of output mismatch {} vs {} in {}.".format(
                len(node_output), outputs_num, op_name
            )
            if outputs_num == 1:
                self._nodes[node_output[0]] = fold_constant(op)
            else:
                op = _expr.TupleWrapper(fold_constant(op.astuple()), len(op))
                for k, i in zip(list(node_output), range(len(node_output))):
                    self._nodes[k] = op[i]

        '''--------------------- Step4：解析输出 ----------------------'''
        outputs = [self._nodes[self._parse_value_proto(i)] for i in graph.output]
        outputs = outputs[0] if len(outputs) == 1 else _expr.Tuple(outputs)
        # 如果需要直接返回转换后的表达式，在这里return
        if get_output_expr:
            return outputs
        free_vars = analysis.free_vars(outputs)
        nodes = {v: k for k, v in self._nodes.items()}
        free_vars = [nodes[var] for var in free_vars]
        for i_name in self._params:
            if i_name in free_vars and i_name not in self._inputs:
                self._inputs[i_name] = self._nodes[i_name]

        '''--------------------- Step5：创建返回值 ----------------------'''
        # 根据输出表达式和所有输入变量创建函数
        func = _function.Function([v for k, v in self._inputs.items()], outputs)
        # 把这个函数用IRModule包起来，同时返回权重参数
        return IRModule.from_expr(func), self._params
```

**遍历onnx graph，依次解析权重/input/ops/output，通过每个op特有的** **_convert_operator()** **函数逐个转换op。**

```python
def _convert_operator(self, op_name, inputs, attrs, opset):
	"""Convert ONNX operator into a Relay operator.
	The converter must specify conversions explicitly for incompatible name, and
	apply handlers to operator attributes.

	Parameters
	----------
	op_name : str
		Operator name, such as Convolution, FullyConnected
	inputs : list of tvm.relay.function.Function
		List of inputs.
	attrs : dict
		Dict of operator attributes
	opset : int
		Opset version

	Returns
	-------
	sym : tvm.relay.function.Function
		Converted relay function
	"""
	# 获取tvm支持的op map
	convert_map = _get_convert_map(opset)
	if op_name in _identity_list:
		sym = get_relay_op(op_name)(*inputs, **attrs)
	elif op_name in convert_map:
		sym = convert_map[op_name](inputs, attrs, self._params)
	else:
		raise NotImplementedError("Operator {} not implemented.".format(op_name))
	return sym
```

**从** **_get_convert_map(opset)** **中获取支持的op map，key是onnx的op name，value是转换之后的relay ir。**

```python
def _get_convert_map(opset):
    return {
        "Constant": Constant.get_converter(opset),
        "Add": Add.get_converter(opset),
        "Sub": Sub.get_converter(opset),
        "Mul": Mul.get_converter(opset),
        "Div": Div.get_converter(opset),
        "Sqrt": Renamer("sqrt"),
        "Relu": Renamer("relu"),
        "Celu": Celu.get_converter(opset),
        "LeakyRelu": Renamer("leaky_relu"),
        "Exp": Renamer("exp"),
        "Greater": Renamer("greater"),
        "Less": Renamer("less"),
        "Log": Renamer("log"),
        "Cos": Renamer("cos"),
        "Tan": Renamer("tan"),
        "AveragePool": AveragePool.get_converter(opset),
        "MaxPool": MaxPool.get_converter(opset),
        "Conv": Conv.get_converter(opset),
        "ConvTranspose": ConvTranspose.get_converter(opset),
        "BatchNormalization": BatchNorm.get_converter(opset),
        "RNN": RNN.get_converter(opset),
        "LSTM": LSTM.get_converter(opset),
        "MaxRoiPool": MaxRoiPool.get_converter(opset),
        "ReduceMax": ReduceMax.get_converter(opset),
        "ReduceMin": ReduceMin.get_converter(opset),
        "Transpose": AttrCvt("transpose", {"perm": "axes"}),
        "Loop": Loop.get_converter(opset),
        "If": If.get_converter(opset),
        ... ...
    }
```

**以****Conv**为例分析onnx op转换为relay ir的过程。

```python
class Conv(OnnxOpConverter):
    """Operator converter for Conv."""

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        # input[0]为输入的relay ir
        data = inputs[0]
        input_shape = infer_shape(data)
        ndim = len(input_shape)
        # auto_pad是ONNX特有属性，Relay的Conv不支持该属性
        # 需要将Pad的数值计算出来并分情况进行处理
        if "auto_pad" in attr:
            attr["auto_pad"] = attr["auto_pad"].decode("utf-8")
            if attr["auto_pad"] in ("SAME_UPPER", "SAME_LOWER"):
                data = autopad(data, attr["strides"], attr["kernel_shape"], attr["dilations"], ndim)
            elif attr["auto_pad"] == "VALID":
                attr["pads"] = tuple([0 for i in range(ndim - 2)])
            elif attr["auto_pad"] == "NOTSET":
                pass
            else:
                msg = 'Value {} in attribute "auto_pad" of operator Conv is invalid.'
                raise tvm.error.OpAttributeInvalid(msg.format(attr["auto_pad"]))
            attr.pop("auto_pad")

        out = AttrCvt(
            op_name=dimension_picker("conv"),
            transforms={
                "kernel_shape": "kernel_size",
                "dilations": ("dilation", 1),
                "pads": ("padding", 0),
                "group": ("groups", 1),
            },
            custom_check=dimension_constraint(),
        )([data, inputs[1]], attr, params)

        use_bias = len(inputs) == 3
        if use_bias:
            out = _op.nn.bias_add(out, inputs[2])
        return out
```

**至此，完成了整个onnx模型到Relay IR的转换。**

# Relay Passes

![](https://cdn.nlark.com/yuque/0/2024/png/38586013/1717488543661-677bdf32-14fa-4253-83bc-d9e9c982730d.png)

**Pass是TVM中基于Relay IR进行的一系列优化，可以对计算图进行变换，去除冗余算子，提高模型的推理效率。**

## Infrastructure

**Relay pass infrastructure 的设计很大程度上受到 LLVM 中使用的分层pass manager和流行的深度学习框架中使用的block-style容器的启发，代码位于tvm/include/tvm/ir/transform.h 中，主要目标包括：**

* **实现更好的optimizer编程编排，允许用户灵活地定制和构建自己的pipeline**
* **提供一种用户友好的方式来调试passes**
* **减轻开发人员解决passes之间的依赖关系，简化实现新passes的难度**

### C++ backend

**TVM的核心数据类型一般会分成 XXX，与****XXXNode**两个类，其中XXX类是对XXXNode的引用，真正的成员变量存储在XXXNode中。

#### PassInfo

**Relay提供了 ****PassInfo** 对象来包含一个pass所需的基本信息。

```cpp
class PassInfoNode : public Object {
// pass 名称
String name;
// 启用 pass 的优化级别
int opt_level;
// 执行某个 pass 所需的 pass
Array<String> required;
};
```

#### PassContext

**PassContext是****PassContextNode**的引用，带有用于优化pass的有用信息。

```cpp
class PassContextNode : public Object {
public:
// 用于pass作者提供有关优化失败原因的注释
ErrorReporter err_reporter;
// 优化级别，默认为2
int opt_level{2};
// 必需的pass
tvm::Array<tvm::Expr> required_pass;
// 禁用的pass
tvm::Array<tvm::Expr> disabled_pass;
};
```

#### PassConstructs

**Pass Infrastructure可以在****不同粒度**的Relay程序下工作。引入了一个抽象类 **PassNode** 作为不同优化pass的基础。子类在modules, functions, or sequences of passes级别重写方法的实现，子类本身都可以充当pass管理器，可以收集所需的passes并执行它们或基于给定的元数据构建依赖关系图。

```cpp
class PassNode : Object {
  virtual PassInfo Info() const = 0;
  virtual Module operator()(const IRModule& mod
                            const PassContext& pass_ctx) const = 0;
};
```

**社区源码中已经创建了几个子类来实现不同类型的优化pass：**

* **Module Level Passes**，主要用于全局和过程间优化 (IPO)，用户可以在此级别从一个module中添加和/或删除function。

```cpp
class ModulePassNode : PassNode {
  // 维护module-level pass所需的信息
  PassInfo pass_info;
  // optimization功能的实现
  runtime::TypedPackedFunc<Module(Module, PassContext)> pass_func;
  Module operator()(const Module& mod, const PassContext& pass_ctx) const final;
};
```

* **Function Level Passes**，用于为给定的 Relay module实现各种内部函数级优化，它一次从module的函数列表中获取一个函数以进行优化，并生成一个重写的 Relay Function 或 tir PrimFunc。

```cpp
class FunctionPassNode : PassNode {
  PassInfo pass_info;
  runtime::TypedPackedFunc<Function(Function, Module, PassContext)> pass_func;
  Module operator()(const Module& mod, const PassContext& pass_ctx) const final;
  // 可选择忽略的Function
  bool SkipFunction(const Function& func) const;
};
```

* **Sequential Passes**，一个pass序列，包含许多用于执行的passes。

```cpp
class SequentialPassNode : PassNode {
  PassInfo pass_info;
  // 需要执行的passes
  Array<Pass> passes;
  bool PassEnabled(const PassInfo& info) const;
  Module operator()(const Module& mod, const PassContext& pass_ctx) const final;
};
```

#### Pass Registration

**以FoldConstant为例，将这个pass注册到pass infra。**

```cpp
namespace transform {

Pass FoldConstant() {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
    [=](Function f, IRModule m, PassContext pc) {
      return Downcast<Function>(FoldConstant(f));
  };
  // 创建Function级别的pass, 最后的{}表示不需要先决条件
  return CreateFunctionPass(pass_func, 2, "FoldConstant", {});
}

TVM_REGISTER_GLOBAL("relay._transform.FoldConstant")
.set_body_typed(FoldConstant);

}  // namespace transform
```

### Python Frontend

**python前端为用户提供以下 APIs 来创建和执行一个 pass，后端接收信息并决定它应该使用哪个函数来创建 Pass 对象。**

**以PassContext为例，Python前端为其提供了一个包装器，通过覆盖 __enter__ 和 __exit__ 来启用 with 语法。为用户提供了一个 current 静态方法来获取在特定范围内使用的上下文。**

```python
@tvm._ffi.register_object("transform.PassContext")
class PassContext(tvm.runtime.Object):
    def __enter__(self):
        _transform.EnterPassContext(self)
        return self

    def __exit__(self, ptype, value, trace, config):
        _transform.ExitPassContext(self)

    @staticmethod
    def current():
        """Return the current pass context."""
        return _transform.GetCurrentPassContext()
```

**使用方式：**

```python
with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(func, "llvm", params=params)
```

## Graph Structure

**神经网络的计算图是一种树状结构，在编译器前端称之为抽象语法树（AST）。**

### Node

**Relay的图节点定义为在 include/tvm/relay/expr.h 中，主要有以下几种类型：ConstantNode、VarNode、TupleNode、CallNode、LetNode、IfNode。以****IfNode**和**CallNode**为例， 其实现如下：

```cpp
class IfNode : public ExprNode {
 public:
  /*! \brief The condition */
  Expr cond;
  /*! \brief The expression evaluated when condition is true. */
  Expr true_branch;
  /*! \brief The expression evaluated when condition is false */
  Expr false_branch;
};

class CallNode : public ExprNode {
 public:
  /*!
   * \brief The operator(function) being invoked
   */
  Expr op;
  /*! \brief The arguments(inputs) of the call */
  tvm::Array<relay::Expr> args;
  /*! \brief The additional attributes */
  Attrs attrs;
};
```

### Node Visit

**对图结构节点的访问使用****ExprVisitor**类，不修改程序只是执行程序分析和收集信息。

**ExprVisitor继承自ExprFunctor，ExprFunctor设置了** **VisitExpr_** **的虚函数，在解析时会回到ExprVisitor来解析节点。ExprFunctor提供了一个public接口方法** **VisitExpr** **，它接受一个表达式和零个或多个参数并返回某种类型的实例。**

**VisitExpr和VisitExpr_之间的关系与调度有关，每个VisitExpr_定义针对特定类型的表达式，但访问时并不总是知道将访问节点是哪种类型。因此ExprFunctor提供了一个VisitExpr函数，定义了自己的 vtable由VisitExpr使用，它从给定的表达式路由到处理特定类型的VisitExpr。**

```cpp
class ExprFunctor<R(const Expr& n, Args...)> {
public:
/*! \brief the result type of this functor */
using result_type = R;
/*!
   * \brief The functor call.
   * \param n The expression node.
   * \param args Additional arguments.
   * \return The result of the call
   */
virtual R VisitExpr(const Expr& n, Args... args) {
    ICHECK(n.defined()) << "Found null pointer node while traversing AST. The previous pass may "
        "have generated invalid data.";
    static FType vtable = InitVTable();
    return vtable(n, this, std::forward<Args>(args)...);
}
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
virtual R VisitExprDefault_(const Object* op, Args...) {
    LOG(FATAL) << "Do not have a default for " << op->GetTypeKey();
    throw;
}
} // class ExprFunctor
```

**ExprVisitor类继承自ExprFunctor 并**提供 **VisitExpr_** **的默认实现**，用于捕获每个表达式类型的常见遍历模式。

```cpp
class ExprVisitor : public ::tvm::relay::ExprFunctor<void(const Expr& n)> {
 public:
  void VisitExpr(const Expr& expr) override;
  void VisitExpr_(const VarNode* op) override;
  void VisitExpr_(const GlobalVarNode* op) override;
  void VisitExpr_(const ConstantNode* op) override;
  void VisitExpr_(const TupleNode* op) override;
  void VisitExpr_(const FunctionNode* op) override;
  void VisitExpr_(const CallNode* op) override;
  void VisitExpr_(const LetNode* op) override;
  void VisitExpr_(const IfNode* op) override;
  void VisitExpr_(const OpNode* op) override;
  void VisitExpr_(const TupleGetItemNode* op) override;
  void VisitExpr_(const RefCreateNode* op) override;
  void VisitExpr_(const RefReadNode* op) override;
  void VisitExpr_(const RefWriteNode* op) override;
  void VisitExpr_(const ConstructorNode* op) override;
  void VisitExpr_(const MatchNode* op) override;
  virtual void VisitType(const Type& t);
  virtual void VisitClause(const Clause& c);
  virtual void VisitPattern(const Pattern& c);
  virtual void VisitSpan(const Span& span);
} // class ExprVisitor
```

### Node Mutate

**对图结构节点的修改使用****ExprMutator**子类，它和ExprVisitor一样继承自ExprFunctor。

```cpp
class ExprMutator : public ::tvm::relay::ExprFunctor<Expr(const Expr&)> {
 public:
  /*!
   * \brief Mutate is alias for VisitExpr
   * \return expr.
   */
  Expr Mutate(const Expr& expr) { return this->VisitExpr(expr); }
  Expr VisitExpr(const Expr& expr) override;
  Expr VisitExpr_(const VarNode* op) override;
  Expr VisitExpr_(const ConstantNode* op) override;
  Expr VisitExpr_(const GlobalVarNode* op) override;
  Expr VisitExpr_(const OpNode* op) override;
  Expr VisitExpr_(const TupleNode* op) override;
  Expr VisitExpr_(const FunctionNode* op) override;
  Expr VisitExpr_(const CallNode* call_node) override;
  Expr VisitExpr_(const LetNode* op) override;
  Expr VisitExpr_(const IfNode* op) override;
  Expr VisitExpr_(const TupleGetItemNode* op) override;
  Expr VisitExpr_(const RefCreate来表记Node* op) override;
  Expr VisitExpr_(const RefReadNode* op) override;
  Expr VisitExpr_(const RefWriteNode* op) override;
  Expr VisitExpr_(const ConstructorNode* op) override;
  Expr VisitExpr_(const MatchNode* op) override;

  /*!
   * \brief Used to visit the types inside of expressions.
   *
   * Can be overloaded to transform the types in arbitrary
   * ways, one way would be to define a sub-class of type
   * visitor for types which transform them appropriately.
   */
  virtual Type VisitType(const Type& t);
  virtual Clause VisitClause(const Clause& c);
  virtual Pattern VisitPattern(const Pattern& c);

 protected:
  /*! \brief Internal map used for memoization. */
  std::unordered_map<Expr, Expr, ObjectPtrHash, ObjectPtrEqual> memo_;
};
```

 **memo_** **成员变量存储了图中的各个节点，其VisitExpr实现如下：**

```cpp
Expr ExprMutator::VisitExpr(const Expr& expr) {
  auto it = this->memo_.find(expr);
  if (it != this->memo_.end()) {
    return it->second;
  } else {
    Expr new_expr = ExprFunctor::VisitExpr(expr);
    memo_[expr] = new_expr;
    return new_expr;
  }
}
```

**VisitExpr会根据Expr的类型分发到各个VisitExpr_上执行，以****IfNode**为例：

```cpp
Expr ExprMutator::VisitExpr_(const IfNode* op) {
  auto guard = this->Mutate(op->cond);
  auto true_b = this->Mutate(op->true_branch);
  auto false_b = this->Mutate(op->false_branch);
  if (op->cond.same_as(guard) && op->true_branch.same_as(true_b) &&
      op->false_branch.same_as(false_b)) {
    // 如果IFNode的子节点都没有被修改，那么就返回这个节点本身
    return GetRef<Expr>(op);
  } else {
    // 否则说明IFNode的子节点被修改了，创建新的If节点并返回
    return If(guard, true_b, false_b, op->span);
  }
}
```

## General optimization

### EliminateCommonSubexpr

**公共子表达式消除Pass，公共子表达式指的是具有相同的**OP类型**以及相同的**参数**，并且参数的**顺序**都是完全相同的，那么这些表达式就可以合成一个公共子表达式。例如下面的例子，经过公共子表达式消除Pass后计算图就会消除其中一个表达式：**

**a = b + c**

**d = b + c**

**CommonSubexprEliminator类的VisitExpr_成员函数又调用了Rewrite_，重载了两个** **Rewrite_** **函数来对expr进行遍历和重写，以重写CallNode为例：**

```cpp
Expr Rewrite_(const CallNode* call, const Expr& post) final {
    static auto op_stateful = Op::GetAttrMap<TOpIsStateful>("TOpIsStateful");
    Expr new_expr = post;
    const CallNode* new_call = new_expr.as<CallNode>();
    ICHECK(new_call);
    const OpNode* op = new_call->op.as<OpNode>();
    StructuralEqual attrs_equal;

    if (new_call->args.size() == 0 || op == nullptr || op_stateful.get(GetRef<Op>(op), false)) {
      return new_expr;
    }
    if (fskip_ != nullptr && fskip_(new_expr)) {
      return new_expr;
    }

    // expr_map_记录了遍历过的具有相同op的expr
    auto it = expr_map_.find(new_call->op);
    // 遇到相同op的表达式，匹配OP的attrs及参数列表
    if (it != expr_map_.end()) {
      for (const Expr& candidate_expr : it->second) {
        if (const CallNode* candidate = candidate_expr.as<CallNode>()) {
          bool is_equivalent = true;
          // attrs匹配
          if (!attrs_equal(new_call->attrs, candidate->attrs)) {
            continue;
          }
          // args匹配
          for (size_t i = 0; i < new_call->args.size(); i++) {
            if (!new_call->args[i].same_as(candidate->args[i]) &&
                !IsEqualScalar(new_call->args[i], candidate->args[i])) {
              is_equivalent = false;
              break;
            }
          }
          if (!is_equivalent) continue;
          // 确认两个表达式是公共表达式，不返回新的表达式
          return GetRef<Call>(candidate);
        }
      }
    }
    // 创建新的表达式
    expr_map_[new_call->op].push_back(new_expr);
    return new_expr;
  }
```

### FoldConstant

**常量折叠Pass对ir中能提前计算的常量数据进行计算，并构建出新的const节点用来存储计算后的常量。FoldConstant类的VisitExpr_成员函数又调用了** **Rewrite_** **，以重写CallNode为例：**

```cpp
Expr Rewrite_(const CallNode* call, const Expr& post) final {
    if (inside_primitive) {
      return GetRef<Expr>(call);
    }
    static auto op_stateful = Op::GetAttrMap<TOpIsStateful>("TOpIsStateful");

    auto origin_args = call->args;
    call = post.as<CallNode>();
    // We don't constant fold function with zero arguments.
    // This is a heuristic that is useful.
    // For example it is harmful to fold ones(shape=(4, 5)).
    if (call->args.size() == 0) return post;
    const OpNode* op = call->op.as<OpNode>();
    if (op == nullptr) return post;
    // skip stateful ops.
    if (op_stateful.get(GetRef<Op>(op), false)) return post;
    // Try to evaluate shape_of op
    if (call->op == shape_of_op_ || call->op == vm_shape_of_op_) {
      return EvaluateShapeOf(post, origin_args, call->attrs);
    }

    if (call->op == ndarray_size_op_) {
      return EvaluateNdarraySize(post, origin_args, call->attrs);
    }

    // We should think about potentially constant evaluation over these ops too.
    static auto fnoncomputational = Op::GetAttrMap<TNonComputational>("TNonComputational");
    if (const auto* call_node = call->op.as<OpNode>()) {
      Op op = GetRef<Op>(call_node);
      if ((fnoncomputational.count(op) && fnoncomputational[op]) || (call->op == device_copy_op_)) {
        return GetRef<Call>(call);
      }
    }

    bool all_const_args = true;
    for (Expr arg : call->args) {
      if (!checker_.Check(arg)) {
        all_const_args = false;
      }
    }
    if (all_const_args) {
      return ConstEvaluate(post);
    } else {
      return post;
    }
  }
```

## Specific optimization

**我们在基于原生TVM的开发中，针对算子库限制及硬件特性加入了一些新的优化pass，比如：**

* **各种算子融合**
* **AlterOpLayout**
* **MultiHeadAttentionOpt**
