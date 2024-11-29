# 变更历史

# 接口分析

[https://tvm.apache.org/docs/reference/api/python/relay/transform.html#tvm.relay.transform.DynamicToStatic](https://tvm.apache.org/docs/reference/api/python/relay/transform.html#tvm.relay.transform.DynamicToStatic)

1. **概述**

**将符合条件的动态算子转换为静态的。例如，**

```plain
                  %y(ny,cy,hy,wy)
                         |              DynamicToStatic         %x(nx,cx,hx,wx)
    %x(nx,cx,hx,wx)   shape_of                ===>                     |
                   \ /                                        reshape(ny,cy,hy,wy)
                 reshape
```

2. **接口参数**

`<span class="ne-text">tvm.relay.transform.DynamicToStatic()</span>`

**无参数**

# 实现分析

**python 接口如下：**

```python
def DynamicToStatic():
	# python 侧没有操作，直接调用C++侧实现
    return _ffi_api.DynamicToStatic()
```

**往下看 C++ 侧的代码实现：**

```cpp
Expr DynamicToStatic(Function f, IRModule m) {
  // 初始化 DynamicToStaticMutator
  // 创建 op_map_，将动态算子映射为静态算子
  // 主要有 reshape, squeeze, tile, topk, one_hot, resize2d 等
  DynamicToStaticMutator mutator(m, f);
  Expr expr = mutator.Mutate(f);
  // 处理输入的动态变量，将符合条件的转换为常量
  Expr out = mutator.PrepareInput(expr);
  return out;
}

Pass DynamicToStatic() {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) {
        return Downcast<Function>(DynamicToStatic(f, m));
      };
  return CreateFunctionPass(pass_func, 2, "DynamicToStatic", {});
}
```

**具体的转换过程是在 DynamicToStaticMutator 中实现的，我们具体来看，**

```cpp
class DynamicToStaticMutator : public MixedModeMutator {
 public:
  DynamicToStaticMutator(IRModule mod, Function func) : mod_(mod), func_(func) {
    // 初始化 op_map_
    op_map_ = {
        {Op::Get("dyn.reshape"),
         [this](const CallNode* call_node) {
           auto args = PrepareArgs(call_node);
           if (const ConstantNode* shape = args[1].as<ConstantNode>()) {
             ICHECK_EQ(shape->data->ndim, 1);
             return MakeReshape(call_node->args[0], ToVector(shape->data));
           }
           return Expr(nullptr);
         }},
    	// ...
    };
    Map<BaseFunc, GlobalVar> vars;
    for (auto kv : mod_->functions) {
      vars.Set(kv.second, kv.first);
    }
    gv_ = vars[func_];
  }

  Expr GetCurExpr(const Expr& original_expr) {
    if (original_expr.as<FunctionNode>()) {
      return mod_->Lookup(gv_);
    } else {
      return mod_->Lookup(gv_).as<FunctionNode>()->body;
    }
  }

  Expr PrepareInput(const Expr& expr) {
    BaseFunc func;
    if (auto* func_node = expr.as<BaseFuncNode>()) {
      func = GetRef<BaseFunc>(func_node);
    } else {
      func =
          relay::Function(relay::FreeVars(expr), expr, Type(), relay::FreeTypeVars(expr, mod_), {});
    }
    mod_->Update(gv_, func);

    mod_ = transform::FoldConstant()(mod_);
    transform::InferTypeLocal(GetCurExpr(expr));
    mod_ = transform::FoldConstant()(mod_);
    transform::InferTypeLocal(GetCurExpr(expr));

    Expr out;
    if (expr.as<FunctionNode>()) {
      out = mod_->Lookup(gv_);
    } else {
      out = mod_->Lookup(gv_).as<FunctionNode>()->body;
    }
    return out;
  }

  std::vector<Expr> PrepareArgs(const CallNode* call_node) {
    std::vector<Expr> args;
    for (auto arg : call_node->args) {
      if (arg.as<ConstantNode>()) {
        args.emplace_back(arg);
      } else {
        args.emplace_back(PrepareInput(arg));
      }
    }
    return args;
  }

 private:
  Expr Rewrite_(const CallNode* pre, const Expr& post) override {
    if (const CallNode* call_node = post.as<CallNode>()) {
      // 如果算子在 op_map_ 中，进行 dynamic -> static 转换
      if (op_map_.count(call_node->op)) {
        auto out = op_map_[call_node->op](call_node);
        if (out.defined()) {
          return out;
        }
      }
    }
    return post;
  }
```

# 示例

**相关测试代码路径为 **`<span class="ne-text">tests/python/contrib/test_swai/relay/test_pass_dynamic_to_static.py</span>`，挑选其中的 `<span class="ne-text">test_dynamic_to_static_reshape</span>`测试case具体来看。

`<span class="ne-text">test_dynamic_to_static_reshape</span>`测试代码如下：

```python
@tvm.testing.uses_gpu
def test_dynamic_to_static_reshape():
    def verify_reshape(shape, newshape, oshape):
        x = relay.var("x", relay.TensorType(shape, "float32"))
        y = relay.var("y", relay.TensorType(newshape, "float32"))
        z = relay.reshape(x, relay.shape_of(y))
        func = run_infer_type(relay.Function([x, y], z))
        func2 = run_opt_pass(run_opt_pass(func, transform.DynamicToStatic()), transform.InferType())

        zz = func2.body
        assert isinstance(zz, relay.Call)
        assert zz.op == relay.op.get("reshape")
        assert "newshape=" in zz.astext()
        assert zz.checked_type == relay.ty.TensorType(oshape, "float32")

        x_data = np.random.uniform(low=-1, high=1, size=shape).astype("float32")
        y_data = np.random.uniform(low=-1, high=1, size=newshape).astype("float32")
        ref_res = np.reshape(x_data, oshape)
        verify_func(func2, [x_data, y_data], ref_res)

    verify_reshape((2, 3, 4), (8, 3), (8, 3))
    verify_reshape((4, 7), (2, 7, 2), (2, 7, 2))
```

1. **原来的 IR 表达**

```plain
fn (%x: Tensor[(4, 7), float32] /* ty=Tensor[(4, 7), float32] */, %y: Tensor[(2, 7, 2), float32] /* ty=Tensor[(2, 7, 2), float32] */) -> Tensor[(?, ?, ?), float32] {
  %0 = shape_of(%y, dtype="int32") /* ty=Tensor[(3), int32] */;
  dyn.reshape(%x, %0, newshape=[]) /* ty=Tensor[(?, ?, ?), float32] */
} /* ty=fn (Tensor[(4, 7), float32], Tensor[(2, 7, 2), float32]) -> Tensor[(?, ?, ?), float32] */
```

2. **做如下改动**

```python
def run_opt_pass(expr, opt_pass, params=None):
    assert isinstance(opt_pass, tvm.transform.Pass)

    mod = tvm.IRModule.from_expr(expr)
    if params is not None:
        mod["main"] = bind_params_by_name(mod["main"], params)
    mod = opt_pass(mod)
    entry = mod["main"]
    return entry if isinstance(expr, relay.Function) else entry.body

func2 = run_opt_pass(run_opt_pass(func, transform.DynamicToStatic()), transform.InferType())
```

3. **转换过后的 IR 表达**

```plain
fn (%x: Tensor[(4, 7), float32] /* ty=Tensor[(4, 7), float32] */, %y: Tensor[(2, 7, 2), float32] /* ty=Tensor[(2, 7, 2), float32] */) -> Tensor[(2, 7, 2), float32] {
  reshape(%x, newshape=[2, 7, 2]) /* ty=Tensor[(2, 7, 2), float32] */
} /* ty=fn (Tensor[(4, 7), float32], Tensor[(2, 7, 2), float32]) -> Tensor[(2, 7, 2), float32] */
```

**优化前后的 IR 结构示意如下，**

```plain
                     %y(2,7,2)
                         |              DynamicToStatic            %x(4,7)
           %x(4,7)   shape_of                ===>                     |
                   \ /                                         reshape(2,7,2)
                 reshape
```

**可以看到，原来 reshape 的 output shape 需要使用 shape_of 动态计算得到，使用 DynamicToStatic 优化后，reshape 的 output shape 被固定为了常量。**

# 适配SWAI

**无需适配即可支持。**
