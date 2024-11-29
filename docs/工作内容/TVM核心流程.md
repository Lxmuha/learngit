## 【源码学习】TVM核心流程

## 一 算子接入流程

**输入：算子的attribute, type relation, fcompute, fschedule**

**输出：无**

**效果：包含该算子的Relay IR可以编译**

[https://tvm.apache.org/docs/dev/how_to/relay_add_op.html#](https://tvm.apache.org/docs/dev/how_to/relay_add_op.html#)

## 二 Relay Build流程

**输入：Relay IR + TOPI库 + Target**

**输出：runtime::module**

**1 Relay IR -> IRModule**

**通过from_expr函数可以将Relay IR转成IR Module**

```plain
mod = tvm.IRModule.from_expr(relay.Function([x, y], x + y))
```

**核心函数：**

```plain
std::pair<IRModule, GlobalVar> IRModule::FromExprInContext(
    const RelayExpr& expr, const tvm::Map<GlobalVar, BaseFunc>& global_funcs,
    const tvm::Map<GlobalTypeVar, TypeData>& type_definitions,
    std::unordered_set<String> import_set)
```

**核心逻辑是将各种Relay IR转成统一的GlobalVar -> Function的map的形式。**

**2 IRModule(Relay Function) -> IRModule(TIR Statement) **

**流程入口：**

```plain
void BuildRelay(IRModule relay_module, const String& mod_name)
```

**Lower这一步用一个Pass作为入口：**

```plain
Pass LowerTE(String module_name, CompilationConfig complilation_config, ProcessFn process_fn) {
  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func = [=](IRModule module,
                                                                            PassContext ctx) {
    return LowerTE(module, module_name, process_fn, complilation_config);
  };

  return tvm::transform::Sequential(
      {tvm::relay::transform::RelayToTIRTargetHook(complilation_config),
       tvm::transform::CreateModulePass(pass_func, 0, "LowerTE", {"InferType"}), InferType(),
       tvm::tir::transform::ExtractPrimFuncConstants()});
}
```

**以Function作为单位完成Lower:**

```plain
IRModule lowered_module = compiler->GetLoweredFunctions();
```

**以CallNode作为Relay与TIR的层次分界点：**

```plain

class LowerTensorExprMutator : public DeviceAwareExprMutator {
  Expr DeviceAwareVisitExpr_(const CallNode* call_node) override {
    // Look for (possibly indirect) calls to primitives.
    BaseFunc primitive_func = ResolveToPrimitive(call_node->op);
  }
}
```

**最终通过TE的ScheduleBuilder完成单个算子的Lower:**

```plain
/*!
 * \brief Create schedule for target.
 * \param source_func The primitive function to be lowered.
 * \param target The target we want to create schedule for.
 * \return Pair of schedule and cache.
 *  The funcs field in cache is not yet populated.
 */
CachedFunc PrimFuncFor(const Function& source_func, const Target& target,
                       GlobalVarSupply global_var_supply, NameSupply constant_name_supply) {
  return ScheduleBuilder(target).Create(source_func, global_var_supply, constant_name_supply);
}
```

**3 IRModule(TIR Statement) -> runtime::Module**

**在BuildRelay中完成Lower之后：**

```plain
ret_.mod = tvm::TIRToRuntime(lowered_funcs, host_target);
```

**codegen完成从TIR到binary的转换：**

```plain
runtime::Module mhost = codegen::Build(mhost_all, target_host);
```

## 三 用户自定义Compiler（以cutlass为例）

**输入：Relay IR + 对应节点的Lower逻辑**

**输出：runtime::Module，其中对应节点的实现会对应到专门的逻辑，如Cutlass代码。**

**入口：在Relay层面完成子图切分**

```plain
def partition_for_cutlass(mod, params=None):
  cutlass_patterns = relay.op.contrib.get_pattern_table("cutlass")

    seq = Sequential(
        [
            transform.InferType(),
            transform.MergeComposite(cutlass_patterns),
            transform.AnnotateTarget(["cutlass"], include_non_call_ops=False),
            transform.PartitionGraph(bind_constants=False),
        ]
    )
```

**划分完成后，用自定义的codegen类完成Relay Function到字符串的转换：**

```plain
class CutlassModuleCodegen {
 public:
  explicit CutlassModuleCodegen(IRModule mod) : mod_(std::move(mod)) {}

  runtime::Module CreateCSourceModule() {
    for (const auto& entry : mod_->functions) {
      if (const auto* function_node = GetCutlassFunctionNode(entry.second)) {
        GenCutlassFunc(GetRef<Function>(function_node));
      }
    }
    return Finalize(code_stream_.str(), func_names_);
  }

  void GenCutlassFunc(const Function& function) {
    CodegenCutlass builder(sid, dict);
    auto out = builder.VisitExpr(function->body);
    auto code = builder.JIT(out);
    for (const auto& header : builder.GetHeaders()) {
      code_stream_ << "#include <" << header << ">\n";
    }
    code_stream_ << "\n" + code;
  }

  }
```

**TVM提供了从C代码字符串到runtime::Module，也就是Binary的流程：**

```plain
class CSourceCodegen : public CSourceModuleCodegenBase {
 public:
  // Pass a subgraph function, and generate the C code.
  void GenCFunc(const Function& func) { ; }

  // Use GenCFunc to generate the C code and wrap it as a C source module.
  runtime::Module CreateCSourceModule(const NodeRef& ref) override { ; }

 private:
  std::ostringstream code_stream_;
};
```
