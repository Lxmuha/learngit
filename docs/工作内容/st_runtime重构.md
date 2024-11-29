# 1 需求

* **h2d/compute/d2h分离**
* **支持异步(多stream)**
* **支持多核组推理**

# 2 当前实现

## 2.1 compile过程

![](https://cdn.nlark.com/yuque/0/2023/png/29482156/1690941863884-dcde9b44-5a16-461f-b7db-7b3fad0ffc0f.png)

**编译出来的****runtime.Module可以理解成一个map结构，key是op name，value是op实现**

**参考：**

* [深度学习编译器 TVM 代码串讲](https://zhuanlan.zhihu.com/p/446976730)

## 2.2 graph executor实现

**整体流程如下：**

1. `<span class="ne-text">Init</span>`：构建图，构建op执行流。计算图显存共享。
2. `<span class="ne-text">SetInput</span>`: h2d，将输入copy至Init图时确定的地址处
3. `<span class="ne-text">Run</span>`：执行op，无memory操作
4. `<span class="ne-text">GetOutput</span>`：d2h, 将输出从Init图时确定的地址出copy至host端

**具体代码实现如下：**

### 2.2.1 Init

```cpp
/*!
 * \brief Initialize the graph executor with graph and device.
 * \param graph_json The execution graph.
 * \param module The module containing the compiled functions for the host
 * processor.
 * \param devs The devices of the host and devices where graph nodes will be
 * executed on.
 * \param lookup_linked_param_func Linked parameter lookup function. Default is nullptr.
 */
void GraphExecutor::Init(const std::string& graph_json, tvm::runtime::Module module,
                         const std::vector<Device>& devs,
                         const PackedFunc lookup_linked_param_func) {
  VLOG(1) << "[start] init graph executor";
  std::istringstream is(graph_json);
  dmlc::JSONReader reader(&is);
  this->Load(&reader);
  module_ = module;
  devices_ = devs;
  lookup_linked_param_ = lookup_linked_param_func;
  if (lookup_linked_param_ == nullptr) {
    lookup_linked_param_ = PackedFunc(
        [this](TVMArgs args, TVMRetValue* rv) { this->DefaultLookupLinkedParam(args, rv); });
  }
  // 开graph的空间，拷贝权重至device
  this->SetupStorage();
  // 构建op执行流
  this->SetupOpExecs();
  // 设置graph input
  for (size_t i = 0; i < input_nodes_.size(); i++) {
    const uint32_t nid = input_nodes_[i];
    std::string& name = nodes_[nid].name;
    input_map_[name] = i;
  }
  // 设置graph output
  for (size_t i = 0; i < outputs_.size(); i++) {
    const uint32_t nid = outputs_[i].node_id;
    std::string& name = nodes_[nid].name;
    std::stringstream ss;
    ss << name << ":" << i;
    output_map_[ss.str()] = i;
  }
  VLOG(1) << "[end] init graph executor";
}
```

**进一步具体看** `<span class="ne-text">SetStorage()</span>`和 `<span class="ne-text">SetupOpExecs()</span>`这两个接口做了什么事。

```cpp
void GraphExecutor::SetupStorage() {

  // Size and device type of each storage pool entry.
  std::vector<PoolEntry> pool_entry;
  // Find the maximum space size.
  for (size_t i = 0; i < attrs_.shape.size(); ++i) {
    // ...
  }

  // Allocate the space.
  for (const auto& pit : pool_entry) {
    // ...
  }

  // Assign the pooled entries. A unified memory pool is used to simplifiy
  // memory assignment for each node entry. The allocated memory on each device
  // is mapped to this pool.
  data_entry_.resize(num_node_entries());
  data_alignment_.resize(num_node_entries());
  for (size_t i = 0; i < data_entry_.size(); ++i) {
    int storage_id = attrs_.storage_id[i];
    ICHECK_LT(static_cast<size_t>(storage_id), storage_pool_.size());
    data_entry_[i] = storage_pool_[storage_id].CreateView(attrs_.shape[i], vtype[i]);

    const DLTensor* tmp = data_entry_[i].operator->();
    data_alignment_[i] = details::GetDataAlignment(*tmp);
  }
}
```

```cpp
void GraphExecutor::SetupOpExecs() {
  // 输入节点
  std::unordered_set<uint32_t> input_node_eids;
  for (size_t i = 0; i < input_nodes_.size(); i++) {
    uint32_t nid = input_nodes_[i];
    input_node_eids.insert(entry_id(nid, 0));
  }
  // 输出节点
  std::unordered_set<uint32_t> output_node_eids;
  for (size_t i = 0; i < outputs_.size(); i++) {
    output_node_eids.insert(entry_id(outputs_[i]));
  }

  // setup the array and requirements.
  // 执行流
  for (uint32_t nid = 0; nid < this->GetNumOfNodes(); ++nid) {
    const auto& inode = nodes_[nid];
    if (inode.op_type == "null") continue;
    std::vector<DLTensor> args;
    for (const auto& e : inode.inputs) {
      uint32_t eid = this->entry_id(e);
      args.push_back(*(data_entry_[eid].operator->()));
    }
    for (uint32_t index = 0; index < inode.param.num_outputs; ++index) {
      uint32_t eid = this->entry_id(nid, index);
      args.push_back(*(data_entry_[eid].operator->()));
    }
    ICHECK(inode.op_type == "tvm_op") << "Can only take tvm_op as op";

    std::shared_ptr<OpArgs> op_args = nullptr;
    std::tie(op_execs_[nid], op_args) = CreateTVMOp(inode.param, args);

    for (size_t i = 0; i < inode.inputs.size(); i++) {
      uint32_t input_eid = this->entry_id(inode.inputs[i]);
      // check if op input is model input
      if (input_node_eids.count(input_eid) > 0) {
        input_dltensors_[input_eid].push_back(
            static_cast<DLTensor*>(op_args->arg_values[i].v_handle));
      }
      // check if any model output is the input of the op
      if (output_node_eids.count(input_eid) > 0) {
        both_output_opinput_dltensors_[input_eid].push_back(
            static_cast<DLTensor*>(op_args->arg_values[i].v_handle));
      }
    }

    for (uint32_t i = inode.inputs.size(); i < inode.inputs.size() + inode.param.num_outputs; ++i) {
      uint32_t output_eid = this->entry_id(nid, i - inode.inputs.size());
      // check if op output is model output
      if (output_node_eids.count(output_eid) > 0) {
        output_dltensors_[output_eid].push_back(
            static_cast<DLTensor*>(op_args->arg_values[i].v_handle));
      }
    }
  }
}
```

### 2.2.2 SetInput

```cpp
void GraphExecutor::SetInput(int index, DLTensor* data_in) {
  VLOG(1) << "[start] graph executor set input";
  ICHECK_LT(static_cast<size_t>(index), input_nodes_.size());
  uint32_t eid = this->entry_id(input_nodes_[index], 0);
  // 输入tensor copy至data_entry里确定好的地址上
  data_entry_[eid].CopyFrom(data_in);
  VLOG(1) << "[end] graph executor set input";
}
```

### 2.2.3 Run

```cpp
void GraphExecutor::Run() {
  VLOG(1) << "[start] graph executor run";
  // setup the array and requirements.
  // 直接执行Init时确定的op_exec
  for (size_t i = 0; i < op_execs_.size(); ++i) {
    if (op_execs_[i]) op_execs_[i]();
  }
  VLOG(1) << "[end] graph executor run";
}
```

### 2.2.4 GetOutput

```cpp
NDArray GraphExecutor::GetOutput(int index) const {
  VLOG(1) << "[start] get output " << index;
  ICHECK_LT(static_cast<size_t>(index), outputs_.size());
  // 从data entry中确定的地址copy至host端
  uint32_t eid = this->entry_id(outputs_[index]);
  return data_entry_[eid];
  VLOG(1) << "[end] get output " << index;
}
```

# 3 方案

![](https://cdn.nlark.com/yuque/0/2023/png/29482156/1690942605465-a7fc3489-2b37-484d-84f0-8301c95c4c21.png)

[基于 Context 算子接入设计文档](https://taichu-platform.yuque.com/xy3e23/kmihv5/wsn9o0kglaeq3edv)
