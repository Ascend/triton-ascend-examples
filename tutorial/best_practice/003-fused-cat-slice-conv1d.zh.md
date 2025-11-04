# 004-fused-cat-slice-conv1d.py说明

## 简介
该文件借由Qwen3-Next中的`causal_conv1d_update`triton kernel的Ascend亲和重构，介绍开源triton算子在昇腾硬件上的常见优化点和综合优化手段。

## 算子功能：
该算子的原始功能见代码文件中`causal_conv1d_update_ref`，简化如下：(仅取cache_seqlens is None and num_accepted_tokens is None)的分支：
```python
def func(
    x: (batch, dim, seqlen),
    conv_state: (batch + 1, dim, state_len),
    weight: (dim, width),
    bias: (dim,),
    activation: Union[None, "silu", swish],
    conv_state_indices: (batch,) dtype=torch.int32
):
    x_new = torch.cat([conv_state[conv_state_indices], x], dim=-1).to(
    weight.dtype)  # (batch, dim, state_len + seqlen)
    to_copy = x_new[:, :, -state_len:]
    conv_state[conv_state_indices] =(to_copy)
    out = F.conv1d(x_new, weight.unsqueeze(1), bias, padding=0, groups=dim)[:, :, -seqlen:]
    return  (out if activation is None else F.silu(out)).to(dtype=dtype_in)
```


## 优化点分析：
### 1.cat写法
由于triton社区没有提供多维tensor的cat接口，因此，原生实现中使用了`tl.arange(0, N)-M`的方式，实现了 "将数据搬运到buffer的右侧" 的效果，又通过`tl.where`实现了"将左右两部分数据选择到同一buffer中"，最终实现了concat功能：
```python
idx_seq = tl.program_id(0)
idx_feats = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
VAL = state_len - seqlen
x_base = x_ptr + (idx_seq * stride_x_seq) + (idx_feats * stride_x_dim)  # [BLOCK_N]

mask = (
    (conv_state_batch_coord < num_cache_lines)
    & ((idx_tokens + seqlen) < state_len)[:, None]
    & (idx_feats < dim)[None, :]
)
conv_state = tl.load(conv_state_ptrs_source, mask, other=0.0)

x_ptrs = (
    x_base[None, :] + ((idx_tokens - VAL) * stride_x_token)[:, None]
)  # [BLOCK_M, BLOCK_N]

mask_x = (
    (idx_tokens - VAL >= 0)[:, None]
    & (idx_tokens - VAL < seqlen)[:, None]
    & (idx_feats < dim)[None, :]
)  # token-index  # token-index  # feature-index
loaded_x = tl.load(x_ptrs, mask_x, 0.0)

new_conv_state = tl.where(mask, conv_state, loaded_x)
```
- 这种写法由于产生了负数的offset，会被当前的triton-ascend分析为离散访存场景，从DMA整块读取退化到标量读取，导致性能劣化严重。
- 同时由于使用了`tl.where`，其mask的运算会占用较长时间(load/store中的连续mask会被编译器分析并优化，不会真的在运行时计算)，导致性能变差；

以上两点会在后续的triton-ascend版本中自动识别并优化，本文主要考虑利用现有接口重新写出昇腾亲和的高性能实现。

新代码中，我们利用triton-ascend的扩展Op`tl.insert_slice`来实现UB上的cat功能，规避了负数访问的读取，和冗余的mask计算，同时通过transpose操作将低维concat的不连续搬运转换到了高维的连续搬运，从而进一步提升性能：
```python
x = tl.load(x_ptr + pid * x_batch_stride + doffs * seq_len + tl.arange(0, DIM_BLOCK * seq_len))
x_T = x.trans()

x_new_T = tl.full([cat_len * DIM_BLOCK], 0, x_ptr.dtype.element_ty)
x_new_T = tl.insert_slice(x_new_T, conv_state_T, offsets = (0,), sizes = (state_len * DIM_BLOCK,), strides = (1,)) # [cat_len , DIM_BLOCK].view(-1)
x_new_T = tl.insert_slice(x_new_T, x_T, offsets = (state_len * DIM_BLOCK,), sizes = (seq_len * DIM_BLOCK,), strides = (1,))
```

### 2. 32B对齐导致的性能劣化
昇腾硬件的UB要求tensor的尾轴大小能被32Byte整除，若尾轴长度不足则会自动补齐。
在此前提下，对模型中shape为(2048,3)和(2048,1)Tensor的种种操作，都会因为自动补齐导致性能成倍恶化，此时可考虑通过转置操作将对齐轴转到低维，直到store时再转置为原始状态，从而规避自动补齐，优化计算速度。
同时由于转置操作本身也受自动补齐规则的影响，因此同样需要特殊技巧来规避补齐。
这里列出一个"借轴转置"的tip，适用于**`tensor.numel() % 256Byte == 0`**的场景，具体操作如下：
```python
# conv_state = tensor([2048, 3], bfloat16)
conv_state = tl.load(conv_state_ptr + conv_batch_offs * conv_batch_stride + doffs * 3 + tl.arange(0, 2048 * 3)) # 当成1D tensor load，此时由于numel对齐，不会自动补齐。
conv_state_T = conv_state.reshape(128, 16 * 3).trans().reshape(16, 3 * 128).trans().reshape(3 * 2048,) # 长轴(2048)裂出一根对齐轴(16)借给短轴(3)，从而让两个轴都对齐
```

### 3. 分核优化
原始分核是`grid = (batch, dim // BLOCK_N)`，在 `batch = 32, dim = 2048, BLOCK_N = 128`时，会产生512个任务，但是Ascend 910系列通常只有40/48个vector core，超过此数目的grid会排队下发，造成很长的等待时间。因此高性能实现的分核通常不会超过num vector core。

结合以上三点，重写后的算子完整逻辑见`_causal_conv1d_update_kernel_no_cache_len_no_mtp`，在AtlasA3硬件上(Ascend910),运行时间从1400us优化到了12us。
