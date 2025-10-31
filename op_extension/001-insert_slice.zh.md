# 002-insert_slice.py说明

## 功能
大模型训练/推理中MOE Token重排场景下，新增insert_slice接口，实现数据合并写出到GM，提升性能

## 接口描述
```html
"""
    Insert a tensor to another tensor as specified by the operation’s offsets, sizes and strides arguments.

    :param ful: The tensor to receive tensor.
    :type ful: Tensor
    :param sub: The tensor to be inserted.
    :type sub: Tensor
    :param offsets:
    :type offsets: tuple of ints
    :param sizes:
    :type sizes: tuple of ints
    :param strides:
    :type strides: tuple of ints
"""
def insert_slice(ful, sub, offsets, sizes, strides, _builder=None, _generator=None) -> tensor
```

## 差异点概述
1. Moe Token重排，从indices指定的位置读取数据，顺序存放，该场景下，数据随机读取，写出的位置是顺序连续的   
GPU实现：每个kernel处理一个Token，利用多核优势能够达成很好的性能   
NPU实现：NPU核数少，需要增加单Kernel处理数据量，才能达到性能最佳，针对Moe重排，写出数据连续存放，该场景可以使用insert_slice接口，将多个从不同位置读取的Tensor数据合并后一次写出，提升性能   


## 差异点详解

Code diff of NPU and CUDA

```python
@triton.jit
def gpu_token_rearrangement_kernel(x_ptr, indices, 
                                   output_ptr, n_elements, 
                                   S : tl.constexpr, 
                                   D : tl.constexpr, 
                                   BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)

    # 1.load rearrangement index
    idx_start = pid * BLOCK_SIZE
    idx_mask = idx_start < S
    index = tl.load(indices + idx_start, mask=idx_mask)

    # 2.load token data
    offsets = index * D + tl.arange(0, D)
    data_mask = offsets < n_elements
    data = tl.load(x_ptr + offsets, mask=data_mask)

    # 3.calc the store offset & store
    out_offset = pid * BLOCK_SIZE * D + tl.arange(0, D)
    out_msk = out_offset < n_elements
    tl.store(output_ptr + out_offset, data, mask=out_msk)
```

<pre><code>
@triton.jit
def npu_token_rearrangement_kernel(x_ptr, indices, 
                                   output_ptr, n_elements, 
                                   S : tl.constexpr, 
                                   D : tl.constexpr, 
                                   BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    dtype = output_ptr.type.element_ty
    out_start = pid * BLOCK_SIZE * D

    # 1.prepare output tensor
    output = tl.full((BLOCK_SIZE, D), 0, dtype=dtype)

    # 2.batch load rearrangement indices
    idx_offset = pid * BLOCK_SIZE  + tl.arange(0, BLOCK_SIZE)
    idx_mask = idx_offset < S
    idx = tl.load(indices + idx_offset, idx_mask)
<b><i>
    # 3.load data by index & insert into output tensor in loop
    for i in tl.range(0, BLOCK_SIZE):
        data_offset = D * tl.get_element(idx, (i,))+ tl.arange(0, D)[None,:]
        data_mask = data_offset < n_elements
        data = tl.load(x_ptr + data_offset, data_mask)
        output = tl.insert_slice(output, data, [i,D], [1,D], [1,1])
</i></b>
    # 4.batch store to gm
    out_offset = out_start + tl.arange(0, BLOCK_SIZE)[:,None] + tl.arange(0, D)[None, :]
    out_mask = out_offset < n_elements
    tl.store(output_ptr + out_offset, output, out_mask)
</code></pre>
