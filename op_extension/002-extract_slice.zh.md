# 002-extract_slicee.py说明

## 功能
大模型训练/推理中MOE Token反重排场景下，新增extract_slice接口，实现数据批量读取到UB，从UB中截取部分处理，加速npu计算流程  

## 接口说明
```html
"""
    Extract a tensor from another tensor as specified by the operation’s offsets, sizes and strides arguments.

    :param ful: The tensor to split.
    :type ful: Tensor
    :param offsets:
    :type offsets: tuple of ints
    :param sizes:
    :type sizes: tuple of ints
    :param strides:
    :type strides: tuple of ints
"""
def extract_slice(ful, offsets, sizes, strides, _builder=None, _generator=None) -> tensor:
```

## 差异点概述
1. Moe Token反重排，读取一个连续的数据块，根据indices，将Token提取出来后放置到指定的位置，该场景下，数据读取连续，写出是分散的   
GPU实现：每个kernel处理一个Token，利用多核优势能够达成很好的性能   
NPU实现：NPU核数少，需要增加单Kernel处理数据量，才能达到性能最佳，针对Moe反重排，读取数据连续，每段写出到不同的位置，提供extract_slice接口，支持从一个大的Tensor中，读取部分数据，然后对该数据操作，达成批量读取，分散操作的目的   


## 差异点详解

Code diff of NPU and CUDA
```diff


@triton.jit
def gpu_token_reverse_kernel(x_ptr, indices, output_ptr, n_elements, S: tl.constexpr, D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE * D

    # 1.load data
    offsets = block_start + tl.arange(0, D)
    data_mask = offsets < n_elements
    data = tl.load(x_ptr + offsets, mask=data_mask)

    # 2.load index
    index_offset = pid * BLOCK_SIZE
    idx_mask = index_offset < S
    index = tl.load(indices + index_offset, mask=idx_mask)

    # 3.calc output offset by index & store
    out_offset = index * D + tl.arange(0, D)
    out_msk = out_offset < n_elements
    tl.store(output_ptr + out_offset, data, mask=out_msk)


@triton.jit
def npu_token_reverse_kernel(x_ptr, indices, output_ptr, n_elements, S : tl.constexpr, D : tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE * D

    # 1. batch load data
    data_offset = D * tl.arange(0, BLOCK_SIZE)[:,None] + tl.arange(0, BLOCK_SIZE)[None, :]
    data_mask = data_offset < n_elements
    data = tl.load(x_ptr + block_start + data_offset, data_mask)


    # 2. batch load indices
    idx_start = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    idx_mask = idx_start < S
    idx = tl.load(indices + idx_start, idx_mask)

    # 3. extract token one by one and store
    for i in tl.arange(0, BLOCK_SIZE):
        x_sub = tl.extract_slice(data, [i,0], [1,D], [1,1])
        output_offset = D * tl.get_element(idx, (i,))+ tl.arange(0, D)[None,:]
        out_mask = output_offset < n_elements
        tl.store(output_ptr + output_offset, x_sub, out_mask)


def run(device = "cuda"):
    S = 1024
    D = 32
    x = torch.rand(S, D, device=device)
    indices = torch.randperm(S).to(device=device)
    output = torch.empty_like(x)

    if device == "cuda":
        print("begin to run cuda!")
        gpu_token_reverse_kernel[(S,1,1)](x, indices, output, x.numel(), S, D , BLOCK_SIZE=1)
    elif device == "npu":
        print("begin to run npu!")
        npu_token_reverse_kernel[(48,1,1)](x, indices, output, x.numel(), S, D , BLOCK_SIZE=22)

```
