# 002-vector_cmp.py说明

## 功能
以layerNorm为例说明实现向量化cmp加速npu计算流程  
该cmp操作用于处理layerNorm中的尾块处理

## 差异点概述
1. 原始gpu计算流程i64/i32 cmp在npu设备上无法使能vector，退化为scalar计算效率降低；通过转化为fp32来利用vec_cast和vec_cmp实现vector操作加速  
需要注意的是，在tl.load和tl.save中的mask使用cmp功能，大部分情况下编译器可以自动优化为vec操作，本例中tl.where则需要手动转换

## 差异点详解

Code diff of NPU and CUDA
```diff
@triton.jit
def npu_vector_cmp_kernel(
    X,                 # [Tensor] input tensor (row x col)
    Out,               # [Tensor] output tensor (row x col)
    Mean,              # [Vector] mean tensor (row, ) of X
    Rstd,              # [Vector] std tensor (row, ) of X
    stride_x_row,      # [Scalar] stride of row in X
    stride_out_row,    # [Scalar] stride of row in Out, normally equals to stride_x_row
    M,                 # [Scalar] row number
    N,                 # [Scalar] col number
    eps,               # [Scalar] epsilon to aviod division by zeros
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr
):
    """
    an example of layernorm to checkout Vector Cmp
    Out = ((X - E[X]) / sqrt(V[X] + eps)) on dim -1
    
    just for easy case, we assume that:
    1. BLOCK_N >= X.shape(-1), group_n = 0 only
    2. BLOCK_M = 1, group_m = range(0, row, 1)
    """
    group_m = tl.program_id(0)
    group_n = tl.program_id(1)
    row = group_m

    # calculate index & offset
    Mean = Mean + group_n * M
    Rstd = Rstd + group_n * M
    X = X + row * stride_x_row + group_n * N
    Out = Out + row * stride_out_row + group_n * N

    cols = tl.arange(0, BLOCK_N)  # cols is int64
    x = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)

    # calculate mean & rstd
    mean = tl.sum(x, axis=0) / N
    tl.store(Mean + row, mean)
    
-   xbar = tl.where(cols < N, X - mean, 0.0)
+   # change cols(i64) into cols_cmp(f32) to enable vector processing
+   cols_cmp = cols.to(tl.float32)
+   xbar = tl.where(cols_cmp < N, x - mean, 0.0)

    var = tl.sum(xbar * xbar, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    tl.store(Rstd + row, rstd)

    # calculate Out
    mask = cols < N
    out = (x - mean) * rstd
    tl.store(Out + cols, out, mask=mask)

```

