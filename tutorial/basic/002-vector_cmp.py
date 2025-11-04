# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

"""
Vector Cmp
1. Case study on performance optimization of Triton operators on NPU, compared with GPU.
2. Ascend do not support Cmp on i32/i64, which leads to vector compute fall back into scalar
"""
import os
import time

import torch
import torch_npu
import triton
import triton.language as tl



def is_npu() -> bool:
    return hasattr(torch, "npu") and torch.npu.is_available()


@triton.jit
def gpu_vector_cmp_kernel(
    X,                 # [Tensor] input tensor (row x col)
    Out,               # [Tensor] output tensor (row x col)
    Mean,              # [Vector] mean tensor (row, ) of X
    Rstd,              # [Vector] std tensor (row, ) of X
    stride_x_row,      # [Scalar] stride of row of x
    stride_out_row,    # [Scalar] stride of row of out, normally equals to stride_x_row
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
    xbar = tl.where(cols < N, x - mean, 0.0)
    var = tl.sum(xbar * xbar, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    tl.store(Rstd + row, rstd)

    # calculate Out
    mask = cols < N
    out = (x - mean) * rstd
    tl.store(Out + cols, out, mask=mask)


@triton.jit
def npu_vector_cmp_kernel(
    X,                 # [Tensor] input tensor (row x col)
    Out,               # [Tensor] output tensor (row x col)
    Mean,              # [Vector] mean tensor (row, ) of X
    Rstd,              # [Vector] std tensor (row, ) of X
    stride_x_row,      # [Scalar] stride of row of x
    stride_out_row,    # [Scalar] stride of row of out, normally equals to stride_x_row
    M,                 # [Scalar] row number
    N,                 # [Scalar] col number
    eps,               # [Scalar] epsilon to aviod division by zeros
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr
):
    """
    NPU example shows how to use vec cmp from upper gpu original triton
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
    # [Changed begin]
    # xbar = tl.where(cols < N, X - mean, 0.0)
    cols_cmp = cols.to(tl.float32)
    xbar = tl.where(cols_cmp < N, x - mean, 0.0)
    # [Changed end]

    var = tl.sum(xbar * xbar, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    tl.store(Rstd + row, rstd)

    # calculate Out
    mask = cols < N
    out = (x - mean) * rstd
    tl.store(Out + cols, out, mask=mask)


def run(device = "cuda"):
    """run random test along with native norm"""
    def run_ref(X):
        mean = X.mean(dim=-1, keepdim=True)
        std = X.std(dim=-1, keepdim=True)
        out =  (X.to(torch.float32) - mean) / (std + eps)
        return out.to(device_dtype)

    batch_size = 256
    feature_dim = 128
    eps = 1e-6

    if device == "npu":
        vector_cmp_kernel = npu_vector_cmp_kernel
        sync_device = torch.npu
        device_dtype = torch.float32
    else:
        vector_cmp_kernel = gpu_vector_cmp_kernel
        sync_device = torch.cuda
        device_dtype = torch.float32
    
    X = torch.rand(batch_size, feature_dim, device=device, dtype=device_dtype)
    Out = torch.empty_like(X)
    Mean = torch.empty(batch_size, device=device, dtype=device_dtype)
    Rstd = torch.empty(batch_size, device=device, dtype=device_dtype)
    
    BLK_M = 1
    BLK_N = triton.next_power_of_2(feature_dim)
    num_warps = min(max(BLK_N // 256, 1), 8)
    # warm up 
    vector_cmp_kernel[(batch_size // BLK_M, 1)](
        X, Out, Mean, Rstd, X.stride(0), Out.stride(0), batch_size, feature_dim, eps, BLK_M, BLK_N, num_warps=num_warps)
    sync_device.synchronize()

    spend_time = 0
    iter_times = 100
    sync_device.synchronize()
    start_time = time.time()
    for _ in range(iter_times):
        vector_cmp_kernel[(batch_size // BLK_M, 1)](
            X, Out, Mean, Rstd, X.stride(0), Out.stride(0), batch_size, feature_dim, eps, BLK_M, BLK_N, num_warps=num_warps)
    sync_device.synchronize()
    spend_time += (time.time() - start_time)

    print(f"==== {device} spend_time: {spend_time / iter_times * 1000} ms")

    Out_ref = run_ref(X)
    torch.testing.assert_close(Out, Out_ref, rtol=1e-3, atol=1e-2)
    print(f"==== {device} acc check passed!")


if __name__ == "__main__":
    if is_npu():
        run("npu")
    else:
        run("cuda")

