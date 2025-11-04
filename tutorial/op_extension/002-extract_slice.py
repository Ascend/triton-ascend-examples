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

import os
import torch
import time
try:
    import torch_npu
except Exception as e:
    print("import torch_npu failed.")
import triton
import triton.language as tl


def is_npu() -> bool:
    return hasattr(torch, "npu") and torch.npu.is_available()

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
    for i in tl.range(0, BLOCK_SIZE):
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

if __name__ == "__main__":
    if is_npu():
        run("npu")
    else:
        run("cuda")