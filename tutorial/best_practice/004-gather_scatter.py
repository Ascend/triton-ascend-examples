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

'''
本代码为对megablocks(https://github.com/databricks/megablocks/blob/main/megablocks/backend/kernels.py)中gather、scatter以及scatter_wgrad在Ascend上的亲和实现，以提升在Ascend上的性能
'''

import os
import sys
import numpy as np
import torch
import triton
import triton.language as tl

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import is_npu, to_numpy, get_npu_properties



@triton.jit
def _gather_kernel(
    x,
    out,
    indices,
    INDICES_LENGTH: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    SUB_BLOCK_SIZE: tl.constexpr,
    NUM_COLUMNS: tl.constexpr,
    BLOCK_X: tl.constexpr,
    TOP_K: tl.constexpr,
):
    pid = tl.program_id(0)
    idx_begin = pid * BLOCK_SIZE
    idx_end = tl.minimum((pid + 1) * BLOCK_SIZE, INDICES_LENGTH)
    for idx_in_block in range(0, BLOCK_SIZE, SUB_BLOCK_SIZE):
        idx_offset = idx_begin + idx_in_block
        if idx_offset < idx_end:
            idx_offsets = tl.arange(0, SUB_BLOCK_SIZE) + idx_offset
            idx_mask = idx_offsets < idx_end
            cur_indices = tl.load(indices + idx_offsets, idx_mask, other=0)
            for col_offset in range(0, NUM_COLUMNS, BLOCK_X):
                tmp_buf = tl.zeros((SUB_BLOCK_SIZE, BLOCK_X), x.dtype.element_ty)
                col_offsets = tl.arange(0, BLOCK_X) + col_offset
                col_mask = col_offsets < NUM_COLUMNS
                for i in range(0, SUB_BLOCK_SIZE):
                    idx = tl.get_element(cur_indices, (i,)) // TOP_K * NUM_COLUMNS
                    val = tl.load(x + idx + col_offsets, col_mask)
                    tmp_buf = tl.insert_slice(tmp_buf, val[None,:], offsets=(i, 0), sizes=(1, BLOCK_X), strides=(1, 1))
                tl.store(out + idx_offsets[:, None] * NUM_COLUMNS + col_offsets[None, :],
                         tmp_buf, idx_mask[:, None] & col_mask[None, :])


def gather(x, indices, top_k):
    # create output buffer
    out = torch.empty((indices.shape[0], x.shape[1]), dtype=x.dtype, device=x.device)

    # tiling
    # NOTE: tiling泛化，如果ub overflow，调小max_block_x和sub_block_size
    num_core = get_npu_properties()["num_vectorcore"]
    indices_length = indices.shape[0]
    block_size = (indices_length - 1) // num_core + 1
    num_columns = x.shape[1]
    # ub一共192KB，double-buffer后，每个流水96KB，当hidden_size很大时，只能对一个token的hidden_states做分块处理，
    # 此时只有一个int32的idx和block_x * 2个bf16/fp16的数据需要占用ub
    # max_block_x * 2 * 2 + 4 <= 80KB（预留16k内存）
    max_block_x = 20480
    # 输入bf16/fp16, ub首地址需要32字节对齐
    block_x = (min(num_columns, max_block_x) + 15) // 16 * 16
    enable_multi_buffer = True
    # sub_block_size个int32的cur_indices, block_x个bf16/fp16的val, sub_block_size * block_x个bf16/fp16的tmp_buf
    # sub_block_size * 4 + block_x * 2 + sub_block_size * block_x * 2 <= 80KB (预留16KB)
    sub_block_size = max((80 * 1024 - block_x * 2) // (block_x * 2 + 4), 1)

    _gather_kernel[(num_core,)](
        x,
        out,
        indices,
        indices_length,
        block_size,
        sub_block_size,
        num_columns,
        block_x,
        top_k,
        multibuffer=enable_multi_buffer
    )
    return out

@triton.jit
def _scatter_kernel(
    x,
    out,
    weights,
    indices,
    INDICES_LENGTH: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    SUB_BLOCK_SIZE: tl.constexpr,
    NUM_COLUMNS: tl.constexpr,
    BLOCK_X: tl.constexpr,
    TOP_K: tl.constexpr,
    SCALE:  tl.constexpr,
):
    pid = tl.program_id(0)
    idx_begin = pid * BLOCK_SIZE
    idx_end = tl.minimum((pid + 1) * BLOCK_SIZE, INDICES_LENGTH)
    for idx_in_block in range(0, BLOCK_SIZE, SUB_BLOCK_SIZE):
        idx_offset = idx_begin + idx_in_block
        idx_offsets = tl.arange(0, SUB_BLOCK_SIZE) + idx_offset
        idx_mask = idx_offsets < idx_end
        cur_indices = tl.load(indices + idx_offsets, idx_mask, other=0)
        for col_offset in range(0, NUM_COLUMNS, BLOCK_X):
            col_offsets = tl.arange(0, BLOCK_X) + col_offset
            col_mask = col_offsets < NUM_COLUMNS
            cur_x = tl.load(x + idx_offsets[:, None] * NUM_COLUMNS + col_offsets[None, :],
                            idx_mask[:, None] & col_mask[None, :])
            for i in range(0, SUB_BLOCK_SIZE):
                if i + idx_offset < idx_end:
                    idx = tl.get_element(cur_indices, (i,))
                    val = tl.extract_slice(cur_x, offsets=(i, 0), sizes=(1, BLOCK_X), strides=(1, 1))
                    if SCALE:
                        scale = tl.load(weights + idx)
                        val = val.to(tl.float32) * scale.to(tl.float32)

                    tl.store(out + idx * NUM_COLUMNS + col_offsets, val.to(out.dtype.element_ty).reshape(BLOCK_X), col_mask)


def scatter(x: torch.Tensor, indices: torch.Tensor, weights: torch.Tensor, top_k: int):
    # create output buffer
    tokens = indices.shape[0] // top_k
    out = torch.zeros((tokens, top_k, x.shape[1]), dtype=x.dtype, device=x.device)

    # tiling
    # NOTE: tiling泛化，如果ub overflow，调小max_block_x和sub_block_size
    num_core = get_npu_properties()["num_vectorcore"]
    indices_length = indices.shape[0]
    block_size = (indices_length - 1) // num_core + 1
    num_columns = x.shape[1]
    max_block_x = 6144
    block_x = (min(num_columns, max_block_x) + 15) // 16 * 16
    enable_multi_buffer = True
    sub_block_size = max((80 * 1024 - block_x * 12) // (block_x * 2 + 4), 1)

    scale = weights is not None
    _scatter_kernel[(num_core,)](
        x,
        out,
        weights,
        indices,
        indices.shape[0],
        block_size,
        sub_block_size,
        num_columns,
        block_x,
        TOP_K=top_k,
        SCALE=scale,
        multibuffer=enable_multi_buffer
    )
    return out.sum(dim=1) if top_k > 1 else out.view(tokens, x.shape[1])


@triton.jit
def _scatter_wgrad_kernel(
    x,
    grads,
    wgrad,
    indices,
    INDICES_LENGTH: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    SUB_BLOCK_SIZE: tl.constexpr,
    NUM_COLUMNS: tl.constexpr,
    BLOCK_X: tl.constexpr,
    TOP_K: tl.constexpr,
):
    pid = tl.program_id(0)
    idx_begin = pid * BLOCK_SIZE
    idx_end = tl.minimum((pid + 1) * BLOCK_SIZE, INDICES_LENGTH)

    col_offsets = tl.arange(0, BLOCK_X)
    col_mask = col_offsets < NUM_COLUMNS
    for idx_in_block in range(0, BLOCK_SIZE, SUB_BLOCK_SIZE):
        idx_offset = idx_begin + idx_in_block
        if SUB_BLOCK_SIZE > 1:
            idx_offsets = tl.arange(0, SUB_BLOCK_SIZE) + idx_offset
            idx_mask = idx_offsets < idx_end
            cur_indices = tl.load(indices + idx_offsets, idx_mask, other=0)
            cur_x = tl.load(x + idx_offsets[:, None] * NUM_COLUMNS + col_offsets[None, :],
                            idx_mask[:, None] & col_mask[None, :], other=0)
            for i in range(0, SUB_BLOCK_SIZE):
                if i + idx_offset < idx_end:
                    idx = tl.get_element(cur_indices, (i,))
                    data = tl.extract_slice(cur_x, offsets=(i, 0), sizes=(1, BLOCK_X), strides=(1, 1)).to(tl.float32)
                    grad = tl.load(grads + (idx // TOP_K) * NUM_COLUMNS + col_offsets, col_mask, other=0).to(tl.float32)
                    out = tl.sum(data.reshape(BLOCK_X) * grad.reshape(BLOCK_X))
                    tl.store(wgrad+idx, out.to(wgrad.dtype.element_ty))

        elif idx_offset < idx_end:
            idx = tl.load(indices + idx_offset)
            acc = tl.zeros((BLOCK_X,), dtype=tl.float32)
            for col_offset in range(0, NUM_COLUMNS, BLOCK_X):
                data = tl.load(x + idx * NUM_COLUMNS + col_offsets, col_mask, other=0)
                grad = tl.load(grads + (idx // TOP_K) * NUM_COLUMNS + col_offsets, col_mask, other=0)
                acc += data.to(tl.float32) * grad.to(tl.float32)
            acc = tl.sum(acc)
            tl.store(wgrad + idx, acc.to(wgrad.dtype.element_ty))

def scatter_wgrad(
    x: torch.Tensor,
    grads: torch.Tensor,
    indices: torch.Tensor,
    top_k: int
):
    # create output buffer
    tokens = indices.shape[0] // top_k
    out = torch.empty((tokens * top_k,), dtype=x.dtype, device=x.device)
    # tiling
    # NOTE: tiling泛化，如果ub overflow，调小max_block_x和sub_block_size
    num_core = get_npu_properties()["num_vectorcore"]
    indices_length = indices.shape[0]
    block_size = (indices_length - 1) // num_core + 1
    num_columns = x.shape[1]
    max_block_x = 5120
    block_x = (min(num_columns, max_block_x) + 15) // 16 * 16
    enable_multi_buffer = True
    sub_block_size = max((80 * 1024 - block_x * 12) // (block_x * 2 + 4), 1)
    if block_x < num_columns:
        sub_block_size = 1

    _scatter_wgrad_kernel[(num_core,)](
        x,
        grads,
        out,
        indices,
        indices.shape[0],
        block_size,
        sub_block_size,
        num_columns,
        block_x,
        TOP_K=top_k,
        multibuffer=enable_multi_buffer
    )
    return out


def test_fn(shape):
    assert is_npu()

    def generate_data(sl: int, hs: int, ne: int, top_k: int):
        # Create the data and indices.
        x = torch.randn((sl, hs)).npu().half()
        # Randomly assign tokens to experts.
        top_expert = torch.randint(0, ne, (sl * top_k,)).npu().int()
        bin_ids, indices = torch.sort(top_expert)
        weights = torch.rand((sl * top_k,)).npu().half()
        grads =  torch.randn((sl, hs)).npu().half()
        return x, indices, weights, grads

    def gather_numpy(
        x: torch.Tensor,
        indices: torch.Tensor,
        top_k: int,
    ):
        x = to_numpy(x)
        indices = to_numpy(indices)

        out = np.zeros((indices.shape[0], x.shape[1]))
        for i in range(indices.shape[0]):
            load_idx = indices[i] // top_k
            out[i, :] = x[load_idx, :]
        return torch.from_numpy(out).npu().half()
    
    def scatter_numpy(
        x: torch.Tensor,
        indices: torch.Tensor,
        weights: torch.Tensor,
        top_k: int,
    ):
        x = to_numpy(x)
        weights = to_numpy(weights)
        indices = to_numpy(indices)

        out = np.zeros((indices.shape[0] // top_k, x.shape[1]))
        for i in range(indices.shape[0]):
            idx = indices[i]
            scale = weights[idx]
            store_idx = idx // top_k
            out[store_idx, :] += scale * x[i]
        return torch.from_numpy(out).npu().half()
    
    def scatter_wgrad_numpy(
        x: torch.Tensor,
        grads: torch.Tensor,
        indices: torch.Tensor,
        top_k: int,
    ):
        x = to_numpy(x)
        grads = to_numpy(grads)
        indices = to_numpy(indices)
        
        out = np.zeros(indices.shape).astype(np.float32)
        for i in range(indices.shape[0]):
            data = x[i, :]
            grad_idx  = indices[i] // top_k
            grad = grads[grad_idx]
            store_idx = indices[i]
            out[store_idx] = np.sum(data.astype(np.float32) * grad.astype(np.float32))
        return torch.from_numpy(out).npu().half()


    top_k = shape[-1]
    x, indices, weights, grads = generate_data(*shape)
    
    # gather
    gather_ans = gather_numpy(x, indices, top_k)
    gather_results = gather(x, indices, top_k)
    assert torch.all(torch.eq(gather_ans, gather_results))

    # scatter
    scatter_ans = scatter_numpy(gather_ans, indices, weights, top_k)
    scatter_results = scatter(gather_ans, indices, weights, top_k)
    assert torch.all(torch.eq(scatter_ans, scatter_results))
    # scatter_wgrad
    scatter_wgrad_ans = scatter_wgrad_numpy(gather_ans, grads, indices, top_k)
    scatter_wgrad_results = scatter_wgrad(gather_ans, grads, indices, top_k)
    assert np.testing.assert_allclose(
                    to_numpy(scatter_wgrad_ans),
                    to_numpy(scatter_wgrad_results),
                    rtol=5e-2,
                ) is None


if __name__ == "__main__":
    shapes = (
        (1024, 4, 64, 4),
        (1024, 1536, 64, 4),
        (1024, 1536, 128, 4),
        (16384, 768, 64, 4),
        (16384, 768, 128, 4)
    )

    for shape in shapes:
        test_fn(shape)