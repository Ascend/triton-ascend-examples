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
    本代码为对megablocks(https://github.com/databricks/megablocks/blob/main/megablocks/backend/kernels.py)中binned_gather、binned_scatter
    以及binned_scatter_wgrad在Ascend上的亲和实现, 以提升在Ascend上的性能
'''

import torch
import triton
import triton.language as tl
import numpy as np

import triton.runtime.driver as driver



def get_npu_properties():
    device = torch.npu.current_device()
    return driver.active.utils.get_device_properties(device)



def assert_is_tensor(x, ndim):
    if x.ndim != ndim:
        raise ValueError(f'Expected {ndim}-tensor but got {x.ndim}-tensor')



def assert_is_matrix(x):
    assert_is_tensor(x, 2)



def assert_is_vector(x):
    if x.ndim != 1:
        raise ValueError(f'Expected 1-tensor but got {x.ndim}-tensor')



def assert_equal(a, b):
    if a != b:
        raise ValueError(f'Expected dimensions to be equal but got {a} and {b}.',)



@triton.jit
def _binned_copy_gather_npu(
                            a: torch.Tensor,
                            b: torch.Tensor,
                            indices: torch.Tensor,
                            bins: torch.Tensor,
                            bin_ids: torch.Tensor,
                            expert_capacity: tl.constexpr,
                            INDICES_LENGTH: tl.constexpr,
                            BLOCK_SIZE: tl.constexpr,
                            SUB_BLOCK_SIZE: tl.constexpr,
                            NUM_COLUMNS: tl.constexpr,
                            BLOCK_X: tl.constexpr,
                            TOP_K: tl.constexpr,
                        ):
    idx_begin = tl.program_id(0) * BLOCK_SIZE

    for idx_in_block in range(0, BLOCK_SIZE, SUB_BLOCK_SIZE):
        idx_offset = idx_begin + idx_in_block
        idx_offsets = tl.arange(0, SUB_BLOCK_SIZE) + idx_offset
        idx_mask = idx_offsets < INDICES_LENGTH
        if idx_offset < INDICES_LENGTH:
            cur_indices = tl.load(indices + idx_offsets, idx_mask)
            cur_bin_ids = tl.load(bin_ids + idx_offsets, idx_mask)
            base_bin_idx = tl.get_element(cur_bin_ids, (0,))

            for col_offset in range(0, NUM_COLUMNS, BLOCK_X):
                tmp_buf = tl.zeros((SUB_BLOCK_SIZE, BLOCK_X), a.dtype.element_ty)
                col_offsets = tl.arange(0, BLOCK_X) + col_offset
                col_mask = col_offsets < NUM_COLUMNS

                offset_in_bin = idx_offset
                if base_bin_idx > 0:
                    offset_in_bin -= tl.load(bins + base_bin_idx - 1)
                idx_b = base_bin_idx * expert_capacity + offset_in_bin

                COUNT = 0
                for item_offset in range(0, SUB_BLOCK_SIZE):
                    inner_offset = idx_offset + item_offset
                    if inner_offset < (idx_begin + BLOCK_SIZE):
                        bin_idx = tl.get_element(cur_bin_ids, (item_offset,))

                        if bin_idx != base_bin_idx:
                            b_offsets = tl.arange(0, SUB_BLOCK_SIZE) + idx_b
                            b_mask = b_offsets < idx_b + COUNT
                            tl.store(b + (b_offsets[:, None] * NUM_COLUMNS + col_offsets[None, :]),
                                    tmp_buf, b_mask[:, None] & col_mask[None, :])

                            # 更新起始偏移量和当前专家号
                            base_bin_idx = bin_idx
                            offset_in_bin = inner_offset
                            if base_bin_idx > 0:
                                offset_in_bin -= tl.load(bins + base_bin_idx - 1)
                            idx_b = base_bin_idx * expert_capacity + offset_in_bin
                            COUNT = 0

                        if offset_in_bin + COUNT < expert_capacity:
                            idx_a = (tl.get_element(cur_indices, (item_offset,)) // TOP_K) * NUM_COLUMNS
                            val = tl.load(a + idx_a + col_offsets, col_mask)

                            tmp_buf = tl.insert_slice(tmp_buf, val[None, :], offsets=(COUNT, 0), sizes=(1, BLOCK_X),
                                                        strides=(1, 1))
                            COUNT += 1
                # end for SUB_BLOCK_SIZE
                # store last block
                if COUNT != 0:
                    b_offsets = tl.arange(0, SUB_BLOCK_SIZE) + idx_b
                    b_mask = b_offsets < idx_b + COUNT
                    tl.store(b + (b_offsets[:, None] * NUM_COLUMNS + col_offsets[None, :]),
                            tmp_buf, b_mask[:, None] & col_mask[None, :])
            # end for col_offset
    # end for SUB_BLOCK loop


def binned_gather_npu(x, indices, weights, bins, expert_capacity, top_k, bin_ids):

    # Validate the input shapes.
    assert_is_matrix(x)
    assert_is_vector(indices)
    assert_is_vector(bins)
    assert_is_vector(bin_ids)
    assert_equal(indices.shape[0], x.shape[0] * top_k)

    if weights is not None:
        assert_equal(weights.shape[0], x.shape[0] * top_k)

    num_experts = bins.shape[0]
    out = torch.zeros((num_experts, expert_capacity, x.shape[1]), dtype=x.dtype, device=x.device)

    # +++ tiling
    num_core = get_npu_properties()["num_vectorcore"]
    indices_length = indices.shape[0]
    block_size = (indices_length - 1) // num_core + 1
    num_columns = x.shape[1]

    max_block_x = 24560
    block_x = (min(num_columns, max_block_x) + 15) // 16 * 16
    sub_block_size = max((64 * 1024 - block_x * 2) // (block_x * 2 + 4), 1)   # change if ub overflow

    enable_multi_buffer = True
    _binned_copy_gather_npu[(num_core, )](
            x,
            out,
            indices,
            bins,
            bin_ids,
            expert_capacity,
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
def _binned_copy_scatter_npu(
                                a,
                                b,
                                indices,
                                bins,
                                bin_ids,
                                weights,
                                expert_capacity: tl.constexpr,
                                INDICES_LENGTH: tl.constexpr,
                                BLOCK_SIZE: tl.constexpr,
                                SUB_BLOCK_SIZE: tl.constexpr,
                                NUM_COLUMNS: tl.constexpr,
                                BLOCK_X: tl.constexpr,
                                EXPERT_NUM: tl.constexpr,
                                SCALE: tl.constexpr,
                            ):
    pid = tl.program_id(0)
    idx_begin = pid * BLOCK_SIZE
    idx_end = tl.minimum((pid + 1) * BLOCK_SIZE, INDICES_LENGTH)

    exp_offset = tl.arange(0, EXPERT_NUM)
    loaded_bins = tl.load(bins + exp_offset)

    for idx_in_block in range(0, BLOCK_SIZE, SUB_BLOCK_SIZE):
        idx_offset = idx_begin + idx_in_block
        if idx_offset < INDICES_LENGTH:
            idx_offsets = tl.arange(0, SUB_BLOCK_SIZE) + idx_offset
            idx_mask = idx_offsets < idx_end
            cur_indices = tl.load(indices + idx_offsets, idx_mask, other=0)
            cur_bin_ids = tl.load(bin_ids + idx_offsets, idx_mask, other=0)
            cur_sub_block_size = tl.minimum(SUB_BLOCK_SIZE, idx_end - idx_offset)

            base_bin_idx = tl.get_element(cur_bin_ids, (0,))
            last_bin_idx = tl.get_element(cur_bin_ids, (cur_sub_block_size - 1,))
            bin_count = last_bin_idx - base_bin_idx + 1

            begin = 0
            for i in range(bin_count):
                start_pos = 0
                if base_bin_idx > 0:
                    start_pos = tl.get_element(loaded_bins, (base_bin_idx - 1,))
                cur_bin = tl.get_element(loaded_bins, (base_bin_idx,))
                cur_bin_cut = tl.minimum(cur_bin, start_pos + expert_capacity)
                count = tl.minimum(cur_bin_cut - idx_offset - begin, cur_sub_block_size - begin)
                if count > 0:
                    x_offset = idx_offset + begin
                    if base_bin_idx > 0:
                        x_offset -= start_pos

                    for col_offset in range(0, NUM_COLUMNS, BLOCK_X):
                        col_offsets = tl.arange(0, BLOCK_X) + col_offset
                        col_mask = col_offsets < NUM_COLUMNS

                        idx_a = base_bin_idx * expert_capacity + x_offset
                        load_offsets = tl.arange(0, SUB_BLOCK_SIZE) + idx_a
                        load_mask = load_offsets < idx_a + count
                        cur_a = tl.load(a + (load_offsets[:, None] * NUM_COLUMNS + col_offsets[None, :]),
                                        load_mask[:, None] & col_mask[None, :])

                        end = begin + count
                        for j in range(begin, end):
                            idx = tl.get_element(cur_indices, (j,))
                            val = tl.extract_slice(cur_a, offsets=(j-begin, 0), sizes=(1, BLOCK_X), strides=(1, 1))
                            if SCALE:
                                scale = tl.load(weights + idx)
                                val = val.to(tl.float32) * scale.to(tl.float32)
                            tl.store(b + (idx * NUM_COLUMNS + col_offsets), val.to(b.dtype.element_ty).reshape(BLOCK_X), col_mask)

                base_bin_idx += 1
                begin = cur_bin - idx_offset
            # end for
    # end for


def binned_scatter_npu(x, indices, weights, bins, top_k, bin_ids):

    # Validate the input shapes.
    assert_is_tensor(x,3)
    assert_is_vector(indices)
    assert_is_vector(bins)
    assert_is_vector(bin_ids)
    assert_equal(bins.shape[0], x.shape[0])

    if weights is not None:
        assert_equal(indices.shape[0], weights.shape[0])

    num_experts, expert_capacity, hidden_size = x.shape
    tokens = indices.shape[0] // top_k
    out = torch.zeros((tokens, top_k, hidden_size), dtype=x.dtype, device=x.device)

    num_core = get_npu_properties()["num_vectorcore"]
    indices_length = indices.shape[0]
    block_size = (indices_length - 1) // num_core + 1
    num_columns = x.shape[2]

    max_block_x = 24560
    block_x = (min(num_columns, max_block_x) + 15) // 16 * 16
    sub_block_size = max((92 * 1024 - block_x * 2) // (block_x * 2 + 4), 1)  # change if ub overflow

    enable_multi_buffer = True
    _binned_copy_scatter_npu[(num_core, )](
        x,
        out,
        indices,
        bins,
        bin_ids,
        weights,
        expert_capacity,
        indices_length,
        block_size,
        sub_block_size,
        num_columns,
        block_x,
        x.shape[0],
        weights is not None,
        multibuffer=enable_multi_buffer
    )

    # Reduce along the top-k dimension, if needed.
    return out.sum(dim=1) if top_k > 1 else out.view(tokens, hidden_size)


@triton.jit
def _binned_copy_wgrad_npu(
                            x,
                            grad,
                            wgrad,
                            expert_capacity,
                            indices,
                            bins,
                            NUM_COLUMNS: tl.constexpr,
                            TOP_K: tl.constexpr,
                            BLOCK_X: tl.constexpr,
                            BLOCK_SIZE: tl.constexpr,
                            ):
    expert_idx = tl.program_id(0)
    entry_idx = tl.program_id(1)

    bins_start = 0
    if expert_idx > 0:
        bins_start = tl.load(bins + expert_idx - 1)
    bins_end = tl.load(bins + expert_idx)
    expert_num_tokens = bins_end - bins_start

    for idx_in_block in range(0, BLOCK_SIZE):
        # Calculate our offset into the output.
        index_x = expert_idx * expert_capacity + entry_idx * BLOCK_SIZE + idx_in_block
        out_token_num = entry_idx * BLOCK_SIZE + idx_in_block + 1
        if not (out_token_num > expert_num_tokens):
            index_out = tl.load(indices + bins_start + entry_idx * BLOCK_SIZE + idx_in_block)

            wgrad_ptr = wgrad + index_out
            grad_ptr = grad + tl.multiple_of((index_out // TOP_K) * NUM_COLUMNS, NUM_COLUMNS)
            x_ptr = x + tl.multiple_of(index_x * NUM_COLUMNS, NUM_COLUMNS)
            offsets = tl.max_contiguous(tl.arange(0, BLOCK_X), BLOCK_X)

            acc = tl.zeros((BLOCK_X,), dtype=tl.float32)
            iterations = tl.cdiv(NUM_COLUMNS, BLOCK_X)

            for _ in range(iterations):
                mask = offsets < NUM_COLUMNS
                data = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)
                scale = tl.load(grad_ptr + offsets, mask=mask).to(tl.float32)
                acc += data * scale
                offsets += BLOCK_X

            # Reduce to get the final result and store.
            out = tl.sum(acc).to(wgrad.dtype.element_ty)
            tl.store(wgrad_ptr, out)


def binned_scatter_wgrad_npu(x, grad, indices, bins, top_k):
    # Validate the input shapes.
    assert_is_tensor(x, 3)
    assert_is_matrix(grad)
    assert_is_vector(indices)
    assert_is_vector(bins)
    assert_equal(bins.shape[0], x.shape[0])

    num_experts, expert_capacity, hidden_size = x.shape
    tokens = indices.shape[0] // top_k
    out = torch.zeros((tokens * top_k), dtype=x.dtype, device=x.device)

    # +++ tiling
    import math
    enable_multi_buffer = True
    num_tokens_per_task = math.ceil(expert_capacity / 16)
    num_task = math.ceil(expert_capacity / num_tokens_per_task)

    _binned_copy_wgrad_npu[(num_experts, num_task)](
        x,
        grad,
        out,
        expert_capacity,
        indices,
        bins,
        NUM_COLUMNS=hidden_size,
        TOP_K=top_k,
        BLOCK_X=hidden_size,
        BLOCK_SIZE=num_tokens_per_task,
        multibuffer=enable_multi_buffer
    )
    return out


#=============================TEST============================
def test_gather_npu():

    def binned_gather_generate_data(sl: int, hs: int, ne: int, top_k: int):
        # NOTE: Capacity factor == 1.
        ec = (sl * top_k) // ne  # ？每个专家接收的token数量 ？

        # Create the data and indices.
        x = torch.randn((sl, hs)).npu().half()

        # Randomly assign tokens to experts.
        top_expert = torch.randint(0, ne, (sl * top_k,)).npu().int()  # 每个token选择的专家id

        # _, indices = ops.sort(top_expert)
        bin_ids, indices = torch.sort(top_expert)  # token重排

        # bins = ops.inclusive_cumsum(ops.histogram(top_expert, ne), 0)
        tokens_per_expert = torch.histc(top_expert, ne, 0, ne - 1)
        bins = torch.cumsum(tokens_per_expert, dim=0).to(torch.int32)

        return x, indices.to(torch.int32), bins, ec, bin_ids.to(torch.int32)

    def binned_gather_numpy(
        x: torch.Tensor,
        indices: torch.Tensor,
        bins: torch.Tensor,
        ec: int,
        top_k: int,
        ne: int,
        hs: int
    ):

        x = x.cpu().numpy()
        indices = indices.cpu().numpy()
        bins = bins.cpu().numpy()
        start = 0
        out = np.zeros((ne, ec, hs))  # [专家数量， 每个专家处理的token数量，hidden dim]
        for i in range(ne):
            end = bins[i]
            for j in range(min(ec, end - start)):
                index = indices[start + j] // top_k
                out[i, j, :] = x[index, :]
            start = end
        return torch.from_numpy(out).npu().half()

    print(" >>> running binned_gather test >>>")
    test_shapes = (
        (16, 16, 4, 4),
        (1024, 4, 64, 4),  # 验证ub 32B对齐功能
        (1024, 1536, 64, 4),
        (1024, 1536, 128, 4),
        (16384, 768, 64, 4),
        (16384, 768, 128, 4),
    )
    for shape in test_shapes:
        for _ in range(1):
            top_k = shape[-1]
            x, indices, bins, ec, bin_ids = binned_gather_generate_data(*shape)

            numpy_golden = binned_gather_numpy(x, indices, bins, ec, top_k, shape[2], shape[1])

            triton_results = binned_gather_npu(x, indices, None, bins, ec, top_k, bin_ids)

            print("case: \t", shape, "\t======> triton compare numpy: ",
                torch.all(torch.eq(numpy_golden.cpu(), triton_results.cpu())), "======")

            assert torch.all(torch.eq(numpy_golden.cpu(), triton_results.cpu()))


def test_scatter_npu():
    torch.manual_seed(23)

    def binned_scatter_generate_data(sl: int, hs: int, ne: int, top_k: int):
        # NOTE: Capacity factor == 1.
        ec = (sl * top_k) // ne

        # Create the data and indices.
        x = torch.randn((sl, hs)).npu().half()

        # Randomly assign tokens to experts.
        top_expert = torch.randint(0, ne, (sl * top_k,)).npu().int()

        # _, indices = ops.sort(top_expert)
        bin_ids, indices = torch.sort(top_expert)

        # bins = ops.inclusive_cumsum(ops.histogram(top_expert, ne), 0)
        tokens_per_expert = torch.histc(top_expert, ne, 0, ne - 1)
        bins = torch.cumsum(tokens_per_expert, dim=0).to(torch.int32)

        # Sample weights for the scatter reduce.
        weights = torch.rand((sl * top_k,)).npu().half()

        x = binned_gather_npu(x, indices, None, bins, ec, top_k, bin_ids)

        return x, indices, bins, ec, weights, bin_ids

    def binned_scatter_numpy(
        x: torch.Tensor,
        indices: torch.Tensor,
        weights: torch.Tensor,
        bins: torch.Tensor,
        top_k: int,
        sl: int,
        hs: int,
        ne: int
    ):
        x = x.cpu().numpy()
        indices = indices.cpu().numpy()
        weights = weights.cpu().numpy()
        bins = bins.cpu().numpy()
        start = 0
        out = np.zeros((sl, hs))
        for i in range(ne):
            end = bins[i]
            for j in range(min(ec, end - start)):
                index = indices[start + j]
                scale = weights[index]
                index //= top_k

                out[index, :] += scale * x[i, j, :]
            start = end
        return torch.from_numpy(out).npu().half()

    def binned_scatter_wgrad_numpy(
        x: torch.Tensor,
        grads: torch.Tensor,
        indices: torch.Tensor,
        bins: torch.Tensor,
        top_k: int,
        ne: int,
        ec: int
    ):
        x = x.cpu().numpy()
        grads = grads.cpu().numpy()
        indices = indices.cpu().numpy()
        bins = bins.cpu().numpy()

        out = np.zeros(indices.shape).astype(np.float32)
        start = 0
        for i in range(ne):
            end = bins[i]
            for j in range(min(ec, end - start)):
                index = indices[start + j]
                grad_idx = index // top_k
                grad = grads[grad_idx]
                out[index] = np.sum(grad.astype(np.float32) * x[i, j, :].astype(np.float32))
            start = end
        return torch.from_numpy(out).npu().half()

    test_shapes = (
        (1024, 4, 64, 4),  # 验证ub 32B对齐功能
        (1024, 1536, 64, 4),
        (1024, 1536, 128, 4),
        (16384, 768, 64, 4),
        (16384, 768, 128, 4),
    )

    print(" >>> running binned_scatter test >>>")
    for shape in test_shapes:
        for _ in range(1):
            top_k = shape[-1]
            x, indices, bins, ec, weights, bin_ids = binned_scatter_generate_data(*shape)

            numpy_golden = binned_scatter_numpy(x, indices, weights, bins, top_k, shape[0], shape[1], shape[2])

            triton_results = binned_scatter_npu(x, indices, weights, bins, top_k, bin_ids)
            print("forward case: \t", shape, "\t======> triton compare numpy: ",
                torch.all(torch.eq(numpy_golden.cpu(), triton_results.cpu())), "======")

            assert torch.all(torch.eq(numpy_golden.cpu(), triton_results.cpu()))

            # wgrad
            grad = torch.randn_like(triton_results)

            numpy_wgrad_out = binned_scatter_wgrad_numpy(x, grad, indices, bins, top_k, shape[2], ec)

            npu_wgrad_out = binned_scatter_wgrad_npu(x, grad, indices, bins, top_k)

            print("wgrad case: \t", shape, "\t======> triton compare numpy: ",
                torch.allclose(numpy_wgrad_out.cpu(), npu_wgrad_out.cpu(), rtol=1e-3, atol=1e-3), "======")


if __name__ == "__main__":
    test_gather_npu()
    test_scatter_npu()