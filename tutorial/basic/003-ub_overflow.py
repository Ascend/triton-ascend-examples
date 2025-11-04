"""
The purpose of this example is to demonstrate how to avoid ub-overflow on NPU devices.

This example is a simplified version to allocate physical memory slots (pages) in a paged KV-cache for Sglang.

The difference between the gpu and npu version is to load large data from global memory.
The NPU version uses a loop-based approach to conform to hardware constraints, while the GPU version uses full vectorized parallelism for maximum performance.
"""

import numpy as np
import torch

import triton
import triton.language as tl

from ..utils import is_npu, next_power_of_2

_is_npu = is_npu()
if _is_npu:
    import torch_npu

    device = torch.device('npu')
else:
    device = torch.device('cuda')


@triton.jit
def gpu_alloc_extend_kernel(
        pre_lens_ptr,
        seq_lens_ptr,
        free_page_ptr,
        out_indices,
        bs_upper: tl.constexpr,
        page_size: tl.constexpr,
        max_num_extend_tokens: tl.constexpr,
):
    pid = tl.program_id(0)

    load_offset = tl.arange(0, bs_upper)
    seq_lens = tl.load(seq_lens_ptr + load_offset, mask=load_offset <= pid)
    pre_lens = tl.load(pre_lens_ptr + load_offset, mask=load_offset <= pid)
    extend_lens = seq_lens - pre_lens

    seq_len = tl.load(seq_lens_ptr + pid)
    pre_len = tl.load(pre_lens_ptr + pid)
    extend_len = seq_len - pre_len

    sum_extend_lens = tl.sum(extend_lens)
    output_start_loc = sum_extend_lens - extend_len

    num_pages_after = (seq_lens + page_size - 1) // page_size
    num_pages_before = (pre_lens + page_size - 1) // page_size
    num_new_pages = num_pages_after - num_pages_before

    num_page_start_loc_self = (seq_len + page_size - 1) // page_size - (
            pre_len + page_size - 1
    ) // page_size
    sum_num_new_pages = tl.sum(num_new_pages)
    new_page_start_loc = sum_num_new_pages - num_page_start_loc_self

    # Part 2: fill the new full pages
    num_part2 = (
            seq_len // page_size * page_size
            - (pre_len + page_size - 1) // page_size * page_size
    )

    offset_many_page = tl.arange(0, max_num_extend_tokens)
    page_start = tl.load(
        free_page_ptr + new_page_start_loc + offset_many_page // page_size,
        mask=offset_many_page < num_part2,
    )
    tl.store(
        out_indices + output_start_loc + offset_many_page,
        page_start * page_size + offset_many_page % page_size,
        mask=offset_many_page < num_part2,
    )


@triton.jit
def npu_alloc_extend_kernel(
        pre_lens_ptr,
        seq_lens_ptr,
        free_page_ptr,
        out_indices,
        bs_upper: tl.constexpr,
        page_size: tl.constexpr,
        max_num_extend_tokens: tl.constexpr,
        BLOCK_SIZE: tl.constexpr = 1024,
):
    pid = tl.program_id(0)

    load_offset = tl.arange(0, bs_upper)
    seq_lens = tl.load(seq_lens_ptr + load_offset, mask=load_offset <= pid)
    pre_lens = tl.load(pre_lens_ptr + load_offset, mask=load_offset <= pid)
    extend_lens = seq_lens - pre_lens

    seq_len = tl.load(seq_lens_ptr + pid)
    pre_len = tl.load(pre_lens_ptr + pid)
    extend_len = seq_len - pre_len

    sum_extend_lens = tl.sum(extend_lens)
    output_start_loc = sum_extend_lens - extend_len

    num_pages_after = (seq_lens + page_size - 1) // page_size
    num_pages_before = (pre_lens + page_size - 1) // page_size
    num_new_pages = num_pages_after - num_pages_before

    num_page_start_loc_self = (seq_len + page_size - 1) // page_size - (
            pre_len + page_size - 1
    ) // page_size
    sum_num_new_pages = tl.sum(num_new_pages)
    new_page_start_loc = sum_num_new_pages - num_page_start_loc_self

    # Part 2: fill the new full pages
    num_part2 = (
            seq_len // page_size * page_size
            - (pre_len + page_size - 1) // page_size * page_size
    )

    num_loop = tl.cdiv(max_num_extend_tokens, BLOCK_SIZE)
    blk_offset = tl.arange(0, BLOCK_SIZE)
    for i in range(num_loop):
        offset_many_page = blk_offset + i * BLOCK_SIZE
        page_start = tl.load(
            free_page_ptr + new_page_start_loc + offset_many_page // page_size,
            mask=offset_many_page < num_part2,
        )
        tl.store(
            out_indices + output_start_loc + offset_many_page,
            page_start * page_size + offset_many_page % page_size,
            mask=offset_many_page < num_part2,
        )


def _gen_test_date(batch_size: int,
                   max_context_len: int,
                   max_free_page: int):
    free_pages = torch.arange(max_free_page, dtype=torch.int32, device=device)

    prefix_lens = torch.zeros(batch_size, dtype=torch.int32, device=device)  # shape:(batch)

    seq_lens = torch.from_numpy(
        np.random.choice(range(max_context_len), size=batch_size, replace=False)
    ).to(torch.int32).to(device=device)  # shape:(batch)

    extend_num_tokens = torch.sum(seq_lens).item()

    out_indices = torch.empty(
        (extend_num_tokens,), dtype=torch.int64, device=device
    )

    max_num_extend_tokens = next_power_of_2(extend_num_tokens)

    return prefix_lens, seq_lens, free_pages, out_indices, max_num_extend_tokens


def run(device_name="npu"):
    kwargs = {}
    if device_name == "npu":
        alloc_extend_kernel = npu_alloc_extend_kernel
        kwargs = {
            'BLOCK_SIZE': 1024
        }
    else:
        alloc_extend_kernel = gpu_alloc_extend_kernel

    batch_size = 2
    max_context_len = 2000
    max_free_page = 2048

    (
        prefix_lens,
        seq_lens,
        free_pages,
        out_indices,
        max_num_extend_tokens
    ) = _gen_test_date(batch_size, max_context_len, max_free_page)

    bs = prefix_lens.shape[0]
    alloc_extend_kernel[(bs,)](
        prefix_lens,
        seq_lens,
        free_pages,
        out_indices,
        next_power_of_2(bs),
        page_size=128,
        max_num_extend_tokens=max_num_extend_tokens,
        **kwargs
    )

    print(f'{out_indices=}', flush=True)


if __name__ == "__main__":
    run("npu" if _is_npu else "cuda")
