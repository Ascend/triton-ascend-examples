# 003-ub-overflow.py说明

本示例旨在演示如何在NPU设备上避免UB溢出。

## 功能

本实例是Sglang中alloc_extend_kernel的简化版，功能是为Token分配KV Slot

## 差异点概述

GPU版本与NPU版本的主要差异在于从Global Memory上搬入大量数据到Shared Memory的区别
GPU版本可以直接搬入，NPU版本受限于UB大小，在长序列的场景下需要循环搬入

## 差异点详解

Code diff of NPU and CUDA

```diff
@triton.jit
def alloc_extend_kernel(
        pre_lens_ptr,
        seq_lens_ptr,
        free_page_ptr,
        out_indices,
        bs_upper: tl.constexpr,
        page_size: tl.constexpr,
        max_num_extend_tokens: tl.constexpr,
+       BLOCK_SIZE: tl.constexpr = 1024,
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

-   # gpu load data at once
-   offset_many_page = tl.arange(0, max_num_extend_tokens)
-   page_start = tl.load(
-       free_page_ptr + new_page_start_loc + offset_many_page // page_size,
-       mask=offset_many_page < num_part2,
-   )
-   tl.store(
-       out_indices + output_start_loc + offset_many_page,
-       page_start * page_size + offset_many_page % page_size,
-       mask=offset_many_page < num_part2,
-   )

+   # npu load data using loop
+   num_loop = tl.cdiv(max_num_extend_tokens, BLOCK_SIZE)
+   blk_offset = tl.arange(0, BLOCK_SIZE)
+   for i in range(num_loop):
+       offset_many_page = blk_offset + i * BLOCK_SIZE
+       page_start = tl.load(
+           free_page_ptr + new_page_start_loc + offset_many_page // page_size,
+           mask=offset_many_page < num_part2,
+       )
+       tl.store(
+           out_indices + output_start_loc + offset_many_page,
+           page_start * page_size + offset_many_page % page_size,
+           mask=offset_many_page < num_part2,
+       )

```

