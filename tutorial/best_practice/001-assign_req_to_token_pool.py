"""
1. Case study on performance optimization of Triton operators on NPU, compared with GPU.
2. The core business logic of this operator is to assign request data to the corresponding 
token pool on the device. It first calculates the KV (key-value) start and end positions for 
the current request using the program ID. Then, it computes the output cache offset by 
summing the lengths of all previous requests (derived from their start and end positions). 
Using block-based processing (with a block size of 32), it loads data from 
the output cache and stores it into the target token pool in loops, adjusting the load and 
save offsets incrementally until all data within the KV range is processed.
"""


import time
import triton
import torch
import triton.language as tl


def is_npu() -> bool:
    return hasattr(torch, "npu") and torch.npu.is_available()


# source: python/sglang/srt/speculative/spec_utils.py
@triton.jit
def gpu_assign_req_to_token_pool(
    req_pool_indices,
    req_to_token,
    start_offset,
    end_offset,
    out_cache_loc,
    pool_len: tl.constexpr,
    bs_upper: tl.constexpr,
):
    BLOCK_SIZE: tl.constexpr = 32
    pid = tl.program_id(axis=0)
    kv_start = tl.load(start_offset + pid)
    kv_end = tl.load(end_offset + pid)
    token_pool = req_to_token + tl.load(req_pool_indices + pid) * pool_len

    length_offset = tl.arange(0, bs_upper)
    start = tl.load(start_offset + length_offset, mask=length_offset < pid, other=0) # int64
    end = tl.load(end_offset + length_offset, mask=length_offset < pid, other=0) # int64
    out_offset = tl.sum(end - start, axis=0)

    out_cache_ptr = out_cache_loc + out_offset

    save_offset = tl.arange(0, BLOCK_SIZE) + kv_start
    load_offset = tl.arange(0, BLOCK_SIZE)

    num_loop = tl.cdiv(kv_end - kv_start, BLOCK_SIZE)
    for _ in range(num_loop):
        mask = save_offset < kv_end
        data = tl.load(out_cache_ptr + load_offset, mask=mask)
        tl.store(token_pool + save_offset, data, mask=mask)
        save_offset += BLOCK_SIZE
        load_offset += BLOCK_SIZE


@triton.jit
def npu_assign_req_to_token_pool(
    req_pool_indices,
    req_to_token,
    start_offset,
    end_offset,
    out_cache_loc,
    pool_len: tl.constexpr,
    bs_upper: tl.constexpr,
):
    BLOCK_SIZE: tl.constexpr = 32
    pid = tl.program_id(axis=0)
    kv_start = tl.load(start_offset + pid)
    kv_end = tl.load(end_offset + pid)
    token_pool = req_to_token + tl.load(req_pool_indices + pid) * pool_len

    length_offset = tl.arange(0, bs_upper)
    start = tl.load(start_offset + length_offset, mask=length_offset < pid, other=0) # Ensure start_offset uses int32 dtype
    end = tl.load(end_offset + length_offset, mask=length_offset < pid, other=0) # Ensure end_offset uses int32 dtype 
    out_offset = tl.sum(end - start, axis=0) 
    # Using int32 as the input dtype improves performance for vector subtraction (end - start) and vector summation (tl.sum(...)) operations 

    out_cache_ptr = out_cache_loc + out_offset

    save_offset = tl.arange(0, BLOCK_SIZE) + kv_start
    load_offset = tl.arange(0, BLOCK_SIZE)

    num_loop = tl.cdiv(kv_end - kv_start, BLOCK_SIZE)
    for i in range(num_loop):
        l_offset = i * BLOCK_SIZE + load_offset  # Prefer using 'l_offset = i * BLOCK_SIZE + load_offset ' over 'load_offset += BLOCK_SIZE'
        s_offset = i * BLOCK_SIZE + save_offset
        mask = save_offset < kv_end
        data = tl.load(out_cache_ptr + l_offset, mask=mask)
        tl.store(token_pool + s_offset, data, mask=mask)


def run(device_name = "npu"):
    max_batch_size = 512
    max_context_len = 8192
    batch_size = 32
    offset_len = 2

    if device_name == "npu":
        dtype_name = torch.int32
        assign_req_to_token_pool = npu_assign_req_to_token_pool
        sync_device = torch.npu
    else:
        dtype_name = torch.int64
        assign_req_to_token_pool = gpu_assign_req_to_token_pool
        sync_device = torch.cuda


    req_to_token = torch.zeros(max_batch_size, max_context_len, device=device_name, dtype=dtype_name)
    req_pool_indices = torch.randint(0, max_batch_size, (batch_size,), device=device_name, dtype=dtype_name)
    start_offset = torch.randint(0, max_context_len - offset_len, (batch_size,), device=device_name, dtype=dtype_name)
    end_offset = start_offset + torch.randint(1, offset_len + 1, (batch_size,), device=device_name, dtype=dtype_name)
    out_cache_loc = torch.randint(0, 8 * max_context_len, ((end_offset - start_offset).sum(),), device=device_name, dtype=dtype_name)
    pool_len = max_context_len
    bs_upper = batch_size

    #warmup
    assign_req_to_token_pool[(batch_size,)](req_pool_indices, req_to_token, start_offset, end_offset, out_cache_loc, pool_len, bs_upper)
    sync_device.synchronize()

    spend_time = 0
    iter_times = 100
    for i in range(iter_times):
        sync_device.synchronize()
        start_time = time.time()
        assign_req_to_token_pool[(batch_size,)](req_pool_indices, req_to_token, start_offset, end_offset, out_cache_loc, pool_len, bs_upper)
        sync_device.synchronize()
        spend_time += (time.time() - start_time)

    print(f"==== {device_name} spend_time: {spend_time / iter_times * 1000} ms")


if __name__ == "__main__":
    if is_npu():
        run("npu")
    else:
        run("cuda")