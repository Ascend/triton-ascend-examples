"""
The purpose of this example is to demonstrate accessing discrete memory on NPU devices.

The GPU version load discrete data with 'tl.load' and 'mask' from global memory.
The NPU version first load all data into shared memory contiguously, then pick the elements from the shared memory.
"""

import torch

import triton
import triton.language as tl

from ..utils import is_npu

_is_npu = is_npu()
if _is_npu:
    import torch_npu

    device = torch.device('npu')
else:
    device = torch.device('cuda')


@triton.jit
def npu_pick_kernel(
        x_ptr,
        idx_ptr,
        y_ptr,
        stride_x,
        stride_idx,
        stride_y,
        M: tl.constexpr,
        N: tl.constexpr
):
    pid = tl.program_id(0)  # 1 block
    rm = tl.arange(0, M)  # [M]
    rn = tl.arange(0, N)  # [N]

    idx = tl.load(idx_ptr + rn * stride_idx)  # [N]
    mask = idx < M

    x_shared = tl.load(x_ptr + rm * stride_x)  # [M]
    val = tl.gather(x_shared, idx, 0)

    tl.store(y_ptr + rn * stride_y, val, mask=mask)


@triton.jit
def gpu_pick_kernel(
        x_ptr,
        idx_ptr,
        y_ptr,
        stride_x,
        stride_idx,
        stride_y,
        M: tl.constexpr,
        N: tl.constexpr
):
    pid = tl.program_id(0)
    rn = tl.arange(0, N)  # [0..N)

    idx = tl.load(idx_ptr + rn * stride_idx)
    mask = idx < M

    val = tl.load(x_ptr + idx * stride_x, mask=mask)

    tl.store(y_ptr + rn * stride_y, val, mask=mask)


def run(device_name="npu"):
    if device_name == "npu":
        pick_kernel = npu_pick_kernel
    else:
        pick_kernel = gpu_pick_kernel

    M = 1024
    N = 256

    x = torch.randn(M, device=device)
    indices = torch.randint(0, M, (N,), device=device)
    y = torch.empty(N, dtype=x.dtype, device=x.device)

    grid = (1,)
    pick_kernel[grid](
        x,
        indices,
        y,
        x.stride(0),
        indices.stride(0),
        y.stride(0),
        M=M,
        N=triton.next_power_of_2(N),
    )
    torch.npu.synchronize()


if __name__ == "__main__":
    run("npu" if _is_npu else "cuda")
