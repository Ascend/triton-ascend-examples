"""
The purpose of this example is to demonstrate making highly use of NPU resources by tiling.

The difference between GPU and NPU is launched kernel number.
On the NPU, we are trying to match the logical kernel number and physical core number to maximum the utilization
and avoid overheads like scheduling between cores.
"""

import triton
import triton.language as tl
import torch

from ..utils import is_npu

_is_npu = is_npu()
if _is_npu:
    import torch_npu

    device = torch.device('npu')
else:
    device = torch.device('cuda')


@triton.jit
def gpu_gather_dim1_kernel(
        x_ptr,  # *x  [B, C]
        idx_ptr,  # *idx[B, K]
        out_ptr,  # *out[B, K]
        stride_xb, stride_xc,
        stride_ib, stride_ik,
        stride_ob, stride_ok,
        B, K,
        BLOCK_B: tl.constexpr,
        BLOCK_K: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_k = tl.program_id(1)

    k_off = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    mask = k_off < K

    idx = tl.load(idx_ptr + pid_b * stride_ib + k_off * stride_ik, mask=mask)

    x_val = tl.load(x_ptr + pid_b * stride_xb + idx * stride_xc, mask=mask)

    tl.store(out_ptr + pid_b * stride_ob + k_off * stride_ok, x_val, mask=mask)


@triton.jit
def npu_gather_dim1_kernel(
        x_ptr,  # *x  [B, C]
        idx_ptr,  # *idx[B, K]
        out_ptr,  # *out[B, K]
        stride_xb, stride_xc,
        stride_ib, stride_ik,
        stride_ob, stride_ok,
        B, K,
        BLOCK_B: tl.constexpr,
        BLOCK_K: tl.constexpr,
):
    pid_b = tl.program_id(0)

    b_idx = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    b_mask = b_idx < B

    # 对 K 维进行循环
    for k_start in range(0, K, BLOCK_K):
        ks = tl.arange(0, BLOCK_K)
        k_mask = ks < K - k_start

        idx_off = (b_idx[:, None] * stride_ib +
                   (k_start + ks)[None, :] * stride_ik)
        col_idx = tl.load(idx_ptr + idx_off, mask=b_mask[:, None] & k_mask)

        x_off = (b_idx[:, None] * stride_xb +
                 col_idx * stride_xc)
        x_val = tl.load(x_ptr + x_off, mask=b_mask[:, None] & k_mask)

        out_off = (b_idx[:, None] * stride_ob +
                   (k_start + ks)[None, :] * stride_ok)
        tl.store(out_ptr + out_off, x_val, mask=b_mask[:, None] & k_mask)


def run(device_name="npu"):
    B = 128  # batch dim
    C = 1024  # column dim
    K = 64  # index dim

    x = torch.randn(B, C, dtype=torch.float32, device=device)
    idx = torch.randint(0, C, (B, K), dtype=torch.int64, device=device)
    out = torch.empty(B, K, dtype=x.dtype, device=x.device)

    BLOCK_B = 4
    BLOCK_K = 128

    if device_name == "npu":
        gather_dim1_kernel = npu_gather_dim1_kernel
        grid = (triton.cdiv(B, BLOCK_B),)
    else:
        gather_dim1_kernel = gpu_gather_dim1_kernel
        grid = (B, triton.cdiv(K, BLOCK_K))

    gather_dim1_kernel[grid](
        x, idx, out,
        x.stride(0), x.stride(1),
        idx.stride(0), idx.stride(1),
        out.stride(0), out.stride(1),
        B, K,
        BLOCK_B=BLOCK_B,
        BLOCK_K=BLOCK_K,
    )


if __name__ == "__main__":
    run("npu" if _is_npu else "cuda")
