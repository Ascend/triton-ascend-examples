# 006-tiling.py说明

## 功能

使用triton实现

``` python
out = torch.gather(x, dim=1, index=idx)
```

输入:

| Input | Shape  |
|-------|--------|
| x     | (B, C) |
| idx   | (B, K) |

输出

| Input | Shape  |
|-------|--------|
| out   | (B, K) |

## 差异点概述

npu 设备的物理核数一般为40或48，当发射的逻辑核数量远大于物理核时，会有严重的启动及调度开销
建议在编写 npu版本的triton kernel时，尽量使发射的逻辑核数量等于物理核

## 差异点详解

Code diff of NPU and CUDA

```diff
@triton.jit
def gather_dim1_kernel(
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
    pid_b = tl.program_id(0)  # 1 block per batch row
-   # GPU实现
-   pid_k = tl.program_id(1)  # 1 block per K-tile

-   k_off = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
-   mask = k_off < K

-   idx = tl.load(idx_ptr + pid_b * stride_ib + k_off * stride_ik, mask=mask)  # [BLOCK_K]

-   x_val = tl.load(x_ptr + pid_b * stride_xb + idx * stride_xc, mask=mask)

-   tl.store(out_ptr + pid_b * stride_ob + k_off * stride_ok, x_val, mask=mask)

+   #NPU实现
+   b_idx = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
+   b_mask = b_idx < B

+   # 对 K 维进行循环
+   for k_start in range(0, K, BLOCK_K):
+       ks = tl.arange(0, BLOCK_K)
+       k_mask = ks < K - k_start

+       idx_off = (b_idx[:, None] * stride_ib +
+                  (k_start + ks)[None, :] * stride_ik)
+       col_idx = tl.load(idx_ptr + idx_off, mask=b_mask[:, None] & k_mask)

+       x_off = (b_idx[:, None] * stride_xb +
+                col_idx * stride_xc)
+       x_val = tl.load(x_ptr + x_off, mask=b_mask[:, None] & k_mask)

+       out_off = (b_idx[:, None] * stride_ob +
+                  (k_start + ks)[None, :] * stride_ok)
+       tl.store(out_ptr + out_off, x_val, mask=b_mask[:, None] & k_mask)

# 调用
B = 128  # batch dim
K = 64  

BLOCK_B = 4
BLOCK_K = 128

— # GPU  
- grid = (B, triton.cdiv(K, BLOCK_K))
+ # NPU
+ grid = (triton.cdiv(B, BLOCK_B),)

gather_dim1_kernel[grid](
    x, idx, out,
    x.stride(0), x.stride(1),
    idx.stride(0), idx.stride(1),
    out.stride(0), out.stride(1),
    B, K,
    BLOCK_B=BLOCK_B,
    BLOCK_K=BLOCK_K,
)

```
