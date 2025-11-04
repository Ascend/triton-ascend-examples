# 004-discrete-memory-access.py说明

此事例旨在说明，在离散访存场景下，如何高效load数据

## 功能

使用triton实现

```python
out = x[idx]
```

输入:

| Input | Shape |
|-------|-------|
| x     | (M,)  |
| idx   | (N,)  |

输出

| Input | Shape |
|-------|-------|
| out   | (N,)  |

## 差异点概述

- gpu直接从global中离散访问取数
- npu先将数据从global搬至share，再从share中select目标值

## 差异点详解

Code diff of NPU and CUDA

```diff
@triton.jit
def pick_kernel(
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
+   rm = tl.arange(0, M)
    rn = tl.arange(0, N)

    idx = tl.load(idx_ptr + rn * stride_idx)
    mask = idx < M

-   # gpu
-   val = tl.load(x_ptr + idx * stride_x, mask=mask)  # gpu直接从global中离散访问取数
+   # npu
+   x_shared = tl.load(x_ptr + rm * stride_x)  # [M]  # npu先将数据从global搬至share
+   val = tl.gather(x_shared, idx, 0)  # 再从share中select目标值

    tl.store(y_ptr + rn * stride_y, val, mask=mask)

```
