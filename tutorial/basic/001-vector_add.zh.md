# 001-vector_add.py说明

## 功能
以vector_add为例说明npu上int类型的vector运算，使用int32比int64性能更优


## 差异点详解

```diff
@triton.jit
def npu_vector_add_kernel(
    x,                          # [Tensor] input tensor (1 x col)     
    y,                          # [Tensor] input tensor (1 x col)
    z,                          # [Tensor] output tensor (1 x col)
    vector_len: tl.constexpr,   # len of the vector
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    offset = pid * BLOCK_SIZE + tl.arange(BLOCK_SIZE)
    len_mask = offset < vector_len
    x1 = tl.load(x + offset, mask=len_mask)
    y1 = tl.load(y + offset, mask=len_mask)
    z1 = x1 + y1
    tl.store(z + offset, z1, mask=len_mask)


def run(dtype_name):
    vector_len = 64
    BLOCK_SIZE = 32
    BLOCK_DIM = 16
    device_name = "npu"

    x = torch.randint(0, 100, (1, vector_len), device=device_name, dtype=dtype_name)
    y = torch.randint(0, 100, (1, vector_len), device=device_name, dtype=dtype_name)
    z = torch.zeros((1, vector_len), device=device_name, dtype=dtype_name)
    npu_vector_add_kernel([BLOCK_DIM,])(x, y, z, vector_len, BLOCK_SIZE)
    torch.npu.synchronize()
    print("vector add result of x {x} and y {y} is: {z}")

    spend_time = 0
    iter_times = 100
    for i in range(iter_times):
        start_time = time.time()
        npu_vector_add_kernel([BLOCK_DIM,])(x, y, z, vector_len, BLOCK_SIZE)
        torch.npu.synchronize()
        spend_time += (time.time() - start_time)

    print(f"==== {dtype_name} spend_time: {spend_time / iter_times * 1000} ms")

if __name__ == "__main__":
-   run(torch.int64)
+   run(torch.int32) # Ensure inputs uses int32 dtype
```

