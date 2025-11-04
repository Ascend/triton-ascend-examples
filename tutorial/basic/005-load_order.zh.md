# 005-load-order.py说明

## 功能

该文件旨在说明，编译器不会修改用户load指令的顺序，在前面的load指令被其他指令阻塞时，可以将没有数据依赖的load语句放在前面以提前发射，提升并行度。

## 差异点概述

在循环中，原本的语句顺序是

```
load B
load A
calc
store O
store B
```
由于当前的load B会等待上一次循环的store B，load A不能提前与load B执行，所以load A与store B不能并行。

将语句顺序改为
```
load A
load B
calc
store O
store B
```
load A即可与上一次循环的store B并行。


## 差异点详解

Code diff
```diff
@triton.jit
def AB_load_kernel(
    A,
    B,
    B_index,
    O,
    B_DIM: tl.constexpr,
    HEAD_NUM: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    i_n = tl.program_id(0)
    i_range = tl.arange(0, B_DIM)

    for i in range(HEAD_NUM):
        # calc index
        p_A = A + i * HEAD_DIM + i_n * B_DIM + i_range
        p_O = O + i * HEAD_DIM + i_n * B_DIM + i_range
        p_B_index = B_index + i

-        # load B
-        idx_B = tl.load(p_B_index)
-        p_B = B + idx_B
-        b_B = tl.load(p_B)
-
-        # load A
-        b_A = tl.load(p_A)

+        # load A
+        b_A = tl.load(p_A)
+
+        # load B
+        idx_B = tl.load(p_B_index)
+        p_B = B + idx_B
+        b_B = tl.load(p_B)

        # calculation
        b_B += tl.sum(b_A)
        b_O = b_A * b_B

        # store O
        tl.store(p_O, b_O)

        # store B
        idx_B = tl.load(p_B_index)
        p_B = B + idx_B
        tl.store(p_B, b_B)

```

