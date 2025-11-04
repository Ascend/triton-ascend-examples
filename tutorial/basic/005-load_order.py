'''
The purpose of this example is to demonstrate how to avoid a load with data dependencies blocking a load without
data dependencies.

In the loop, load A and load B have no data dependencies, but load B waits for the previous loop's store B/
If load A follows load B, load A is blocked by load B and cannot execute in parallel with the previous loop's store B.
In this case, the order of load A and load B can be swapped, allowing load A to execute earlier and in parallel with
store B, thus boost our efficiency. 
''' 
import time

import torch
import triton
import triton.language as tl


@triton.jit
def BA_load_kernel(
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

        # load B
        idx_B = tl.load(p_B_index)
        p_B = B + idx_B
        b_B = tl.load(p_B)

        # load A
        b_A = tl.load(p_A)

        # calculation
        b_B += tl.sum(b_A)
        b_O = b_A * b_B

        # store O
        tl.store(p_O, b_O)

        # store B
        idx_B = tl.load(p_B_index)
        p_B = B + idx_B
        tl.store(p_B, b_B)


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

        # load A
        b_A = tl.load(p_A)

        # load B
        idx_B = tl.load(p_B_index)
        p_B = B + idx_B
        b_B = tl.load(p_B)

        # calculation
        b_B += tl.sum(b_A)
        b_O = b_A * b_B

        # store O
        tl.store(p_O, b_O)

        # store B
        idx_B = tl.load(p_B_index)
        p_B = B + idx_B
        tl.store(p_B, b_B)


def load_order_example(
    A,
    B,
    B_index,
    mode: str
):
    HEAD_NUM, HEAD_DIM = A.shape
    B_DIM = 8
    O = torch.empty_like(A)

    N_DIM = HEAD_DIM // B_DIM

    grid = (N_DIM, )
    if mode == "BA":
        BA_load_kernel[grid](
            A,
            B,
            B_index,
            O,
            B_DIM,
            HEAD_NUM,
            HEAD_DIM,
        )
    else:
        AB_load_kernel[grid](
            A,
            B,
            B_index,
            O,
            B_DIM,
            HEAD_NUM,
            HEAD_DIM,
        )

def run(device_name="npu", mode="BA"):
    HEAD_NUM = 4
    HEAD_DIM = 32
    iter_times = 10001

    A = torch.arange(HEAD_DIM, dtype=torch.float32, device=device_name)
    A = A.repeat(HEAD_NUM, 1).contiguous()

    B = torch.arange(HEAD_NUM, dtype=torch.float32, device=device_name)

    B_index = torch.arange(HEAD_NUM, dtype=torch.int32, device=device_name)

    spend_time = 0
    for i in range(iter_times):
        start_time = time.time()
        O = load_order_example(A, B, B_index, mode)
        if i:
            spend_time += (time.time() - start_time)

    print(f"load_order: {mode}, spend_time: {spend_time / (iter_times - 1) * 1000 * 1000} us")


if __name__ == "__main__":
    run("npu", "BA")
    run("npu", "AB")