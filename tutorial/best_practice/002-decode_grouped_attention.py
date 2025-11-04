import time
import triton
import torch
import triton.language as tl


@triton.jit
def tanh(x):
    return 2 * tl.sigmoid(2 * x) - 1


@triton.jit
def grouped_attention_kernel_stage1(
    Q,                              # [Tensor] input tensor query
    K_Buffer,                       # [Tensor] input tensor k cache
    V_Buffer,                       # [Tensor] input tensor v cache
    sm_scale,                       # [Scalar] scale for qk
    kv_indptr,                      # [Tensor] cumsum of kv seq_lens of the batch
    kv_indices,                     # [Tensor] kv indices of each token in kv cache
    Att_Out,                        # [Tensor] output tensor
    Att_Lse,                        # [Tensor] output tensor
    num_kv_splits,                  # [Tensor] kv split num of each input in the batch
    stride_qbs,                     # [Scalar] stride of q dim(0)
    stride_qh,                      # [Scalar] stride of q dim(1)
    stride_buf_kbs,                 # [Scalar] stride of k dim(0)
    stride_buf_kh,                  # [Scalar] stride of k dim(1)
    stride_buf_vbs,                 # [Scalar] stride of v dim(0)
    stride_buf_vh,                  # [Scalar] stride of v dim(1)
    stride_mid_ob,                  # [Scalar] stride of attention_out dim(0)
    stride_mid_oh,                  # [Scalar] stride of attention_out dim(1)
    stride_mid_os,                  # [Scalar] stride of attention_out dim(2)
    kv_group_num: tl.constexpr,
    q_head_num: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DPE: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
    MIN_BLOCK_KV: tl.constexpr,
    Lk: tl.constexpr,
    Lv: tl.constexpr
):
    cur_batch = tl.program_id(0)
    cur_head_id = tl.program_id(1)
    cur_kv_head = cur_head_id // tl.cdiv(kv_group_num, BLOCK_H)
    split_kv_id = tl.program_id(2)

    if BLOCK_H < kv_group_num:
        VALID_BLOCK_H: tl.constexpr = BLOCK_H
    else:
        VALID_BLOCK_H: tl.constexpr = kv_group_num
    cur_head = cur_head_id * VALID_BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = cur_head < (cur_head_id + 1) * VALID_BLOCK_H
    mask_h = mask_h & (cur_head < q_head_num)

    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dv = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lk
    mask_dv = offs_dv < Lv

    cur_batch_kv_start_idx = tl.load(kv_indptr + cur_batch)
    cur_batch_seq_len = tl.load(kv_indptr + cur_batch + 1) - cur_batch_kv_start_idx
    kv_splits = tl.load(num_kv_splits + cur_batch)

    offs_q = cur_batch * stride_qbs + cur_head[:, None] * stride_qh + offs_d[None, :]

    if BLOCK_DPE > 0:
        offs_dpe = BLOCK_DMODEL + tl.arange(0, BLOCK_DPE)
        mask_dpe = offs_dpe < Lk
        off_qpe = (
            cur_batch * stride_qbs + cur_head[:, None] * stride_qh + offs_dpe[None, :]
        )

    kv_len_per_split = (
        tl.cdiv(tl.cdiv(cur_batch_seq_len, kv_splits), MIN_BLOCK_KV) * MIN_BLOCK_KV
    )
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

    e_max = tl.zeros([BLOCK_H], dtype=tl.float32) - float("inf")
    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, BLOCK_DV], dtype=tl.float32)

    if split_kv_end > split_kv_start:
        q = tl.load(Q + offs_q, mask=(mask_h[:, None]) & (mask_d[None, :]), other=0.0)
        if BLOCK_DPE > 0:
            qpe = tl.load(
                Q + off_qpe, mask=(mask_h[:, None]) & (mask_dpe[None, :]), other=0.0
            )
        for start_n in range(split_kv_start, split_kv_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            kv_loc = tl.load(
                kv_indices + cur_batch_kv_start_idx + offs_n,
                mask=offs_n < split_kv_end,
                other=0,
            )
            # GPU原生处理，高纬连续低纬离散访存
            # offs_buf_k = (
            #     kv_loc[None, :] * stride_buf_kbs
            #     + cur_kv_head * stride_buf_kh
            #     + offs_d[:, None]
            # )
            # k = tl.load(
            #     K_Buffer + offs_buf_k,
            #     mask=(offs_n[None, :] < split_kv_end) & (mask_d[:, None]),
            #     other=0.0,
            # )

            k = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=q.dtype)
            for i in range(start_n, min(BLOCK_N + start_n, split_kv_end)):
                ind = i - start_n
                offs_buf_k = (
                    tl.get_element(kv_loc, (ind, ))  * stride_buf_kbs
                    + cur_kv_head * stride_buf_kh
                    + offs_d[None, :]
                )
                k_tmp = tl.load(K_Buffer + offs_buf_k, mask=(mask_d[None, :]), other=0.0)
                k = tl.insert_slice(k, k_tmp, (ind, 0), (1, BLOCK_DMODEL), (1, 1))
            k = tl.trans(k, (1, 0))

            qk = tl.dot(q, k.to(q.dtype))
            if BLOCK_DPE > 0:
                # GPU原生处理，高纬连续低纬离散访存
                # offs_buf_kpe = (
                #     kv_loc[None, :] * stride_buf_kbs
                #     + cur_kv_head * stride_buf_kh
                #     + offs_dpe[:, None]
                # )
                # kpe = tl.load(
                #     K_Buffer + offs_buf_kpe,
                #     mask=(offs_n[None, :] < split_kv_end) & (mask_dpe[:, None]),
                #     other=0.0,
                # )

                kpe = tl.zeros([BLOCK_N, BLOCK_DPE], dtype=qpe.dtype)
                for i in range(start_n, min(BLOCK_N + start_n, split_kv_end)):
                    ind = i - start_n
                    offs_buf_kpe = (
                        tl.get_element(kv_loc, (ind, ))  * stride_buf_kbs
                        + cur_kv_head * stride_buf_kh
                        + offs_dpe[None, :]
                    )
                    kpe_tmp = tl.load(K_Buffer + offs_buf_kpe, mask=(mask_dpe[None, :]), other=0.0)
                    kpe = tl.insert_slice(kpe, kpe_tmp, (ind, 0), (1, BLOCK_DPE), (1, 1))
                kpe = tl.trans(kpe, (1, 0))

                qk += tl.dot(qpe, kpe.to(qpe.dtype))
            qk *= sm_scale

            qk = tl.where(
                mask_h[:, None] & (offs_n[None, :] < split_kv_end), qk, float("-inf")
            )

            # 高纬离散低纬连续访存，编译器自动优化
            offs_buf_v = (
                kv_loc[:, None] * stride_buf_vbs
                + cur_kv_head * stride_buf_vh
                + offs_dv[None, :]
            )
            v = tl.load(
                V_Buffer + offs_buf_v,
                mask=(offs_n[:, None] < split_kv_end) & (mask_dv[None, :]),
                other=0.0,
            )

            n_e_max = tl.maximum(tl.max(qk, 1), e_max)
            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max[:, None])
            acc *= re_scale[:, None]
            acc += tl.dot(p.to(v.dtype), v)

            e_sum = e_sum * re_scale + tl.sum(p, 1)
            e_max = n_e_max

        offs_mid_o = (
            cur_batch * stride_mid_ob
            + cur_head[:, None] * stride_mid_oh
            + split_kv_id * stride_mid_os
            + offs_dv[None, :]
        )

        tl.store(
            Att_Out + offs_mid_o,
            acc / e_sum[:, None],
            mask=(mask_h[:, None]) & (mask_dv[None, :]),
        )

        offs_mid_o_1 = (
            cur_batch * stride_mid_ob
            + cur_head * stride_mid_oh
            + split_kv_id * stride_mid_os
        ) // Lv

        tl.store(
            Att_Lse + offs_mid_o_1,
            e_max + tl.log(e_sum),
            mask=mask_h,
        )


# For stage1 of GQA/MQA/MLA 
def decode_grouped_attention_stage1(
    q,
    k_buffer,
    v_buffer,
    att_out,
    att_lse,
    kv_indptr,
    kv_indices,
    num_kv_splits,
    max_kv_splits,
    sm_scale,
):
    Lk = k_buffer.shape[-1]
    Lv = v_buffer.shape[-1]

    if Lk == 576:
        BLOCK_DMODEL = 512
        BLOCK_DPE = 64
    elif Lk == 288:
        BLOCK_DMODEL = 256
        BLOCK_DPE = 32
    else:
        BLOCK_DMODEL = triton.next_power_of_2(Lk)
        BLOCK_DPE = 0
    BLOCK_DV = triton.next_power_of_2(Lv)

    batch, head_num = q.shape[0], q.shape[1]
    kv_group_num = q.shape[1] // k_buffer.shape[1]

    # tiling
    BLOCK_N = 16 # BLOCK_SIZE of kv seq_len
    BLOCK_H = 16 # BLOCK_SIZE of q head_num
    MIN_BLOCK_KV = 32 # min token num of each kv split
    MAX_KV_SPLITS = max_kv_splits
    grid = (
        batch,
        triton.cdiv(head_num, min(BLOCK_H, kv_group_num)),
        MAX_KV_SPLITS,
    )

    num_stages = 2
    grouped_attention_kernel_stage1[grid](
        q,
        k_buffer,
        v_buffer,
        sm_scale,
        kv_indptr,
        kv_indices,
        att_out,
        att_lse,
        num_kv_splits,
        q.stride(0),
        q.stride(1),
        k_buffer.stride(0),
        k_buffer.stride(1),
        v_buffer.stride(0),
        v_buffer.stride(1),
        att_out.stride(0),
        att_out.stride(1),
        att_out.stride(2),
        kv_group_num=kv_group_num,
        q_head_num=head_num,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_DPE=BLOCK_DPE,
        BLOCK_DV=BLOCK_DV,
        BLOCK_N=BLOCK_N,
        BLOCK_H=BLOCK_H,
        MIN_BLOCK_KV=MIN_BLOCK_KV,
        num_warps=4,
        num_stages=num_stages,
        Lk=Lk,
        Lv=Lv
    )


def get_device() -> torch.device:
    if hasattr(torch, "npu") and torch.npu.is_available():
        # 华为昇腾 NPU 优先
        return torch.device('npu:0'), 'npu'
    elif torch.cuda.is_available():
        # NVIDIA CUDA
        return torch.device('cuda:0'), 'cuda'
    else:
        # 默认使用 CPU
        return torch.device('cpu'), 'cpu'


def test_grouped_decode_attention_kernel(B, S, H_Q, H_KV, D, D_V):
    device, device_name = get_device()
    dtype = torch.bfloat16
    seq_len = S  # This represents the number of tokens already in the sequence
    total_tokens = B * seq_len
    
    sm_scale = 1.0 / (D**0.5)
    max_kv_splits = 4
    num_kv_splits = torch.full((B,), 4, dtype=torch.int32, device=device)

    # q represents the new token being generated, one per batch
    q = torch.randn(B, H_Q, D, dtype=dtype, device=device)
    # k_buffer and v_buffer represent all previous tokens
    k_buffer = torch.randn(total_tokens, H_KV, D, dtype=dtype, device=device)
    v_buffer = torch.randn(total_tokens, H_KV, D_V, dtype=dtype, device=device)

    b_seq_len = torch.full((B,), seq_len, device=device)

    kv_indptr = torch.zeros((B + 1,), dtype=torch.int32, device=device)
    kv_indptr[1 : B + 1] = torch.cumsum(b_seq_len[:B], dim=0)
    kv_indices = torch.arange(total_tokens, device=device)

    attn_logits = torch.empty(
        (B, H_Q, max_kv_splits, D_V),
        dtype=torch.float32,
        device=device,
    )
    attn_lse = torch.empty(
        (B, H_Q, max_kv_splits),
        dtype=torch.float32,
        device=device,
    )

    spend_time = 0
    iter_times = 100
    sync_device = torch.npu if device_name == 'npu' else torch.cuda
    sync_device.synchronize()
    for i in range(100):
        start_time = time.time()
        decode_grouped_attention_stage1(
            q,
            k_buffer,
            v_buffer,
            attn_logits,
            attn_lse,
            kv_indptr,
            kv_indices,
            num_kv_splits,
            max_kv_splits,
            sm_scale
        )
        sync_device.synchronize()
        spend_time += (time.time() - start_time)

    print(f"==== {device_name} spend_time: {spend_time / iter_times * 1000} ms")


if __name__ == '__main__':
    configs = [(1, 128, 32, 1, 576, 512)]
    for B, S, H_Q, H_KV, D, D_V in configs:
        test_grouped_decode_attention_kernel(B, S, H_Q, H_KV, D, D_V)