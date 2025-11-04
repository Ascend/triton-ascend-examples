# 004-decode_grouped_attention.py说明

## 功能
以decode_grouped_attention为例说明，在npu上，如果存在低维度离散高纬度连续的矩阵，使用triton时的优化方式


## 差异点详解
1. 目标矩阵如果高纬度离散低纬度连续，在访存时，编译器会自动优化，仅对高维离散轴展开，低纬保存向量化处理；开发者也可以更加灵活地选择手动使用for循环展开高维离散轴，对低纬进行连续访存。
2. 目标矩阵如果低纬度离散高纬度连续，在访存时，需要先按照转置方式，先变化为高纬离散低纬连续进行访存，然后在转置成目标矩阵


```diff
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
            # 高维连续低纬离散
-           offs_buf_k = (
-               kv_loc[None, :] * stride_buf_kbs
-               + cur_kv_head * stride_buf_kh
-               + offs_d[:, None]
-           )
-           k = tl.load(
-               K_Buffer + offs_buf_k,
-               mask=(offs_n[None, :] < split_kv_end) & (mask_d[:, None]),
-               other=0.0,
-           )

+           k = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=q.dtype)
+           for i in range(start_n, min(BLOCK_N + start_n, split_kv_end)):
+               ind = i - start_n
+               offs_buf_k = (
+                   tl.get_element(kv_loc, (ind, ))  * stride_buf_kbs
+                   + cur_kv_head * stride_buf_kh
+                  + offs_d[None, :]
+               )
+               k_tmp = tl.load(K_Buffer + offs_buf_k, mask=(mask_d[None, :]), other=0.0)
+               k = tl.insert_slice(k, k_tmp, (ind, 0), (1, BLOCK_DMODEL), (1, 1))
+           k = tl.trans(k, (1, 0))

            qk = tl.dot(q, k.to(q.dtype))
            if BLOCK_DPE > 0:
                # 高维连续低纬离散
-               offs_buf_kpe = (
-                   kv_loc[None, :] * stride_buf_kbs
-                   + cur_kv_head * stride_buf_kh
-                   + offs_dpe[:, None]
-               )
-               kpe = tl.load(
-                   K_Buffer + offs_buf_kpe,
-                   mask=(offs_n[None, :] < split_kv_end) & (mask_dpe[:, None]),
-                   other=0.0,
-               )

+               kpe = tl.zeros([BLOCK_N, BLOCK_DPE], dtype=qpe.dtype)
+               for i in range(start_n, min(BLOCK_N + start_n, split_kv_end)):
+                   ind = i - start_n
+                   offs_buf_kpe = (
+                       tl.get_element(kv_loc, (ind, ))  * stride_buf_kbs
+                       + cur_kv_head * stride_buf_kh
+                       + offs_dpe[None, :]
+                   )
+                   kpe_tmp = tl.load(K_Buffer + offs_buf_kpe, mask=(mask_dpe[None, :]), other=0.0)
+                   kpe = tl.insert_slice(kpe, kpe_tmp, (ind, 0), (1, BLOCK_DPE), (1, 1))
+               kpe = tl.trans(kpe, (1, 0))

                qk += tl.dot(qpe, kpe.to(qpe.dtype))
            qk *= sm_scale

            qk = tl.where(
                mask_h[:, None] & (offs_n[None, :] < split_kv_end), qk, float("-inf")
            )

            # 高维离散低纬连续
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
```

