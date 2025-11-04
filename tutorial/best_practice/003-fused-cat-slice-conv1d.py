from typing import Optional

import torch
import torch.nn.functional as F

PAD_SLOT_ID = -1

from typing import Optional, Union

import pytest
import torch
import torch.nn.functional as F
from einops import rearrange
import triton
import triton.language as tl

PAD_SLOT_ID = -1

# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/mamba/ops/causal_conv1d.py
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2024, Tri Dao.
# Adapted from https://github.com/Dao-AILab/causal-conv1d/blob/main/causal_conv1d/causal_conv1d_interface.py
# and https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/mamba/ops/causal_conv1d.py

from typing import Optional, Union

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

def is_npu() -> bool:
    return hasattr(torch, "npu") and torch.npu.is_available()

def causal_conv1d_update_ref(x,
                             conv_state,
                             weight,
                             bias=None,
                             activation=None,
                             cache_seqlens=None,
                             conv_state_indices=None):
    """
    x: (batch, dim) or (batch, dim, seqlen)
    conv_state: (batch, dim, state_len), where state_len >= width - 1
    weight: (dim, width)
    bias: (dim,)
    cache_seqlens: (batch,), dtype int32.
        If not None, the conv_state is treated as a circular buffer.
        The conv_state will be updated by copying x to the
        conv_state starting at the index
        @cache_seqlens % state_len before performing the convolution.

    out: (batch, dim) or (batch, dim, seqlen)
    """
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")
    dtype_in = x.dtype
    unsqueeze = x.dim() == 2
    if unsqueeze:
        x = x.unsqueeze(-1)
    batch, dim, seqlen = x.shape
    width = weight.shape[1]
    state_len = conv_state.shape[-1]
    assert weight.shape == (dim, width)
    if cache_seqlens is None:
        x_new = torch.cat([conv_state[conv_state_indices], x], dim=-1).to(
            weight.dtype)  # (batch, dim, state_len + seqlen)
        to_copy = x_new[:, :, -state_len:]

        conv_state[conv_state_indices] =(to_copy)
    else:
        width_idx = torch.arange(
            -(width - 1), 0, dtype=torch.long,
            device=x.device).unsqueeze(0) + cache_seqlens.unsqueeze(1)
        width_idx = (torch.remainder(width_idx, state_len).unsqueeze(1).expand(
            -1, dim, -1))
        x_new = torch.cat([conv_state.gather(2, width_idx), x],
                          dim=-1).to(weight.dtype)
        copy_idx = torch.arange( 
            seqlen, dtype=torch.long,
            device=x.device).unsqueeze(0) + cache_seqlens.unsqueeze(1)
        copy_idx = torch.remainder(copy_idx,
                                   state_len).unsqueeze(1).expand(-1, dim, -1)
        conv_state.scatter_(2, copy_idx, x)

    out = F.conv1d(x_new, weight.unsqueeze(1), bias, padding=0,
                   groups=dim)[:, :, -seqlen:]

    if unsqueeze:
        out = out.squeeze(-1)

    return (out if activation is None else F.silu(out)).to(dtype=dtype_in)


def causal_conv1d_update_(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    activation: Optional[str] = None,
    cache_seqlens: Optional[torch.Tensor] = None,
    conv_state_indices: Optional[torch.Tensor] = None,
    pad_slot_id: int = PAD_SLOT_ID,
    use_triton: bool = False,
):
    """
    x: (batch, dim) or (batch, dim, seqlen)
    conv_state: (batch, dim, state_len), where state_len >= width - 1
    weight: (dim, width)
    bias: (dim,)
    cache_seqlens: (batch,), dtype int32.
        If not None, the conv_state is treated as a circular buffer.
        The conv_state will be updated by copying x to the conv_state
        starting at the index
        @cache_seqlens % state_len.
    conv_state_indices: (batch,), dtype int32
        If not None, the conv_state is a larger tensor along the batch dim,
        and we are selecting the batch coords specified by conv_state_indices.
        Useful for a continuous batching scenario.
    pad_slot_id: int
            if cache_indices is passed, lets the kernel identify padded
            entries that will not be processed,
            for example: cache_indices = [pad_slot_id, 1 ,20 ,pad_slot_id]
            in this case, the kernel will not process entries at
            indices 0 and 3
    out: (batch, dim) or (batch, dim, seqlen)
    """
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError(
            f"activation must be None, silu, or swish, actual: {activation}")
    activation_val = activation in ["silu", "swish"]
    unsqueeze = x.dim() == 2
    if unsqueeze:
        x = x.unsqueeze(-1)


    x = causal_conv1d_update_ref(x,
                                 conv_state,
                                 weight,
                                 bias,
                                 activation=activation,
                                 conv_state_indices=conv_state_indices)
    if unsqueeze:
        x = x.squeeze(-1)
    return x,conv_state


# 6*2048 *5 = 30
# 2 * 2048
@triton.jit()
def _causal_conv1d_update_kernel_no_cache_len_no_mtp(
    x_ptr, conv_state_ptr, weight_ptr, bias_ptr, conv_state_indices_ptr,
    out_ptr,
    batch: tl.constexpr,
    dim: tl.constexpr,
    align_val: tl.constexpr, 
    state_len: tl.constexpr, # 3 4 5 
    seq_len: tl.constexpr, # 1 2 
    width: tl.constexpr, # 4, <= seq_len + state_len
    out_len: tl.constexpr,
    x_batch_stride: tl.constexpr,
    conv_batch_stride: tl.constexpr,
    out_batch_stride: tl.constexpr,
    DIM_BLOCK: tl.constexpr, # dim % DIM_BLOCK must be 0
    HAS_BIAS: tl.constexpr,
    SILU_ACTIVATION: tl.constexpr,
    IS_CONTINUOUS_BATCHING: tl.constexpr,
):
    pid = tl.program_id(0)
    cat_len: tl.constexpr = state_len + seq_len # 4
    sub_state_len: tl.constexpr = state_len - seq_len # 3
    sub_align_dim: tl.constexpr = DIM_BLOCK // align_val

    conv_begin: tl.constexpr = (cat_len - width + 1) - seq_len

    if IS_CONTINUOUS_BATCHING:
        conv_batch_offs = tl.load(conv_state_indices_ptr + pid)
    else: 
        conv_batch_offs = pid
    
    for doffs in range(0, dim, DIM_BLOCK):
        '''
        conv_state = tl.load(conv_state_ptr + conv_batch_offs * conv_batch_stride + doffs * 3 + tl.arange(0, 2048 * 3))
        conv_state_T = conv_state.reshape(128, 16 * 3).trans().reshape(16, 3 * 128).trans().reshape(3 * 2048,)
        '''
        conv_state = tl.load(conv_state_ptr + conv_batch_offs * conv_batch_stride + doffs * state_len + tl.arange(0, DIM_BLOCK * state_len))
        conv_state_T = conv_state.reshape(sub_align_dim, align_val * state_len).trans().reshape(align_val, state_len * sub_align_dim).trans().reshape(state_len * DIM_BLOCK,)

        x = tl.load(x_ptr + pid * x_batch_stride + doffs * seq_len + tl.arange(0, DIM_BLOCK * seq_len))
        x_T = x.reshape(sub_align_dim, align_val * seq_len).trans().reshape(align_val, seq_len * sub_align_dim).trans().reshape(seq_len * DIM_BLOCK,)

        x_new_T = tl.full([cat_len * DIM_BLOCK], 0, x_ptr.dtype.element_ty)
        x_new_T = tl.insert_slice(x_new_T, conv_state_T, offsets = (0,), sizes = (state_len * DIM_BLOCK,), strides = (1,)) # [cat_len , DIM_BLOCK].view(-1)
        x_new_T = tl.insert_slice(x_new_T, x_T, offsets = (state_len * DIM_BLOCK,), sizes = (seq_len * DIM_BLOCK,), strides = (1,))

        new_conv_state_T = tl.extract_slice(x_new_T, (seq_len * DIM_BLOCK,), (state_len * DIM_BLOCK,), (1,)) # [state_len, DIM_BLOCK].view(-1)
        new_conv_state = new_conv_state_T.reshape(state_len * align_val, sub_align_dim).trans().reshape(sub_align_dim * state_len, align_val).trans().reshape(DIM_BLOCK * state_len,) # [DIM_BLOCK, state_len].view(-1)
        tl.store(conv_state_ptr + conv_batch_offs * conv_batch_stride + doffs * state_len + tl.arange(0, DIM_BLOCK * state_len), new_conv_state)

        weight = tl.load(weight_ptr + doffs * width + tl.arange(0, DIM_BLOCK * width))
        weight_T = weight.reshape(sub_align_dim, align_val * width).trans().reshape(align_val, width * sub_align_dim).trans().reshape(width * DIM_BLOCK,) # [width, DIM_BLOCK].view(-1)

        if HAS_BIAS:
            bias = tl.load(bias_ptr + doffs + tl.arange(0, DIM_BLOCK))
        else:
            bias = 0

        if width == cat_len:
            result = tl.sum((x_new_T.to(tl.float32) * weight_T).reshape(width, DIM_BLOCK), 0) + bias
            if SILU_ACTIVATION:
                result = result / (1 + tl.exp(-result))
            tl.store(out_ptr + pid * out_batch_stride + (doffs + tl.arange(0, DIM_BLOCK)) * out_len, result)
        else:
            for i in range(seq_len):
                x_conv_part = tl.extract_slice(x_new_T, ((conv_begin + i) * DIM_BLOCK), (width * DIM_BLOCK), (1,)).to(tl.float32)
                result = tl.sum((x_conv_part * weight_T).reshape(width, DIM_BLOCK), 0) + bias
                if SILU_ACTIVATION:
                    result = result / (1 + tl.exp(-result))
                tl.store(out_ptr + pid * out_batch_stride + (doffs + tl.arange(0, DIM_BLOCK)) * out_len, result)

@triton.jit()
def _causal_conv1d_update_kernel_gpu(
    # Pointers to matrices
    x_ptr,  # (batch, dim, seqlen)
    w_ptr,  # (dim, width)
    bias_ptr,
    conv_state_ptr,
    cache_seqlens_ptr,  # circular buffer
    conv_state_indices_ptr,
    num_accepted_tokens_ptr,
    intermediate_conv_window_ptr,
    o_ptr,  # (batch, dim, seqlen)
    # Matrix dimensions
    batch: int,
    dim: tl.constexpr,
    seqlen: tl.constexpr,
    state_len: tl.constexpr,
    num_cache_lines: tl.constexpr,  # added to support vLLM larger cache lines
    # Strides
    stride_x_seq: tl.constexpr,
    stride_x_dim: tl.constexpr,
    stride_x_token: tl.constexpr,
    stride_w_dim: tl.constexpr,
    stride_w_width: tl.constexpr,
    stride_conv_state_seq: tl.constexpr,
    stride_conv_state_dim: tl.constexpr,
    stride_conv_state_tok: tl.constexpr,
    stride_state_indices: tl.constexpr,
    stride_inter_seq: tl.constexpr,
    stride_inter_step: tl.constexpr,
    stride_inter_dim: tl.constexpr,
    stride_inter_win: tl.constexpr,
    stride_o_seq: tl.constexpr,
    stride_o_dim: tl.constexpr,
    stride_o_token: tl.constexpr,
    # others
    pad_slot_id: tl.constexpr,
    # Meta-parameters
    HAS_BIAS: tl.constexpr,
    KERNEL_WIDTH: tl.constexpr,
    SILU_ACTIVATION: tl.constexpr,
    IS_CONTINUOUS_BATCHING: tl.constexpr,
    IS_SPEC_DECODING: tl.constexpr,
    NP2_STATELEN: tl.constexpr,
    USE_PAD_SLOT: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SAVE_INTERMEDIATE: tl.constexpr,
):
    # ruff: noqa: E501
    idx_seq = tl.program_id(0)
    if idx_seq >= batch:
        return

    # [BLOCK_N,] elements along the feature-dimension (channel)
    idx_feats = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)

    if IS_CONTINUOUS_BATCHING:
        # mask = idx_seq < batch
        conv_state_batch_coord = tl.load(
            conv_state_indices_ptr + idx_seq * stride_state_indices
        ).to(tl.int64)
    else:
        conv_state_batch_coord = idx_seq
    if USE_PAD_SLOT:  # noqa
        if conv_state_batch_coord == pad_slot_id:
            # not processing as this is not the actual sequence
            return

    if IS_SPEC_DECODING:
        # The rolling of conv state:
        #
        # Before forward, the conv_state is:
        # [history1, history2, ..., historyM].
        #
        # After forward, the conv_state becomes:
        # [history2, ..., historyM, draft1, draft2, ..., draftN].
        #
        # After acceptance, it becomes:
        #
        # - accept 1 tokens: [history2, ..., historyM, draft1]
        # - accept 2 tokens: [history3, ..., historyM, draft1, draft2]
        # - and so on.
        conv_state_token_offset = tl.load(num_accepted_tokens_ptr + idx_seq) - 1
    else:
        conv_state_token_offset = 0

    # STEP 1: READ init_state data
    conv_states_base = (
        conv_state_ptr
        + (conv_state_batch_coord * stride_conv_state_seq)
        + (idx_feats * stride_conv_state_dim)
    )
    mask_w = idx_feats < dim

    prior_tokens = conv_states_base + conv_state_token_offset * stride_conv_state_tok
    if KERNEL_WIDTH >= 2:
        conv_states_ptrs = prior_tokens  # [BLOCK_N]
        col0 = tl.load(conv_states_ptrs, mask_w, 0.0)
    if KERNEL_WIDTH >= 3:
        conv_states_ptrs = prior_tokens + 1 * stride_conv_state_tok  # [BLOCK_N]
        col1 = tl.load(conv_states_ptrs, mask_w, 0.0)
    if KERNEL_WIDTH >= 4:
        conv_states_ptrs = prior_tokens + 2 * stride_conv_state_tok  # [BLOCK_N]
        col2 = tl.load(conv_states_ptrs, mask_w, 0.0)
    if KERNEL_WIDTH == 5:
        conv_states_ptrs = prior_tokens + 3 * stride_conv_state_tok  # [BLOCK_N]
        col3 = tl.load(conv_states_ptrs, mask_w, 0.0)

    # STEP 2: assume state_len > seqlen
    idx_tokens = tl.arange(0, NP2_STATELEN)  # [BLOCK_M]

    # The conv_state updates works in a sliding window manner,
    # at each forward pass, the tokens are shift by 1, so we
    # load since idx_tokens + 1.
    conv_state_ptrs_source = (
        conv_state_ptr
        + (conv_state_batch_coord * stride_conv_state_seq)
        + conv_state_token_offset * stride_conv_state_tok
        + (idx_feats * stride_conv_state_dim)[None, :]
        + ((idx_tokens + 1) * stride_conv_state_tok)[:, None]
    )  # [BLOCK_M, BLOCK_N]
    mask = (
        (conv_state_batch_coord < num_cache_lines)
        & ((idx_tokens + seqlen) < state_len)[:, None]
        & (idx_feats < dim)[None, :]
    )
    conv_state = tl.load(conv_state_ptrs_source, mask, other=0.0)

    VAL = state_len - seqlen
    x_base = x_ptr + (idx_seq * stride_x_seq) + (idx_feats * stride_x_dim)  # [BLOCK_N]

    x_ptrs = (
        x_base[None, :] + ((idx_tokens - VAL) * stride_x_token)[:, None]
    )  # [BLOCK_M, BLOCK_N]

    mask_x = (
        (idx_tokens - VAL >= 0)[:, None]
        & (idx_tokens - VAL < seqlen)[:, None]
        & (idx_feats < dim)[None, :]
    )  # token-index  # token-index  # feature-index
    loaded_x = tl.load(x_ptrs, mask_x, 0.0)
    tl.debug_barrier()

    new_conv_state = tl.where(mask, conv_state, loaded_x)

    conv_state_base = (
        conv_state_ptr
        + (conv_state_batch_coord * stride_conv_state_seq)
        + (idx_feats * stride_conv_state_dim)
    )  # [BLOCK_N,]
    conv_state_ptrs_target = (
        conv_state_base + (idx_tokens * stride_conv_state_tok)[:, None]
    )  # [BLOCK_M, BLOCK_N]
    mask = (idx_tokens < state_len)[:, None] & (idx_feats < dim)[None, :]
    tl.store(conv_state_ptrs_target, new_conv_state, mask)

    # STEP 3: init accumulator
    if HAS_BIAS:
        bias = bias_ptr + idx_feats
        mask_bias = idx_feats < dim
        acc_preload = tl.load(bias, mask=mask_bias, other=0.0).to(
            tl.float32
        )  # [BLOCK_N]
    else:
        acc_preload = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # STEP 4:
    # PRE-LOAD WEIGHTS
    # first kernel column, configured for weights to handle BLOCK_N features in range
    w_base = w_ptr + (idx_feats * stride_w_dim)  # [BLOCK_N,]
    mask_w = idx_feats < dim
    if KERNEL_WIDTH >= 2:
        w_ptrs = w_base + (0 * stride_w_width)  # [BLOCK_N] tensor
        w_col0 = tl.load(w_ptrs, mask_w, other=0.0)
        w_ptrs = w_base + (1 * stride_w_width)  # [BLOCK_N] tensor
        w_col1 = tl.load(w_ptrs, mask_w, other=0.0)
    if KERNEL_WIDTH >= 3:
        w_ptrs = w_base + (2 * stride_w_width)  # [BLOCK_N] tensor
        w_col2 = tl.load(w_ptrs, mask_w, other=0.0)
    if KERNEL_WIDTH >= 4:
        w_ptrs = w_base + (3 * stride_w_width)  # [BLOCK_N] tensor
        w_col3 = tl.load(w_ptrs, mask_w, other=0.0)

    x_base_1d = x_base  # starting of chunk [BLOCK_N]
    mask_x_1d = idx_feats < dim

    # STEP 5: compute each token
    for idx_token in tl.static_range(seqlen):
        acc = acc_preload

        matrix_w = w_col0
        matrix_x = col0
        for j in tl.static_range(KERNEL_WIDTH):
            if KERNEL_WIDTH == 2:
                if j == 1:  # KERNEL_WIDTH-1:
                    matrix_w = w_col1
                    x_ptrs_1d = x_base_1d + idx_token * stride_x_token  # [BLOCK_N]
                    matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)
            elif KERNEL_WIDTH == 3:
                if j == 1:
                    matrix_w = w_col1
                    matrix_x = col1
                elif j == 2:
                    matrix_w = w_col2
                    x_ptrs_1d = x_base_1d + idx_token * stride_x_token  # [BLOCK_N]
                    matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)
            elif KERNEL_WIDTH == 4:
                if j == 1:
                    matrix_w = w_col1
                    matrix_x = col1
                elif j == 2:
                    matrix_w = w_col2
                    matrix_x = col2
                elif j == 3:
                    matrix_w = w_col3
                    x_ptrs_1d = x_base_1d + idx_token * stride_x_token  # [BLOCK_N]
                    matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)

            acc += matrix_x * matrix_w  # [BLOCK_N]

        if KERNEL_WIDTH == 2:
            col0 = matrix_x
        elif KERNEL_WIDTH == 3:
            col0 = col1
            col1 = matrix_x
        elif KERNEL_WIDTH == 4:
            col0 = col1
            col1 = col2
            col2 = matrix_x

        if SILU_ACTIVATION:
            acc = acc / (1 + tl.exp(-acc))
        # mask_1d = (idx_token < seqlen) & (
        #     idx_feats < dim
        # )  # token-index  # feature-index
        maskL = idx_feats < dim
        maskR = tl.full(maskL.shape, False, tl.int1)
        mask_1d = tl.where(idx_token < seqlen, maskL, maskR)

        o_ptrs = (
            o_ptr
            + (idx_seq) * stride_o_seq
            + idx_token * stride_o_token
            + (idx_feats * stride_o_dim)
        )

        tl.store(o_ptrs, acc, mask=mask_1d)

        if SAVE_INTERMEDIATE:
            # Save the window state after consuming this token
            # Layout: [seq(cache line), step, dim, win(K-1)]
            base_ptr = (
                intermediate_conv_window_ptr
                + conv_state_batch_coord * stride_inter_seq
                + idx_token * stride_inter_step
                + idx_feats * stride_inter_dim
            )
            if KERNEL_WIDTH >= 2:
                tl.store(base_ptr + 0 * stride_inter_win, col0, mask=mask_w)
            if KERNEL_WIDTH >= 3:
                tl.store(base_ptr + 1 * stride_inter_win, col1, mask=mask_w)
            if KERNEL_WIDTH >= 4:
                tl.store(base_ptr + 2 * stride_inter_win, col2, mask=mask_w)


def causal_conv1d_update_device(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    activation: Union[bool, str, None] = None,
    cache_seqlens: Optional[torch.Tensor] = None,
    conv_state_indices: Optional[torch.Tensor] = None,
    num_accepted_tokens: Optional[torch.Tensor] = None,
    intermediate_conv_window: Optional[torch.Tensor] = None,
    pad_slot_id: int = PAD_SLOT_ID,
    metadata=None,
    validate_data=False,
):
    """
    x: (batch, dim) or (batch, dim, seqlen)
        [shape=2: single token prediction]
        [shape=3: single or multiple tokens prediction]
    conv_state: (..., dim, state_len), where state_len >= width - 1
    weight: (dim, width)
    bias: (dim,)
    cache_seqlens: (batch,), dtype int32.
        If not None, the conv_state is treated as a circular buffer.
        The conv_state will be updated by copying x to the conv_state
        starting at the index
        @cache_seqlens % state_len.
    conv_state_indices: (batch,), dtype int32
        If not None, the conv_state is a larger tensor along the batch dim,
        and we are selecting the batch coords specified by conv_state_indices.
        Useful for a continuous batching scenario.
    pad_slot_id: int
            if cache_indices is passed, lets the kernel identify padded
            entries that will not be processed,
            for example: cache_indices = [pad_slot_id, 1 ,20 ,pad_slot_id]
            in this case, the kernel will not process entries at
            indices 0 and 3
    out: (batch, dim) or (batch, dim, seqlen)
    """
    if validate_data:
        assert cache_seqlens is None  # not implemented yet - ok for vLLM
        assert pad_slot_id is not None
        assert x.stride(1) == 1
    if isinstance(activation, bool):
        activation = "silu" if activation is True else None
    elif activation is not None:
        assert activation in ["silu", "swish"]
    unsqueeze = x.dim() == 2
    if unsqueeze:
        # make it (batch, dim, seqlen) with seqlen == 1
        x = x.unsqueeze(-1)
    batch, dim, seqlen = x.shape
    _, width = weight.shape
    # conv_state: (..., dim, state_len), where state_len >= width - 1
    num_cache_lines, _, state_len = conv_state.size()

    if validate_data:
        assert dim == weight.size(0)
        assert (
            conv_state.stride(-2) == 1
        ), f"ERROR: expect contiguous along feat-dim of conv_state (currently stride={conv_state.stride()})"
        assert state_len >= width - 1
        # when above happens, we don't shift-left to keep any records in conv_state
        assert dim == conv_state.size(1)
        if conv_state_indices is None:
            assert conv_state.size(0) >= batch
        else:
            assert (batch,) == conv_state_indices.shape

        assert num_cache_lines >= batch
        assert weight.stride(1) == 1  # Need this
        assert cache_seqlens is None  # not needed for vLLM - circular buffer

    # adopt the strategy in vLLM that overwrite on 'x' directly, rather than creating a new tensor 'o'
    out = x
    stride_w_dim, stride_w_width = weight.stride()

    stride_x_seq, stride_x_dim, stride_x_token = x.stride()  # X (batch, dim, seqlen)

    stride_o_seq, stride_o_dim, stride_o_token = out.stride()
    stride_istate_seq, stride_istate_dim, stride_istate_token = conv_state.stride()
    stride_state_indices = (
        conv_state_indices.stride(0) if conv_state_indices is not None else 0
    )
    state_len = width - 1 + (seqlen - 1)  # effective state_len needed
    np2_statelen = triton.next_power_of_2(state_len)

    def grid(META):
        return (
            batch,
            triton.cdiv(dim, META["BLOCK_N"]),
        )

    # prepare intermediate buffer strides if provided
    if intermediate_conv_window is not None:
        stride_inter_seq, stride_inter_step, stride_inter_dim, stride_inter_win = (
            intermediate_conv_window.stride(0),
            intermediate_conv_window.stride(1),
            intermediate_conv_window.stride(2),
            intermediate_conv_window.stride(3),
        )
    else:
        stride_inter_seq = stride_inter_step = stride_inter_dim = stride_inter_win = 0
    
    if is_npu() and cache_seqlens is None and num_accepted_tokens is None:
        DIM_BLOCK = 2048
        print(f"device = npu")
        _causal_conv1d_update_kernel_no_cache_len_no_mtp[(batch, 1, 1)](
            x, conv_state, weight, bias, conv_state_indices,
            out,
            batch = batch,
            dim = dim,
            align_val = 16,
            state_len = conv_state.shape[-1], # 3 4 5 
            seq_len = x.shape[-1], # 1 2 
            width = width, # 4, <= seq_len + state_len
            out_len = out.shape[-1],
            x_batch_stride = x.stride()[0],
            conv_batch_stride = conv_state.stride()[0],
            out_batch_stride = out.stride()[0],
            DIM_BLOCK = DIM_BLOCK, # dim % DIM_BLOCK must be 0
            HAS_BIAS = bias is not None,
            SILU_ACTIVATION = activation in ["silu", "swish"],
            IS_CONTINUOUS_BATCHING=conv_state_indices is not None,
        )
    else:
        BLOCK_N = 128
        _causal_conv1d_update_kernel_gpu[grid]( # FIXME
            # Pointers to matrices
            x,
            weight,
            bias,
            conv_state,
            cache_seqlens,
            conv_state_indices,
            num_accepted_tokens,
            intermediate_conv_window if intermediate_conv_window is not None else x,
            out,
            # Matrix dimensions
            batch,
            dim,
            seqlen,
            state_len,
            num_cache_lines,
            # stride
            stride_x_seq,
            stride_x_dim,
            stride_x_token,
            stride_w_dim,
            stride_w_width,
            stride_istate_seq,
            stride_istate_dim,
            stride_istate_token,
            stride_state_indices,
            stride_inter_seq,
            stride_inter_step,
            stride_inter_dim,
            stride_inter_win,
            stride_o_seq,
            stride_o_dim,
            stride_o_token,
            # others
            pad_slot_id,
            # META
            HAS_BIAS=bias is not None,
            KERNEL_WIDTH=width,
            SILU_ACTIVATION=activation in ["silu", "swish"],
            IS_CONTINUOUS_BATCHING=conv_state_indices is not None,
            IS_SPEC_DECODING=num_accepted_tokens is not None,
            NP2_STATELEN=np2_statelen,
            USE_PAD_SLOT=pad_slot_id is not None,
            BLOCK_N=BLOCK_N,
            SAVE_INTERMEDIATE=intermediate_conv_window is not None,
        )

    if unsqueeze:
        out = out.squeeze(-1)
    return out,conv_state



def test_fn():
    torch.manual_seed(23)

    device = "npu" if is_npu() else "cuda"

    mixed_qkv_non_spec = torch.randn((32, 2048), dtype=torch.bfloat16, device=device)
    conv_state = torch.randn((33, 2048, 3), dtype=torch.bfloat16, device=device)
    conv_weights = torch.randn((2048, 4), dtype=torch.bfloat16, device=device)
    conv1d_bias = None
    non_spec_state_indices_tensor = torch.tensor([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
        19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,  0],
       device=device, dtype=torch.int32)

    mixed_qkv_non_spec_bak = mixed_qkv_non_spec.clone()

    mixed_qkv_non_spec,stdconv = causal_conv1d_update_(
                mixed_qkv_non_spec,
                conv_state.clone(),
                conv_weights,
                conv1d_bias,
                "silu",
                conv_state_indices=non_spec_state_indices_tensor,
            )

    mixed_qkv_non_spec_bak,retconv = causal_conv1d_update_device(
                mixed_qkv_non_spec_bak,
                conv_state,
                conv_weights,
                conv1d_bias,
                "silu",
                conv_state_indices=non_spec_state_indices_tensor,
            )

    torch.testing.assert_close(stdconv, retconv, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(mixed_qkv_non_spec, mixed_qkv_non_spec_bak, atol=1e-2, rtol=1e-2)

if __name__ == "__main__":
    test_fn()
