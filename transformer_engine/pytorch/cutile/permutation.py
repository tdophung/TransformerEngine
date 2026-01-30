# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""PyTorch wrapper functions for Permutation cuTile kernels."""

import math
from typing import Optional, Tuple

import torch
import cuda.tile as ct

from transformer_engine.common.cutile import permutation as cutile_permutation

# =============================================================================
# Helper Functions
# =============================================================================


def next_power_of_2(n: int) -> int:
    """Return the smallest power of 2 greater than or equal to n."""
    if n <= 0:
        return 1
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    n |= n >> 32
    n += 1
    return n


def ceil_div(a: int, b: int) -> int:
    """Ceiling division."""
    return (a + b - 1) // b


# =============================================================================
# cuTile Kernel Definitions
# =============================================================================

ConstInt = ct.Constant[int]
ConstBool = ct.Constant[bool]


@ct.kernel
def _row_id_map_pass_1_kernel(
    routing_map,
    row_id_map,
    workspace,
    num_tokens: ConstInt,
    BLOCK_SIZE: ConstInt,
):
    """
    Pass 1: Compute cumulative sum of expert assignments within each block.
    """
    pid_m = ct.bid(0)  # Expert index
    pid_n = ct.bid(1)  # Token block index

    # Calculate token offsets for this block
    offset = pid_n * BLOCK_SIZE + ct.arange(BLOCK_SIZE, dtype=ct.int32)
    mask = offset < num_tokens

    # Load routing map: routing_map[pid_m, offset]
    # routing_map is [num_tokens, num_experts] with strides [num_experts, 1]
    # So routing_map[token, expert] = routing_map_flat[token * num_experts + expert]
    routing_indices = offset * routing_map.shape[1] + pid_m
    expert_token_mask = ct.gather(routing_map, routing_indices, padding_value=0)
    expert_token_mask = ct.astype(expert_token_mask, ct.int32)

    # Compute cumulative sum within block
    row_id_within_token_block = ct.cumsum(expert_token_mask, axis=0) * expert_token_mask

    # Store row IDs: row_id_map[offset, pid_m]
    # row_id_map is [num_tokens, num_experts * 2 + 1]
    row_id_stride = row_id_map.shape[1]
    row_id_indices = offset * row_id_stride + pid_m
    ct.scatter(row_id_map, row_id_indices, row_id_within_token_block)

    # Store the count of tokens in this block for this expert
    n_tokens_per_block = ct.sum(expert_token_mask)
    num_blocks = ct.cdiv(num_tokens, BLOCK_SIZE)
    workspace_idx = pid_m * num_blocks + pid_n
    workspace_idx_tile = ct.full((1,), workspace_idx, dtype=ct.int32)
    ct.scatter(workspace, workspace_idx_tile, n_tokens_per_block.reshape((1,)))


@ct.kernel
def _row_id_map_pass_2_kernel(
    row_id_map,
    workspace,
    num_tokens: ConstInt,
    num_experts: ConstInt,
    WORKSPACE_LOAD_WIDTH: ConstInt,
    BLOCK_SIZE: ConstInt,
):
    """
    Pass 2: Add prefix sums from previous blocks to complete row ID mapping.
    """
    pid_m = ct.bid(0)  # Expert index
    pid_n = ct.bid(1)  # Token block index

    num_blocks = ct.cdiv(num_tokens, BLOCK_SIZE)
    chunk_idx = pid_m * num_blocks + pid_n

    # Load current block's row IDs
    offset = pid_n * BLOCK_SIZE + ct.arange(BLOCK_SIZE, dtype=ct.int32)
    row_id_stride = num_experts * 2 + 1
    row_id_indices = offset * row_id_stride + pid_m
    row_id_within_token_block = ct.gather(row_id_map, row_id_indices, padding_value=0)

    # Load prefix sums from previous chunks
    workspace_off = ct.arange(WORKSPACE_LOAD_WIDTH, dtype=ct.int32)
    n_tokens_per_chunk = ct.gather(workspace, workspace_off, padding_value=0)

    # Mask out chunks >= current chunk
    chunk_mask = ct.where(workspace_off < chunk_idx, 1, 0)
    n_tokens_per_chunk = n_tokens_per_chunk * chunk_mask

    # Compute prefix sum
    prefix_sum = ct.sum(n_tokens_per_chunk)

    # Update row IDs: -1 for unrouted, else add prefix sum
    row_id = ct.where(
        row_id_within_token_block == 0,
        ct.full((BLOCK_SIZE,), -1, dtype=ct.int32),
        row_id_within_token_block + prefix_sum - 1,
    )

    # Store updated row IDs
    ct.scatter(row_id_map, row_id_indices, row_id)


@ct.kernel
def _permute_kernel_simple(
    input_tensor,
    row_id_map,
    output,
    num_experts: ConstInt,
    hidden_size: ConstInt,
    BLOCK_SIZE: ConstInt,
):
    """
    Simple permute kernel without probabilities or scaling.
    """
    pid_t = ct.bid(0)  # Token index
    pid_h = ct.bid(1)  # Hidden dimension block index

    # Current hidden dimension offsets
    cur_off = pid_h * BLOCK_SIZE + ct.arange(BLOCK_SIZE, dtype=ct.int32)

    # Load input data for this token
    input_off = pid_t * hidden_size + cur_off
    inp = ct.gather(input_tensor, input_off, padding_value=0.0)

    # Load n_routed for this token
    row_id_stride = num_experts * 2 + 1
    n_routed_idx = ct.full((1,), pid_t * row_id_stride + num_experts * 2, dtype=ct.int32)
    n_routed = ct.gather(row_id_map, n_routed_idx, padding_value=0)

    # Process each routed expert destination
    for idx in range(num_experts):
        # Check if this expert slot is used
        cond = ct.full((1,), idx, dtype=ct.int32) < n_routed
        if cond[0]:
            # Load destination row
            dst_row_idx = ct.full((1,), pid_t * row_id_stride + idx, dtype=ct.int32)
            dst_row = ct.gather(row_id_map, dst_row_idx, padding_value=-1)

            if dst_row[0] >= 0:
                # Store output
                output_off = dst_row[0] * hidden_size + cur_off
                ct.scatter(output, output_off, inp)


@ct.kernel
def _unpermute_kernel_simple(
    input_tensor,
    row_id_map,
    merging_probs,
    output,
    num_experts: ConstInt,
    hidden_size: ConstInt,
    WITH_MERGING_PROBS: ConstBool,
    BLOCK_SIZE: ConstInt,
):
    """
    Simple unpermute kernel with optional merging probabilities.
    """
    pid_t = ct.bid(0)  # Token index
    pid_h = ct.bid(1)  # Hidden dimension block index

    current_offset = pid_h * BLOCK_SIZE + ct.arange(BLOCK_SIZE, dtype=ct.int32)

    # Initialize accumulator
    accumulator = ct.zeros((BLOCK_SIZE,), dtype=ct.float32)

    # Load n_routed for this token
    row_id_stride = num_experts * 2 + 1
    n_routed_idx = ct.full((1,), pid_t * row_id_stride + num_experts * 2, dtype=ct.int32)
    n_routed = ct.gather(row_id_map, n_routed_idx, padding_value=0)

    # Process each routed source
    for idx in range(num_experts):
        cond = ct.full((1,), idx, dtype=ct.int32) < n_routed
        if cond[0]:
            # Load source row
            src_row_idx = ct.full((1,), pid_t * row_id_stride + idx, dtype=ct.int32)
            src_row = ct.gather(row_id_map, src_row_idx, padding_value=-1)

            if src_row[0] >= 0:
                # Load expert index
                expert_idx_offset = ct.full(
                    (1,), pid_t * row_id_stride + num_experts + idx, dtype=ct.int32
                )
                expert_idx = ct.gather(row_id_map, expert_idx_offset, padding_value=0)

                # Load input
                input_off = src_row[0] * hidden_size + current_offset
                inp = ct.gather(input_tensor, input_off, padding_value=0.0)
                inp = ct.astype(inp, ct.float32)

                # Apply merging probability if needed
                if WITH_MERGING_PROBS:
                    merging_prob_off = ct.full(
                        (1,), pid_t * num_experts + expert_idx[0], dtype=ct.int32
                    )
                    merging_prob = ct.gather(merging_probs, merging_prob_off, padding_value=0.0)
                    inp = inp * merging_prob[0]

                accumulator = accumulator + inp

    # Store accumulated result
    output_off = pid_t * hidden_size + current_offset
    accumulator_out = ct.astype(accumulator, output.dtype)
    ct.scatter(output, output_off, accumulator_out)


@ct.kernel
def _sort_chunks_by_map_kernel(
    input_tensor,
    row_id_map,
    output,
    hidden_size: ConstInt,
    BLOCK_SIZE: ConstInt,
    FORWARD: ConstBool,
):
    """
    Sort chunks according to a precomputed mapping.
    """
    pid_t = ct.bid(0)  # Token index
    pid_h = ct.bid(1)  # Hidden dimension block index

    # Determine source and destination rows based on direction
    pid_t_idx = ct.full((1,), pid_t, dtype=ct.int32)
    map_val = ct.gather(row_id_map, pid_t_idx, padding_value=0)

    if FORWARD:
        src_row = pid_t
        dst_row = map_val[0]
    else:
        src_row = map_val[0]
        dst_row = pid_t

    # Copy hidden dimension block
    current_offset = pid_h * BLOCK_SIZE + ct.arange(BLOCK_SIZE, dtype=ct.int32)

    input_offsets = src_row * hidden_size + current_offset
    output_offsets = dst_row * hidden_size + current_offset

    inp = ct.gather(input_tensor, input_offsets, padding_value=0.0)
    ct.scatter(output, output_offsets, inp)


@ct.kernel
def _make_chunk_sort_map_kernel(
    split_sizes,
    sorted_indices,
    dst_rows,
    num_splits: ConstInt,
    IDX_LOAD_WIDTH: ConstInt,
):
    """
    Create a mapping for sorting chunks based on split sizes.
    """
    pid = ct.bid(0)  # Token index

    load_split_offset = ct.arange(IDX_LOAD_WIDTH, dtype=ct.int32)

    # Load sorted indices
    sorted_idx = ct.gather(sorted_indices, load_split_offset, padding_value=0)

    # Load input split sizes
    input_split_sizes = ct.gather(split_sizes, load_split_offset, padding_value=0)
    input_split_sizes = ct.astype(input_split_sizes, ct.int32)

    # Compute cumsum to find chunk boundaries
    input_split_sizes_cumsum = ct.cumsum(input_split_sizes, axis=0)

    # Find which input chunk this token belongs to
    input_split_sizes_mask = ct.where(input_split_sizes_cumsum <= pid, 1, 0)
    input_chunk_idx = ct.sum(input_split_sizes_mask)

    # Compute offset within input chunk
    input_split_sizes_presum = ct.sum(input_split_sizes * input_split_sizes_mask)
    in_chunk_offset = pid - input_split_sizes_presum

    # Find output chunk index
    output_chunk_mask = ct.where(sorted_idx == input_chunk_idx, 1, 0)
    output_chunk_idx = ct.argmax(output_chunk_mask, axis=0)

    # Compute output split sizes
    output_split_sizes = ct.gather(split_sizes, sorted_idx, padding_value=0)
    output_split_sizes = ct.astype(output_split_sizes, ct.int32)

    # Compute prefix sum for output chunks before this one
    output_pre_mask = ct.where(load_split_offset < output_chunk_idx, 1, 0)
    output_pre_split_sizes = output_split_sizes * output_pre_mask
    dst_row = ct.sum(output_pre_split_sizes) + in_chunk_offset

    # Store destination row
    dst_row_idx = ct.full((1,), pid, dtype=ct.int32)
    dst_row_tile = ct.full((1,), dst_row, dtype=ct.int32)
    ct.scatter(dst_rows, dst_row_idx, dst_row_tile)


# =============================================================================
# Wrapper Functions
# =============================================================================


def make_row_id_map(
    routing_map: torch.Tensor,
    num_tokens: int,
    num_experts: int,
) -> torch.Tensor:
    """
    Prepare the row_id_map for the permutation.

    Parameters
    ----------
    routing_map : torch.Tensor
        Input tensor of shape `[num_tokens, num_experts]`. It is a mask tensor that indicates
        which experts are routed to which tokens. The values in it: 1 means the token is routed to
        this expert and 0 means not.
    num_tokens : int
        Number of tokens in the input tensor.
    num_experts : int
        Number of experts in the input tensor.

    Returns
    -------
    row_id_map : torch.Tensor
        The row_id_map for the permutation of shape `[num_tokens, num_experts * 2 + 1]`.
        For each token, the last item is the number of experts that are routed (n_routed).
        The first n_routed items are the destination row indices in the permuted tokens.
        The [num_experts, num_experts + n_routed) items are the indices of the experts corresponding
        to the first n_routed row indices above.
    """
    routing_map_expert_major = routing_map.bool().T.contiguous()
    return cutile_permutation.compute_row_id_map(
        routing_map_expert_major, num_tokens, num_experts, block_size=1024
    ).view(num_tokens, num_experts * 2 + 1)


def permute_with_mask_map(
    inp: torch.Tensor,
    row_id_map: torch.Tensor,
    probs: Optional[torch.Tensor],
    scale: Optional[torch.Tensor],
    pad_offsets: Optional[torch.Tensor],
    num_tokens: int,
    num_experts: int,
    num_out_tokens: int,
    hidden_size: int,
    scale_hidden_dim: Optional[int],
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Permute the input tensor based on the row_id_map.

    Parameters
    ----------
    inp : torch.Tensor
        Input tensor of shape `[num_tokens, hidden_size]`, on which permutation will be applied.
    row_id_map : torch.Tensor
        The token to expert mapping tensor of shape `[num_tokens, num_experts * 2 + 1]`.
    probs : torch.Tensor
        The probabilities of the input tensor. If it is not None, it will be permuted.
    scale : torch.Tensor
        The scale of the input tensor. If it is not None, it will be permuted.
    pad_offsets : torch.Tensor
        Per-expert padding offsets of shape `[num_experts]` for FP8 fused padding.
    num_tokens : int
        Number of tokens in the input tensor.
    num_experts : int
        Number of experts in the input tensor.
    num_out_tokens : int
        Number of tokens in the permuted tensor.
    hidden_size : int
        Hidden size of the input tensor.
    scale_hidden_dim : int
        Hidden size of the scale tensor.
    """
    output, permuted_probs, permuted_scale = cutile_permutation.permute(
        input_tensor=inp,
        row_id_map=row_id_map.reshape(-1),
        probs=probs,
        scale=scale,
        pad_offsets=pad_offsets,
        num_out_tokens=num_out_tokens,
        num_experts=num_experts,
        permute_probs=probs is not None,
        permute_scale=scale is not None,
        fusion_pad=pad_offsets is not None,
    )
    return output, permuted_scale, permuted_probs


def unpermute_with_mask_map(
    inp: torch.Tensor,
    row_id_map: torch.Tensor,
    merging_probs: Optional[torch.Tensor],
    permuted_probs: Optional[torch.Tensor],
    pad_offsets: Optional[torch.Tensor],
    num_tokens: int,
    num_experts: int,
    hidden_size: int,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Unpermute the input tensor based on the row_id_map.

    Parameters
    ----------
    inp : torch.Tensor
        Input tensor of shape `[num_out_tokens, hidden_size]`.
    row_id_map : torch.Tensor
        The token to expert mapping tensor of shape `[num_tokens, num_experts * 2 + 1]`.
    merging_probs : torch.Tensor
        The merging probabilities of the input tensor.
    permuted_probs : torch.Tensor
        The permuted probabilities of the input tensor.
    pad_offsets : torch.Tensor
        Per-expert padding offsets of shape `[num_experts]` for FP8 fused unpadding.
    num_tokens : int
        Number of tokens in the permuted tensor.
    num_experts : int
        Number of experts in the permuted tensor.
    hidden_size : int
        Hidden size of the permuted tensor.
    """
    output, unpermuted_probs = cutile_permutation.unpermute(
        input_tensor=inp,
        row_id_map=row_id_map.reshape(-1),
        merging_probs=merging_probs,
        permuted_probs=permuted_probs,
        pad_offsets=pad_offsets,
        num_tokens=num_tokens,
        num_experts=num_experts,
        with_merging_probs=merging_probs is not None,
        permute_probs=permuted_probs is not None,
        fusion_unpad=pad_offsets is not None,
    )
    return output, unpermuted_probs


def unpermute_with_mask_map_bwd_with_merging_probs(
    fwd_output_grad: torch.Tensor,
    row_id_map: torch.Tensor,
    fwd_input: torch.Tensor,
    merging_probs: torch.Tensor,
    pad_offsets: Optional[torch.Tensor],
    num_tokens: int,
    num_experts: int,
    num_out_tokens: int,
    hidden_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Unpermute backward pass kernel with merging probs.

    Parameters
    ----------
    fwd_output_grad : torch.Tensor
        The gradient of the output tensor of shape `[num_tokens, hidden_size]`.
    row_id_map : torch.Tensor
        The token to expert mapping tensor of shape `[num_tokens, num_experts * 2 + 1]`.
    fwd_input : torch.Tensor
        The input tensor of the forward pass of shape `[num_out_tokens, hidden_size]`.
    merging_probs : torch.Tensor
        The merging probabilities of the input tensor of shape `[num_tokens, num_experts]`.
    pad_offsets : torch.Tensor
        Per-expert padding offsets of shape `[num_experts]` for FP8 fused padding.
    num_tokens : int
        Number of tokens in the permuted tensor.
    num_experts : int
        Number of experts in the permuted tensor.
    num_out_tokens : int
        Number of tokens in the output tensor.
    hidden_size : int
        Hidden size of the output tensor.
    """
    return cutile_permutation.unpermute_with_mask_map_bwd_with_merging_probs(
        fwd_output_grad=fwd_output_grad,
        row_id_map=row_id_map.reshape(-1),
        fwd_input=fwd_input,
        merging_probs=merging_probs,
        pad_offsets=pad_offsets,
        num_tokens=num_tokens,
        num_experts=num_experts,
        num_out_tokens=num_out_tokens,
        hidden_size=hidden_size,
    )


def make_chunk_sort_map(
    split_sizes: torch.Tensor,
    sorted_indices: torch.Tensor,
    num_tokens: int,
    num_splits: int,
) -> torch.Tensor:
    """
    Make a row_id_map for chunk sort.

    Parameters
    ----------
    split_sizes : torch.Tensor
        The sizes of the chunks of shape `[num_splits,]`.
    sorted_indices : torch.Tensor
        The indices of the sorted chunks of shape `[num_splits,]`.
    num_tokens : int
        Number of tokens in the input tensor.
    num_splits : int
        Number of splits of split_sizes and sorted_indices.
    """
    device = split_sizes.device
    stream = torch.cuda.current_stream()

    row_id_map = torch.empty((num_tokens,), dtype=torch.int32, device=device)
    idx_load_width = next_power_of_2(num_splits)

    ct.launch(
        stream,
        (num_tokens,),
        _make_chunk_sort_map_kernel,
        (
            split_sizes.to(torch.int32),
            sorted_indices.to(torch.int32),
            row_id_map,
            num_splits,
            idx_load_width,
        ),
    )

    return row_id_map


def sort_chunks_by_map(
    inp: torch.Tensor,
    row_id_map: torch.Tensor,
    probs: Optional[torch.Tensor],
    num_tokens: int,
    hidden_size: int,
    is_forward: bool,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Sort chunks with row_id_map.

    Parameters
    ----------
    inp : torch.Tensor
        Input tensor of shape `[num_tokens, hidden_size]`.
    row_id_map : torch.Tensor
        The token to expert mapping tensor of shape `[num_tokens,]`.
    probs : torch.Tensor
        The probabilities of the input tensor. If it is not None, it will be permuted.
    num_tokens : int
        Number of tokens in the input tensor.
    hidden_size : int
        Hidden size of the input tensor.
    is_forward : bool
        Whether the sort is for forward or backward.
    """
    device = inp.device
    stream = torch.cuda.current_stream()

    output = torch.empty((num_tokens, hidden_size), dtype=inp.dtype, device=device)

    if probs is not None:
        permuted_probs = torch.empty((num_tokens,), dtype=probs.dtype, device=device)
    else:
        permuted_probs = None

    block_size = 256
    num_hidden_blocks = ceil_div(hidden_size, block_size)
    grid = (num_tokens, num_hidden_blocks)

    ct.launch(
        stream,
        grid,
        _sort_chunks_by_map_kernel,
        (
            inp.reshape(-1),
            row_id_map,
            output.reshape(-1),
            hidden_size,
            block_size,
            is_forward,
        ),
    )

    if probs is not None and permuted_probs is not None:
        for t in range(num_tokens):
            if is_forward:
                src_row = t
                dst_row = row_id_map[t].item()
            else:
                src_row = row_id_map[t].item()
                dst_row = t
            permuted_probs[dst_row] = probs[src_row]

    return output, permuted_probs
