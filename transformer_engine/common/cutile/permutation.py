# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Efficient Permutation kernels written with NVIDIA cuTile."""

import math
import cuda.tile as ct
import torch

ConstInt = ct.Constant[int]
ConstBool = ct.Constant[bool]


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
# Bitonic Sort / Argsort Kernels
# =============================================================================


@ct.function
def _compare_and_swap_cutile(
    x: ct.Tile,
    indices: ct.Tile,
    flip: ct.Tile,
    i: ConstInt,
    n_dims: ConstInt,
):
    """
    Compare and swap operation for bitonic sort in cuTile.

    This mirrors the Triton implementation logic in
    `transformer_engine/common/triton/permutation.py`.
    """
    n_outer = x.numel >> n_dims
    shape_0 = n_outer * (2 ** i)
    shape_2 = 2 ** (n_dims - i - 1)

    y = ct.reshape(x, (shape_0, 2, shape_2))
    z = ct.reshape(indices, (shape_0, 2, shape_2))

    mask = ct.arange(2, dtype=ct.int32)
    mask = ct.reshape(mask, (1, 2, 1))
    mask = ct.broadcast_to(mask, (shape_0, 2, shape_2))
    inv_mask = 1 - mask

    y_left = ct.sum(y * ct.astype(inv_mask, y.dtype), axis=1)
    y_right = ct.sum(y * ct.astype(mask, y.dtype), axis=1)
    z_left = ct.sum(z * inv_mask, axis=1)
    z_right = ct.sum(z * mask, axis=1)

    y_left = ct.expand_dims(y_left, axis=1)
    y_right = ct.expand_dims(y_right, axis=1)
    z_left = ct.expand_dims(z_left, axis=1)
    z_right = ct.expand_dims(z_right, axis=1)

    l_value = ct.broadcast_to(y_left, (shape_0, 2, shape_2))
    r_value = ct.broadcast_to(y_right, (shape_0, 2, shape_2))
    l_indice = ct.broadcast_to(z_left, (shape_0, 2, shape_2))
    r_indice = ct.broadcast_to(z_right, (shape_0, 2, shape_2))

    l_value = ct.reshape(l_value, x.shape)
    r_value = ct.reshape(r_value, x.shape)
    l_indice = ct.reshape(l_indice, x.shape)
    r_indice = ct.reshape(r_indice, x.shape)

    swap_cond = (l_value > r_value) ^ (flip != 0)
    mask_lr = ct.reshape(mask, x.shape)

    left_val = ct.where(swap_cond, r_value, l_value)
    right_val = ct.where(swap_cond, l_value, r_value)
    left_idx = ct.where(swap_cond, r_indice, l_indice)
    right_idx = ct.where(swap_cond, l_indice, r_indice)

    ret = ct.where(mask_lr == 0, left_val, right_val)
    ind = ct.where(mask_lr == 0, left_idx, right_idx)

    return ret, ind


@ct.function
def _bitonic_merge_cutile(
    x: ct.Tile,
    indices: ct.Tile,
    stage: ConstInt,
    order: ConstInt,
    n_dims: ConstInt,
):
    """
    Bitonic merge operation for bitonic sort in cuTile.
    """
    n_outer = x.numel >> n_dims
    if order == 2:
        shape_0 = n_outer * (2 ** (n_dims - 1 - stage))
        shape_2 = 2 ** stage
        flip_base = ct.arange(2, dtype=ct.int32)
        flip_base = ct.reshape(flip_base, (1, 2, 1))
        flip = ct.broadcast_to(flip_base, (shape_0, 2, shape_2))
        flip = ct.reshape(flip, x.shape)
    else:
        flip = ct.full(x.shape, order, dtype=ct.int32)

    for i in range(stage):
        x, indices = _compare_and_swap_cutile(x, indices, flip, i + (n_dims - stage), n_dims)

    return x, indices


@ct.function
def _argsort_cutile(x: ct.Tile, indices: ct.Tile, n_dims: ConstInt):
    """
    Bitonic argsort implementation in cuTile.
    """
    for i in range(1, n_dims + 1):
        order = 2 if i < n_dims else 1
        x, indices = _bitonic_merge_cutile(x, indices, i, order, n_dims)
    return x, indices


@ct.kernel
def _row_id_map_pass_1_kernel(
    routing_map,
    row_id_map,
    workspace,
    num_tokens: ConstInt,
    num_experts: ConstInt,
    BLOCK_SIZE: ConstInt,
):
    """
    Pass 1: Compute cumulative sum of expert assignments within each block.

    For each expert (pid_m), iterates over token blocks (pid_n) and computes
    the local row ID for tokens assigned to that expert within the block.
    """
    pid_m = ct.bid(0)  # Expert index
    pid_n = ct.bid(1)  # Token block index

    # Calculate token offsets for this block
    offset = pid_n * BLOCK_SIZE + ct.arange(BLOCK_SIZE, dtype=ct.int32)

    # Load routing map for this expert and token block
    # routing_map shape: [num_experts, num_tokens]
    routing_indices = pid_m * num_tokens + offset
    expert_token_mask = ct.gather(routing_map, routing_indices, padding_value=0)
    expert_token_mask = ct.astype(expert_token_mask, ct.int32)

    # Compute cumulative sum within block - tokens assigned to this expert
    row_id_within_token_block = ct.cumsum(expert_token_mask, axis=0) * expert_token_mask

    # Store row IDs back
    row_id_indices = pid_m * num_tokens + offset
    ct.scatter(row_id_map, row_id_indices, row_id_within_token_block)

    # Store the count of tokens in this block for this expert
    n_tokens_per_block = ct.sum(expert_token_mask)
    workspace_idx = pid_m * ct.cdiv(num_tokens, BLOCK_SIZE) + pid_n
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

    Takes the local row IDs from pass 1 and adds the total count from all
    previous blocks to get the final global row ID.
    """
    pid_m = ct.bid(0)  # Expert index
    pid_n = ct.bid(1)  # Token block index

    num_blocks = ct.cdiv(num_tokens, BLOCK_SIZE)
    chunk_idx = pid_m * num_blocks + pid_n

    # Load current block's row IDs
    offset = pid_n * BLOCK_SIZE + ct.arange(BLOCK_SIZE, dtype=ct.int32)
    row_id_indices = pid_m * num_tokens + offset
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
def _row_id_map_pass_3_kernel(
    row_id_map_temp,
    row_id_map,
    num_tokens: ConstInt,
    num_experts: ConstInt,
    LOAD_SIZE: ConstInt,
    N_DIMS: ConstInt,
):
    """
    Pass 3: Sort row IDs by expert for each token.

    For each token, loads all expert row IDs, sorts them, and stores back
    the sorted row IDs along with the corresponding expert indices.
    """
    pid = ct.bid(0)  # Token index

    off = ct.arange(LOAD_SIZE, dtype=ct.int32)
    row_id_map_load_indices = pid + off * num_tokens
    row_id_map_tile = ct.gather(row_id_map_temp, row_id_map_load_indices, padding_value=-1)

    routed_mask = row_id_map_tile != -1
    n_routed = ct.sum(
        ct.where(
            routed_mask,
            ct.ones(routed_mask.shape, dtype=ct.int32),
            ct.zeros(routed_mask.shape, dtype=ct.int32),
        )
    )

    indices = off
    sorted_map, sorted_indices = _argsort_cutile(row_id_map_tile, indices, N_DIMS)

    base = pid * (num_experts * 2 + 1)
    ct.scatter(row_id_map, base + off, sorted_map)
    ct.scatter(row_id_map, base + num_experts + off, sorted_indices)
    ct.scatter(row_id_map, base + num_experts * 2, n_routed)


@ct.kernel
def _permute_kernel(
    input_tensor,
    row_id_map,
    probs,
    scale,
    permuted_scale,
    pad_offsets,
    output,
    permuted_probs,
    scale_hidden_dim: ConstInt,
    num_tokens: ConstInt,
    num_out_tokens: ConstInt,
    num_experts: ConstInt,
    hidden_size: ConstInt,
    PERMUTE_PROBS: ConstBool,
    PERMUTE_SCALE: ConstBool,
    FUSION_PAD: ConstBool,
    BLOCK_SIZE: ConstInt,
):
    """
    Permute kernel: Reorders tokens according to expert assignments.

    Each block handles one token (pid_t) and one hidden dimension chunk (pid_h).
    Reads the row ID mapping to determine destination positions and copies
    input data to the permuted output.
    """
    pid_t = ct.bid(0)  # Token index
    pid_h = ct.bid(1)  # Hidden dimension block index

    # Current hidden dimension offsets
    cur_off = pid_h * BLOCK_SIZE + ct.arange(BLOCK_SIZE, dtype=ct.int32)
    mask = cur_off < hidden_size

    # Load input data for this token
    src_row = pid_t
    input_off = src_row * hidden_size + cur_off
    inp = ct.gather(input_tensor, input_off, padding_value=0.0)

    # Load scale if needed
    if PERMUTE_SCALE:
        mask_scale = cur_off < scale_hidden_dim
        scale_off = pid_t * scale_hidden_dim + cur_off
        scale_val = ct.gather(scale, scale_off, padding_value=0.0)

    # Load n_routed for this token
    n_routed_idx = ct.full((1,), pid_t * (num_experts * 2 + 1) + num_experts * 2, dtype=ct.int32)
    n_routed = ct.gather(row_id_map, n_routed_idx, padding_value=0)
    n_routed_scalar = n_routed[0]

    # Process each routed expert destination
    # Note: cuTile doesn't support dynamic loop bounds directly
    # We use a fixed iteration count and mask
    for idx in range(num_experts):
        # Check if this expert slot is used
        if idx < n_routed_scalar:
            # Load destination row
            dst_row_idx = ct.full((1,), pid_t * (num_experts * 2 + 1) + idx, dtype=ct.int32)
            dst_row = ct.gather(row_id_map, dst_row_idx, padding_value=-1)

            if dst_row[0] >= 0:
                expert_idx_offset = ct.full(
                    (1,), pid_t * (num_experts * 2 + 1) + num_experts + idx, dtype=ct.int32
                )
                expert_idx = ct.gather(row_id_map, expert_idx_offset, padding_value=0)

                actual_dst_row = dst_row[0]

                if FUSION_PAD:
                    pad_off_idx = expert_idx
                    pad_off = ct.gather(pad_offsets, pad_off_idx, padding_value=0)
                    actual_dst_row = actual_dst_row + pad_off[0]

                # Store output
                output_off = actual_dst_row * hidden_size + cur_off
                ct.scatter(output, output_off, inp)

                if PERMUTE_SCALE:
                    permuted_scale_off = actual_dst_row * scale_hidden_dim + cur_off
                    ct.scatter(permuted_scale, permuted_scale_off, scale_val)

                if PERMUTE_PROBS:
                    if pid_h == 0:
                        # Load and store probability
                        prob_off = ct.full(
                            (1,), pid_t * num_experts + expert_idx[0], dtype=ct.int32
                        )
                        prob = ct.gather(probs, prob_off, padding_value=0.0)
                        permuted_prob_off = ct.full((1,), actual_dst_row, dtype=ct.int32)
                        ct.scatter(permuted_probs, permuted_prob_off, prob)

                        # Handle zero probability padding
                        if prob[0] == 0.0:
                            zero_tile = ct.zeros((BLOCK_SIZE,), dtype=inp.dtype)
                            ct.scatter(output, output_off, zero_tile)


@ct.kernel
def _unpermute_kernel(
    input_tensor,
    row_id_map,
    merging_probs,
    permuted_probs,
    pad_offsets,
    output,
    unpermuted_probs,
    num_tokens: ConstInt,
    num_experts: ConstInt,
    hidden_size: ConstInt,
    PROBS_LOAD_WIDTH: ConstInt,
    WITH_MERGING_PROBS: ConstBool,
    PERMUTE_PROBS: ConstBool,
    FUSION_UNPAD: ConstBool,
    BLOCK_SIZE: ConstInt,
):
    """
    Unpermute kernel: Reverses the permutation to restore original token order.

    Gathers data from permuted positions and accumulates (with optional weighting)
    back to original token positions.
    """
    pid_t = ct.bid(0)  # Token index
    pid_h = ct.bid(1)  # Hidden dimension block index

    current_offset = pid_h * BLOCK_SIZE + ct.arange(BLOCK_SIZE, dtype=ct.int32)
    mask = current_offset < hidden_size

    # Initialize probs grad to zero if needed
    if PERMUTE_PROBS:
        if pid_h == 0:
            map_load_off = ct.arange(PROBS_LOAD_WIDTH, dtype=ct.int32)
            unpermuted_prob_off = pid_t * num_experts + map_load_off
            zero_probs = ct.zeros((PROBS_LOAD_WIDTH,), dtype=ct.float32)
            ct.scatter(unpermuted_probs, unpermuted_prob_off, zero_probs)

    # Initialize accumulator
    accumulator = ct.zeros((BLOCK_SIZE,), dtype=ct.float32)

    # Load n_routed for this token
    n_routed_idx = ct.full((1,), pid_t * (num_experts * 2 + 1) + num_experts * 2, dtype=ct.int32)
    n_routed = ct.gather(row_id_map, n_routed_idx, padding_value=0)
    n_routed_scalar = n_routed[0]

    # Process each routed source
    for idx in range(num_experts):
        if idx < n_routed_scalar:
            # Load source row
            src_row_idx = ct.full((1,), pid_t * (num_experts * 2 + 1) + idx, dtype=ct.int32)
            src_row = ct.gather(row_id_map, src_row_idx, padding_value=-1)

            if src_row[0] >= 0:
                expert_idx_offset = ct.full(
                    (1,), pid_t * (num_experts * 2 + 1) + num_experts + idx, dtype=ct.int32
                )
                expert_idx = ct.gather(row_id_map, expert_idx_offset, padding_value=0)

                actual_src_row = src_row[0]

                if FUSION_UNPAD:
                    pad_off_idx = expert_idx
                    pad_off = ct.gather(pad_offsets, pad_off_idx, padding_value=0)
                    actual_src_row = actual_src_row + pad_off[0]

                # Load input
                input_off = actual_src_row * hidden_size + current_offset
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

                # Handle probs permutation
                if PERMUTE_PROBS:
                    if pid_h == 0:
                        unpermuted_prob_off = ct.full(
                            (1,), pid_t * num_experts + expert_idx[0], dtype=ct.int32
                        )
                        permuted_prob_off = ct.full((1,), actual_src_row, dtype=ct.int32)
                        prob = ct.gather(permuted_probs, permuted_prob_off, padding_value=0.0)
                        ct.scatter(unpermuted_probs, unpermuted_prob_off, prob)

    # Store accumulated result
    dst_row = pid_t
    output_off = dst_row * hidden_size + current_offset
    accumulator_out = ct.astype(accumulator, output.dtype)
    ct.scatter(output, output_off, accumulator_out)


@ct.kernel
def _unpermute_bwd_with_merging_probs_kernel(
    fwd_output_grad,
    fwd_input,
    merging_probs,
    row_id_map,
    pad_offsets,
    fwd_input_grad,
    merging_probs_grad,
    num_tokens: ConstInt,
    num_experts: ConstInt,
    hidden_size: ConstInt,
    PROBS_LOAD_WIDTH: ConstInt,
    FUSION_UNPAD: ConstBool,
    BLOCK_SIZE: ConstInt,
):
    """
    Backward pass for unpermute with merging probabilities.

    Computes gradients for both the input and the merging probabilities.
    """
    pid = ct.bid(0)  # Token index

    # Initialize probs grad to zero
    map_load_off = ct.arange(PROBS_LOAD_WIDTH, dtype=ct.int32)
    token_probs_grad_off = pid * num_experts + map_load_off
    zero_probs = ct.zeros((PROBS_LOAD_WIDTH,), dtype=ct.float32)
    ct.scatter(merging_probs_grad, token_probs_grad_off, zero_probs)

    # Load n_routed for this token
    n_routed_idx = ct.full((1,), pid * (num_experts * 2 + 1) + num_experts * 2, dtype=ct.int32)
    n_routed = ct.gather(row_id_map, n_routed_idx, padding_value=0)
    n_routed_scalar = n_routed[0]

    # Process each routed destination
    for idx in range(num_experts):
        if idx < n_routed_scalar:
            # Load destination row
            dst_row_idx = ct.full((1,), pid * (num_experts * 2 + 1) + idx, dtype=ct.int32)
            dst_row = ct.gather(row_id_map, dst_row_idx, padding_value=-1)

            if dst_row[0] >= 0:
                expert_idx_offset = ct.full(
                    (1,), pid * (num_experts * 2 + 1) + num_experts + idx, dtype=ct.int32
                )
                expert_idx = ct.gather(row_id_map, expert_idx_offset, padding_value=0)

                actual_dst_row = dst_row[0]

                if FUSION_UNPAD:
                    pad_off_idx = expert_idx
                    pad_off = ct.gather(pad_offsets, pad_off_idx, padding_value=0)
                    actual_dst_row = actual_dst_row + pad_off[0]

                # Load merging probability
                merging_prob_off = ct.full((1,), pid * num_experts + expert_idx[0], dtype=ct.int32)
                merging_prob = ct.gather(merging_probs, merging_prob_off, padding_value=0.0)

                prob_grad_accum = ct.zeros((1,), dtype=ct.float32)

                # Process hidden dimensions in blocks
                current_start = 0
                while current_start < hidden_size:
                    current_offset = current_start + ct.arange(BLOCK_SIZE, dtype=ct.int32)
                    h_mask = current_offset < hidden_size

                    # Load forward output gradient
                    src_row = pid
                    input_off = src_row * hidden_size + current_offset
                    inp = ct.gather(fwd_output_grad, input_off, padding_value=0.0)
                    inp = ct.astype(inp, ct.float32)

                    # Compute and store input gradient
                    output_grad = inp * merging_prob[0]
                    output_grad_typed = ct.astype(output_grad, fwd_input_grad.dtype)
                    output_off = actual_dst_row * hidden_size + current_offset
                    ct.scatter(fwd_input_grad, output_off, output_grad_typed)

                    # Accumulate probability gradient
                    fwd_input_off = actual_dst_row * hidden_size + current_offset
                    fwd_input_val = ct.gather(fwd_input, fwd_input_off, padding_value=0.0)
                    fwd_input_val = ct.astype(fwd_input_val, ct.float32)
                    prob_grad_accum = prob_grad_accum + ct.sum(fwd_input_val * inp)

                    current_start = current_start + BLOCK_SIZE

                # Store probability gradient
                probs_grad_off = ct.full((1,), pid * num_experts + expert_idx[0], dtype=ct.int32)
                prob_grad_typed = ct.astype(prob_grad_accum, merging_probs_grad.dtype)
                ct.scatter(merging_probs_grad, probs_grad_off, prob_grad_typed)


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

    For each token, determines which output chunk it belongs to based on
    the sorted expert order.
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

    # Find output chunk index (where input_chunk_idx appears in sorted_indices)
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


@ct.kernel
def _sort_chunks_by_map_kernel(
    input_tensor,
    row_id_map,
    probs,
    output,
    permuted_probs,
    hidden_size: ConstInt,
    PERMUTE_PROBS: ConstBool,
    BLOCK_SIZE: ConstInt,
    FORWARD: ConstBool,
):
    """
    Sort chunks according to a precomputed mapping.

    Used for reordering data between different expert orderings.
    """
    pid_t = ct.bid(0)  # Token index
    pid_h = ct.bid(1)  # Hidden dimension block index

    # Determine source and destination rows based on direction
    if FORWARD:
        src_row = pid_t
        dst_row_idx = ct.full((1,), pid_t, dtype=ct.int32)
        dst_row = ct.gather(row_id_map, dst_row_idx, padding_value=0)
        dst_row = dst_row[0]
    else:
        src_row_idx = ct.full((1,), pid_t, dtype=ct.int32)
        src_row = ct.gather(row_id_map, src_row_idx, padding_value=0)
        src_row = src_row[0]
        dst_row = pid_t

    # Copy hidden dimension block
    current_offset = pid_h * BLOCK_SIZE + ct.arange(BLOCK_SIZE, dtype=ct.int32)
    mask = current_offset < hidden_size

    input_offsets = src_row * hidden_size + current_offset
    output_offsets = dst_row * hidden_size + current_offset

    inp = ct.gather(input_tensor, input_offsets, padding_value=0.0)
    ct.scatter(output, output_offsets, inp)

    # Handle probability permutation
    if PERMUTE_PROBS:
        if pid_h == 0:
            prob_off = ct.full((1,), src_row, dtype=ct.int32)
            prob = ct.gather(probs, prob_off, padding_value=0.0)
            permuted_prob_off = ct.full((1,), dst_row, dtype=ct.int32)
            ct.scatter(permuted_probs, permuted_prob_off, prob)


# =============================================================================
# Launcher Functions
# =============================================================================


def compute_row_id_map(
    routing_map: torch.Tensor,
    num_tokens: int,
    num_experts: int,
    block_size: int = 256,
) -> torch.Tensor:
    """
    Compute row ID mapping for token permutation.

    Args:
        routing_map: Boolean tensor of shape [num_experts, num_tokens] indicating
                     which tokens are assigned to which experts.
        num_tokens: Number of tokens.
        num_experts: Number of experts.
        block_size: Block size for kernel execution.

    Returns:
        row_id_map: Tensor containing sorted row IDs and expert indices for each token.
    """
    device = routing_map.device
    stream = torch.cuda.current_stream()

    # Allocate intermediate tensors
    num_blocks = ceil_div(num_tokens, block_size)
    workspace = torch.zeros(num_experts * num_blocks, dtype=torch.int32, device=device)
    row_id_map_temp = torch.zeros(num_experts * num_tokens, dtype=torch.int32, device=device)

    # Pass 1: Local cumsum within blocks
    grid = (num_experts, num_blocks)
    ct.launch(
        stream,
        grid,
        _row_id_map_pass_1_kernel,
        (routing_map.reshape(-1), row_id_map_temp, workspace, num_tokens, num_experts, block_size),
    )

    # Pass 2: Add prefix sums from previous blocks
    workspace_load_width = next_power_of_2(num_experts * num_blocks)
    ct.launch(
        stream,
        grid,
        _row_id_map_pass_2_kernel,
        (
            row_id_map_temp,
            workspace,
            num_tokens,
            num_experts,
            workspace_load_width,
            block_size,
        ),
    )

    # Pass 3: Sort row IDs for each token using cuTile bitonic sort
    # Output layout: [num_tokens, 2*num_experts + 1]
    # For each token: [sorted_row_ids..., expert_indices..., n_routed]
    load_size = next_power_of_2(num_experts)
    n_dims = int(math.log2(load_size))
    row_id_map = torch.zeros(num_tokens * (2 * num_experts + 1), dtype=torch.int32, device=device)

    ct.launch(
        stream,
        (num_tokens,),
        _row_id_map_pass_3_kernel,
        (row_id_map_temp, row_id_map, num_tokens, num_experts, load_size, n_dims),
    )

    return row_id_map


def permute(
    input_tensor: torch.Tensor,
    row_id_map: torch.Tensor,
    probs: torch.Tensor = None,
    scale: torch.Tensor = None,
    pad_offsets: torch.Tensor = None,
    num_out_tokens: int = None,
    num_experts: int = 8,
    permute_probs: bool = False,
    permute_scale: bool = False,
    fusion_pad: bool = False,
    block_size: int = 256,
) -> tuple:
    """
    Permute tokens according to expert assignments.

    Args:
        input_tensor: Input tensor of shape [num_tokens, hidden_size].
        row_id_map: Row ID mapping from compute_row_id_map.
        probs: Optional probability tensor of shape [num_tokens, num_experts].
        scale: Optional scale tensor.
        pad_offsets: Optional padding offsets for fused padding.
        num_out_tokens: Number of output tokens (after padding if fusion_pad=True).
        num_experts: Number of experts.
        permute_probs: Whether to permute probabilities.
        permute_scale: Whether to permute scales.
        fusion_pad: Whether to fuse padding into permutation.
        block_size: Block size for kernel execution.

    Returns:
        Tuple of (output, permuted_probs, permuted_scale).
    """
    device = input_tensor.device
    stream = torch.cuda.current_stream()

    num_tokens, hidden_size = input_tensor.shape
    scale_hidden_dim = scale.shape[1] if scale is not None else 0

    if num_out_tokens is None:
        num_out_tokens = num_tokens

    # Allocate output tensors
    output = torch.zeros(num_out_tokens, hidden_size, dtype=input_tensor.dtype, device=device)
    permuted_probs = (
        torch.zeros(num_out_tokens, dtype=probs.dtype, device=device)
        if permute_probs and probs is not None
        else None
    )
    permuted_scale = (
        torch.zeros(num_out_tokens, scale_hidden_dim, dtype=scale.dtype, device=device)
        if permute_scale and scale is not None
        else None
    )

    # Handle None tensors for kernel
    if probs is None:
        probs = torch.empty(0, dtype=torch.float32, device=device)
    if scale is None:
        scale = torch.empty(0, dtype=torch.float32, device=device)
    if permuted_scale is None:
        permuted_scale = torch.empty(0, dtype=torch.float32, device=device)
    if permuted_probs is None:
        permuted_probs = torch.empty(0, dtype=torch.float32, device=device)
    if pad_offsets is None:
        pad_offsets = torch.empty(0, dtype=torch.int32, device=device)

    # Launch kernel
    num_hidden_blocks = ceil_div(hidden_size, block_size)
    grid = (num_tokens, num_hidden_blocks)

    ct.launch(
        stream,
        grid,
        _permute_kernel,
        (
            input_tensor.reshape(-1),
            row_id_map,
            probs.reshape(-1) if probs.numel() > 0 else probs,
            scale.reshape(-1) if scale.numel() > 0 else scale,
            permuted_scale.reshape(-1) if permuted_scale.numel() > 0 else permuted_scale,
            pad_offsets,
            output.reshape(-1),
            permuted_probs,
            scale_hidden_dim,
            num_tokens,
            num_out_tokens,
            num_experts,
            hidden_size,
            permute_probs,
            permute_scale,
            fusion_pad,
            block_size,
        ),
    )

    return output, permuted_probs if permute_probs else None, permuted_scale if permute_scale else None


def unpermute(
    input_tensor: torch.Tensor,
    row_id_map: torch.Tensor,
    merging_probs: torch.Tensor = None,
    permuted_probs: torch.Tensor = None,
    pad_offsets: torch.Tensor = None,
    num_tokens: int = None,
    num_experts: int = 8,
    with_merging_probs: bool = False,
    permute_probs: bool = False,
    fusion_unpad: bool = False,
    block_size: int = 256,
) -> tuple:
    """
    Unpermute tokens to restore original order.

    Args:
        input_tensor: Permuted input tensor.
        row_id_map: Row ID mapping from compute_row_id_map.
        merging_probs: Optional merging probabilities.
        permuted_probs: Optional permuted probabilities (for gradient computation).
        pad_offsets: Optional padding offsets.
        num_tokens: Number of original tokens.
        num_experts: Number of experts.
        with_merging_probs: Whether to apply merging probabilities.
        permute_probs: Whether to unpermute probabilities.
        fusion_unpad: Whether to fuse unpadding.
        block_size: Block size for kernel execution.

    Returns:
        Tuple of (output, unpermuted_probs).
    """
    device = input_tensor.device
    stream = torch.cuda.current_stream()

    if num_tokens is None:
        num_tokens = row_id_map.shape[0] // (2 * num_experts + 1)

    _, hidden_size = input_tensor.shape

    # Allocate output tensors
    output = torch.zeros(num_tokens, hidden_size, dtype=input_tensor.dtype, device=device)
    unpermuted_probs = (
        torch.zeros(num_tokens, num_experts, dtype=torch.float32, device=device)
        if permute_probs
        else None
    )

    # Handle None tensors
    if merging_probs is None:
        merging_probs = torch.empty(0, dtype=torch.float32, device=device)
    if permuted_probs is None:
        permuted_probs = torch.empty(0, dtype=torch.float32, device=device)
    if pad_offsets is None:
        pad_offsets = torch.empty(0, dtype=torch.int32, device=device)
    if unpermuted_probs is None:
        unpermuted_probs = torch.empty(0, dtype=torch.float32, device=device)

    probs_load_width = next_power_of_2(num_experts)

    # Launch kernel
    num_hidden_blocks = ceil_div(hidden_size, block_size)
    grid = (num_tokens, num_hidden_blocks)

    ct.launch(
        stream,
        grid,
        _unpermute_kernel,
        (
            input_tensor.reshape(-1),
            row_id_map,
            merging_probs.reshape(-1) if merging_probs.numel() > 0 else merging_probs,
            permuted_probs,
            pad_offsets,
            output.reshape(-1),
            unpermuted_probs.reshape(-1) if unpermuted_probs.numel() > 0 else unpermuted_probs,
            num_tokens,
            num_experts,
            hidden_size,
            probs_load_width,
            with_merging_probs,
            permute_probs,
            fusion_unpad,
            block_size,
        ),
    )

    return output, unpermuted_probs if permute_probs else None


def unpermute_with_mask_map_bwd_with_merging_probs(
    fwd_output_grad: torch.Tensor,
    row_id_map: torch.Tensor,
    fwd_input: torch.Tensor,
    merging_probs: torch.Tensor,
    pad_offsets: torch.Tensor = None,
    num_tokens: int = None,
    num_experts: int = 8,
    num_out_tokens: int = None,
    hidden_size: int = None,
    block_size: int = 256,
) -> tuple:
    """
    Backward pass for unpermute with merging probabilities.

    Args:
        fwd_output_grad: Gradient of unpermuted output, shape [num_tokens, hidden_size].
        row_id_map: Row ID mapping from compute_row_id_map.
        fwd_input: Forward input tensor, shape [num_out_tokens, hidden_size].
        merging_probs: Merging probabilities, shape [num_tokens, num_experts].
        pad_offsets: Optional padding offsets.
        num_tokens: Number of tokens.
        num_experts: Number of experts.
        num_out_tokens: Number of permuted tokens.
        hidden_size: Hidden size.
        block_size: Block size for kernel execution.

    Returns:
        Tuple of (fwd_input_grad, merging_probs_grad).
    """
    device = fwd_output_grad.device
    stream = torch.cuda.current_stream()

    if num_tokens is None:
        num_tokens = row_id_map.shape[0] // (2 * num_experts + 1)
    if num_out_tokens is None:
        num_out_tokens = fwd_input.shape[0]
    if hidden_size is None:
        hidden_size = fwd_output_grad.shape[1]

    alloc = torch.zeros if pad_offsets is not None else torch.empty
    fwd_input_grad = alloc((num_out_tokens, hidden_size), dtype=fwd_output_grad.dtype, device=device)
    merging_probs_grad = torch.empty((num_tokens, num_experts), dtype=merging_probs.dtype, device=device)

    if pad_offsets is None:
        pad_offsets = torch.empty(0, dtype=torch.int32, device=device)

    probs_load_width = next_power_of_2(num_experts)

    ct.launch(
        stream,
        (num_tokens,),
        _unpermute_bwd_with_merging_probs_kernel,
        (
            fwd_output_grad.reshape(-1),
            fwd_input.reshape(-1),
            merging_probs.reshape(-1),
            row_id_map,
            pad_offsets,
            fwd_input_grad.reshape(-1),
            merging_probs_grad.reshape(-1),
            num_tokens,
            num_experts,
            hidden_size,
            probs_load_width,
            pad_offsets.numel() > 0,
            block_size,
        ),
    )

    return fwd_input_grad, merging_probs_grad


def sort_chunks_by_map(
    input_tensor: torch.Tensor,
    split_sizes: torch.Tensor,
    sorted_indices: torch.Tensor,
    probs: torch.Tensor = None,
    permute_probs: bool = False,
    forward: bool = True,
    block_size: int = 256,
) -> tuple:
    """
    Sort chunks of data according to a mapping.

    Args:
        input_tensor: Input tensor to sort.
        split_sizes: Sizes of each chunk.
        sorted_indices: New order of chunks.
        probs: Optional probabilities to permute.
        permute_probs: Whether to permute probabilities.
        forward: Direction of sorting.
        block_size: Block size for kernel execution.

    Returns:
        Tuple of (output, permuted_probs).
    """
    device = input_tensor.device
    stream = torch.cuda.current_stream()

    num_tokens, hidden_size = input_tensor.shape
    num_splits = split_sizes.shape[0]

    # Compute destination row mapping
    idx_load_width = next_power_of_2(num_splits)
    dst_rows = torch.zeros(num_tokens, dtype=torch.int32, device=device)

    ct.launch(
        stream,
        (num_tokens,),
        _make_chunk_sort_map_kernel,
        (split_sizes, sorted_indices, dst_rows, num_splits, idx_load_width),
    )

    # Sort using the mapping
    output = torch.zeros_like(input_tensor)
    permuted_probs = (
        torch.zeros(num_tokens, dtype=probs.dtype, device=device)
        if permute_probs and probs is not None
        else None
    )

    if probs is None:
        probs = torch.empty(0, dtype=torch.float32, device=device)
    if permuted_probs is None:
        permuted_probs = torch.empty(0, dtype=torch.float32, device=device)

    num_hidden_blocks = ceil_div(hidden_size, block_size)
    grid = (num_tokens, num_hidden_blocks)

    ct.launch(
        stream,
        grid,
        _sort_chunks_by_map_kernel,
        (
            input_tensor.reshape(-1),
            dst_rows,
            probs if probs.numel() > 0 else probs,
            output.reshape(-1),
            permuted_probs,
            hidden_size,
            permute_probs,
            block_size,
            forward,
        ),
    )

    return output, permuted_probs if permute_probs else None
