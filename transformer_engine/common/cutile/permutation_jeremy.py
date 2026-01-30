# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Efficient Permutation kernels written with cuTile Python."""

import cuda.tile as ct

# Type aliases for constants
ConstInt = ct.Constant[int]
ConstBool = ct.Constant[bool]


# =============================================================================
# Bitonic Sort Helper Functions
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
    
    This function compares pairs of elements and swaps them based on the
    flip mask and comparison result.
    """
    n_outer = x.numel >> n_dims
    
    # Create shape for reshape: [n_outer * (2**i), 2, 2 ** (n_dims - i - 1)]
    shape_0 = n_outer * (2 ** i)
    shape_2 = 2 ** (n_dims - i - 1)
    
    # Reshape x and indices
    y = ct.reshape(x, (shape_0, 2, shape_2))
    z = ct.reshape(indices, (shape_0, 2, shape_2))
    
    # Create mask [0, 1] along dimension 1
    mask = ct.arange(2, dtype=ct.int32)
    mask = ct.reshape(mask, (1, 2, 1))
    mask = ct.broadcast_to(mask, (shape_0, 2, shape_2))
    
    # Extract left and right values using mask
    # Left: where mask == 0, Right: where mask == 1
    inv_mask = 1 - mask
    
    # Sum along axis 1 to get left/right values
    # y * inv_mask gives left values (mask=0), y * mask gives right values (mask=1)
    y_left = ct.sum(y * ct.astype(inv_mask, y.dtype), axis=1)
    y_right = ct.sum(y * ct.astype(mask, y.dtype), axis=1)
    z_left = ct.sum(z * inv_mask, axis=1)
    z_right = ct.sum(z * mask, axis=1)
    
    # Broadcast back to original shape
    y_left = ct.expand_dims(y_left, axis=1)
    y_right = ct.expand_dims(y_right, axis=1)
    z_left = ct.expand_dims(z_left, axis=1)
    z_right = ct.expand_dims(z_right, axis=1)
    
    l_value = ct.broadcast_to(y_left, (shape_0, 2, shape_2))
    r_value = ct.broadcast_to(y_right, (shape_0, 2, shape_2))
    l_indice = ct.broadcast_to(z_left, (shape_0, 2, shape_2))
    r_indice = ct.broadcast_to(z_right, (shape_0, 2, shape_2))
    
    # Reshape back to original shape
    l_value = ct.reshape(l_value, x.shape)
    r_value = ct.reshape(r_value, x.shape)
    l_indice = ct.reshape(l_indice, x.shape)
    r_indice = ct.reshape(r_indice, x.shape)
    
    # Bitcast for comparison (treat as signed int for proper comparison)
    il_value = ct.bitcast(l_value, ct.int32)
    ir_value = ct.bitcast(r_value, ct.int32)
    ix = ct.bitcast(x, ct.int32)
    
    # Compute swap condition: (l_value > r_value) XOR flip
    swap_cond = (l_value > r_value) ^ (flip != 0)
    
    # XOR-based swap
    zero_int = ct.zeros(x.shape, dtype=ct.int32)
    flag1 = ct.where(swap_cond, il_value ^ ir_value, zero_int)
    ret = ix ^ flag1
    
    flag2 = ct.where(swap_cond, l_indice ^ r_indice, zero_int)
    ind = indices ^ flag2
    
    return ct.bitcast(ret, x.dtype), ind


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
    
    order_type 0 == ascending
    order_type 1 == descending
    order_type 2 == alternating
    """
    n_outer = x.numel >> n_dims
    
    if order == 2:
        # Alternating order - create flip pattern
        shape_0 = n_outer * (2 ** (n_dims - 1 - stage))
        shape_2 = 2 ** stage
        flip_base = ct.arange(2, dtype=ct.int32)
        flip_base = ct.reshape(flip_base, (1, 2, 1))
        flip = ct.broadcast_to(flip_base, (shape_0, 2, shape_2))
        flip = ct.reshape(flip, x.shape)
    else:
        flip = ct.full(x.shape, order, dtype=ct.int32)
    
    # Perform compare and swap for each sub-stage
    for i in range(stage):
        x, indices = _compare_and_swap_cutile(x, indices, flip, i + (n_dims - stage), n_dims)
    
    return x, indices


@ct.function
def _argsort_cutile(x: ct.Tile, indices: ct.Tile, n_dims: ConstInt):
    """
    Bitonic argsort implementation in cuTile.
    
    Sorts the input tile and returns both sorted values and their original indices.
    """
    for i in range(1, n_dims + 1):
        order = 2 if i < n_dims else 1
        x, indices = _bitonic_merge_cutile(x, indices, i, order, n_dims)
    return x, indices


# =============================================================================
# Row ID Map Kernels
# =============================================================================


@ct.kernel
def _row_id_map_pass_1_kernel_cutile(
    # input arrays
    routing_map: ct.Array,
    # output arrays
    row_id_map: ct.Array,
    workspace: ct.Array,
    # sizes
    num_tokens: int,
    num_experts: int,
    # metas
    BLOCK_SIZE: ConstInt,
):
    """
    First pass of row ID mapping in cuTile.
    
    Computes cumulative sum of expert token masks within each block
    and stores the count of tokens per block.
    """
    pid_m = ct.bid(0)  # expert index
    pid_n = ct.bid(1)  # token block index
    
    num_blocks = ct.cdiv(num_tokens, BLOCK_SIZE)
    
    # Create offset indices for this block
    offset = pid_n * BLOCK_SIZE + ct.arange(BLOCK_SIZE, dtype=ct.int32)
    
    # Load routing map values for this expert and token block
    # Using gather for non-contiguous access pattern
    expert_token_mask = ct.gather(
        routing_map,
        pid_m * num_tokens + offset,
        padding_value=0,
    ).astype(ct.int32)
    
    # Mask out-of-bounds tokens
    valid_mask = offset < num_tokens
    expert_token_mask = ct.where(valid_mask, expert_token_mask, ct.zeros(expert_token_mask.shape, dtype=ct.int32))
    
    # Compute cumulative sum within block
    row_id_within_token_block = ct.cumsum(expert_token_mask, axis=0) * expert_token_mask
    
    # Store row IDs
    ct.scatter(
        row_id_map,
        pid_m * num_tokens + offset,
        row_id_within_token_block,
    )
    
    # Store token count for this block
    n_tokens_per_block = ct.sum(expert_token_mask)
    ct.scatter(workspace, pid_m * num_blocks + pid_n, n_tokens_per_block)


@ct.kernel
def _row_id_map_pass_2_kernel_cutile(
    # input/output arrays
    row_id_map: ct.Array,
    workspace: ct.Array,
    # sizes
    num_tokens: int,
    num_experts: int,
    # metas
    WORKSPACE_LOAD_WIDTH: ConstInt,
    BLOCK_SIZE: ConstInt,
):
    """
    Second pass of row ID mapping in cuTile.
    
    Adds prefix sum offsets from previous blocks to get global row IDs.
    """
    pid_m = ct.bid(0)  # expert index
    pid_n = ct.bid(1)  # token block index
    
    num_blocks = ct.cdiv(num_tokens, BLOCK_SIZE)
    chunk_idx = pid_m * num_blocks + pid_n
    
    # Create offset indices for this block
    offset = pid_n * BLOCK_SIZE + ct.arange(BLOCK_SIZE, dtype=ct.int32)
    
    # Load row IDs from previous pass
    row_id_within_token_block = ct.gather(
        row_id_map,
        pid_m * num_tokens + offset,
        padding_value=0,
    )
    
    # Load workspace to compute prefix sum from previous chunks
    workspace_off = ct.arange(WORKSPACE_LOAD_WIDTH, dtype=ct.int32)
    workspace_mask = workspace_off < chunk_idx
    n_tokens_per_chunk = ct.gather(workspace, workspace_off, padding_value=0)
    n_tokens_per_chunk = ct.where(workspace_mask, n_tokens_per_chunk, ct.zeros(n_tokens_per_chunk.shape, dtype=ct.int32))
    
    prefix_sum = ct.sum(n_tokens_per_chunk)
    
    # Compute global row IDs (-1 for non-routed tokens)
    row_id = ct.where(
        row_id_within_token_block == 0,
        ct.full(row_id_within_token_block.shape, -1, dtype=ct.int32),
        row_id_within_token_block + prefix_sum - 1,
    )
    
    # Store updated row IDs
    valid_mask = offset < num_tokens
    # Only scatter valid positions
    ct.scatter(
        row_id_map,
        pid_m * num_tokens + offset,
        row_id,
    )


@ct.kernel
def _row_id_map_pass_3_kernel_cutile(
    # input/output arrays
    row_id_map: ct.Array,
    # sizes
    num_tokens: int,
    # metas
    num_experts: ConstInt,
    LOAD_SIZE: ConstInt,
):
    """
    Third pass of row ID mapping in cuTile.
    
    Sorts the row ID map for each token and stores sorted indices.
    """
    pid = ct.bid(0)  # token index
    
    # Compute n_dims for argsort (log2 of LOAD_SIZE)
    n_dims = 0
    temp = LOAD_SIZE
    while temp > 1:
        temp = temp >> 1
        n_dims = n_dims + 1
    
    # Load row IDs for this token across all experts
    off = ct.arange(LOAD_SIZE, dtype=ct.int32)
    row_id_map_tile = ct.gather(
        row_id_map,
        pid * (num_experts * 2 + 1) + off,
        padding_value=-1,
    )
    
    # Count routed experts
    routed_mask = row_id_map_tile != -1
    n_routed = ct.sum(ct.where(routed_mask, ct.ones(routed_mask.shape, dtype=ct.int32), ct.zeros(routed_mask.shape, dtype=ct.int32)))
    
    # Initialize indices
    indices = off
    
    # Perform argsort
    sorted_map, sorted_indices = _argsort_cutile(row_id_map_tile, indices, n_dims)
    
    # Store sorted map (only n_routed elements)
    ct.scatter(
        row_id_map,
        pid * (num_experts * 2 + 1) + off,
        sorted_map,
    )
    
    # Store sorted indices at offset num_experts
    ct.scatter(
        row_id_map,
        pid * (num_experts * 2 + 1) + num_experts + off,
        sorted_indices,
    )
    
    # Store n_routed at offset num_experts * 2
    ct.scatter(
        row_id_map,
        pid * (num_experts * 2 + 1) + num_experts * 2,
        n_routed,
    )


# =============================================================================
# Permute Kernel
# =============================================================================


@ct.kernel
def _permute_kernel_cutile(
    # input arrays
    input_arr: ct.Array,
    row_id_map: ct.Array,
    probs: ct.Array,
    scale: ct.Array,
    pad_offsets: ct.Array,
    # output arrays
    output_arr: ct.Array,
    permuted_probs: ct.Array,
    permuted_scale: ct.Array,
    # sizes
    num_tokens: int,
    hidden_size: int,
    scale_hidden_dim: int,
    num_out_tokens: int,
    # metas
    num_experts: ConstInt,
    PERMUTE_PROBS: ConstInt,
    PERMUTE_SCALE: ConstInt,
    FUSION_PAD: ConstInt,
    BLOCK_SIZE: ConstInt,
):
    """
    Permute kernel in cuTile.
    
    Note: When FUSION_PAD=True, output buffers should be pre-zeroed by the caller
    to ensure padding positions contain zeros.
    """
    pid_t = ct.bid(0)  # token index
    pid_h = ct.bid(1)  # hidden dimension block index
    
    # Compute current offset for hidden dimension
    cur_off = pid_h * BLOCK_SIZE + ct.arange(BLOCK_SIZE, dtype=ct.int32)
    mask = cur_off < hidden_size
    
    src_row = pid_t
    
    # Load input data for this token and hidden block
    input_off = src_row * hidden_size + cur_off
    inp = ct.gather(input_arr, input_off, padding_value=0)
    
    # Load scale if needed
    if PERMUTE_SCALE:
        mask_scale = cur_off < scale_hidden_dim
        scale_off = pid_t * scale_hidden_dim + cur_off
        scale_tile = ct.gather(scale, scale_off, padding_value=0)
    
    # Load n_routed for this token
    n_routed = ct.gather(
        row_id_map,
        pid_t * (num_experts * 2 + 1) + num_experts * 2,
        padding_value=0,
    )
    
    # Iterate over routed experts
    for idx in range(num_experts):
        # Early exit if we've processed all routed experts
        # Note: cuTile doesn't have dynamic loop bounds, so we check inside
        if idx < n_routed:
            # Load destination row
            dst_row = ct.gather(
                row_id_map,
                pid_t * (num_experts * 2 + 1) + idx,
                padding_value=0,
            )
            
            if FUSION_PAD or PERMUTE_PROBS:
                expert_idx = ct.gather(
                    row_id_map,
                    pid_t * (num_experts * 2 + 1) + num_experts + idx,
                    padding_value=0,
                )
            
            if FUSION_PAD:
                pad_off = ct.gather(pad_offsets, expert_idx, padding_value=0)
                dst_row = dst_row + pad_off
            
            output_off = dst_row * hidden_size + cur_off
            
            if PERMUTE_SCALE:
                permuted_scale_off = dst_row * scale_hidden_dim + cur_off
                ct.scatter(permuted_scale, permuted_scale_off, scale_tile)
            
            if PERMUTE_PROBS:
                prob_off = pid_t * num_experts + expert_idx
                prob = ct.gather(probs, prob_off, padding_value=0)
                
                if pid_h == 0:
                    permuted_prob_off = dst_row
                    ct.scatter(permuted_probs, permuted_prob_off, prob)
                
                # Handle routing map padding (prob == 0 means padded slot)
                zero_tile = ct.zeros(inp.shape, dtype=inp.dtype)
                is_zero_prob = prob == 0.0
                output_tile = ct.where(
                    ct.broadcast_to(is_zero_prob, inp.shape),
                    zero_tile,
                    inp,
                )
                ct.scatter(output_arr, output_off, output_tile)
            else:
                ct.scatter(output_arr, output_off, inp)


# =============================================================================
# Unpermute Kernel
# =============================================================================


@ct.kernel
def _unpermute_kernel_cutile(
    # input arrays
    input_arr: ct.Array,
    row_id_map: ct.Array,
    merging_probs: ct.Array,
    permuted_probs: ct.Array,
    pad_offsets: ct.Array,
    # output arrays
    output_arr: ct.Array,
    unpermuted_probs: ct.Array,
    # sizes
    num_tokens: int,
    hidden_size: int,
    num_out_tokens: int,
    # metas
    num_experts: ConstInt,
    PROBS_LOAD_WIDTH: ConstInt,
    WITH_MERGING_PROBS: ConstInt,
    PERMUTE_PROBS: ConstInt,
    FUSION_UNPAD: ConstInt,
    BLOCK_SIZE: ConstInt,
):
    """
    Unpermute kernel in cuTile.
    
    Reverses the permutation and optionally applies merging probabilities.
    """
    pid_t = ct.bid(0)  # token index
    pid_h = ct.bid(1)  # hidden dimension block index
    
    current_offset = pid_h * BLOCK_SIZE + ct.arange(BLOCK_SIZE, dtype=ct.int32)
    mask = current_offset < hidden_size
    
    # Initialize probs_grad to zero if needed
    if PERMUTE_PROBS:
        if pid_h == 0:
            map_load_off = ct.arange(PROBS_LOAD_WIDTH, dtype=ct.int32)
            unpermuted_prob_off = pid_t * num_experts + map_load_off
            zero_probs = ct.zeros((PROBS_LOAD_WIDTH,), dtype=ct.float32)
            ct.scatter(unpermuted_probs, unpermuted_prob_off, zero_probs)
    
    # Initialize accumulator
    accumulator = ct.zeros((BLOCK_SIZE,), dtype=ct.float32)
    
    # Load n_routed for this token
    n_routed = ct.gather(
        row_id_map,
        pid_t * (num_experts * 2 + 1) + num_experts * 2,
        padding_value=0,
    )
    
    for idx in range(num_experts):
        if idx < n_routed:
            # Load source row
            src_row = ct.gather(
                row_id_map,
                pid_t * (num_experts * 2 + 1) + idx,
                padding_value=0,
            )
            
            if FUSION_UNPAD or WITH_MERGING_PROBS:
                expert_idx = ct.gather(
                    row_id_map,
                    pid_t * (num_experts * 2 + 1) + num_experts + idx,
                    padding_value=0,
                )
            
            if FUSION_UNPAD:
                pad_off = ct.gather(pad_offsets, expert_idx, padding_value=0)
                src_row = src_row + pad_off
            
            input_off = src_row * hidden_size + current_offset
            inp = ct.gather(input_arr, input_off, padding_value=0)
            inp = ct.astype(inp, ct.float32)
            
            if WITH_MERGING_PROBS:
                merging_prob_off = pid_t * num_experts + expert_idx
                merging_prob = ct.gather(merging_probs, merging_prob_off, padding_value=0)
                merging_prob = ct.astype(merging_prob, ct.float32)
                inp = inp * merging_prob
            
            accumulator = accumulator + inp
            
            if PERMUTE_PROBS:
                if pid_h == 0:
                    expert_idx_probs = ct.gather(
                        row_id_map,
                        pid_t * (num_experts * 2 + 1) + num_experts + idx,
                        padding_value=0,
                    )
                    unpermuted_prob_off = pid_t * num_experts + expert_idx_probs
                    permuted_prob_off = src_row
                    prob = ct.gather(permuted_probs, permuted_prob_off, padding_value=0)
                    ct.scatter(unpermuted_probs, unpermuted_prob_off, prob)
    
    # Convert back to output dtype and store
    accumulator = ct.astype(accumulator, output_arr.dtype)
    dst_row = pid_t
    output_off = dst_row * hidden_size + current_offset
    ct.scatter(output_arr, output_off, accumulator)


# =============================================================================
# Unpermute Backward with Merging Probs Kernel
# =============================================================================


@ct.kernel
def _unpermute_bwd_with_merging_probs_kernel_cutile(
    # input arrays
    fwd_output_grad: ct.Array,
    fwd_input: ct.Array,
    merging_probs: ct.Array,
    row_id_map: ct.Array,
    pad_offsets: ct.Array,
    # output arrays
    fwd_input_grad: ct.Array,
    merging_probs_grad: ct.Array,
    # sizes
    num_tokens: int,
    hidden_size: int,
    # metas
    num_experts: ConstInt,
    PROBS_LOAD_WIDTH: ConstInt,
    FUSION_UNPAD: ConstInt,
    BLOCK_SIZE: ConstInt,
):
    """
    Backward pass for unpermute with merging probs in cuTile.
    
    Computes gradients for both the input and the merging probabilities.
    """
    pid = ct.bid(0)  # token index
    
    # Initialize probs_grad to zero
    map_load_off = ct.arange(PROBS_LOAD_WIDTH, dtype=ct.int32)
    token_probs_grad_off = pid * num_experts + map_load_off
    zero_probs = ct.zeros((PROBS_LOAD_WIDTH,), dtype=ct.float32)
    ct.scatter(merging_probs_grad, token_probs_grad_off, zero_probs)
    
    # Load n_routed for this token
    n_routed = ct.gather(
        row_id_map,
        pid * (num_experts * 2 + 1) + num_experts * 2,
        padding_value=0,
    )
    
    for idx in range(num_experts):
        if idx < n_routed:
            # Load destination row and expert index
            dst_row = ct.gather(
                row_id_map,
                pid * (num_experts * 2 + 1) + idx,
                padding_value=0,
            )
            expert_idx = ct.gather(
                row_id_map,
                pid * (num_experts * 2 + 1) + num_experts + idx,
                padding_value=0,
            )
            
            if FUSION_UNPAD:
                pad_off = ct.gather(pad_offsets, expert_idx, padding_value=0)
                dst_row = dst_row + pad_off
            
            # Load merging probability
            merging_prob_off = pid * num_experts + expert_idx
            merging_prob = ct.gather(merging_probs, merging_prob_off, padding_value=0)
            merging_prob = ct.astype(merging_prob, ct.float32)
            
            prob_grad_accum = ct.zeros((BLOCK_SIZE,), dtype=ct.float32)
            
            # Process hidden dimension in blocks
            current_start = 0
            while current_start < hidden_size:
                current_offset = current_start + ct.arange(BLOCK_SIZE, dtype=ct.int32)
                block_mask = current_offset < hidden_size
                
                src_row = pid
                input_off = src_row * hidden_size + current_offset
                inp = ct.gather(fwd_output_grad, input_off, padding_value=0)
                inp = ct.astype(inp, ct.float32)
                
                # Compute input grad
                output = inp * merging_prob
                output = ct.astype(output, fwd_input_grad.dtype)
                output_off = dst_row * hidden_size + current_offset
                ct.scatter(fwd_input_grad, output_off, output)
                
                # Accumulate prob grad
                fwd_input_off = dst_row * hidden_size + current_offset
                fwd_input_tile = ct.gather(fwd_input, fwd_input_off, padding_value=0)
                fwd_input_tile = ct.astype(fwd_input_tile, ct.float32)
                prob_grad_accum = prob_grad_accum + fwd_input_tile * inp
                
                current_start = current_start + BLOCK_SIZE
            
            # Reduce and store prob grad
            probs_grad = ct.sum(prob_grad_accum)
            probs_grad = ct.astype(probs_grad, merging_probs_grad.dtype)
            probs_grad_off = pid * num_experts + expert_idx
            ct.scatter(merging_probs_grad, probs_grad_off, probs_grad)


# =============================================================================
# Chunk Sort Map Kernel
# =============================================================================


@ct.kernel
def _make_chunk_sort_map_kernel_cutile(
    # input arrays
    split_sizes: ct.Array,
    sorted_indices: ct.Array,
    # output arrays
    dst_rows: ct.Array,
    # sizes
    num_splits: ConstInt,
    num_tokens: int,
    # metas
    IDX_LOAD_WIDTH: ConstInt,
):
    """
    Create chunk sort map in cuTile.
    
    Maps each token from its input position to its output position
    based on the sorted chunk order.
    """
    pid = ct.bid(0)  # token index
    
    load_split_offset = ct.arange(IDX_LOAD_WIDTH, dtype=ct.int32)
    
    # Load sorted indices
    sorted_idx = ct.gather(sorted_indices, load_split_offset, padding_value=0)
    
    # Load input split sizes
    input_split_sizes = ct.gather(split_sizes, load_split_offset, padding_value=0)
    input_split_sizes = ct.astype(input_split_sizes, ct.int32)
    
    # Mask out-of-bounds
    valid_mask = load_split_offset < num_splits
    input_split_sizes = ct.where(valid_mask, input_split_sizes, ct.zeros(input_split_sizes.shape, dtype=ct.int32))
    
    # Compute cumulative sum of split sizes
    input_split_sizes_cumsum = ct.cumsum(input_split_sizes, axis=0)
    
    # Find which chunk the current token belongs to
    input_split_sizes_mask = ct.where(
        input_split_sizes_cumsum <= pid,
        ct.ones(input_split_sizes_cumsum.shape, dtype=ct.int32),
        ct.zeros(input_split_sizes_cumsum.shape, dtype=ct.int32),
    )
    input_chunk_idx = ct.sum(input_split_sizes_mask)
    
    # Compute offset within chunk
    input_split_sizes_presum = ct.sum(input_split_sizes * input_split_sizes_mask)
    in_chunk_offset = pid - input_split_sizes_presum
    
    # Find output chunk index (where does input_chunk_idx map to in output?)
    output_chunk_mask = ct.where(
        sorted_idx == input_chunk_idx,
        ct.ones(sorted_idx.shape, dtype=ct.int32),
        ct.zeros(sorted_idx.shape, dtype=ct.int32),
    )
    output_chunk_idx = ct.argmax(output_chunk_mask, axis=0)
    
    # Compute output split sizes (reordered by sorted_indices)
    output_split_sizes = ct.gather(split_sizes, sorted_idx, padding_value=0)
    output_split_sizes = ct.astype(output_split_sizes, ct.int32)
    
    # Compute prefix sum up to output_chunk_idx
    output_pre_split_mask = ct.where(
        load_split_offset < output_chunk_idx,
        ct.ones(load_split_offset.shape, dtype=ct.int32),
        ct.zeros(load_split_offset.shape, dtype=ct.int32),
    )
    output_pre_split_sizes = output_split_sizes * output_pre_split_mask
    
    dst_row = ct.sum(output_pre_split_sizes) + in_chunk_offset
    ct.scatter(dst_rows, pid, dst_row)


# =============================================================================
# Sort Chunks by Map Kernel
# =============================================================================


@ct.kernel
def _sort_chunks_by_map_kernel_cutile(
    # input arrays
    input_arr: ct.Array,
    row_id_map: ct.Array,
    probs: ct.Array,
    # output arrays
    output_arr: ct.Array,
    permuted_probs: ct.Array,
    # sizes
    num_tokens: int,
    hidden_size: int,
    # metas
    PERMUTE_PROBS: ConstInt,
    FORWARD: ConstInt,
    BLOCK_SIZE: ConstInt,
):
    """
    Sort chunks by map in cuTile.
    
    Reorders data according to the row_id_map, either forward or backward.
    """
    pid_t = ct.bid(0)  # token index
    pid_h = ct.bid(1)  # hidden dimension block index
    
    if FORWARD:
        src_row = pid_t
        dst_row = ct.gather(row_id_map, pid_t, padding_value=0)
    else:
        src_row = ct.gather(row_id_map, pid_t, padding_value=0)
        dst_row = pid_t
    
    current_offset = pid_h * BLOCK_SIZE + ct.arange(BLOCK_SIZE, dtype=ct.int32)
    mask = current_offset < hidden_size
    
    # Load input
    input_offsets = src_row * hidden_size + current_offset
    inp = ct.gather(input_arr, input_offsets, padding_value=0)
    
    # Store output
    output_offsets = dst_row * hidden_size + current_offset
    ct.scatter(output_arr, output_offsets, inp)
    
    # Handle probs permutation
    if PERMUTE_PROBS:
        if pid_h == 0:
            prob = ct.gather(probs, src_row, padding_value=0)
            ct.scatter(permuted_probs, dst_row, prob)