# Copyright (c) 2025-2028, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""JAX wrapper functions for Permutation Triton kernels."""

from typing import Optional, Tuple, Union
import jax
import jax.numpy as jnp
from jax import ShapeDtypeStruct
import triton
import jax_triton as jt

from transformer_engine.common.triton.permutation import (
    _row_id_map_pass_1_kernel,
    _row_id_map_pass_2_kernel,
    _row_id_map_pass_3_kernel,
    _permute_kernel,
    _unpermute_kernel,
    _unpermute_bwd_with_merging_probs_kernel,
    _make_chunk_sort_map_kernel,
    _sort_chunks_by_map_kernel,
)


def make_row_id_map(
    routing_map: jnp.ndarray,
    num_tokens: int,
    num_experts: int,
) -> jnp.ndarray:
    """
    Prepare the row_id_map for the permutation using JAX-Triton.
    
    Parameters
    ----------
    routing_map : jnp.ndarray
        Input tensor of shape `[num_tokens, num_experts]`. It is a mask tensor that indicates
        which experts are routed to which tokens. The values in it: 1 means the token is routed to
        this expert and 0 means not.
    num_tokens : int
        Number of tokens in the input tensor.
    num_experts : int
        Number of experts in the input tensor.
    
    Returns
    -------
    row_id_map : jnp.ndarray
        The row_id_map for the permutation of shape `[num_tokens, num_experts * 2 + 1]`.
        For each token, the last item is the number of experts that are routed (n_routed).
        The first n_routed items are the destination row indices in the permuted tokens.
        The [num_experts, num_experts + n_routed) items are the indices of the experts corresponding
        to the first n_routed row indices above.
    """
    row_id_map = jnp.zeros((num_tokens, num_experts * 2 + 1), dtype=jnp.int32)
    block_size = 1024
    grid = (num_experts, triton.cdiv(num_tokens, block_size))
    workspace_tensor = jnp.zeros(grid, dtype=jnp.int32)
    
    # Pass 1: block cumsum
    row_id_map, workspace_tensor = jt.triton_call(
        routing_map,
        row_id_map,
        workspace_tensor,
        kernel=_row_id_map_pass_1_kernel,
        out_shape=[
            ShapeDtypeStruct(row_id_map.shape, row_id_map.dtype),
            ShapeDtypeStruct(workspace_tensor.shape, workspace_tensor.dtype),
        ],
        grid=grid,
        num_tokens=num_tokens,
        routing_stride_0=routing_map.strides[0] // routing_map.dtype.itemsize,
        routing_stride_1=routing_map.strides[1] // routing_map.dtype.itemsize,
        row_id_stride_0=num_experts * 2 + 1,  # Stride for row dimension
        row_id_stride_1=1,  # Stride for column dimension
        block_size=block_size,
    )
    
    # Pass 2: cumsum all and process the mask
    row_id_map, workspace_tensor = jt.triton_call(
        row_id_map,
        workspace_tensor,
        kernel=_row_id_map_pass_2_kernel,
        out_shape=[
            ShapeDtypeStruct(row_id_map.shape, row_id_map.dtype),
            ShapeDtypeStruct(workspace_tensor.shape, workspace_tensor.dtype),
        ],
        grid=grid,
        num_tokens=num_tokens,
        row_id_stride_0=num_experts * 2 + 1,
        row_id_stride_1=1,
        workspace_stride=triton.next_power_of_2(num_experts * triton.cdiv(num_tokens, block_size)),
        block_size=block_size,
    )
    
    # Pass 3: make the row_id_map from sparse to dense structure
    grid = (num_tokens,)
    row_id_map = jt.triton_call(
        row_id_map,
        kernel=_row_id_map_pass_3_kernel,
        out_shape=[ShapeDtypeStruct(row_id_map.shape, row_id_map.dtype)],
        grid=grid,
        num_experts=num_experts,
        row_id_stride_0=num_experts * 2 + 1,
        row_id_stride_1=1,
        EXPERTS_LOAD_WIDTH=triton.next_power_of_2(num_experts),
    )[0]
    
    return row_id_map


def permute_with_mask_map(
    inp: jnp.ndarray,
    row_id_map: jnp.ndarray,
    probs: Optional[jnp.ndarray],
    num_tokens: int,
    num_experts: int,
    num_out_tokens: int,
    hidden_size: int,
) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
    """
    Permute the input tensor based on the row_id_map using JAX-Triton.
    
    Parameters
    ----------
    inp : jnp.ndarray
        Input tensor of shape `[num_tokens, hidden_size]`, on which permutation will be applied.
    row_id_map : jnp.ndarray
        The token to expert mapping tensor of shape `[num_tokens, num_experts * 2 + 1]`.
    probs : Optional[jnp.ndarray]
        The probabilities of the input tensor. If it is not None, it will be permuted.
    num_tokens : int
        Number of tokens in the input tensor.
    num_experts : int
        Number of experts in the input tensor.
    num_out_tokens : int
        Number of tokens in the permuted tensor.
    hidden_size : int
        Hidden size of the input tensor.
    
    Returns
    -------
    output : jnp.ndarray
        Permuted output tensor of shape `[num_out_tokens, hidden_size]`.
    permuted_probs : Optional[jnp.ndarray]
        Permuted probabilities if probs was provided, None otherwise.
    """
    output = jnp.zeros((num_out_tokens, hidden_size), dtype=inp.dtype)
    
    if probs is not None:
        permuted_probs = jnp.zeros((num_out_tokens,), dtype=probs.dtype)
    else:
        permuted_probs = None
    
    # Grid: one block per token, multiple blocks for hidden dimension
    def grid_fn(meta):
        return (num_tokens, triton.cdiv(hidden_size, meta['BLOCK_SIZE']))
    
    if probs is not None:
        output, permuted_probs = jt.triton_call(
            inp,
            output,
            row_id_map,
            probs,
            None,  # scale (not used for JAX, FP8 support different)
            permuted_probs,
            None,  # permuted_scale
            kernel=_permute_kernel,
            out_shape=[
                ShapeDtypeStruct(output.shape, output.dtype),
                ShapeDtypeStruct(permuted_probs.shape, permuted_probs.dtype),
            ],
            grid=grid_fn,
            num_experts=num_experts,
            hidden_size=hidden_size,
            scale_hidden_dim=0,  # Not used
            row_id_stride_0=row_id_map.strides[0] // row_id_map.dtype.itemsize,
            row_id_stride_1=row_id_map.strides[1] // row_id_map.dtype.itemsize,
            inp_stride_0=inp.strides[0] // inp.dtype.itemsize,
            inp_stride_1=inp.strides[1] // inp.dtype.itemsize,
            output_stride_0=hidden_size,
            output_stride_1=1,
            probs_stride_0=probs.strides[0] // probs.dtype.itemsize if probs.ndim > 1 else 1,
            probs_stride_1=probs.strides[1] // probs.dtype.itemsize if probs.ndim > 1 else 1,
            scale_stride_0=0,
            scale_stride_1=0,
            permuted_probs_stride_0=1,
            permuted_scale_stride_0=0,
            permuted_scale_stride_1=0,
            PERMUTE_PROBS=True,
            PERMUTE_SCALE=False,
        )
    else:
        output = jt.triton_call(
            inp,
            output,
            row_id_map,
            None,  # probs
            None,  # scale
            None,  # permuted_probs
            None,  # permuted_scale
            kernel=_permute_kernel,
            out_shape=[ShapeDtypeStruct(output.shape, output.dtype)],
            grid=grid_fn,
            num_experts=num_experts,
            hidden_size=hidden_size,
            scale_hidden_dim=0,
            row_id_stride_0=row_id_map.strides[0] // row_id_map.dtype.itemsize,
            row_id_stride_1=row_id_map.strides[1] // row_id_map.dtype.itemsize,
            inp_stride_0=inp.strides[0] // inp.dtype.itemsize,
            inp_stride_1=inp.strides[1] // inp.dtype.itemsize,
            output_stride_0=hidden_size,
            output_stride_1=1,
            probs_stride_0=0,
            probs_stride_1=0,
            scale_stride_0=0,
            scale_stride_1=0,
            permuted_probs_stride_0=0,
            permuted_scale_stride_0=0,
            permuted_scale_stride_1=0,
            PERMUTE_PROBS=False,
            PERMUTE_SCALE=False,
        )[0]
    
    return output, permuted_probs


def unpermute_with_mask_map(
    inp: jnp.ndarray,
    row_id_map: jnp.ndarray,
    merging_probs: Optional[jnp.ndarray],
    permuted_probs: Optional[jnp.ndarray],
    num_tokens: int,
    num_experts: int,
    hidden_size: int,
) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
    """
    Unpermute the input tensor based on the row_id_map using JAX-Triton.
    
    Parameters
    ----------
    inp : jnp.ndarray
        Input tensor of shape `[num_out_tokens, hidden_size]`.
    row_id_map : jnp.ndarray
        The token to expert mapping tensor of shape `[num_tokens, num_experts * 2 + 1]`.
    merging_probs : Optional[jnp.ndarray]
        The merging probabilities of the input tensor. If it is not None, it will be used as weights
        to reduce the unpermuted tokens.
    permuted_probs : Optional[jnp.ndarray]
        The permuted probabilities of the input tensor. If it is not None, it will be unpermuted.
    num_tokens : int
        Number of tokens in the permuted tensor.
    num_experts : int
        Number of experts in the permuted tensor.
    hidden_size : int
        Hidden size of the permuted tensor.
    
    Returns
    -------
    output : jnp.ndarray
        Unpermuted output tensor of shape `[num_tokens, hidden_size]`.
    unpermuted_probs : Optional[jnp.ndarray]
        Unpermuted probabilities if permuted_probs was provided, None otherwise.
    """
    output = jnp.zeros((num_tokens, hidden_size), dtype=inp.dtype)
    
    if permuted_probs is not None:
        unpermuted_probs = jnp.zeros((num_tokens, num_experts), dtype=permuted_probs.dtype)
    else:
        unpermuted_probs = None
    
    # Grid: one block per token, multiple blocks for hidden dimension
    def grid_fn(meta):
        return (num_tokens, triton.cdiv(hidden_size, meta['BLOCK_SIZE']))
    
    if permuted_probs is not None:
        output, unpermuted_probs = jt.triton_call(
            inp,
            output,
            row_id_map,
            merging_probs,
            permuted_probs,
            unpermuted_probs,
            kernel=_unpermute_kernel,
            out_shape=[
                ShapeDtypeStruct(output.shape, output.dtype),
                ShapeDtypeStruct(unpermuted_probs.shape, unpermuted_probs.dtype),
            ],
            grid=grid_fn,
            num_experts=num_experts,
            hidden_size=hidden_size,
            row_id_stride_0=row_id_map.strides[0] // row_id_map.dtype.itemsize,
            row_id_stride_1=row_id_map.strides[1] // row_id_map.dtype.itemsize,
            inp_stride_0=inp.strides[0] // inp.dtype.itemsize,
            inp_stride_1=inp.strides[1] // inp.dtype.itemsize,
            output_stride_0=output.strides[0] // output.dtype.itemsize,
            output_stride_1=output.strides[1] // output.dtype.itemsize,
            merging_probs_stride_0=merging_probs.strides[0] // merging_probs.dtype.itemsize if merging_probs is not None else 0,
            merging_probs_stride_1=merging_probs.strides[1] // merging_probs.dtype.itemsize if merging_probs is not None else 0,
            permuted_probs_stride_0=1,
            unpermuted_probs_stride_0=unpermuted_probs.strides[0] // unpermuted_probs.dtype.itemsize,
            unpermuted_probs_stride_1=unpermuted_probs.strides[1] // unpermuted_probs.dtype.itemsize,
            PROBS_LOAD_WIDTH=triton.next_power_of_2(num_experts),
            WITH_MERGING_PROBS=merging_probs is not None,
            PERMUTE_PROBS=True,
        )
    else:
        output = jt.triton_call(
            inp,
            output,
            row_id_map,
            merging_probs,
            None,  # permuted_probs
            None,  # unpermuted_probs
            kernel=_unpermute_kernel,
            out_shape=[ShapeDtypeStruct(output.shape, output.dtype)],
            grid=grid_fn,
            num_experts=num_experts,
            hidden_size=hidden_size,
            row_id_stride_0=row_id_map.strides[0] // row_id_map.dtype.itemsize,
            row_id_stride_1=row_id_map.strides[1] // row_id_map.dtype.itemsize,
            inp_stride_0=inp.strides[0] // inp.dtype.itemsize,
            inp_stride_1=inp.strides[1] // inp.dtype.itemsize,
            output_stride_0=output.strides[0] // output.dtype.itemsize,
            output_stride_1=output.strides[1] // output.dtype.itemsize,
            merging_probs_stride_0=merging_probs.strides[0] // merging_probs.dtype.itemsize if merging_probs is not None else 0,
            merging_probs_stride_1=merging_probs.strides[1] // merging_probs.dtype.itemsize if merging_probs is not None else 0,
            permuted_probs_stride_0=0,
            unpermuted_probs_stride_0=0,
            unpermuted_probs_stride_1=0,
            PROBS_LOAD_WIDTH=triton.next_power_of_2(num_experts),
            WITH_MERGING_PROBS=merging_probs is not None,
            PERMUTE_PROBS=False,
        )[0]
    
    return output, unpermuted_probs


def make_chunk_sort_map(
    split_sizes: jnp.ndarray,
    sorted_indices: jnp.ndarray,
    num_tokens: int,
    num_splits: int,
) -> jnp.ndarray:
    """
    Make a row_id_map for chunk sort using JAX-Triton.
    
    Parameters
    ----------
    split_sizes : jnp.ndarray
        The sizes of the chunks of shape `[num_splits,]`.
    sorted_indices : jnp.ndarray
        The indices of the sorted chunks of shape `[num_splits,]`.
    num_tokens : int
        Number of tokens in the input tensor.
    num_splits : int
        Number of splits of split_sizes and sorted_indices.
    
    Returns
    -------
    row_id_map : jnp.ndarray
        Row ID map for chunk sorting of shape `[num_tokens,]`.
    """
    row_id_map = jnp.zeros((num_tokens,), dtype=jnp.int32)
    grid = (num_tokens,)
    
    row_id_map = jt.triton_call(
        split_sizes,
        sorted_indices,
        row_id_map,
        kernel=_make_chunk_sort_map_kernel,
        out_shape=[ShapeDtypeStruct(row_id_map.shape, row_id_map.dtype)],
        grid=grid,
        num_splits=num_splits,
        IDX_LOAD_WIDTH=triton.next_power_of_2(num_splits),
    )[0]
    
    return row_id_map


def sort_chunks_by_map(
    inp: jnp.ndarray,
    row_id_map: jnp.ndarray,
    probs: Optional[jnp.ndarray],
    num_tokens: int,
    hidden_size: int,
    is_forward: bool,
) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
    """
    Sort chunks with row_id_map using JAX-Triton.
    
    Parameters
    ----------
    inp : jnp.ndarray
        Input tensor of shape `[num_tokens, hidden_size]`.
    row_id_map : jnp.ndarray
        The token to expert mapping tensor of shape `[num_tokens,]`.
    probs : Optional[jnp.ndarray]
        The probabilities of the input tensor. If it is not None, it will be permuted.
    num_tokens : int
        Number of tokens in the input tensor.
    hidden_size : int
        Hidden size of the input tensor.
    is_forward : bool
        Whether the sort is for forward or backward.
    
    Returns
    -------
    output : jnp.ndarray
        Sorted output tensor of shape `[num_tokens, hidden_size]`.
    permuted_probs : Optional[jnp.ndarray]
        Sorted probabilities if probs was provided, None otherwise.
    """
    output = jnp.zeros((num_tokens, hidden_size), dtype=inp.dtype)
    
    if probs is not None:
        permuted_probs = jnp.zeros((num_tokens,), dtype=probs.dtype)
    else:
        permuted_probs = None
    
    # Grid: one block per token, multiple blocks for hidden dimension
    def grid_fn(meta):
        return (num_tokens, triton.cdiv(hidden_size, meta['BLOCK_SIZE']))
    
    if probs is not None:
        output, permuted_probs = jt.triton_call(
            inp,
            output,
            row_id_map,
            probs,
            permuted_probs,
            kernel=_sort_chunks_by_map_kernel,
            out_shape=[
                ShapeDtypeStruct(output.shape, output.dtype),
                ShapeDtypeStruct(permuted_probs.shape, permuted_probs.dtype),
            ],
            grid=grid_fn,
            hidden_size=hidden_size,
            inp_stride_0=inp.strides[0] // inp.dtype.itemsize,
            inp_stride_1=inp.strides[1] // inp.dtype.itemsize,
            output_stride_0=output.strides[0] // output.dtype.itemsize,
            output_stride_1=output.strides[1] // output.dtype.itemsize,
            probs_stride_0=1,
            permuted_probs_stride_0=1,
            PERMUTE_PROBS=True,
            FORWARD=is_forward,
        )
    else:
        output = jt.triton_call(
            inp,
            output,
            row_id_map,
            None,  # probs
            None,  # permuted_probs
            kernel=_sort_chunks_by_map_kernel,
            out_shape=[ShapeDtypeStruct(output.shape, output.dtype)],
            grid=grid_fn,
            hidden_size=hidden_size,
            inp_stride_0=inp.strides[0] // inp.dtype.itemsize,
            inp_stride_1=inp.strides[1] // inp.dtype.itemsize,
            output_stride_0=output.strides[0] // output.dtype.itemsize,
            output_stride_1=output.strides[1] // output.dtype.itemsize,
            probs_stride_0=0,
            permuted_probs_stride_0=0,
            PERMUTE_PROBS=False,
            FORWARD=is_forward,
        )[0]
    
    return output, permuted_probs
