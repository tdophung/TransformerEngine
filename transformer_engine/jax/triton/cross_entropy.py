# Copyright (c) 2025-2028, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""JAX wrapper functions for Cross Entropy Triton kernels."""

import jax
import jax.numpy as jnp
from jax import ShapeDtypeStruct
from typing import Optional, Tuple
from functools import reduce, partial
from operator import mul

import triton
import jax_triton as jt

from transformer_engine.common.triton.cross_entropy import (
    online_softmax_kernel,
    cross_entropy_kernel,
    element_mul_kernel,
)

# The optimal maximum block size depends on your hardware, your kernel, and your dtype
MAX_FUSED_SIZE = 65536 // 2


def _get_process_group_info(axis_name: Optional[str]) -> Tuple[int, int]:
    """Get rank and world size from JAX's collective axis."""
    if axis_name is None:
        return 0, 1
    
    # Get the axis index (rank) and axis size (world_size)
    try:
        axis_index = jax.lax.axis_index(axis_name)
        axis_size = jax.lax.psum(1, axis_name)
        return axis_index, axis_size
    except:
        # If not in a pmap/shard_map context, return defaults
        return 0, 1


@partial(jax.custom_vjp, nondiff_argnums=(2, 3, 4, 5))
def cross_entropy_loss(
    _input: jnp.ndarray,
    target: jnp.ndarray,
    label_smoothing: float = 0.0,
    reduce_loss: bool = True,
    ignore_idx: int = -100,
    axis_name: Optional[str] = None,
) -> jnp.ndarray:
    """
    Compute cross entropy loss using Triton kernels.
    
    Parameters
    ----------
    _input : jnp.ndarray
        Input logits of shape (B, SQ, V) where B is batch size, SQ is sequence length,
        and V is vocabulary size.
    target : jnp.ndarray
        Target indices of shape (B, SQ) or (B * SQ,).
    label_smoothing : float, default=0.0
        Amount of label smoothing to apply.
    reduce_loss : bool, default=True
        Whether to reduce the loss to a scalar (mean) or return per-token loss.
    ignore_idx : int, default=-100
        Index to ignore in the loss computation.
    axis_name : Optional[str], default=None
        Name of the collective axis for distributed training (tensor parallelism).
    
    Returns
    -------
    jnp.ndarray
        The computed cross entropy loss. Scalar if reduce_loss=True, otherwise shape (B, SQ).
    """
    loss, _ = _cross_entropy_fwd(_input, target, label_smoothing, reduce_loss, ignore_idx, axis_name)
    return loss


def _cross_entropy_fwd(
    _input: jnp.ndarray,
    target: jnp.ndarray,
    label_smoothing: float,
    reduce_loss: bool,
    ignore_idx: int,
    axis_name: Optional[str], # For when we need to handle distributed training
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Forward implementation of Cross Entropy kernel."""
    
    # Get shape information
    B, SQ, V = _input.shape
    n_rows = B * SQ
    
    # Flatten target if needed
    target_flat = target.reshape(-1)
    assert target_flat.shape[0] == n_rows, "Each token needs a target token ID."

    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(V))
    
    rank, world_size = _get_process_group_info(axis_name)
    
    # Allocate outputs for the first kernel on the same device as input
    # Since I am assuming single GPU, this is correct. But otherwise, for explicit control:
    # device = _input.devices().pop()
    # m_d_X_y = jax.device_put(jnp.zeros(...), device)
    m_d_X_y = jnp.zeros(n_rows * 3, dtype=jnp.float32)

    # TEDDY: JAX arrays are contiguous by default, so no explicit contiguous conversion is needed if you are wondering why this is different from the Pytorch lines
    
    # Compute strides (JAX arrays don't have .strides attribute)
    # For a 3D array (B, SQ, V), stride to move along SQ dimension is V
    X_stride = V  # Stride to move to next sequence position
    Y_stride = 1  # Target is 1D after flatten, always contiguous
    m_d_X_y_stride = 1  # 1D array, always contiguous

    # Call online_softmax_kernel to compute m, d, and X_y values
    # The kernel writes to m_d_X_y_ptr (modifies it in-place)
    m_d_X_y = jt.triton_call(
        _input,          # Array arg 0: Read-only
        X_stride,        # Scalar arg
        target_flat,     # Array arg 1: Read-only
        Y_stride,        # Scalar arg
        m_d_X_y,         # Array arg 2: Modified in-place (becomes output)
        m_d_X_y_stride,  # Scalar arg
        rank,            # Scalar arg
        V,               # Scalar arg: n_cols
        kernel=online_softmax_kernel,
        out_shape=[ShapeDtypeStruct(m_d_X_y.shape, m_d_X_y.dtype)],
        input_output_aliases={2: 0},  # Array index 2 (m_d_X_y) → Output 0
        grid=(n_rows,),
        num_warps=32,
        BLOCK_SIZE=BLOCK_SIZE,  # Constexpr - keep as keyword arg
    )[0]
    
    # Gather m_d_X_y across all ranks if using tensor parallelism
    # Assume sinle GPU for now so the if case is not effective
    # if world_size > 1 and axis_name is not None:
    #     # All-gather the m/d/X_y values
    #     m_d_X_y_gathered = jax.lax.all_gather(m_d_X_y.reshape(n_rows, 3), axis_name, axis=0, tiled=True)
    #     m_d_X_y_gathered = m_d_X_y_gathered.reshape(-1)
    # else:
    m_d_X_y_gathered = m_d_X_y
    
    # Allocate outputs for the cross entropy kernel
    loss_1d = jnp.zeros(n_rows, dtype=jnp.float32)
    grad_input = jnp.zeros_like(_input)
    
    # Strides for the second kernel call
    loss_stride = 1  # 1D array, always contiguous
    m_d_X_y_stride_gathered = 1  # 1D array, always contiguous
    
    # Call cross_entropy_kernel to compute loss and gradients
    # The kernel modifies X_ptr (gradients) and loss_ptr in-place
    # Use input_output_aliases to map modified inputs to outputs
    loss_1d, grad_input = jt.triton_call(
        _input,          # Array arg 0: Modified to become grad_input (output 1)
        X_stride,        # Scalar arg
        target_flat,     # Array arg 1: Read-only
        Y_stride,        # Scalar arg
        loss_1d,         # Array arg 2: Modified to become loss output (output 0)
        loss_stride,     # Scalar arg
        m_d_X_y_gathered,  # Array arg 3: Read-only
        m_d_X_y_stride_gathered,  # Scalar arg
        rank,            # Scalar arg
        world_size,      # Scalar arg
        ignore_idx,      # Scalar arg
        V,               # Scalar arg: n_cols
        n_rows,          # Scalar arg: n_non_ignore
        kernel=cross_entropy_kernel,
        out_shape=[
            ShapeDtypeStruct(loss_1d.shape, loss_1d.dtype),      # Output 0: loss
            ShapeDtypeStruct(_input.shape, _input.dtype),        # Output 1: gradients
        ],
        input_output_aliases={
            2: 0,  # Array index 2 (loss_1d) → Output 0 (loss)
            0: 1,  # Array index 0 (_input) → Output 1 (grad_input)
        },
        grid=(n_rows,),
        num_warps=32,
        reduce_loss=reduce_loss,        # Constexpr - keep as keyword
        label_smoothing=label_smoothing,  # Constexpr - keep as keyword
        BLOCK_SIZE=BLOCK_SIZE,          # Constexpr - keep as keyword
    )
    
    # Compute final loss
    if reduce_loss:
        loss = jnp.sum(loss_1d) / n_rows
    else:
        loss = loss_1d.reshape(B, SQ)
    
    return loss, grad_input


def _cross_entropy_bwd(
    label_smoothing: float,
    reduce_loss: bool,
    ignore_idx: int,
    axis_name: Optional[str],
    res: jnp.ndarray,
    g: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Backward implementation of cross entropy loss kernel."""
    
    grad_input = res  # The gradient was computed in forward pass
    
    # Scale grad_input by grad_output
    # Always scale to avoid control flow issues with JAX tracing
    # (even if g is 1.0, multiplication is harmless)
    B, SQ, V = grad_input.shape
    n_rows = B * SQ
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(V))
    
    # Compute stride for grad_input (same as before: V)
    X_stride = V
    
    # Prepare grad_output for multiplication
    # Handle both scalar and per-token gradients
    grad_output_reshaped = g.reshape(-1) if g.size > 1 else g
    grad_output_stride = 1 if g.size > 1 else 0
    
    grad_input = jt.triton_call(
        grad_input,      # Array arg 0: Modified in-place (becomes output)
        X_stride,        # Scalar arg
        grad_output_reshaped,  # Array arg 1: Read-only
        grad_output_stride,    # Scalar arg
        V,               # Scalar arg: n_cols
        kernel=element_mul_kernel,
        out_shape=[ShapeDtypeStruct(grad_input.shape, grad_input.dtype)],
        input_output_aliases={0: 0},  # Array index 0 (grad_input) → Output 0
        grid=(n_rows,),
        num_warps=32,
        BLOCK_SIZE=BLOCK_SIZE,  # Constexpr - keep as keyword arg
    )[0]
    
    # Return gradients for _input and target (target gradient is None)
    return grad_input, None


# Register the custom VJP
cross_entropy_loss.defvjp(_cross_entropy_fwd, _cross_entropy_bwd)
 