# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Dense layer transformation operations for Transformer Engine in JAX.

This module provides optimized dense layer transformation operations for transformer
architectures, including support for quantization and automatic differentiation.
It implements matrix multiplication with optional bias addition and supports
customizable contracting dimensions for flexible tensor operations.
"""

from typing import Tuple, Sequence
from functools import partial
import warnings
import jax
import jax.numpy as jnp

from . import cpp_extensions as tex
from .cpp_extensions.amax import AmaxScope
from .quantize import (
    ScaledTensor,
    QuantizerSet,
    noop_quantizer_set,
    with_sharding_constraint_by_logical_axes,
    TensorUsage,
)
from .quantize.tensor import GroupedScaledTensor1x


def _all_gather_kernel(kernel, mesh_axis, axis_idx):
    assert mesh_axis is not None
    assert 0 < axis_idx < len(kernel.shape)

    # TODO(Ming Hunag): Add a condition branch for with/without shmap.
    kernel_shape = kernel.shape
    kernel_whole_shape = (*kernel_shape[:axis_idx], -1, *kernel_shape[axis_idx + 1 :])
    global_kernel = jax.lax.all_gather(kernel, mesh_axis, axis=axis_idx)
    global_kernel = global_kernel.reshape(*kernel_whole_shape)
    return global_kernel


def _psum_scatter_kernel(kernel, scattered_kernel_shape, mesh_axis, axis_idx):
    assert mesh_axis is not None
    assert 0 < axis_idx < len(scattered_kernel_shape)

    # TODO(Ming Hunag): Add a condition branch for with/without shmap.
    kernel = kernel.reshape(
        *scattered_kernel_shape[:axis_idx],
        -1,
        scattered_kernel_shape[axis_idx],
        *scattered_kernel_shape[axis_idx + 1 :],
    )
    kernel = jax.lax.psum_scatter(kernel, mesh_axis, scatter_dimension=axis_idx)
    kernel = kernel.reshape(scattered_kernel_shape)
    return kernel


# MXFP8 block layout constants (mirrors BlockScalingModeMetadataImpl).
# block_dims = (1, 32). block_alignment = (128, 4). The 2D
# ``(n_block_x_M_side, n_block_y_K_side)`` scale view layout is:
#
#                          block_x   alignment_x   block_y   alignment_y
#   Rowwise (default)         1         128          32          4
#   Colwise (transposed)     32           4           1        128
#
# In both cases ``block_x * alignment_x = 128`` and
# ``block_y * alignment_y = 128``. So a single ``per-shard size %
# 128 == 0`` guard on the FSDP-sharded axis covers both
# ``is_colwise`` values AND both M-side / K-side FSDP placements. It
# guarantees: (a) no block straddles peers (per-shard rows are a
# multiple of block_x; per-shard cols are a multiple of block_y), and
# (b) per-shard ``n_block_x`` / ``n_block_y`` carries no alignment
# padding, so the per-shard scale layout is exactly
# ``(G * M_per_shard/block_x, K_per_shard/block_y)`` and AG
# composes cleanly.
_MXFP8_PER_SHARD_MIN_ALIGN = 128


def _mxfp8_block_y_alignment_y(is_colwise: bool) -> Tuple[int, int]:
    """Return ``(block_y, alignment_y)`` for the K-side of MXFP8 scales."""
    return (1, 128) if is_colwise else (32, 4)


def _mxfp8_block_x_alignment_x(is_colwise: bool) -> Tuple[int, int]:
    """Return ``(block_x, alignment_x)`` for the M-side of MXFP8 scales."""
    return (32, 4) if is_colwise else (1, 128)


def _all_gather_grouped_scaled_tensor_1x(t, mesh_axis, axis_idx):
    """All-gather a per-shard ``GroupedScaledTensor1x`` along the kernel's FSDP axis.

    Used by ``grouped_dense`` when the **kernel** is FSDP-sharded and the
    caller has already quantized per shard via
    ``tex.grouped_quantize(..., amax_scope=AmaxScope.FSDP)``. ``t`` always
    refers to a kernel-side tensor (the activations are *never* AG'd; they
    flow into the GEMM on their own EP shard).

    On-device, ``GroupedScaledTensor1x.data`` is flat 1D (see the
    ``ndim == 1`` invariant in its ``__post_init__``); the multi-D layout
    is encoded in ``original_shape``. We therefore reshape ``t.data`` to
    ``t.original_shape``, all-gather along ``axis_idx`` (the kernel dim
    that FSDP sliced), reflatten, and rewrap with the full-shape
    ``original_shape``.

    Per scaling mode:

    * **Tensor scaling** (``CurrentScaling``, ``DelayedScaling``): one
      ``scale_inv`` per expert, made FSDP-consistent by the ``pmax`` inside
      ``grouped_quantize``. We reuse it as-is; only ``data`` is gathered.

    * **MXFP8 block scaling** (rowwise *and* colwise, since the forward
      GEMM consumes the colwise tensor as its RHS): per-block
      ``scale_inv``. The single per-shard ``size % 128 == 0`` guard
      enforced by ``MoEBlock._assert_kernel_fsdp_alignment`` is the same
      for both M-side and K-side FSDP placements (see the constants
      block above for why), and unlocks both:

      - **K-side FSDP** (``axis_idx >= flatten_axis``): per-shard
        ``n_block_y = K_per_shard/block_y`` carries no alignment
        padding. Reshape ``scale_inv`` to
        ``(G * M/block_x, n_block_y_per_shard)``, AG along axis 1,
        reflatten.

      - **M-side FSDP** (``axis_idx < flatten_axis``): per-shard
        ``n_block_x = G * M_per_shard/block_x`` carries no per-group
        padding (each group's ``M_per_shard`` is mult of
        ``block_x * alignment_x = 128``). Reshape ``scale_inv`` to
        ``(G, M_per_shard/block_x, n_block_y)``, AG along axis 1
        (the per-group M sub-axis), reflatten. Restricted to the
        single-non-G-M-dim layout ``(G, M, K_dims...)`` (i.e.
        ``axis_idx == 1`` and ``flatten_axis == 2``); other layouts
        require a non-trivial reshape and aren't yet implemented.
    """
    assert isinstance(t, GroupedScaledTensor1x)
    assert mesh_axis is not None
    assert 0 < axis_idx < len(t.original_shape)

    reshaped = t.data.reshape(t.original_shape)
    gathered_data = jax.lax.all_gather(reshaped, mesh_axis, axis=axis_idx, tiled=True)
    full_orig_shape = (
        *t.original_shape[:axis_idx],
        gathered_data.shape[axis_idx],
        *t.original_shape[axis_idx + 1 :],
    )

    if t.scaling_mode.is_tensor_scaling():
        gathered_scale_inv = t.scale_inv
    elif t.scaling_mode.is_mxfp8_scaling:
        per_shard_size = t.original_shape[axis_idx]
        assert per_shard_size % _MXFP8_PER_SHARD_MIN_ALIGN == 0, (
            f"MXFP8 + FSDP requires per-shard size on axis {axis_idx} to be "
            f"a multiple of {_MXFP8_PER_SHARD_MIN_ALIGN} (= block_x * "
            f"alignment_x on the M side, = block_y * alignment_y on the K "
            f"side, both 128 for rowwise and colwise); got per-shard size "
            f"{per_shard_size}. This guarantees no block straddles peers "
            "AND per-shard n_block_x / n_block_y carries no alignment "
            "padding so the scale_inv AG composes cleanly."
        )

        block_y, _alignment_y = _mxfp8_block_y_alignment_y(t.is_colwise)
        block_x, _alignment_x = _mxfp8_block_x_alignment_x(t.is_colwise)
        flattened_k_per_shard = 1
        for d in t.original_shape[t.flatten_axis :]:
            flattened_k_per_shard *= d
        n_block_y_per_shard = flattened_k_per_shard // block_y

        scale_total = t.scale_inv.shape[0]
        assert scale_total % n_block_y_per_shard == 0, (
            f"scale_inv flat size {scale_total} not divisible by "
            f"n_block_y_per_shard={n_block_y_per_shard} "
            f"(is_colwise={t.is_colwise}, block_y={block_y}); cannot "
            "reshape to 2D"
        )
        padded_n_block_x = scale_total // n_block_y_per_shard

        if axis_idx >= t.flatten_axis:
            # K-side: AG along the n_block_y axis of the 2D view.
            scale_2d = t.scale_inv.reshape((padded_n_block_x, n_block_y_per_shard))
            gathered_scale_2d = jax.lax.all_gather(
                scale_2d, mesh_axis, axis=1, tiled=True
            )
            gathered_scale_inv = gathered_scale_2d.reshape(-1)
        else:
            # M-side: split the flattened n_block_x into (G, M_per_shard/block_x)
            # so the AG concatenates per-group M slices in full-kernel row order.
            # Restricted to single-non-G-M-dim layouts (G, M, K_dims...).
            if axis_idx != 1 or t.flatten_axis != 2:
                raise NotImplementedError(
                    f"MXFP8 + FSDP M-side AG only supports the single-M-dim "
                    f"layout (kernel.shape == (G, M, K_dims...) with FSDP on "
                    f"axis 1). Got original_shape={t.original_shape}, "
                    f"axis_idx={axis_idx}, flatten_axis={t.flatten_axis}. "
                    "Multi-M-dim layouts need a non-trivial reshape (un-flatten "
                    "the per-group M dims, AG on the FSDP-target dim, "
                    "reflatten); not implemented yet."
                )
            g = t.original_shape[0]
            m_per_shard = t.original_shape[1]
            n_block_x_per_group_per_shard = m_per_shard // block_x
            expected_scale_total = g * n_block_x_per_group_per_shard * n_block_y_per_shard
            assert scale_total == expected_scale_total, (
                f"per-shard scale_inv layout mismatch under MXFP8 M-side AG: "
                f"got scale_total={scale_total}, expected G * "
                f"(M_per_shard/block_x) * n_block_y = {g} * "
                f"{n_block_x_per_group_per_shard} * {n_block_y_per_shard} = "
                f"{expected_scale_total}. The MoEBlock-level mult-of-128 "
                "alignment guard should have caught this; please file a bug."
            )
            scale_3d = t.scale_inv.reshape(
                (g, n_block_x_per_group_per_shard, n_block_y_per_shard)
            )
            gathered_scale_3d = jax.lax.all_gather(
                scale_3d, mesh_axis, axis=1, tiled=True
            )
            gathered_scale_inv = gathered_scale_3d.reshape(-1)
    else:
        raise NotImplementedError(
            f"FSDP all-gather for scaling_mode={t.scaling_mode} is not "
            "implemented. Currently supported: tensor scaling (CurrentScaling, "
            "DelayedScaling) and MXFP8 (M-side and K-side FSDP)."
        )

    return GroupedScaledTensor1x(
        data=gathered_data.reshape(-1),
        scale_inv=gathered_scale_inv,
        amax=t.amax,
        first_dims=t.first_dims,
        last_dims=t.last_dims,
        scaling_mode=t.scaling_mode,
        dq_dtype=t.dq_dtype,
        _dq_func=t._dq_func,
        is_colwise=t.is_colwise,
        data_layout=t.data_layout,
        flatten_axis=t.flatten_axis,
        original_shape=full_orig_shape,
        pre_swizzled=t.pre_swizzled,
    )


def dense(
    x: jnp.ndarray,
    kernel: jnp.ndarray,
    bias: jnp.ndarray = None,
    contracting_dims: Tuple[Sequence[int], Sequence[int]] = ((1,), (0,)),
    transpose_batch_sequence: bool = False,
    input_axes: Tuple[str, ...] = None,
    kernel_axes: Tuple[str, ...] = None,
    output_axes: Tuple[str, ...] = None,
    collective_op_set: tex.CollectiveOpSet = tex.noop_collective_op_set,
    quantizer_set: QuantizerSet = noop_quantizer_set,
):
    """Perform dense layer transformation with optional quantization.

    This function implements matrix multiplication with optional bias addition,
    supporting quantization and custom contracting dimensions. It's optimized
    for transformer architectures and supports automatic differentiation.

    Args:
        x: Input tensor
        kernel: Weight matrix for the dense layer transformation
        bias: Optional bias tensor to add after the transformation
        contracting_dims: Tuple of sequences specifying which dimensions to contract
        transpose_batch_sequence: Transpose the batch and sequence dimensions of the input tensor.
        input_axes: Logical axes for sharding the activation input
        kernel_axes: Logical axes for sharding the weight matrix
        output_axes: Logical axes for sharding the output
        collective_op_set: A set of CollectiveOp objects for forward and backward passes.
        quantizer_set: QuantizerSet which contains quantizers for different tensor types

    Returns:
        Transformed output tensor
    """
    if transpose_batch_sequence:
        warnings.warn("transpose_batch_sequence is not well tested, use with caution!")

    if collective_op_set != tex.noop_collective_op_set and not output_axes:
        warnings.warn(
            "Collective GEMM with Shardy propagation may produce an incorrect sharding pattern"
            " for the output. Set `output_axes` to apply the correct sharding constraint.",
            UserWarning,
        )

    if quantizer_set == noop_quantizer_set:
        input_dtype = x.dtype
        kernel = kernel.astype(input_dtype)

    output = _dense(
        x,
        kernel,
        bias,
        contracting_dims,
        transpose_batch_sequence,
        input_axes,
        kernel_axes,
        output_axes,
        collective_op_set,
        quantizer_set,
    )
    return output


@partial(jax.custom_vjp, nondiff_argnums=(3, 4, 5, 6, 7, 8))
def _dense(
    x,
    kernel,
    bias,
    contracting_dims,
    transpose_batch_sequence,
    input_axes,
    kernel_axes,
    output_axes,
    collective_op_set,
    quantizer_set,  # need to be a diff_arg for DelayedScaling state management
):
    """Internal implementation of dense layer transformation with custom VJP.

    This function implements the core dense layer transformation logic with support
    for custom vector-Jacobian product (VJP) for automatic differentiation.

    Args:
        x: Input tensor
        kernel: Weight matrix
        bias: Optional bias tensor
        contracting_dims: Contracting dimensions specification
        transpose_batch_sequence: Transpose the batch and sequence dimensions of the input tensor.
        input_axes: Logical axes for sharding the activation input
        output_axes: Logical axes for sharding the output_axes
        kernel_axes: Logical axes for sharding the weight matrix
        collective_op_set: A set of CollectiveOp objects for forward and backward passes.
        quantizer_set: QuantizerSet which contains quantizers for different tensor types

    Returns:
        Transformed output tensor
    """
    output, _ = _dense_fwd_rule(
        x,
        kernel,
        bias,
        contracting_dims,
        transpose_batch_sequence,
        input_axes,
        kernel_axes,
        output_axes,
        collective_op_set,
        quantizer_set,
    )
    return output


def _dense_fwd_rule(
    x,
    kernel,
    bias,
    contracting_dims,
    transpose_batch_sequence,
    input_axes,
    kernel_axes,
    output_axes,
    collective_op_set,
    quantizer_set,
):
    """Forward pass rule for dense layer transformation.

    Returns:
        Tuple of (output, context) for backward pass
    """
    x_contracting_dims, k_contracting_dims = map(
        tex.sanitize_dims, (x.ndim, kernel.ndim), contracting_dims
    )

    # Check supported input layout
    x_is_transposed = x.ndim - 1 not in x_contracting_dims
    k_is_transposed = kernel.ndim - 1 in k_contracting_dims
    assert (
        not x_is_transposed and not k_is_transposed
    ), "Dense layer only supports `NN` layout inputs, i.e. non-transposed X and Kernel."

    flatten_axis_x = -len(x_contracting_dims)
    flatten_axis_k = len(k_contracting_dims) - len(kernel.shape)

    casted_x = tex.quantize(
        x,
        flatten_axis=flatten_axis_x,
        quantizer=quantizer_set.x,
        amax_scope=AmaxScope.TPSP,
        transpose_batch_sequence=transpose_batch_sequence,
    )
    casted_x = with_sharding_constraint_by_logical_axes(casted_x, input_axes)

    casted_kernel = tex.quantize(
        kernel,
        flatten_axis=flatten_axis_k,
        quantizer=quantizer_set.kernel,
        amax_scope=AmaxScope.FSDP,
    )
    casted_kernel = with_sharding_constraint_by_logical_axes(casted_kernel, kernel_axes)

    # GEMM NN
    output = tex.gemm(
        casted_x.get_tensor(usage=TensorUsage.LHS),
        casted_kernel.get_tensor(usage=TensorUsage.RHS),
        bias=bias,
        contracting_dims=(x_contracting_dims, k_contracting_dims),
        transpose_batch_sequence=transpose_batch_sequence,
        collective_op=collective_op_set.forward,
    )
    output = with_sharding_constraint_by_logical_axes(output, output_axes)

    has_bias = bias is not None
    ctx = (
        casted_x.get_tensor(usage=TensorUsage.LHS_TRANS).checkpoint(quantizer_set.x),
        casted_kernel.get_tensor(usage=TensorUsage.RHS_TRANS).checkpoint(quantizer_set.kernel),
        x.shape,
        kernel.shape,
        quantizer_set,
        flatten_axis_k,
        has_bias,
    )
    return output, ctx


def _dense_bwd_rule(
    contracting_dims,
    transpose_batch_sequence,
    input_axes,
    kernel_axes,
    output_axes,
    collective_op_set,
    ctx,
    grad,
):
    """Backward pass rule for dense layer transformation.

    Returns:
        Tuple of gradients with respect to inputs
    """
    (
        casted_x_lhs,
        casted_kernel_rhs,
        x_shape,
        kernel_shape,
        quantizer_set,
        flatten_axis_k,
        has_bias,
    ) = ctx
    grad = with_sharding_constraint_by_logical_axes(grad, output_axes)

    fwd_x_contracting_dims, fwd_k_contracting_dims = map(
        tex.sanitize_dims, (casted_x_lhs.ndim, casted_kernel_rhs.ndim), contracting_dims
    )

    casted_grad, dbias = tex.quantize_dbias(
        grad,
        is_dbias=has_bias,
        flatten_axis=flatten_axis_k,
        quantizer=quantizer_set.dgrad,
        amax_scope=AmaxScope.TPSP,
        transpose_batch_sequence=transpose_batch_sequence,
    )

    # GEMM NT
    # k_non_contracting_dims calibrated with the shape difference of grad.ndim vs kernel.ndim
    g_contracting_dim = tuple(
        range(grad.ndim - len(kernel_shape) + len(fwd_k_contracting_dims), grad.ndim)
    )
    # k_non_contracting_dims
    k_contracting_dim = tuple(
        dim for dim in range(len(kernel_shape)) if dim not in fwd_k_contracting_dims
    )

    dgrad = tex.gemm(
        casted_grad.get_tensor(usage=TensorUsage.LHS),
        casted_kernel_rhs,
        contracting_dims=(g_contracting_dim, k_contracting_dim),
        transpose_batch_sequence=transpose_batch_sequence,
        collective_op=collective_op_set.backward,
    )

    # GEMM TN
    # x_non_contracting_dims
    g_contracting_dim = x_contracting_dim = tuple(
        range(0, len(x_shape) - len(fwd_x_contracting_dims))
    )

    wgrad = tex.gemm(
        casted_x_lhs,
        casted_grad.get_tensor(usage=TensorUsage.RHS),
        contracting_dims=(x_contracting_dim, g_contracting_dim),
        transpose_batch_sequence=transpose_batch_sequence,
    )

    dgrad = with_sharding_constraint_by_logical_axes(dgrad, input_axes)
    wgrad = with_sharding_constraint_by_logical_axes(wgrad, kernel_axes)

    return dgrad, wgrad, dbias, quantizer_set


_dense.defvjp(_dense_fwd_rule, _dense_bwd_rule)


def grouped_dense(
    x: jnp.ndarray,
    kernel: jnp.ndarray,
    group_sizes: jnp.ndarray,
    contracting_dims: Tuple[Sequence[int], Sequence[int]] = ((1,), (1,)),
    bias: jnp.ndarray = None,
    precision: jax.lax.Precision = jax.lax.Precision.DEFAULT,
    preferred_element_type: jnp.dtype = None,
    group_offset: jnp.array = None,
    quantizer_set: QuantizerSet = noop_quantizer_set,
    kernel_fsdp_info: Tuple[str, int] = (None, -1),
):
    """
    Perform grouped dense (linear) layer transformation with optional quantization.

    Args:
        x: Input tensor of shape (M, K)
        kernel: Weight matrix of shape (G, K, N)
        group_sizes: 1D array of shape (G,) specifying the size of each group
        contracting_dims: Tuple of sequences specifying which dimensions to contract
                          (currently only supports ((1,), (1,)))
        bias: Bias tensor of shape (G, N)
        precision: JAX precision for the GEMM operation
        preferred_element_type: Preferred data type for the output tensor
        group_offset: 1D array containing offsets for each group (not yet implemented)
        quantizer_set: Set of quantizers for FP8 quantization of the input and output
        kernel_fsdp_info: A tuple containing FSDP-related information for a weight matrix
                          represented in the format (str, int). The first element is the
                          FSDP mesh axis, and the second element is the dimension along
                          which the weight is sharded.

    Returns:
        A jnp.ndarray containing the result of the grouped linear operation
    """
    output = _grouped_dense(
        x,
        kernel,
        group_sizes,
        contracting_dims,
        bias,
        precision,
        preferred_element_type,
        group_offset,
        quantizer_set,
        kernel_fsdp_info,
    )
    return output


@partial(jax.custom_vjp, nondiff_argnums=(3, 5, 6, 7, 9))
def _grouped_dense(
    x,
    kernel,
    group_sizes,
    contracting_dims,
    bias,
    precision,
    preferred_element_type,
    group_offset,
    quantizer_set,
    kernel_fsdp_info,
):
    output, _ = _grouped_dense_fwd_rule(
        x,
        kernel,
        group_sizes,
        contracting_dims,
        bias,
        precision,
        preferred_element_type,
        group_offset,
        quantizer_set,
        kernel_fsdp_info,
    )
    return output


def _grouped_dense_fwd_rule(
    x,
    kernel,
    group_sizes,
    contracting_dims,
    bias,
    precision,
    preferred_element_type,
    group_offset,
    quantizer_set,
    kernel_fsdp_info,
):
    use_bias = bias is not None

    kernel_fsdp_mesh_axis, kernel_fsdp_axis_idx = kernel_fsdp_info
    kernel_fsdp_enabled = kernel_fsdp_mesh_axis is not None
    has_kernel_quantizer = (
        kernel_fsdp_enabled
        and quantizer_set is not None
        and quantizer_set.kernel is not None
    )
    if kernel_fsdp_enabled and not has_kernel_quantizer:
        # No FP8 quantization (e.g. bf16 autocast disabled): the
        # "quantize before AG" win collapses to "AG inside body vs.
        # outside body", which is semantically identical. Plain-AG the
        # kernel up front so the rest of the fwd code sees a full
        # kernel and follows the legacy path. The bwd still needs
        # ``psum_scatter`` on wgrad because the kernel input was
        # per-shard.
        kernel = jax.lax.all_gather(
            kernel,
            kernel_fsdp_mesh_axis,
            axis=kernel_fsdp_axis_idx,
            tiled=True,
        )
    if has_kernel_quantizer:
        # FP8 / MXFP8 "quantize before AG" path. Supported scaling
        # modes: tensor scaling (CurrentScaling, DelayedScaling) and
        # MXFP8 K-side FSDP. See ``_all_gather_grouped_scaled_tensor_1x``
        # for the per-mode contract and the M-side MXFP8 limitation.
        kernel_scaling_mode = quantizer_set.kernel.scaling_mode
        assert (
            kernel_scaling_mode.is_tensor_scaling() or kernel_scaling_mode.is_mxfp8_scaling
        ), (
            "quantize-before-FSDP-AG (kernel_fsdp_info != (None, -1)) currently "
            "supports tensor-scaling FP8 (CurrentScaling / DelayedScaling) and "
            f"MXFP8 (K-side FSDP only); got kernel scaling_mode={kernel_scaling_mode}"
        )

    x_contracting_dims, k_contracting_dims = contracting_dims
    flatten_axis_x = -len(x_contracting_dims)
    flatten_axis_k = len(k_contracting_dims) - len(kernel.shape) + 1  # +1 for G axis

    casted_x = tex.grouped_quantize(
        x,
        quantizer_set.x,
        group_sizes,
        flatten_axis=flatten_axis_x,
    )

    casted_kernel = tex.grouped_quantize(
        kernel,
        quantizer_set.kernel,
        flatten_axis=flatten_axis_k,
        amax_scope=AmaxScope.FSDP if has_kernel_quantizer else AmaxScope.LOCAL,
    )
    contracting_dims = (x_contracting_dims, k_contracting_dims)

    # For x_contracting_dims == (1,) and k_contracting_dims == (1,), we should have
    # rowwise_casted_x.original_shape == (M, K)
    # colwise_casted_kernel.original_shape == (G, N, K)
    grouped_gemm_x = casted_x.get_tensor(usage=TensorUsage.LHS)
    # Checkpoint the rowwise inputs so that te_grouped_quantize_ffi can be DCE'd in the
    # backward-scan remat block.  Without this, JAX would re-run the quantize kernel to
    # obtain grouped_gemm_x / grouped_gemm_kernel for the forward-GEMM recomputation even
    # though the colwise residuals (ctx_x / ctx_kernel) are already saved.  With both
    # orientations checkpointed, all outputs of the custom-call become dead in the remat trace.
    grouped_gemm_x = (
        grouped_gemm_x.checkpoint(quantizer_set.x)
        if isinstance(grouped_gemm_x, ScaledTensor)
        else grouped_gemm_x
    )
    ctx_x = casted_x.get_tensor(usage=TensorUsage.LHS_TRANS)
    ctx_kernel = casted_kernel.get_tensor(usage=TensorUsage.RHS_TRANS)

    grouped_gemm_kernel = casted_kernel.get_tensor(usage=TensorUsage.RHS)
    if has_kernel_quantizer:
        # All-gather the per-shard FP8 RHS data (rowwise for tensor
        # scaling, colwise for MXFP8 since ``get_tensor(usage=RHS)``
        # returns colwise under MXFP8) along the kernel's FSDP axis.
        # ``scale_inv`` is handled inside the helper: reused as-is for
        # tensor scaling (peer-identical after the ``pmax`` inside
        # ``grouped_quantize``) or AG'd along its K-block axis for
        # MXFP8. ``ctx_kernel`` below is intentionally left per-shard;
        # the bwd rule AGs it separately for the dgrad GEMM.
        grouped_gemm_kernel = _all_gather_grouped_scaled_tensor_1x(
            grouped_gemm_kernel, kernel_fsdp_mesh_axis, kernel_fsdp_axis_idx
        )
    grouped_gemm_kernel = (
        grouped_gemm_kernel.checkpoint(quantizer_set.kernel)
        if isinstance(grouped_gemm_kernel, ScaledTensor)
        else grouped_gemm_kernel
    )
    output = tex.grouped_gemm(
        grouped_gemm_x,
        grouped_gemm_kernel,
        contracting_dims=contracting_dims,
        bias=bias,
        precision=precision,
        preferred_element_type=preferred_element_type,
        group_offset=group_offset,
    )

    ctx = (
        group_sizes,
        ctx_x.checkpoint(quantizer_set.x) if isinstance(ctx_x, ScaledTensor) else ctx_x,
        (
            ctx_kernel.checkpoint(quantizer_set.kernel)
            if isinstance(ctx_kernel, ScaledTensor)
            else ctx_kernel
        ),
        x.shape,
        kernel.shape,
        use_bias,
        quantizer_set,
        flatten_axis_k,
    )
    return output, ctx


def _grouped_dense_bwd_rule(
    contracting_dims, precision, preferred_element_type, group_offset, kernel_fsdp_info, ctx, grad
):
    kernel_fsdp_mesh_axis, kernel_fsdp_axis_idx = kernel_fsdp_info
    kernel_fsdp_enabled = kernel_fsdp_mesh_axis is not None

    fwd_x_contracting_dims, fwd_k_contracting_dims = contracting_dims

    (
        group_sizes,
        ctx_x,
        ctx_kernel,
        x_shape,
        kernel_shape,
        use_bias,
        quantizer_set,
        flatten_axis_k,
    ) = ctx

    # The 1 in range is for excluding the group dimension (shall we use the hardcoded results below?)
    # g_contracting_dim = (1, )
    # k_contracting_dim = (2, )
    g_contracting_dim = tuple(
        range(1 + grad.ndim - len(kernel_shape) + len(fwd_k_contracting_dims), grad.ndim)
    )
    k_contracting_dim = tuple(
        dim for dim in range(1, len(kernel_shape)) if dim not in fwd_k_contracting_dims
    )

    casted_grad = tex.grouped_quantize(
        grad, quantizer_set.dgrad, group_sizes, flatten_axis=flatten_axis_k
    )

    dgrad_contracting_dims = (g_contracting_dim, k_contracting_dim)
    dgrad_grad = casted_grad.get_tensor(usage=TensorUsage.LHS)
    dgrad_kernel_T = ctx_kernel
    if kernel_fsdp_enabled and isinstance(ctx_kernel, GroupedScaledTensor1x):
        # FP8 / MXFP8 path: fwd saved ctx_kernel per-shard. Mirror the
        # fwd AG on the colwise (RHS_TRANS) data so the dgrad GEMM sees
        # the full kernel and produces a full-K dgrad (matching x's
        # full-K spec; x is FSDP-sharded on batch, not on K). The helper
        # handles both rowwise and colwise via its is_colwise branch;
        # original_shape and flatten_axis are layout-invariant, so
        # ``kernel_fsdp_axis_idx`` (defined against the original kernel
        # shape) applies unchanged. For bf16 we already AG'd the kernel
        # up front in fwd, so ctx_kernel is already full -- skip.
        dgrad_kernel_T = _all_gather_grouped_scaled_tensor_1x(
            dgrad_kernel_T, kernel_fsdp_mesh_axis, kernel_fsdp_axis_idx
        )

    # g_contracting_dim = (0, )
    # x_contracting_dim = (0, )
    g_contracting_dim = x_contracting_dim = tuple(
        range(0, len(x_shape) - len(fwd_x_contracting_dims))
    )
    wgrad_contracting_dims = (x_contracting_dim, g_contracting_dim)

    wgrad_x_T = ctx_x
    wgrad_grad = casted_grad.get_tensor(usage=TensorUsage.RHS)
    dgrad = tex.grouped_gemm(
        dgrad_grad,
        dgrad_kernel_T,
        contracting_dims=dgrad_contracting_dims,
        precision=precision,
        preferred_element_type=preferred_element_type,
        group_offset=group_offset,
    )

    wgrad = tex.grouped_gemm(
        wgrad_x_T,
        wgrad_grad,
        contracting_dims=wgrad_contracting_dims,
        precision=precision,
        preferred_element_type=preferred_element_type,
        group_offset=group_offset,
    )

    if kernel_fsdp_enabled:
        # ``wgrad`` is full kernel-shape ``(G, K, N)``; the kernel input was
        # per-shard on ``kernel_fsdp_axis_idx``, so the bwd output must be
        # too. ``psum_scatter`` is the autodiff transpose of the fwd
        # ``all_gather``: it sums per-FSDP-peer partial wgrads (different
        # peers contribute different ctx_x batch slices) and scatters the
        # summed result back along the FSDP axis. Each peer keeps only its
        # K-slice of the global wgrad, matching the per-shard kernel spec.
        wgrad = jax.lax.psum_scatter(
            wgrad,
            kernel_fsdp_mesh_axis,
            scatter_dimension=kernel_fsdp_axis_idx,
            tiled=True,
        )

    group_sizes_grad = None
    dbias = tex.grouped_dbias(grad, group_sizes) if use_bias else None

    return dgrad, wgrad, group_sizes_grad, dbias, quantizer_set


_grouped_dense.defvjp(_grouped_dense_fwd_rule, _grouped_dense_bwd_rule)
