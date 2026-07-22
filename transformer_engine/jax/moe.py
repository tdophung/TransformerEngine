# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Mixture-of-Experts (MoE) layer for TransformerEngine JAX.

This module exposes :func:`moe`, a single fused MoE forward pass + bwd
built on top of TE's NCCL-backed Expert Parallelism primitives
(``tex.ep_dispatch`` / ``tex.ep_combine``). The block runs::

    gate  ->  topk  ->  ep_dispatch  ->  per-expert FFN (grouped GEMMs)
          ->  ep_combine  ->  output

under a single ``jax.custom_vjp`` so the routing, dispatch, FFN and
combine steps fuse cleanly under XLA without leaking intermediate
residuals into the user-facing autograd graph.

Sharding model
--------------
* Inbound activations are 3D ``[B, S, H]`` sharded
  ``((*data_parallelism_axes, ep_axis), None, None)``. The public
  :func:`moe` soft-repins this on entry and warns when a reshard is
  inserted.
* The EP primitives operate at global view (their custom_partitioning
  rules handle per-shard execution). The FFN GEMMs run per-shard inside
  a small ``shard_map`` whose ``in_specs`` and ``out_specs`` mirror the
  same ``((dp, ep), ...)`` layout.

FC1 and FC2 use independent quantizer sets.  The sets are differentiable
``custom_vjp`` arguments, are threaded through the per-shard FFN, and are
returned by the backward rule so stateful recipes follow the same update
semantics as :mod:`transformer_engine.jax.dense`.
"""

from functools import partial
from typing import Any, Optional, Tuple, Union
import os
import warnings

import flax.struct
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P

from . import cpp_extensions as tex
from .quantize import (
    GroupedNoScaleTensor,
    QuantizerSet,
    ScaledTensor,
    ScaledTensorFactory,
    TensorUsage,
    noop_quantizer_set,
    with_sharding_constraint_by_logical_axes,
)
from .flax.module import _convert_to_activation_function
from .router import ScoreFunction, _validate_score_function
from .sharding import _get_mesh

__all__ = ["moe"]


# Per-expert dispatch-slot alignment fed to ``tex.ep_prepare`` as
# ``dispatch_output_per_expert_alignment``. NCCL EP HT requires the
# per-expert recv block to be at least 128-token aligned, and all current
# TE grouped-GEMM recipes (bf16/fp16/fp8/mxfp8) are satisfied by the
# same 128-token tile, so a single constant covers every supported path.
_ALIGN_SIZE = 128
_CUDNN_CUTEDSL_ALIGN_SIZE = 256
_CUDNN_CUTEDSL_ENV = "NVTE_JAX_MOE_USE_CUDNN_CUTEDSL_FUSION"


def _use_cudnn_cutedsl_fusion_from_env() -> bool:
    value = os.getenv(_CUDNN_CUTEDSL_ENV, "0")
    if value not in ("0", "1"):
        raise ValueError(f"{_CUDNN_CUTEDSL_ENV} must be '0' or '1', got {value!r}")
    return value == "1"


def _cudnn_cutedsl_fusion_rejection_reasons(
    x,
    wi_0,
    wi_1,
    wi_0_bias,
    wi_1_bias,
    fc1_quantizer_set,
    fc2_quantizer_set,
    *,
    num_experts,
    activation_type,
    ep_axis,
) -> list[str]:
    """Return reasons this call cannot use the CuTeDSL fusion, or an empty list."""
    from transformer_engine_jax import get_device_compute_capability

    from .cutedsl_extensions.moe import (
        load_grouped_gemm_dswiglu_kernel,
        load_grouped_gemm_swiglu_kernel,
    )
    from .quantize import GroupedQuantizer, ScalingMode

    errors = []
    try:
        compute_capability = get_device_compute_capability(0)
    except RuntimeError as exc:
        errors.append(f"could not query GPU compute capability: {exc}")
    else:
        if compute_capability != 100:
            errors.append(f"requires an SM100 GPU, got SM{compute_capability}")
    if str(activation_type).lower() != "silu":
        errors.append("requires activation_type='silu'")
    if wi_0_bias is not None or wi_1_bias is not None:
        errors.append("does not support FC1 gate/up bias")
    if wi_0.shape != wi_1.shape or wi_0.ndim != 3:
        errors.append(f"requires matching rank-3 wi_0/wi_1, got {wi_0.shape}/{wi_1.shape}")
    elif wi_0.shape[-1] % 32:
        errors.append(f"requires intermediate size divisible by 32, got {wi_0.shape[-1]}")
    if x.dtype not in (jnp.bfloat16, jnp.float16):
        errors.append(f"requires BF16 or FP16 activations, got {x.dtype}")

    required_quantizers = {
        "fc1.x": fc1_quantizer_set.x,
        "fc1.kernel": fc1_quantizer_set.kernel,
        "fc1.dgrad": fc1_quantizer_set.dgrad,
        "fc2.x": fc2_quantizer_set.x,
        "fc2.kernel": fc2_quantizer_set.kernel,
        "fc2.dgrad": fc2_quantizer_set.dgrad,
    }
    for name, quantizer in required_quantizers.items():
        if not isinstance(quantizer, GroupedQuantizer):
            errors.append(f"requires a grouped MXFP8 quantizer for {name}")
        elif quantizer.scaling_mode != ScalingMode.MXFP8_1D_SCALING:
            errors.append(f"requires MXFP8_1D_SCALING for {name}")
        elif not quantizer.q_layout.is_rowwise_colwise:
            errors.append(f"requires rowwise+colwise quantization for {name}")

    if all(isinstance(q, GroupedQuantizer) for q in required_quantizers.values()):
        if fc1_quantizer_set.x.q_dtype != fc1_quantizer_set.kernel.q_dtype:
            errors.append("requires identical FC1 activation and weight MXFP8 payload dtypes")
        if fc2_quantizer_set.dgrad.q_dtype != fc2_quantizer_set.kernel.q_dtype:
            errors.append(
                "requires identical FC2 dgrad and weight MXFP8 payload dtypes for "
                "dSwiGLU backward fusion"
            )
        supported = (jnp.float8_e4m3fn, jnp.float8_e5m2)
        for name, quantizer in required_quantizers.items():
            if quantizer.q_dtype not in supported:
                errors.append(f"unsupported MXFP8 payload dtype {quantizer.q_dtype} for {name}")

    mesh = _get_mesh()
    if mesh is not None and not mesh.empty and ep_axis in mesh.shape:
        num_local_experts = num_experts // mesh.shape[ep_axis]
        if num_local_experts > 1024:
            errors.append(f"requires at most 1024 local experts, got {num_local_experts}")

    try:
        import cutlass.jax as cutlass_jax
    except ImportError as exc:
        errors.append(f"nvidia-cutlass-dsl with JAX bindings is required: {exc}")
    else:
        try:
            if not cutlass_jax.is_available():
                errors.append(
                    "cutlass.jax.is_available() is false; check cute_dsl_runtime.so discovery"
                )
        except (AttributeError, RuntimeError) as exc:
            errors.append(f"CUTLASS JAX runtime check failed: {exc}")

    for kernel_name, loader in (
        ("SwiGLU forward", load_grouped_gemm_swiglu_kernel),
        ("dSwiGLU backward", load_grouped_gemm_dswiglu_kernel),
    ):
        try:
            loader()
        except (ImportError, ModuleNotFoundError, RuntimeError) as exc:
            errors.append(f"could not load the cuDNN frontend {kernel_name} kernel: {exc}")

    return errors


def _with_sharding_constraint_cast_bwd(x: jnp.ndarray, sharding) -> jnp.ndarray:
    """Sharding constraint that keeps bwd cotangents in the primal dtype.

    Plain ``jax.lax.with_sharding_constraint`` is identity on the fwd
    but does not constrain the dtype of the cotangent that flows back
    through it. In this MoE bwd, ``d_x`` is built from two paths:

      * ``d_x_from_dispatch`` from ``ep_dispatch_bwd`` -- primal dtype
        (bf16 in mixed precision).
      * ``d_x_from_gate = d_logits_2d @ gate_kernel.T`` where
        ``d_logits_2d`` is produced by
        ``fused_topk_with_score_function_bwd``. That primitive runs at
        fp32 because the fwd promoted ``logits_2d`` to fp32 (the fused
        topk/softmax/sigmoid kernels are only validated at fp32).

    JAX's type promotion then makes ``d_x_from_gate + d_x_from_dispatch``
    fp32, so the user-visible ``d_x`` ends up wider than ``x``. That
    doubles activation-grad bandwidth and breaks any downstream kernel
    that pins a bf16 input layout. This wrapper inserts an explicit
    cast back to the primal dtype on the bwd side and re-asserts the
    same sharding there as well.
    """

    @jax.custom_vjp
    def _constraint(y):
        return jax.lax.with_sharding_constraint(y, sharding)

    def _constraint_fwd(y):
        return jax.lax.with_sharding_constraint(y, sharding), jnp.zeros((), dtype=y.dtype)

    def _constraint_bwd(dtype_ref, grad):
        return (jax.lax.with_sharding_constraint(grad.astype(dtype_ref.dtype), sharding),)

    _constraint.defvjp(_constraint_fwd, _constraint_bwd)
    return _constraint(x)


# =============================================================================
# Process-level NCCL EP bootstrap (must run eagerly, outside jax.jit)
# =============================================================================
#
# ``tex.ep_bootstrap`` does a NCCL UID allgather over the JAX runtime, which
# cannot run from inside a jit-traced function. The caller must bootstrap
# eagerly once per process before any jitted MoE call, then record the
# bootstrap signature via ``record_ep_bootstrap_signature_for_moe``. The
# per-call check below verifies the recorded signature is wide enough for
# the current MoE invocation (smaller per-call usage is fine since the C++
# backend reserves worst-case buffers at bootstrap time).

_te_ep_bootstrap_signature: Optional[Tuple[int, int, int, int, int]] = None


def record_ep_bootstrap_signature_for_moe(
    num_experts: int,
    max_tokens_per_rank: int,
    recv_capacity_per_rank: int,
    hidden_dim: int,
    ep_size: int,
) -> None:
    """Record the params passed to ``ep_bootstrap`` so the per-call check
    in ``_moe_fwd_rule`` can verify compatibility. Call this once per
    process immediately after ``ep_bootstrap``.
    """
    global _te_ep_bootstrap_signature
    _te_ep_bootstrap_signature = (
        num_experts,
        max_tokens_per_rank,
        recv_capacity_per_rank,
        hidden_dim,
        ep_size,
    )


def _te_ep_assert_compatible_bootstrap(
    num_experts: int,
    max_tokens_per_rank: int,
    recv_capacity_per_rank: int,
    hidden_dim: int,
    ep_size: int,
) -> None:
    """Verify a prior eager ``ep_bootstrap`` is wide enough for this call."""
    if _te_ep_bootstrap_signature is None:
        raise RuntimeError(
            "TE EP was not bootstrapped. Call"
            " transformer_engine.jax.ep.ep_bootstrap(...) eagerly (outside"
            " any jax.jit) once per process, then"
            " transformer_engine.jax.moe.record_ep_bootstrap_signature_for_moe(...)"
            " with the same params, before invoking moe()."
        )
    b_num_experts, b_max_tpr, b_recv_pr, b_hidden, b_ep_size = _te_ep_bootstrap_signature
    if (
        num_experts != b_num_experts
        or hidden_dim != b_hidden
        or ep_size != b_ep_size
        or max_tokens_per_rank > b_max_tpr
        or recv_capacity_per_rank > b_recv_pr
    ):
        raise ValueError(
            "TE EP was already bootstrapped with signature"
            f" (num_experts={b_num_experts}, max_tokens_per_rank={b_max_tpr},"
            f" recv_capacity_per_rank={b_recv_pr}, hidden_dim={b_hidden},"
            f" ep_size={b_ep_size}); this moe() call needs"
            f" (num_experts={num_experts}, max_tokens_per_rank={max_tokens_per_rank},"
            f" recv_capacity_per_rank={recv_capacity_per_rank}, hidden_dim={hidden_dim},"
            f" ep_size={ep_size}). Re-bootstrap with wider params (or matching exact"
            " sizes) is required."
        )


# =============================================================================
# Residual container threaded fwd -> bwd
# =============================================================================


@flax.struct.dataclass
class _Ctx:
    """Residuals carried from the fwd rule into the bwd rule.

    Flattened automatically by jax.custom_vjp; ``cfg`` is the only
    static field (the rest are jnp.ndarray, GroupedNoScaleTensor, or
    None when aux_loss_coeff == 0).
    """

    x: jnp.ndarray
    gate_kernel: jnp.ndarray
    expert_bias: jnp.ndarray
    logits_2d: jnp.ndarray
    saved_scores: jnp.ndarray
    routing_map: jnp.ndarray
    cfg: Any = flax.struct.field(pytree_node=False)
    handle_mem: jnp.ndarray
    token_counts: jnp.ndarray
    recv_topk_weights: jnp.ndarray
    casted_sorted_x_lhs_trans: Any
    casted_wi_rhs_trans: Any
    gate_proj_out: jnp.ndarray
    up_proj_out: jnp.ndarray
    casted_intermediate_lhs_trans: Any
    casted_wo_rhs_trans: Any
    expert_outputs: jnp.ndarray
    local_group_sizes: jnp.ndarray
    fc1_quantizer_set: QuantizerSet
    fc2_quantizer_set: QuantizerSet
    aux_const_buf: Any = None
    aux_tokens_per_expert: Any = None
    aux_saved_scores: Any = None


# =============================================================================
# Per-shard FFN body (runs inside shard_map)
# =============================================================================


def _pack_grouped_tensor(tensor):
    """Flatten a grouped tensor into arrays safe to cross shard_map.

    TE's grouped tensor PyTree constructors validate array shapes during
    ``tree_unflatten``. JAX's shard_map prefix checker temporarily unflattens
    sentinel objects, so carrying the tensor object itself across shard_map
    fails before lowering. A plain tuple keeps the same residual data without
    invoking those constructors.
    """
    if isinstance(tensor, GroupedNoScaleTensor):
        scale_inv = jnp.empty((0,), dtype=jnp.float32)
    else:
        scale_inv = tensor.scale_inv
    first_dims = (
        tensor.first_dims if tensor.first_dims is not None else jnp.empty((0,), dtype=jnp.int32)
    )
    last_dims = (
        tensor.last_dims if tensor.last_dims is not None else jnp.empty((0,), dtype=jnp.int32)
    )
    return tensor.data, scale_inv, tensor.amax, first_dims, last_dims


def _unpack_grouped_tensor(
    packed,
    quantizer,
    *,
    original_shape,
    dq_dtype,
    is_colwise,
    flatten_axis,
):
    """Reconstruct a grouped tensor inside the backward shard_map body."""
    data, scale_inv, amax, first_dims, last_dims = packed
    first_dims = first_dims if first_dims.size else None
    last_dims = last_dims if last_dims.size else None
    if quantizer is None:
        return GroupedNoScaleTensor(
            data=data,
            amax=amax,
            first_dims=first_dims,
            last_dims=last_dims,
            original_shape=original_shape,
        )
    return ScaledTensorFactory.create_1x(
        data=data,
        scale_inv=scale_inv,
        amax=amax,
        scaling_mode=quantizer.scaling_mode,
        dq_dtype=dq_dtype,
        is_colwise=is_colwise,
        data_layout="N",
        flatten_axis=flatten_axis,
        first_dims=first_dims,
        last_dims=last_dims,
        original_shape=original_shape,
        pre_swizzled=quantizer.scaling_mode.is_mxfp8_scaling,
    )


def _token_residual_spec(batch_pspec_axis, quantized):
    """Specs for a packed token-major grouped tensor."""
    data_spec = P(batch_pspec_axis) if quantized else P(batch_pspec_axis, None)
    scale_spec = P(batch_pspec_axis) if quantized else P()
    return data_spec, scale_spec, P(), P(batch_pspec_axis), P()


def _kernel_residual_spec(ep_axis, quantized):
    """Specs for a packed expert-kernel grouped tensor."""
    data_spec = P(ep_axis) if quantized else P(ep_axis, None, None)
    scale_spec = P(ep_axis) if quantized else P()
    return data_spec, scale_spec, P(), P(), P()


def _ffn_fwd_per_shard(
    recv_tokens_local: jnp.ndarray,
    recv_topk_weights_local: jnp.ndarray,
    token_counts_local: jnp.ndarray,
    wi_0: jnp.ndarray,
    wi_1: jnp.ndarray,
    wo: jnp.ndarray,
    wi_0_bias: Optional[jnp.ndarray],
    wi_1_bias: Optional[jnp.ndarray],
    wo_bias: Optional[jnp.ndarray],
    fc1_quantizer_set: QuantizerSet,
    fc2_quantizer_set: QuantizerSet,
    *,
    num_local_experts: int,
    activation_type: str,
    apply_topk_weights_early: bool,
    use_cudnn_cutedsl_fusion: bool,
):
    """Per-shard FFN forward.

    Operates on the shard-local ``[1, recv_pr, H]`` slice that
    ``tex.ep_dispatch`` produces. Returns the expert outputs (shaped
    ``[1, recv_pr, H_out]`` so the surrounding ``shard_map`` reassembles
    them as ``[num_procs, recv_pr, H_out]``) plus the residuals consumed
    by the bwd.

    ``token_counts_local`` (``[1, num_local_experts]``, from
    ``tex.ep_prepare``) is passed to ``grouped_gemm`` as ``group_sizes``
    so cuBLAS skips both 0-token-routed experts and the dispatch
    overalloc tail.
    """
    hidden = recv_tokens_local.shape[-1]
    sorted_x = recv_tokens_local.reshape(-1, hidden)
    recv_w_flat = recv_topk_weights_local.reshape(-1)
    local_group_sizes = token_counts_local.reshape(-1).astype(jnp.int32)

    wi_0 = wi_0.astype(sorted_x.dtype)
    wi_1 = wi_1.astype(sorted_x.dtype)
    wo = wo.astype(sorted_x.dtype)

    # The cuDNN frontend kernel consumes alternating 32-column gate/up
    # blocks. The existing TE grouped-GEMM path consumes the two full
    # projections concatenated along N.
    if use_cudnn_cutedsl_fusion:
        from .cutedsl_extensions.moe import pack_swiglu_pair

        wi_combined = pack_swiglu_pair(wi_0, wi_1)
    else:
        # Concat along the trailing axis (NOT stack on a new axis).
        # grouped_gemm requires the 3D (G, K, N) weight layout with
        # contracting_dims=((1,), (1,)).
        wi_combined = jnp.concatenate([wi_0, wi_1], axis=-1)
    wi_combined_bias = (
        jnp.concatenate([wi_0_bias, wi_1_bias], axis=-1) if wi_0_bias is not None else None
    )

    # EP dispatch zero-fills aligned padding rows in recv_tokens, so
    # inactive slots do not contaminate grouped quantization scales.
    casted_sorted_x = tex.grouped_quantize(
        sorted_x, fc1_quantizer_set.x, local_group_sizes, flatten_axis=-1
    )
    casted_wi = tex.grouped_quantize(wi_combined, fc1_quantizer_set.kernel, flatten_axis=-1)
    casted_intermediate = None
    if use_cudnn_cutedsl_fusion:
        from .cutedsl_extensions.moe import (
            grouped_gemm_swiglu_mxfp8,
            unpack_swiglu_pair,
        )

        casted_sorted_x_lhs = casted_sorted_x.get_tensor(usage=TensorUsage.LHS)
        casted_wi_rhs = casted_wi.get_tensor(usage=TensorUsage.RHS)
        padded_offsets = jnp.cumsum(local_group_sizes, dtype=jnp.int32)
        prob = (
            recv_w_flat[:, None, None]
            if apply_topk_weights_early
            else jnp.ones((sorted_x.shape[0], 1, 1), dtype=jnp.float32)
        )
        (
            combined_out_3d,
            intermediate_row,
            intermediate_col,
            intermediate_scale_row,
            intermediate_scale_col,
        ) = grouped_gemm_swiglu_mxfp8(
            casted_sorted_x_lhs.data.reshape(sorted_x.shape[0], hidden, 1),
            # TE stores the colwise payload in logical [E,K,N] order even
            # though its values were quantized along K. Materialize the
            # K-major [E,N,K] layout consumed by the CuTeDSL kernel.
            casted_wi_rhs.data.reshape(num_local_experts, hidden, wi_combined.shape[-1]).transpose(
                0, 2, 1
            ),
            casted_sorted_x_lhs.scale_inv,
            casted_wi_rhs.scale_inv,
            padded_offsets,
            prob,
            compute_dtype=sorted_x.dtype,
            output_dtype=fc2_quantizer_set.x.q_dtype,
        )
        combined_out = combined_out_3d.reshape(sorted_x.shape[0], wi_combined.shape[-1])
        gate_proj_out, up_proj_out = unpack_swiglu_pair(combined_out)

        intermediate_shape = (sorted_x.shape[0], gate_proj_out.shape[-1])
        scaling_mode = fc2_quantizer_set.x.scaling_mode
        row_scale_size = scaling_mode.get_grouped_scale_shape(
            intermediate_shape,
            num_local_experts,
            False,
            is_padded=True,
            flatten_axis=1,
        )[0]
        col_scale_size = scaling_mode.get_grouped_scale_shape(
            intermediate_shape,
            num_local_experts,
            True,
            is_padded=True,
            flatten_axis=1,
        )[0]
        intermediate_scale_row = jnp.pad(
            intermediate_scale_row,
            (0, row_scale_size - intermediate_scale_row.size),
        )
        intermediate_scale_col = jnp.pad(
            intermediate_scale_col,
            (0, col_scale_size - intermediate_scale_col.size),
        )
        casted_intermediate = ScaledTensorFactory.create(
            data=intermediate_row.reshape(-1),
            scale_inv=intermediate_scale_row,
            colwise_data=intermediate_col.reshape(-1),
            colwise_scale_inv=intermediate_scale_col,
            scaling_mode=scaling_mode,
            dq_dtype=sorted_x.dtype,
            data_layout=fc2_quantizer_set.x.data_layout,
            q_layout=fc2_quantizer_set.x.q_layout,
            flatten_axis=1,
            first_dims=local_group_sizes,
            original_shape=intermediate_shape,
            pre_swizzled=True,
        )
    else:
        combined_out = tex.grouped_gemm(
            casted_sorted_x.get_tensor(usage=TensorUsage.LHS),
            casted_wi.get_tensor(usage=TensorUsage.RHS),
            contracting_dims=((1,), (1,)),
            bias=wi_combined_bias,
        )
        gate_proj_out, up_proj_out = jnp.split(combined_out, 2, axis=-1)
    casted_sorted_x_lhs_trans = casted_sorted_x.get_tensor(usage=TensorUsage.LHS_TRANS)
    casted_sorted_x_lhs_trans = (
        casted_sorted_x_lhs_trans.checkpoint(fc1_quantizer_set.x)
        if isinstance(casted_sorted_x_lhs_trans, ScaledTensor)
        else casted_sorted_x_lhs_trans
    )
    casted_wi_rhs_trans = casted_wi.get_tensor(usage=TensorUsage.RHS_TRANS)
    casted_wi_rhs_trans = (
        casted_wi_rhs_trans.checkpoint(fc1_quantizer_set.kernel)
        if isinstance(casted_wi_rhs_trans, ScaledTensor)
        else casted_wi_rhs_trans
    )

    # Activation inputs (gate_proj_out, up_proj_out) stay in the wi GEMM
    # output dtype; the activation output (`intermediate`) stays in the
    # dtype the wo GEMM / wo's quantized input consumes. For bf16 compute
    # that's all bf16; for FP8/FP4 the downstream grouped_quantize is what
    # transitions to the target precision.
    act_fn = _convert_to_activation_function(activation_type)
    if not use_cudnn_cutedsl_fusion:
        intermediate = act_fn(gate_proj_out) * up_proj_out

    if apply_topk_weights_early and not use_cudnn_cutedsl_fusion:
        # Fold the per-token combine weights into the FFN intermediate;
        # the downstream wo GEMM is linear so this is equivalent to the
        # late-weighting path. EP dispatch zero-fills aligned padding tokens
        # and routing weights; grouped ops exclude any trailing over-allocation,
        # so a separate validity select is unnecessary.
        # ``w_b`` is cast to ``intermediate.dtype`` so the multiply doesn't
        # promote expert_outputs above the EP buffer's element width.
        intermediate = intermediate * recv_w_flat[:, None].astype(intermediate.dtype)

    if not use_cudnn_cutedsl_fusion:
        casted_intermediate = tex.grouped_quantize(
            intermediate, fc2_quantizer_set.x, local_group_sizes, flatten_axis=-1
        )
    assert casted_intermediate is not None
    casted_wo = tex.grouped_quantize(wo, fc2_quantizer_set.kernel, flatten_axis=-1)
    expert_outputs = tex.grouped_gemm(
        casted_intermediate.get_tensor(usage=TensorUsage.LHS),
        casted_wo.get_tensor(usage=TensorUsage.RHS),
        contracting_dims=((1,), (1,)),
        bias=wo_bias,
    )
    casted_intermediate_lhs_trans = casted_intermediate.get_tensor(usage=TensorUsage.LHS_TRANS)
    casted_intermediate_lhs_trans = (
        casted_intermediate_lhs_trans.checkpoint(fc2_quantizer_set.x)
        if isinstance(casted_intermediate_lhs_trans, ScaledTensor)
        else casted_intermediate_lhs_trans
    )
    casted_wo_rhs_trans = casted_wo.get_tensor(usage=TensorUsage.RHS_TRANS)
    casted_wo_rhs_trans = (
        casted_wo_rhs_trans.checkpoint(fc2_quantizer_set.kernel)
        if isinstance(casted_wo_rhs_trans, ScaledTensor)
        else casted_wo_rhs_trans
    )

    expert_outputs_3d = expert_outputs.reshape(1, expert_outputs.shape[0], expert_outputs.shape[1])
    # Reshape local_group_sizes to (1, num_local_experts) so the
    # surrounding shard_map can stitch per-shard counts back into the
    # global (num_procs, num_local_experts) layout matching token_counts.
    local_group_sizes_3d = local_group_sizes.reshape(1, num_local_experts)
    residuals = (
        _pack_grouped_tensor(casted_sorted_x_lhs_trans),
        _pack_grouped_tensor(casted_wi_rhs_trans),
        gate_proj_out,
        up_proj_out,
        _pack_grouped_tensor(casted_intermediate_lhs_trans),
        _pack_grouped_tensor(casted_wo_rhs_trans),
        local_group_sizes_3d,
    )
    return expert_outputs_3d, residuals


def _ffn_bwd_per_shard(
    d_expert_outputs_local: jnp.ndarray,
    casted_sorted_x_lhs_trans,
    casted_wi_rhs_trans,
    gate_proj_out: jnp.ndarray,
    up_proj_out: jnp.ndarray,
    casted_intermediate_lhs_trans,
    casted_wo_rhs_trans,
    local_group_sizes: jnp.ndarray,
    recv_topk_weights_local: jnp.ndarray,
    fc1_quantizer_set: QuantizerSet,
    fc2_quantizer_set: QuantizerSet,
    *,
    activation_type: str,
    apply_topk_weights_early: bool,
    has_bias: bool,
    use_cudnn_cutedsl_fusion: bool,
):
    """Per-shard FFN backward.

    Mirrors :func:`_ffn_fwd_per_shard`. Returns
    ``(d_sorted_x [1, recv_pr, H], d_recv_w [1, recv_pr],
    d_wi_0, d_wi_1, d_wo, d_wi_0_bias, d_wi_1_bias, d_wo_bias)``.
    """
    local_group_sizes = local_group_sizes.reshape(-1).astype(jnp.int32)
    d_eo_2d = d_expert_outputs_local.reshape(-1, d_expert_outputs_local.shape[-1])
    recv_w_flat = recv_topk_weights_local.reshape(-1)
    num_local_experts = local_group_sizes.size
    recv_rows = d_eo_2d.shape[0]
    hidden = d_eo_2d.shape[-1]
    intermediate_size = gate_proj_out.shape[-1]
    casted_sorted_x_lhs_trans = _unpack_grouped_tensor(
        casted_sorted_x_lhs_trans,
        fc1_quantizer_set.x,
        original_shape=(recv_rows, hidden),
        dq_dtype=gate_proj_out.dtype,
        is_colwise=True,
        flatten_axis=1,
    )
    casted_wi_rhs_trans = _unpack_grouped_tensor(
        casted_wi_rhs_trans,
        fc1_quantizer_set.kernel,
        original_shape=(num_local_experts, hidden, 2 * intermediate_size),
        dq_dtype=gate_proj_out.dtype,
        is_colwise=False,
        flatten_axis=2,
    )
    casted_intermediate_lhs_trans = _unpack_grouped_tensor(
        casted_intermediate_lhs_trans,
        fc2_quantizer_set.x,
        original_shape=(recv_rows, intermediate_size),
        dq_dtype=gate_proj_out.dtype,
        is_colwise=True,
        flatten_axis=1,
    )
    casted_wo_rhs_trans = _unpack_grouped_tensor(
        casted_wo_rhs_trans,
        fc2_quantizer_set.kernel,
        original_shape=(num_local_experts, intermediate_size, hidden),
        dq_dtype=gate_proj_out.dtype,
        is_colwise=False,
        flatten_axis=2,
    )
    # Grouped GEMM skips size_g == 0 experts without writing their wgrad
    # slice. Mask those slices to zero so the optimizer never sees garbage;
    # this is unrelated to EP alignment padding (handled by dispatch zero-fill).
    wgrad_group_active = (local_group_sizes > 0)[:, None, None]

    # wo bwd
    casted_d_eo = tex.grouped_quantize(
        d_eo_2d, fc2_quantizer_set.dgrad, local_group_sizes, flatten_axis=-1
    )
    _casted_d_eo_lhs = casted_d_eo.get_tensor(usage=TensorUsage.LHS)
    _casted_d_eo_rhs = casted_d_eo.get_tensor(usage=TensorUsage.RHS)
    d_wo = tex.grouped_gemm(
        casted_intermediate_lhs_trans,
        _casted_d_eo_rhs,
        contracting_dims=((0,), (0,)),
    )
    d_wo = jnp.where(wgrad_group_active, d_wo, jnp.zeros_like(d_wo))
    d_wo_bias = tex.grouped_dbias(d_eo_2d, local_group_sizes) if has_bias else None

    if use_cudnn_cutedsl_fusion:
        from .cutedsl_extensions.moe import (
            grouped_gemm_dswiglu_mxfp8,
            pack_swiglu_pair,
        )

        packed_forward = pack_swiglu_pair(gate_proj_out, up_proj_out)
        padded_offsets = jnp.cumsum(local_group_sizes, dtype=jnp.int32)
        prob = (
            recv_w_flat[:, None, None]
            if apply_topk_weights_early
            else jnp.ones((recv_rows, 1, 1), dtype=jnp.float32)
        )
        (
            d_combined_row,
            d_combined_col,
            d_combined_scale_row,
            d_combined_scale_col,
            _dprob,
        ) = grouped_gemm_dswiglu_mxfp8(
            _casted_d_eo_lhs.data.reshape(recv_rows, hidden, 1),
            casted_wo_rhs_trans.data.reshape(
                num_local_experts, intermediate_size, hidden
            ).transpose(1, 2, 0),
            packed_forward.reshape(recv_rows, 2 * intermediate_size, 1),
            _casted_d_eo_lhs.scale_inv,
            casted_wo_rhs_trans.scale_inv,
            padded_offsets,
            prob,
            output_dtype=fc1_quantizer_set.dgrad.q_dtype,
        )
        d_combined_shape = (recv_rows, 2 * intermediate_size)
        scaling_mode = fc1_quantizer_set.dgrad.scaling_mode
        row_scale_size = scaling_mode.get_grouped_scale_shape(
            d_combined_shape,
            num_local_experts,
            False,
            is_padded=True,
            flatten_axis=1,
        )[0]
        col_scale_size = scaling_mode.get_grouped_scale_shape(
            d_combined_shape,
            num_local_experts,
            True,
            is_padded=True,
            flatten_axis=1,
        )[0]
        d_combined_scale_row = jnp.pad(
            d_combined_scale_row,
            (0, row_scale_size - d_combined_scale_row.size),
        )
        d_combined_scale_col = jnp.pad(
            d_combined_scale_col,
            (0, col_scale_size - d_combined_scale_col.size),
        )
        casted_d_combined = ScaledTensorFactory.create(
            data=d_combined_row.reshape(-1),
            scale_inv=d_combined_scale_row,
            colwise_data=d_combined_col.reshape(-1),
            colwise_scale_inv=d_combined_scale_col,
            scaling_mode=scaling_mode,
            dq_dtype=gate_proj_out.dtype,
            data_layout=fc1_quantizer_set.dgrad.data_layout,
            q_layout=fc1_quantizer_set.dgrad.q_layout,
            flatten_axis=1,
            first_dims=local_group_sizes,
            original_shape=d_combined_shape,
            pre_swizzled=True,
        )
        if apply_topk_weights_early:
            # The cuDNN frontend dSwiGLU kernel accumulates dprob with
            # atomics and expects a zero-initialized output buffer. cutlass_call
            # allocates custom-call outputs uninitialized, so compute this
            # routing-weight cotangent explicitly until we can pass dprob as an
            # initialized input/output buffer.
            d_intermediate_for_prob = tex.grouped_gemm(
                _casted_d_eo_lhs,
                casted_wo_rhs_trans,
                contracting_dims=((1,), (2,)),
            )
            act_fn = _convert_to_activation_function(activation_type)
            intermediate_unweighted = act_fn(gate_proj_out) * up_proj_out
            d_recv_w_from_intermediate = jnp.sum(
                d_intermediate_for_prob * intermediate_unweighted,
                axis=-1,
            )
        else:
            d_recv_w_from_intermediate = jnp.zeros_like(recv_w_flat)
        d_recv_w_from_intermediate = d_recv_w_from_intermediate.astype(recv_w_flat.dtype)
        d_combined_for_bias = None
    else:
        d_intermediate = tex.grouped_gemm(
            _casted_d_eo_lhs,
            casted_wo_rhs_trans,
            contracting_dims=((1,), (2,)),
        )
        act_fn = _convert_to_activation_function(activation_type)
        if apply_topk_weights_early:
            # intermediate' = intermediate * w. Split the cotangent across
            # both factors before the activation bwd consumes it. EP zero-fills
            # aligned padding, and grouped ops exclude trailing over-allocation.
            w_b = recv_w_flat[:, None].astype(d_intermediate.dtype)
            intermediate_unweighted = act_fn(gate_proj_out) * up_proj_out
            d_recv_w_from_intermediate = jnp.sum(
                d_intermediate * intermediate_unweighted,
                axis=-1,
            ).astype(recv_w_flat.dtype)
            d_intermediate = d_intermediate * w_b
        else:
            d_recv_w_from_intermediate = jnp.zeros_like(recv_w_flat)

        # Activation bwd, symmetric with the fwd: silu' and the two
        # elementwise products run in the GEMM dtype (no fp32 island), so
        # the chain rule composes through at the same precision the wi/wo
        # GEMMs consume.
        act_gp, dact_pullback = jax.vjp(act_fn, gate_proj_out)
        d_up_proj_out = d_intermediate * act_gp
        (d_gate_proj_out,) = dact_pullback(d_intermediate * up_proj_out)
        d_combined_for_bias = jnp.concatenate([d_gate_proj_out, d_up_proj_out], axis=-1)
        casted_d_combined = tex.grouped_quantize(
            d_combined_for_bias, fc1_quantizer_set.dgrad, local_group_sizes, flatten_axis=-1
        )
    d_sorted_x = tex.grouped_gemm(
        casted_d_combined.get_tensor(usage=TensorUsage.LHS),
        casted_wi_rhs_trans,
        contracting_dims=((1,), (2,)),
    )
    d_wi_combined = tex.grouped_gemm(
        casted_sorted_x_lhs_trans,
        casted_d_combined.get_tensor(usage=TensorUsage.RHS),
        contracting_dims=((0,), (0,)),
    )
    d_wi_combined = jnp.where(wgrad_group_active, d_wi_combined, jnp.zeros_like(d_wi_combined))
    if use_cudnn_cutedsl_fusion:
        from .cutedsl_extensions.moe import unpack_swiglu_pair

        d_wi_0, d_wi_1 = unpack_swiglu_pair(d_wi_combined)
    else:
        d_wi_0, d_wi_1 = jnp.split(d_wi_combined, 2, axis=-1)
    if has_bias:
        d_wi_combined_bias = tex.grouped_dbias(d_combined_for_bias, local_group_sizes)
        d_wi_0_bias, d_wi_1_bias = jnp.split(d_wi_combined_bias, 2, axis=-1)
    else:
        d_wi_0_bias = None
        d_wi_1_bias = None

    d_sorted_x_3d = d_sorted_x.reshape(1, d_sorted_x.shape[0], d_sorted_x.shape[1])
    d_recv_w_3d = d_recv_w_from_intermediate.reshape(1, -1)
    return (
        d_sorted_x_3d,
        d_recv_w_3d,
        d_wi_0,
        d_wi_1,
        d_wo,
        d_wi_0_bias,
        d_wi_1_bias,
        d_wo_bias,
    )


# =============================================================================
# Full fwd / bwd rules (custom_vjp halves)
# =============================================================================


def _moe_fwd_rule(
    x,
    gate_kernel,
    wi_0,
    wi_1,
    wo,
    wi_0_bias,
    wi_1_bias,
    wo_bias,
    expert_bias,
    fc1_quantizer_set,
    fc2_quantizer_set,
    num_experts,
    num_experts_per_tok,
    activation_type,
    score_function,
    use_pre_softmax,
    num_groups,
    group_topk,
    scaling_factor,
    aux_loss_coeff,
    ep_axis,
    data_parallelism_axes,
    input_axes,
    gate_kernel_axes,
    wi_kernel_axes,
    wo_kernel_axes,
    dtype,
    apply_topk_weights_early,
    use_cudnn_cutedsl_fusion,
):
    """Forward: gate -> topk -> ep_dispatch -> shard_map(FFN) -> ep_combine.

    Returns ``(output, aux_loss)``. ``aux_loss`` is a zero scalar when
    ``aux_loss_coeff == 0``.
    """
    del gate_kernel_axes, wi_kernel_axes, wo_kernel_axes  # used in bwd only
    from jax.experimental.shard_map import shard_map

    x = with_sharding_constraint_by_logical_axes(x, input_axes)

    mesh = _get_mesh()
    if mesh is None or mesh.empty:
        raise ValueError("moe(...) requires an active jax.sharding.Mesh.")
    if ep_axis is None:
        raise ValueError("moe(...) requires ep_axis to be set (TE EP backend).")
    num_ep = mesh.shape[ep_axis]
    if num_experts % num_ep != 0:
        raise ValueError(f"num_experts={num_experts} must be divisible by EP size={num_ep}")
    num_local_experts = num_experts // num_ep

    dp_size = 1
    for ax in data_parallelism_axes:
        dp_size *= mesh.shape[ax]
    num_procs = num_ep * dp_size

    B, S, H = x.shape
    K = num_experts_per_tok
    if B % num_procs != 0:
        raise ValueError(f"batch={B} not divisible by ep*dp={num_procs}")

    # Per-rank send capacity: B/num_procs rows x S tokens per rank.
    max_tokens_per_rank = (B // num_procs) * S
    # Per-rank receive capacity. NCCL EP HT expert-major lays out variable
    # per-expert zones in one flat recv buffer, with each non-empty zone padded
    # to ``dispatch_output_per_expert_alignment``.
    tokens_per_ep_group = num_ep * max_tokens_per_rank
    max_local_assignments = tokens_per_ep_group * min(K, num_local_experts)
    max_nonempty_experts = min(num_local_experts, max_local_assignments)
    dispatch_alignment = _CUDNN_CUTEDSL_ALIGN_SIZE if use_cudnn_cutedsl_fusion else _ALIGN_SIZE
    padded_total_bound = max_local_assignments + (dispatch_alignment - 1) * max_nonempty_experts
    aligned_total_bound = (
        (padded_total_bound + dispatch_alignment - 1) // dispatch_alignment
    ) * dispatch_alignment
    per_expert_bound = (
        num_local_experts
        * ((tokens_per_ep_group + dispatch_alignment - 1) // dispatch_alignment)
        * dispatch_alignment
    )
    recv_pr = min(per_expert_bound, aligned_total_bound)

    _te_ep_assert_compatible_bootstrap(
        num_experts=num_experts,
        max_tokens_per_rank=max_tokens_per_rank,
        recv_capacity_per_rank=recv_pr,
        hidden_dim=H,
        ep_size=num_ep,
    )

    if not data_parallelism_axes:
        batch_pspec_axis: Any = ep_axis
    else:
        # ep must be innermost: ep_bootstrap forms NCCL EP comms from
        # consecutive global ranks (dp_color = rank // ep_size), so the
        # comm only stays within one model replica under (outer_dp, ep).
        batch_pspec_axis = (*data_parallelism_axes, ep_axis)
    ep3_spec = P(batch_pspec_axis, None, None)
    ep2_spec = P(batch_pspec_axis, None)
    x = jax.lax.with_sharding_constraint(x, NamedSharding(mesh, ep3_spec))

    # ---------------- Gate (global view) ----------------
    # tex.fused_topk_with_score_function is only validated against its
    # pytorch reference at fp32 (see tests/pytorch/test_fused_router.py:
    # parametrize gates dtype on torch.float32 only; the tolerance helper
    # raises NotImplementedError for any other dtype). Keeping logits in
    # the activation dtype (e.g. bf16) lets sigmoid / softmax / topk
    # accumulate at low precision and silently produce NaNs on tokens
    # whose normalised weights underflow. Cast to fp32 here to stay in
    # the validated regime.
    gate_kernel_cast = gate_kernel.astype(x.dtype)
    gate_logits = jnp.einsum("bsh,he->bse", x, gate_kernel_cast)
    logits_2d = gate_logits.reshape(-1, num_experts).astype(jnp.float32)

    # ---------------- Routing (global view) ----------------
    # expert_bias is an empty (shape-(0,)) sentinel when the caller did
    # not enable it; the primitive treats that as "no bias".
    eb_arg = expert_bias if expert_bias.shape != (0,) else jnp.zeros((0,), dtype=jnp.float32)
    sparse_probs, routing_map, saved_scores = tex.fused_topk_with_score_function_fwd(
        logits_2d,
        topk=K,
        use_pre_softmax=use_pre_softmax,
        num_groups=-1 if num_groups is None else num_groups,
        group_topk=-1 if group_topk is None else group_topk,
        scaling_factor=scaling_factor,
        score_function=score_function,
        expert_bias=eb_arg,
        compute_aux_scores=False,
    )
    sparse_probs = sparse_probs.astype(dtype)

    # ---------------- Aux loss (global view, replicated) ----------------
    # ``fused_moe_aux_loss_fwd`` sums probs and tokens_per_expert across
    # all tokens, which is wrong when T is sharded. Force-replicate the
    # gate logits and recompute the routing map at global view so the
    # kernel sees a complete [T_global, E] tensor. The replication is a
    # single all-gather over (*dp, ep) and lives off the dispatch
    # critical path.
    if aux_loss_coeff > 0.0:
        global_logits_2d = jax.lax.with_sharding_constraint(logits_2d, NamedSharding(mesh, P()))
        _, global_routing_map, _ = tex.fused_topk_with_score_function_fwd(
            global_logits_2d,
            topk=K,
            use_pre_softmax=use_pre_softmax,
            num_groups=-1 if num_groups is None else num_groups,
            group_topk=-1 if group_topk is None else group_topk,
            scaling_factor=scaling_factor,
            score_function=score_function,
            expert_bias=eb_arg,
            compute_aux_scores=False,
        )
        aux_tokens_per_expert = jnp.sum(global_routing_map.astype(jnp.int32), axis=0)
        # compute_aux_scores=True takes a separate kernel path: clean
        # per-expert softmax, no grouping / bias / scaling.
        aux_probs, _aux_rm, aux_saved_scores = tex.fused_topk_with_score_function_fwd(
            global_logits_2d.astype(jnp.float32),
            topk=K,
            use_pre_softmax=False,
            num_groups=-1,
            group_topk=-1,
            scaling_factor=1.0,
            score_function=score_function,
            expert_bias=jnp.zeros((0,), dtype=jnp.float32),
            compute_aux_scores=True,
        )
        aux_loss, aux_const_buf = tex.fused_moe_aux_loss_fwd(
            aux_probs.astype(jnp.float32),
            aux_tokens_per_expert.astype(jnp.int32),
            topk=K,
            coeff=aux_loss_coeff,
        )
        aux_loss = aux_loss.astype(dtype)
    else:
        aux_loss = jnp.zeros((), dtype=dtype)
        aux_const_buf = None
        aux_tokens_per_expert = None
        aux_saved_scores = None

    # ---------------- Routing -> (topk_idx, topk_w) at 3D ----------------
    # argsort on a bool tensor places True last (False=0 < True=1), so the
    # last K indices are the selected expert IDs.
    selected_experts = jnp.argsort(routing_map, axis=-1)[..., -K:]
    routing_weights = jnp.take_along_axis(sparse_probs, selected_experts, axis=-1)
    topk_idx_3d = selected_experts.reshape(B, S, K).astype(jnp.int32)
    topk_w_3d = routing_weights.reshape(B, S, K).astype(jnp.float32)
    # tex.ep_prepare/dispatch's partition only folds ep_axis into a replicated
    # leading dim, not the outer dp/fsdp axes, so a replicated topk_idx makes
    # each rank see B/ep rows (not B/num_procs) and overrun the bootstrap-sized
    # send buffer. Pin both routing tensors to the (outer, ep) leading sharding
    # so per-rank token counts match max_tokens_per_rank.
    topk_idx_3d = jax.lax.with_sharding_constraint(topk_idx_3d, NamedSharding(mesh, ep3_spec))
    topk_w_3d = jax.lax.with_sharding_constraint(topk_w_3d, NamedSharding(mesh, ep3_spec))

    # ---------------- TE EP dispatch (global view) ----------------
    cfg = tex.EpLayerConfig(
        top_k=K,
        dispatch_output_per_expert_alignment=dispatch_alignment,
    )
    token_counts, handle_mem = tex.ep_prepare(cfg, topk_idx_3d)
    recv_tokens, recv_topk_weights = tex.ep_dispatch_fwd(
        cfg, handle_mem, topk_idx_3d, x, topk_w_3d, recv_pr
    )
    recv_tokens = jax.lax.with_sharding_constraint(recv_tokens, NamedSharding(mesh, ep3_spec))
    recv_topk_weights = jax.lax.with_sharding_constraint(
        recv_topk_weights, NamedSharding(mesh, ep2_spec)
    )

    # ---------------- FFN (per-shard via shard_map) ----------------
    has_bias = wi_0_bias is not None
    kernel_spec = P(ep_axis, None, None)
    bias_spec = P(ep_axis, None) if has_bias else None
    # token_counts is the per-shard (1, num_local_experts) padded
    # per-expert count from ep_prepare; piped into _ffn_fwd_per_shard
    # as the grouped_gemm group_sizes so cuBLAS skips both 0-token
    # experts and the trailing overalloc tail.
    ffn_in_specs = (ep3_spec, ep2_spec, ep2_spec, kernel_spec, kernel_spec, kernel_spec)
    ffn_in_args = [recv_tokens, recv_topk_weights, token_counts, wi_0, wi_1, wo]
    if has_bias:
        ffn_in_specs = ffn_in_specs + (bias_spec, bias_spec, bias_spec)
        ffn_in_args.extend([wi_0_bias, wi_1_bias, wo_bias])
    # QuantizerSet is a JAX pytree. P() is a tree-prefix specification
    # that replicates any recipe state into each FFN shard; stateless
    # recipes such as MXFP8 have no array leaves here.
    ffn_in_specs = ffn_in_specs + (P(), P())
    ffn_in_args.extend([fc1_quantizer_set, fc2_quantizer_set])

    # FFN residuals live entirely on the local ep rank, so the leading
    # "experts" / "rows" dims map to P() (already shard-local). wi is
    # fused via jnp.concatenate along the trailing (output) axis
    # (see _ffn_fwd_per_shard for rationale), so the residual is a
    # single 3D casted_wi_rhs_trans of shape
    # (num_local_experts, hidden, 2*H_inter). local_group_sizes is
    # now per-shard dynamic (= per-shard token_counts), so its
    # residual spec mirrors ep2_spec (one row per ep rank).
    residuals_spec = (
        _token_residual_spec(batch_pspec_axis, fc1_quantizer_set.x is not None),
        _kernel_residual_spec(ep_axis, fc1_quantizer_set.kernel is not None),
        P(batch_pspec_axis, None),  # gate_proj_out
        P(batch_pspec_axis, None),  # up_proj_out
        _token_residual_spec(batch_pspec_axis, fc2_quantizer_set.x is not None),
        _kernel_residual_spec(ep_axis, fc2_quantizer_set.kernel is not None),
        ep2_spec,  # local_group_sizes (1, num_local_experts) per shard
    )
    out_specs = (ep3_spec, residuals_spec)

    def _body(*args):
        if has_bias:
            (r_tok, r_w, tc, w0, w1, w_o, w0b, w1b, wob, fc1_qset, fc2_qset) = args
        else:
            (r_tok, r_w, tc, w0, w1, w_o, fc1_qset, fc2_qset) = args
            w0b = w1b = wob = None
        # NCCL EP zero-fills aligned expert padding. Any trailing
        # over-allocation remains safe because ``tc`` is plumbed into
        # grouped operations and EP consumers use handle metadata to read
        # only valid positions. Zero-token-expert wgrads are masked in
        # ``_ffn_bwd_per_shard`` separately from padding.
        # If a future caller adds a non-group-aware reader of r_tok
        # (e.g. an inspect probe over the full recv tile), re-add the
        # ``jax.lax.cond(jnp.any(r_w != 0), identity, zeros_like)``
        # guard here.
        return _ffn_fwd_per_shard(
            r_tok,
            r_w,
            tc,
            w0,
            w1,
            w_o,
            w0b,
            w1b,
            wob,
            fc1_qset,
            fc2_qset,
            num_local_experts=num_local_experts,
            activation_type=activation_type,
            apply_topk_weights_early=apply_topk_weights_early,
            use_cudnn_cutedsl_fusion=use_cudnn_cutedsl_fusion,
        )

    expert_outputs, ffn_residuals = shard_map(
        _body,
        mesh=mesh,
        in_specs=ffn_in_specs,
        out_specs=out_specs,
        check_rep=False,
    )(*ffn_in_args)
    expert_outputs = jax.lax.with_sharding_constraint(expert_outputs, NamedSharding(mesh, ep3_spec))

    # ---------------- TE EP combine (global view) ----------------
    out_partition_spec = (batch_pspec_axis, None, None)
    if apply_topk_weights_early:
        # expert_outputs is already weighted upstream.
        output = tex.ep_combine_fwd(
            cfg,
            handle_mem,
            expert_outputs,
            num_local_tokens=(B, S),
            out_partition_spec=out_partition_spec,
        )
    else:
        # HT combine is unweighted; apply routing weights before calling it.
        # Padded recv slots are ignored by combine via handle_mem metadata.
        w = recv_topk_weights[..., None].astype(expert_outputs.dtype)
        weighted = expert_outputs * w
        output = tex.ep_combine_fwd(
            cfg,
            handle_mem,
            weighted,
            num_local_tokens=(B, S),
            out_partition_spec=out_partition_spec,
        )

    (
        casted_sorted_x_lhs_trans,
        casted_wi_rhs_trans,
        gate_proj_out,
        up_proj_out,
        casted_intermediate_lhs_trans,
        casted_wo_rhs_trans,
        local_group_sizes,
    ) = ffn_residuals

    ctx = _Ctx(
        x=x,
        gate_kernel=gate_kernel,
        expert_bias=expert_bias,
        logits_2d=logits_2d,
        saved_scores=saved_scores,
        routing_map=routing_map,
        cfg=cfg,
        handle_mem=handle_mem,
        token_counts=token_counts,
        recv_topk_weights=recv_topk_weights,
        casted_sorted_x_lhs_trans=casted_sorted_x_lhs_trans,
        casted_wi_rhs_trans=casted_wi_rhs_trans,
        gate_proj_out=gate_proj_out,
        up_proj_out=up_proj_out,
        casted_intermediate_lhs_trans=casted_intermediate_lhs_trans,
        casted_wo_rhs_trans=casted_wo_rhs_trans,
        expert_outputs=expert_outputs,
        local_group_sizes=local_group_sizes,
        fc1_quantizer_set=fc1_quantizer_set,
        fc2_quantizer_set=fc2_quantizer_set,
        aux_const_buf=aux_const_buf,
        aux_tokens_per_expert=aux_tokens_per_expert,
        aux_saved_scores=aux_saved_scores,
    )
    static = {
        "has_bias": has_bias,
        "x_shape": x.shape,
        "recv_pr": recv_pr,
    }
    return (output, aux_loss), (ctx, static)


def _moe_bwd_rule(
    num_experts,
    num_experts_per_tok,
    activation_type,
    score_function,
    use_pre_softmax,
    num_groups,
    group_topk,
    scaling_factor,
    aux_loss_coeff,
    ep_axis,
    data_parallelism_axes,
    input_axes,
    gate_kernel_axes,
    wi_kernel_axes,
    wo_kernel_axes,
    dtype,
    apply_topk_weights_early,
    use_cudnn_cutedsl_fusion,
    residuals,
    cotangents,
):
    """Backward mirror of :func:`_moe_fwd_rule`."""
    del num_groups, group_topk, dtype  # captured in residuals / unused in bwd
    from jax.experimental.shard_map import shard_map

    d_output, d_aux_loss = cotangents

    ctx, static = residuals
    has_bias = static["has_bias"]
    x_shape = static["x_shape"]
    recv_pr = static["recv_pr"]

    mesh = _get_mesh()
    if mesh is None or mesh.empty:
        raise ValueError("moe(...) requires an active jax.sharding.Mesh.")
    dp_size = 1
    for ax in data_parallelism_axes:
        dp_size *= mesh.shape[ax]

    B, S, _ = x_shape
    K = num_experts_per_tok
    if not data_parallelism_axes:
        batch_pspec_axis: Any = ep_axis
    else:
        batch_pspec_axis = (*data_parallelism_axes, ep_axis)
    ep3_spec = P(batch_pspec_axis, None, None)
    ep2_spec = P(batch_pspec_axis, None)
    out_partition_spec = (batch_pspec_axis, None, None)

    # ---------------- Combine bwd (global view) ----------------
    d_output = jax.lax.with_sharding_constraint(d_output, NamedSharding(mesh, ep3_spec))
    grad_pre_combine = tex.ep_combine_bwd(ctx.cfg, ctx.handle_mem, d_output, recv_pr)
    grad_pre_combine = jax.lax.with_sharding_constraint(
        grad_pre_combine, NamedSharding(mesh, ep3_spec)
    )

    if apply_topk_weights_early:
        # combine_fwd consumed already-weighted expert_outputs; the recv_w
        # cotangent flows through the early-weighting step inside the FFN bwd.
        d_expert_outputs = grad_pre_combine
        d_recv_w_from_combine = jnp.zeros_like(ctx.recv_topk_weights)
    else:
        # Reverse the late-weighting multiply. NCCL EP zero-fills aligned
        # padding, while grouped/EP consumers ignore trailing over-allocation.
        w = ctx.recv_topk_weights[..., None].astype(grad_pre_combine.dtype)
        d_expert_outputs = grad_pre_combine * w
        d_recv_w_from_combine = (grad_pre_combine * ctx.expert_outputs).sum(axis=-1)
        d_recv_w_from_combine = d_recv_w_from_combine.astype(ctx.recv_topk_weights.dtype)

    # ---------------- FFN bwd (per-shard via shard_map) ----------------
    kernel_spec = P(ep_axis, None, None)
    bias_spec = P(ep_axis, None) if has_bias else None

    bwd_in_specs = (
        ep3_spec,  # d_expert_outputs
        _token_residual_spec(batch_pspec_axis, ctx.fc1_quantizer_set.x is not None),
        _kernel_residual_spec(ep_axis, ctx.fc1_quantizer_set.kernel is not None),
        P(batch_pspec_axis, None),  # gate_proj_out
        P(batch_pspec_axis, None),  # up_proj_out
        _token_residual_spec(batch_pspec_axis, ctx.fc2_quantizer_set.x is not None),
        _kernel_residual_spec(ep_axis, ctx.fc2_quantizer_set.kernel is not None),
        ep2_spec,  # local_group_sizes (1, num_local_experts) per shard
        ep2_spec,  # recv_topk_weights
        P(),  # FC1 quantizer-set state, replicated into each shard
        P(),  # FC2 quantizer-set state, replicated into each shard
    )
    bwd_in_args = [
        d_expert_outputs,
        ctx.casted_sorted_x_lhs_trans,
        ctx.casted_wi_rhs_trans,
        ctx.gate_proj_out,
        ctx.up_proj_out,
        ctx.casted_intermediate_lhs_trans,
        ctx.casted_wo_rhs_trans,
        ctx.local_group_sizes,
        ctx.recv_topk_weights,
        ctx.fc1_quantizer_set,
        ctx.fc2_quantizer_set,
    ]
    bwd_out_specs = (
        ep3_spec,  # d_sorted_x
        ep2_spec,  # d_recv_w_from_intermediate
        kernel_spec,  # d_wi_0
        kernel_spec,  # d_wi_1
        kernel_spec,  # d_wo
        bias_spec if has_bias else None,  # d_wi_0_bias
        bias_spec if has_bias else None,  # d_wi_1_bias
        bias_spec if has_bias else None,  # d_wo_bias
    )

    def _bwd_body(*args):
        (
            d_sorted_x_3d,
            d_recv_w_3d,
            d_wi_0,
            d_wi_1,
            d_wo,
            d_wi_0_bias,
            d_wi_1_bias,
            d_wo_bias,
        ) = _ffn_bwd_per_shard(
            *args,
            activation_type=activation_type,
            apply_topk_weights_early=apply_topk_weights_early,
            has_bias=has_bias,
            use_cudnn_cutedsl_fusion=use_cudnn_cutedsl_fusion,
        )
        # Weight grads accumulate per-DP-shard inside the body; psum across
        # DP axes so each replica sees the full sum (matches out_specs
        # P(ep_axis, ...) which is DP-replicated).
        if data_parallelism_axes:
            dp = tuple(data_parallelism_axes)
            d_wi_0 = jax.lax.psum(d_wi_0, axis_name=dp)
            d_wi_1 = jax.lax.psum(d_wi_1, axis_name=dp)
            d_wo = jax.lax.psum(d_wo, axis_name=dp)
            if has_bias:
                d_wi_0_bias = jax.lax.psum(d_wi_0_bias, axis_name=dp)
                d_wi_1_bias = jax.lax.psum(d_wi_1_bias, axis_name=dp)
                d_wo_bias = jax.lax.psum(d_wo_bias, axis_name=dp)
        return (
            d_sorted_x_3d,
            d_recv_w_3d,
            d_wi_0,
            d_wi_1,
            d_wo,
            d_wi_0_bias,
            d_wi_1_bias,
            d_wo_bias,
        )

    (
        d_sorted_x,
        d_recv_w_from_intermediate,
        d_wi_0,
        d_wi_1,
        d_wo,
        d_wi_0_bias,
        d_wi_1_bias,
        d_wo_bias,
    ) = shard_map(
        _bwd_body,
        mesh=mesh,
        in_specs=bwd_in_specs,
        out_specs=bwd_out_specs,
        check_rep=False,
    )(
        *bwd_in_args
    )

    d_recv_w_total = d_recv_w_from_combine + d_recv_w_from_intermediate

    # ---------------- Dispatch bwd (global view) ----------------
    d_sorted_x = jax.lax.with_sharding_constraint(d_sorted_x, NamedSharding(mesh, ep3_spec))
    d_recv_w_total = jax.lax.with_sharding_constraint(d_recv_w_total, NamedSharding(mesh, ep2_spec))
    d_x_from_dispatch, d_topk_w = tex.ep_dispatch_bwd(
        ctx.cfg,
        ctx.handle_mem,
        d_sorted_x,
        d_recv_w_total,
        num_local_tokens=(B, S),
        out_partition_spec=out_partition_spec,
    )

    # ---------------- Routing bwd (global view) ----------------
    # The cotangent on routing_weights is a sparse scatter into sparse_probs
    # at the selected_experts indices.
    selected_experts = jnp.argsort(ctx.routing_map, axis=-1)[..., -K:]
    d_topk_w_flat = d_topk_w.reshape(-1, K)
    d_sparse_probs = jnp.zeros(ctx.routing_map.shape, dtype=d_topk_w_flat.dtype)
    d_sparse_probs = d_sparse_probs.at[
        jnp.arange(ctx.routing_map.shape[0])[:, None], selected_experts
    ].set(d_topk_w_flat)

    d_logits_2d = tex.fused_topk_with_score_function_bwd(
        ctx.routing_map,
        ctx.saved_scores,
        d_sparse_probs.astype(ctx.saved_scores.dtype),
        topk=K,
        use_pre_softmax=use_pre_softmax,
        scaling_factor=scaling_factor,
        score_function=score_function,
        compute_aux_scores=False,
    )

    # ---------------- Aux loss bwd (global view, replicated) ----------------
    # Reverse the fwd's all-gather/aux pipeline: aux_loss_bwd produces
    # d_aux_probs, then topk_bwd(compute_aux_scores=True) produces the
    # extra d_logits contribution. The replicated tensor adds into the
    # T-sharded routing-side d_logits via JAX's normal broadcast.
    if aux_loss_coeff > 0.0:
        T_global = ctx.logits_2d.shape[0]
        d_aux_loss_scalar = d_aux_loss.reshape(()).astype(jnp.float32)
        d_aux_probs = tex.fused_moe_aux_loss_bwd(
            ctx.aux_const_buf,
            ctx.aux_tokens_per_expert.astype(jnp.int32),
            d_aux_loss_scalar,
            num_tokens=int(T_global),
        )
        # routing_map is ignored by the kernel when compute_aux_scores=True,
        # so pass a zero placeholder of the right shape/dtype.
        zero_routing_map = jnp.zeros(ctx.aux_saved_scores.shape, dtype=ctx.routing_map.dtype)
        d_logits_aux = tex.fused_topk_with_score_function_bwd(
            zero_routing_map,
            ctx.aux_saved_scores,
            d_aux_probs.astype(ctx.aux_saved_scores.dtype),
            topk=K,
            use_pre_softmax=False,
            scaling_factor=1.0,
            score_function=score_function,
            compute_aux_scores=True,
        )
        d_logits_2d = d_logits_2d + d_logits_aux.astype(d_logits_2d.dtype)

    # ---------------- Gate bwd (global view) ----------------
    d_gate_logits = d_logits_2d.reshape(B, S, num_experts)
    gate_kernel_cast = ctx.gate_kernel.astype(ctx.x.dtype)
    d_x_from_gate = jnp.einsum("bse,he->bsh", d_gate_logits, gate_kernel_cast)
    d_gate_kernel = jnp.einsum("bsh,bse->he", ctx.x, d_gate_logits).astype(ctx.gate_kernel.dtype)
    d_x = d_x_from_gate + d_x_from_dispatch

    # Pin output grads to the declared logical axes so downstream
    # optimizers see consistent shardings.
    d_x = with_sharding_constraint_by_logical_axes(d_x, input_axes)
    d_gate_kernel = with_sharding_constraint_by_logical_axes(d_gate_kernel, gate_kernel_axes)
    d_wi_0 = with_sharding_constraint_by_logical_axes(d_wi_0, wi_kernel_axes)
    d_wi_1 = with_sharding_constraint_by_logical_axes(d_wi_1, wi_kernel_axes)
    d_wo = with_sharding_constraint_by_logical_axes(d_wo, wo_kernel_axes)

    # expert_bias has no learnable bwd path through fused_topk: the
    # primitive's bwd returns None for the bias slot. Match that with a
    # zero cotangent of the right shape so custom_vjp's arity check
    # passes.
    d_expert_bias = jnp.zeros_like(ctx.expert_bias)

    return (
        d_x,
        d_gate_kernel,
        d_wi_0,
        d_wi_1,
        d_wo,
        d_wi_0_bias if has_bias else None,
        d_wi_1_bias if has_bias else None,
        d_wo_bias if has_bias else None,
        d_expert_bias,
        ctx.fc1_quantizer_set,
        ctx.fc2_quantizer_set,
    )


# =============================================================================
# custom_vjp + public entry
# =============================================================================


@partial(jax.custom_vjp, nondiff_argnums=tuple(range(11, 29)))
def _moe(
    x,
    gate_kernel,
    wi_0,
    wi_1,
    wo,
    wi_0_bias,
    wi_1_bias,
    wo_bias,
    expert_bias,
    fc1_quantizer_set,
    fc2_quantizer_set,
    num_experts,
    num_experts_per_tok,
    activation_type,
    score_function,
    use_pre_softmax,
    num_groups,
    group_topk,
    scaling_factor,
    aux_loss_coeff,
    ep_axis,
    data_parallelism_axes,
    input_axes,
    gate_kernel_axes,
    wi_kernel_axes,
    wo_kernel_axes,
    dtype,
    apply_topk_weights_early,
    use_cudnn_cutedsl_fusion,
):
    primal, _ = _moe_fwd_rule(
        x,
        gate_kernel,
        wi_0,
        wi_1,
        wo,
        wi_0_bias,
        wi_1_bias,
        wo_bias,
        expert_bias,
        fc1_quantizer_set,
        fc2_quantizer_set,
        num_experts,
        num_experts_per_tok,
        activation_type,
        score_function,
        use_pre_softmax,
        num_groups,
        group_topk,
        scaling_factor,
        aux_loss_coeff,
        ep_axis,
        data_parallelism_axes,
        input_axes,
        gate_kernel_axes,
        wi_kernel_axes,
        wo_kernel_axes,
        dtype,
        apply_topk_weights_early,
        use_cudnn_cutedsl_fusion,
    )
    return primal


_moe.defvjp(_moe_fwd_rule, _moe_bwd_rule)


def moe(
    x: jnp.ndarray,
    gate_kernel: jnp.ndarray,
    wi_0: jnp.ndarray,
    wi_1: jnp.ndarray,
    wo: jnp.ndarray,
    wi_0_bias: Optional[jnp.ndarray] = None,
    wi_1_bias: Optional[jnp.ndarray] = None,
    wo_bias: Optional[jnp.ndarray] = None,
    expert_bias: Optional[jnp.ndarray] = None,
    fc1_quantizer_set: QuantizerSet = noop_quantizer_set,
    fc2_quantizer_set: QuantizerSet = noop_quantizer_set,
    *,
    num_experts: int,
    num_experts_per_tok: int,
    activation_type: str = "silu",
    score_function: Union[str, ScoreFunction] = "softmax",
    use_pre_softmax: bool = False,
    num_groups: Optional[int] = None,
    group_topk: Optional[int] = None,
    scaling_factor: float = 1.0,
    aux_loss_coeff: float = 0.0,
    apply_topk_weights_early: bool = False,
    ep_axis: str,
    data_parallelism_axes: Tuple[str, ...] = (),
    input_axes: Tuple[Optional[str], ...] = (),
    gate_kernel_axes: Tuple[Optional[str], ...] = (),
    wi_kernel_axes: Tuple[Optional[str], ...] = ("exp", "embed", "mlp"),
    wo_kernel_axes: Tuple[Optional[str], ...] = ("exp", "mlp", "embed"),
    dtype: jnp.dtype = jnp.float32,
) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
    """Run a full MoE block under a single fused custom_vjp on the TE EP path.

    Returns ``(output, aux_loss)``. ``aux_loss`` is ``None`` when
    ``aux_loss_coeff == 0`` and a 0-d scalar otherwise.

    Parameters
    ----------
    expert_bias : Optional[jnp.ndarray]
        ``[num_experts]`` learnable router bias added before the top-k
        when ``score_function='sigmoid'``. Pass ``None`` to disable.
        The bias has no gradient through the top-k primitive itself (it
        only steers expert selection); a zero cotangent is returned for
        it.
    aux_loss_coeff : float
        Per-step expert-load-balance loss coefficient. ``0.0`` (default)
        disables the aux loss entirely. When non-zero, an extra
        all-gather over the routing-side logits is inserted so the
        ``fused_moe_aux_loss`` kernel sees a global ``[T_global, E]``
        view; this lives off the dispatch critical path.

    The default per-expert dispatch-slot alignment is 128 tokens. Setting
    ``NVTE_JAX_MOE_USE_CUDNN_CUTEDSL_FUSION=1`` requires the call to match
    the eligible MXFP8 SwiGLU surface, uses the cuDNN frontend CuTeDSL FC1
    fusion, and raises dispatch alignment to the kernel-required 256 tokens.
    Ineligible opt-in calls fail with the reasons they cannot use CuTeDSL.

    Axis-name parameters:

    * ``ep_axis`` and ``data_parallelism_axes`` are *physical mesh
      axis names* -- they index ``jax.sharding.Mesh.shape`` directly
      (to compute ``num_ep`` / ``dp_size`` and to construct
      ``P((dp..., ep), None, None)`` for the per-shard
      ``jax.lax.with_sharding_constraint`` calls that JAX requires
      to refer to real mesh axes).
    * ``input_axes``, ``gate_kernel_axes``, ``wi_kernel_axes``,
      ``wo_kernel_axes`` are *logical axis names* (e.g.
      ``"batch"``, ``"embed"``, ``"mlp"``, ``"exp"``) -- they get
      resolved via the active Flax logical-axis rules and consumed
      by ``with_sharding_constraint_by_logical_axes``. They are
      ``Optional[str]`` tuples so a rule of ``None`` means
      "replicated on this axis".

    Logical-axis support for ``ep_axis`` / ``data_parallelism_axes``
    is intentionally out of scope: the EP comm-group construction
    (``dp_color = rank // ep_size``) and the bootstrap signature
    check both require concrete integer sizes, so a logical name
    would have to be resolved to a physical one anyway before any
    EP primitive is called. If a downstream pipeline needs to plumb
    logical names all the way to ``moe()``, do the rule lookup at
    the call site.

    See module docstring for the rest of the parameter semantics and the
    surrounding design rationale.
    """
    score_function = _validate_score_function(score_function)
    use_cudnn_cutedsl_fusion = False
    if _use_cudnn_cutedsl_fusion_from_env():
        rejection_reasons = _cudnn_cutedsl_fusion_rejection_reasons(
            x,
            wi_0,
            wi_1,
            wi_0_bias,
            wi_1_bias,
            fc1_quantizer_set,
            fc2_quantizer_set,
            num_experts=num_experts,
            activation_type=activation_type,
            ep_axis=ep_axis,
        )
        if rejection_reasons:
            raise ValueError(
                f"{_CUDNN_CUTEDSL_ENV}=1 is unsupported for this moe() call:\n- "
                + "\n- ".join(rejection_reasons)
            )
        use_cudnn_cutedsl_fusion = True

    # Enforce ((outer_dp..., ep), None, None) on inbound activations. The
    # EP comm groups consecutive global ranks (dp_color = rank // ep_size),
    # so ep MUST be innermost in the partition spec. Soft re-pin: free if
    # upstream already matches, single reshard otherwise.
    mesh = _get_mesh()
    if mesh is None or mesh.empty:
        raise ValueError("moe(...) requires an active jax.sharding.Mesh.")
    expected_leading: Any = (*data_parallelism_axes, ep_axis) if data_parallelism_axes else ep_axis
    expected_spec = P(expected_leading, None, None)
    actual_spec = getattr(getattr(x, "sharding", None), "spec", None)
    if actual_spec is not None and tuple(actual_spec) != tuple(expected_spec):
        warnings.warn(
            f"moe(...): inbound x sharding {actual_spec} does not match expected "
            f"{expected_spec}; inserting a reshard. Apply "
            "jax.lax.with_sharding_constraint upstream to avoid this overhead.",
            UserWarning,
            stacklevel=2,
        )
    x = _with_sharding_constraint_cast_bwd(x, NamedSharding(mesh, expected_spec))

    # custom_vjp can't trace through None args; lower expert_bias to an
    # empty shape-(0,) tensor that fused_topk_with_score_function treats
    # as "no bias".
    if expert_bias is None:
        expert_bias_arg = jnp.zeros((0,), dtype=jnp.float32)
    else:
        expert_bias_arg = expert_bias.astype(jnp.float32)

    output, aux_loss = _moe(
        x,
        gate_kernel,
        wi_0,
        wi_1,
        wo,
        wi_0_bias,
        wi_1_bias,
        wo_bias,
        expert_bias_arg,
        fc1_quantizer_set,
        fc2_quantizer_set,
        num_experts,
        num_experts_per_tok,
        activation_type,
        score_function,
        use_pre_softmax,
        num_groups,
        group_topk,
        scaling_factor,
        float(aux_loss_coeff),
        ep_axis,
        data_parallelism_axes,
        input_axes,
        gate_kernel_axes,
        wi_kernel_axes,
        wo_kernel_axes,
        dtype,
        apply_topk_weights_early,
        use_cudnn_cutedsl_fusion,
    )
    if aux_loss_coeff <= 0.0:
        aux_loss = None
    assert output.dtype == x.dtype, f"moe() output dtype {output.dtype} != input dtype {x.dtype}"
    return output, aux_loss
