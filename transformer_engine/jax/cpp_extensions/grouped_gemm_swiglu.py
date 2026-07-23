# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""TVM-FFI JAX primitive for cuDNN-FE's fused grouped SwiGLU kernel."""

from __future__ import annotations

import threading
from typing import Any

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec

from .base import BasePrimitive, register_primitive
from .misc import get_padded_spec

__all__ = ["grouped_gemm_swiglu", "grouped_gemm_swiglu_dependencies_available"]


_INPUT_ROLES = ("a", "b", "sfa", "sfb", "padded_offsets", "alpha", "prob", "norm_const")
_OUTPUT_ROLES = ("c", "d", "d_col", "sfd_row", "sfd_col")
_REGISTERED_TARGETS: dict[str, tuple[Any, Any]] = {}
_REGISTRATION_LOCK = threading.Lock()


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _scale_size(rows: int, cols: int, *, colwise: bool) -> int:
    if colwise:
        rows, cols = cols, rows
    return 32 * 4 * _ceil_div(rows, 128) * 4 * _ceil_div(_ceil_div(cols, 32), 4)


def _dtype_name(dtype) -> str:
    dtype = jnp.dtype(dtype)
    supported = {
        jnp.dtype(jnp.float8_e4m3fn): "float8_e4m3fn",
        jnp.dtype(jnp.float8_e5m2): "float8_e5m2",
        jnp.dtype(jnp.float8_e8m0fnu): "float8_e8m0fnu",
        jnp.dtype(jnp.float16): "float16",
        jnp.dtype(jnp.bfloat16): "bfloat16",
        jnp.dtype(jnp.float32): "float32",
        jnp.dtype(jnp.int32): "int32",
    }
    try:
        return supported[dtype]
    except KeyError as exc:
        raise ValueError(f"Unsupported grouped SwiGLU dtype {dtype}") from exc


def grouped_gemm_swiglu_dependencies_available() -> tuple[bool, str]:
    """Check optional runtime dependencies without compiling a kernel."""
    try:
        import jax_tvm_ffi  # noqa: F401  # pylint: disable=unused-import,import-outside-toplevel
        import tvm_ffi  # noqa: F401  # pylint: disable=unused-import,import-outside-toplevel
        from cudnn import compile_grouped_gemm_swiglu  # noqa: F401
    except (ImportError, ModuleNotFoundError, RuntimeError, AttributeError) as exc:
        return False, str(exc)
    return True, ""


def _output_avals(a_aval, b_aval, compute_dtype, output_dtype):
    if a_aval.ndim != 3 or a_aval.shape[-1] != 1:
        raise ValueError(f"Expected A[M,K,1], got {a_aval.shape}")
    if b_aval.ndim != 3:
        raise ValueError(f"Expected physical B[E,N,K], got {b_aval.shape}")
    m, k_a, _ = a_aval.shape
    _, n, k_b = b_aval.shape
    if k_a != k_b:
        raise ValueError(f"A K={k_a} does not match B K={k_b}")
    if m % 256:
        raise ValueError(f"Padded activation rows M={m} must be divisible by 256")
    if n % 2:
        raise ValueError(f"Combined SwiGLU N={n} must be even")
    intermediate = n // 2
    return (
        jax.core.ShapedArray((m, n, 1), compute_dtype),
        jax.core.ShapedArray((m, intermediate, 1), output_dtype),
        jax.core.ShapedArray((m, intermediate, 1), output_dtype),
        jax.core.ShapedArray((_scale_size(m, intermediate, colwise=False),), jnp.float8_e8m0fnu),
        jax.core.ShapedArray((_scale_size(m, intermediate, colwise=True),), jnp.float8_e8m0fnu),
    )


def _compile_and_register(avals_in, avals_out):
    try:
        import jax_tvm_ffi
        import tvm_ffi
        from cudnn import (
            GroupedGemmOperandDesc,
            GroupedGemmSwigluConfig,
            compile_grouped_gemm_swiglu,
        )
    except (ImportError, ModuleNotFoundError, RuntimeError, AttributeError) as exc:
        raise RuntimeError(
            "TVM-FFI grouped SwiGLU requires the patched cuDNN-FE compiler package, "
            "apache-tvm-ffi, and jax-tvm-ffi"
        ) from exc

    layouts = {
        "a": (1, 0, 2),
        "b": (1, 0, 2),
        "sfa": (0,),
        "sfb": (0,),
        "padded_offsets": (0,),
        "alpha": (0,),
        "prob": (1, 0, 2),
        "norm_const": (0,),
        "c": (1, 0, 2),
        "d": (1, 0, 2),
        "d_col": (1, 0, 2),
        "sfd_row": (0,),
        "sfd_col": (0,),
    }
    operands = {}
    for role, aval in zip(_INPUT_ROLES, avals_in):
        shape = aval.shape
        if role == "b":
            # The framework ABI is compact [E,N,K].  The cuDNN-FE compiler
            # describes and restores the kernel's logical [N,K,E] view.
            shape = (aval.shape[1], aval.shape[2], aval.shape[0])
        operands[role] = GroupedGemmOperandDesc(
            role=role,
            dtype=_dtype_name(aval.dtype),
            shape=shape,
            stride_order=layouts[role],
        )
    for role, aval in zip(_OUTPUT_ROLES, avals_out):
        operands[role] = GroupedGemmOperandDesc(
            role=role,
            dtype=_dtype_name(aval.dtype),
            shape=aval.shape,
            stride_order=layouts[role],
        )

    function, abi = compile_grouped_gemm_swiglu(
        operands=operands,
        config=GroupedGemmSwigluConfig(
            sf_vec_size=32,
            mma_tiler_mn=(256, 256),
            cluster_shape_mn=(2, 1),
            discrete_col_sfd=True,
        ),
    )
    if not isinstance(function, tvm_ffi.Function):
        raise TypeError(
            "cuDNN-FE compile_grouped_gemm_swiglu returned a Python wrapper; "
            "the JAX execution target must be a native tvm_ffi.Function"
        )
    expected = [(role, "arg") for role in _INPUT_ROLES] + [
        (role, "ret") for role in _OUTPUT_ROLES
    ] + [("stream", "stream")]
    actual = [(entry.role, entry.kind) for entry in abi.entries]
    if actual != expected:
        raise ValueError(f"Unsupported grouped SwiGLU ABI: expected {expected}, got {actual}")

    target_name = f"te_grouped_gemm_swiglu.{abi.key.rsplit('.', maxsplit=1)[-1]}"
    with _REGISTRATION_LOCK:
        if target_name not in _REGISTERED_TARGETS:
            jax_tvm_ffi.register_ffi_target(
                target_name,
                function,
                arg_spec=list(abi.arg_spec),
                platform="gpu",
                allow_cuda_graph=True,
            )
            # Keep both the native function and its exact ABI alive for the
            # lifetime of the process-global XLA target registration.
            _REGISTERED_TARGETS[target_name] = (function, abi)
    return target_name, abi


class GroupedGemmSwigluPrimitive(BasePrimitive):
    """Fused grouped GEMM, SwiGLU, and dual MXFP8 quantization."""

    name = "te_grouped_gemm_swiglu_tvm_ffi"
    multiple_results = True
    impl_static_args = (8, 9)
    inner_primitive = None
    outer_primitive = None

    @staticmethod
    def abstract(
        a_aval,
        b_aval,
        sfa_aval,
        sfb_aval,
        padded_offsets_aval,
        alpha_aval,
        prob_aval,
        norm_const_aval,
        *,
        compute_dtype,
        output_dtype,
    ):
        del sfa_aval, sfb_aval, alpha_aval, norm_const_aval
        m = a_aval.shape[0]
        experts = b_aval.shape[0]
        if padded_offsets_aval.shape != (experts,):
            raise ValueError(
                f"Expected padded_offsets shape {(experts,)}, got {padded_offsets_aval.shape}"
            )
        if prob_aval.shape != (m, 1, 1):
            raise ValueError(f"Expected prob shape {(m, 1, 1)}, got {prob_aval.shape}")
        return _output_avals(a_aval, b_aval, compute_dtype, output_dtype)

    @staticmethod
    def lowering(ctx, *args, compute_dtype, output_dtype):
        del compute_dtype, output_dtype
        target_name, abi = _compile_and_register(ctx.avals_in, ctx.avals_out)
        operand_layouts = [entry.stride_order for entry in abi.entries if entry.kind == "arg"]
        result_layouts = [entry.stride_order for entry in abi.entries if entry.kind == "ret"]
        return jax.ffi.ffi_lowering(
            target_name,
            operand_layouts=operand_layouts,
            result_layouts=result_layouts,
        )(ctx, *args)

    @staticmethod
    def impl(
        a,
        b,
        sfa,
        sfb,
        padded_offsets,
        alpha,
        prob,
        norm_const,
        compute_dtype,
        output_dtype,
    ):
        if GroupedGemmSwigluPrimitive.inner_primitive is None:
            raise RuntimeError("GroupedGemmSwigluPrimitive has not been registered")
        return GroupedGemmSwigluPrimitive.inner_primitive.bind(
            a,
            b,
            sfa,
            sfb,
            padded_offsets,
            alpha,
            prob,
            norm_const,
            compute_dtype=compute_dtype,
            output_dtype=output_dtype,
        )

    @staticmethod
    def batcher(batched_args, batch_dims, *, compute_dtype, output_dtype):
        del batched_args, compute_dtype, output_dtype
        raise NotImplementedError(
            f"GroupedGemmSwigluPrimitive does not support vmap batch dimensions {batch_dims}"
        )

    @staticmethod
    def partition(compute_dtype, output_dtype, mesh, arg_infos, result_infos):
        del result_infos
        a_spec = get_padded_spec(arg_infos[0])
        m_axis = a_spec[0]
        experts = arg_infos[1].shape[0]
        if experts != 1 and m_axis is not None:
            raise NotImplementedError(
                "Token-axis custom partitioning is currently supported only for the standalone "
                "single-expert experiment; MoEBlock integration remains inside shard_map"
            )

        arg_shardings = (
            NamedSharding(mesh, PartitionSpec(m_axis, None, None)),
            NamedSharding(mesh, PartitionSpec(None, None, None)),
            NamedSharding(mesh, PartitionSpec(m_axis)),
            NamedSharding(mesh, PartitionSpec(None)),
            NamedSharding(mesh, PartitionSpec(None)),
            NamedSharding(mesh, PartitionSpec(None)),
            NamedSharding(mesh, PartitionSpec(m_axis, None, None)),
            NamedSharding(mesh, PartitionSpec(None)),
        )
        out_shardings = [
            NamedSharding(mesh, PartitionSpec(m_axis, None, None)),
            NamedSharding(mesh, PartitionSpec(m_axis, None, None)),
            NamedSharding(mesh, PartitionSpec(m_axis, None, None)),
            NamedSharding(mesh, PartitionSpec(m_axis)),
            NamedSharding(mesh, PartitionSpec(m_axis)),
        ]

        def sharded_impl(a, b, sfa, sfb, padded_offsets, alpha, prob, norm_const):
            if experts == 1:
                padded_offsets = jnp.asarray([a.shape[0]], dtype=jnp.int32)
            return GroupedGemmSwigluPrimitive.impl(
                a,
                b,
                sfa,
                sfb,
                padded_offsets,
                alpha,
                prob,
                norm_const,
                compute_dtype=compute_dtype,
                output_dtype=output_dtype,
            )

        return mesh, sharded_impl, out_shardings, arg_shardings

    @staticmethod
    def shardy_sharding_rule(*args, **kwargs):
        del args, kwargs
        return (
            "m k one, experts n k, sfa, sfb, experts, experts, m prob_one_a prob_one_b, norm -> "
            "m n one, m h one, m h one, sfd_row, sfd_col"
        )


register_primitive(GroupedGemmSwigluPrimitive)


def grouped_gemm_swiglu(
    a: jax.Array,
    b: jax.Array,
    sfa: jax.Array,
    sfb: jax.Array,
    padded_offsets: jax.Array,
    prob: jax.Array,
    *,
    compute_dtype,
    output_dtype,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Run the TVM-FFI fused grouped SwiGLU primitive.

    ``b`` uses TE's compact physical ``[E,N,K]`` representation.  The native
    cuDNN-FE launcher restores the kernel's logical ``[N,K,E]`` view because
    XLA FFI buffers do not expose strides to jax-tvm-ffi.
    """
    if b.ndim != 3:
        raise ValueError(f"Expected physical B[E,N,K], got {b.shape}")
    experts = b.shape[0]
    alpha = jnp.ones((experts,), dtype=jnp.float32)
    norm_const = jnp.ones((1,), dtype=jnp.float32)
    return GroupedGemmSwigluPrimitive.outer_primitive.bind(
        a,
        b,
        sfa.reshape(-1),
        sfb.reshape(-1),
        padded_offsets.astype(jnp.int32),
        alpha,
        prob.astype(jnp.float32),
        norm_const,
        compute_dtype=jnp.dtype(compute_dtype),
        output_dtype=jnp.dtype(output_dtype),
    )
