# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Tests for the JAX binding to cuDNN frontend's CuTeDSL MoE kernel."""

import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from transformer_engine.jax import cpp_extensions as tex
from transformer_engine.jax.cutedsl_extensions.moe import (
    grouped_gemm_dswiglu_mxfp8,
    grouped_gemm_swiglu_mxfp8,
    load_grouped_gemm_swiglu_kernel,
    pack_swiglu_pair,
    unpack_swiglu_pair,
)
from transformer_engine.jax.quantize import (
    QuantizerFactory,
    ScaledTensorFactory,
    ScalingMode,
    TensorUsage,
)


def test_swiglu_block_pack_round_trip():
    """Gate/up packing alternates 32-column blocks and is reversible."""
    gate = jnp.arange(2 * 3 * 64, dtype=jnp.float32).reshape(2, 3, 64)
    up = gate + 1000
    interleaved = pack_swiglu_pair(gate, up)

    np.testing.assert_array_equal(interleaved[..., :32], gate[..., :32])
    np.testing.assert_array_equal(interleaved[..., 32:64], up[..., :32])
    unpacked_gate, unpacked_up = unpack_swiglu_pair(interleaved)
    np.testing.assert_array_equal(unpacked_gate, gate)
    np.testing.assert_array_equal(unpacked_up, up)


def test_direct_kernel_loader_bypasses_torch_api():
    """The direct source loader must not execute cuDNN's Torch API package."""
    kernel_cls = load_grouped_gemm_swiglu_kernel()
    assert kernel_cls.__name__ == "BlockScaledContiguousGroupedGemmKernel"
    assert "cudnn.grouped_gemm.grouped_gemm_swiglu.api" not in sys.modules
    # cuDNN frontend 1.25's shared utility source imports torch only for an
    # unused annotation. The loader supplies a non-executable sentinel rather
    # than importing the Torch package.
    torch_module = sys.modules.get("torch")
    assert torch_module is None or getattr(
        torch_module, "__transformer_engine_cutedsl_stub__", False
    )


def test_swiglu_forward_fused_output_parity():
    """The forward fused call matches TE projection plus JAX SwiGLU reference."""
    try:
        from transformer_engine_jax import get_device_compute_capability

        if get_device_compute_capability(0) != 100:
            pytest.skip("cuDNN frontend grouped GEMM SwiGLU requires SM100")
        load_grouped_gemm_swiglu_kernel()
        import cutlass.jax  # noqa: F401  # pylint: disable=unused-import,import-outside-toplevel
    except (ImportError, RuntimeError) as exc:
        pytest.skip(f"CuTeDSL JAX dependencies are unavailable: {exc}")

    experts, rows, hidden, intermediate = 1, 256, 128, 128
    key = jax.random.PRNGKey(123)
    x = jax.random.normal(key, (rows, hidden), dtype=jnp.bfloat16)
    wi_0 = jax.random.normal(
        jax.random.fold_in(key, 1), (experts, hidden, intermediate), dtype=jnp.bfloat16
    )
    wi_1 = jax.random.normal(
        jax.random.fold_in(key, 2), (experts, hidden, intermediate), dtype=jnp.bfloat16
    )
    wi = pack_swiglu_pair(wi_0, wi_1)
    group_sizes = jnp.asarray([rows], dtype=jnp.int32)
    quantizers = QuantizerFactory.create_set(
        scaling_mode=ScalingMode.MXFP8_1D_SCALING,
        fwd_dtype=jnp.float8_e4m3fn,
        bwd_dtype=jnp.float8_e5m2,
        is_2x2x=True,
        n_groups=experts,
    )

    @jax.jit
    def run(x_arg, wi_arg):
        casted_x = tex.grouped_quantize(
            x_arg, quantizers.x, group_sizes, flatten_axis=-1
        ).get_tensor(TensorUsage.LHS)
        casted_wi = tex.grouped_quantize(wi_arg, quantizers.kernel, flatten_axis=-1).get_tensor(
            TensorUsage.RHS
        )
        reference = tex.grouped_gemm(
            casted_x,
            casted_wi,
            contracting_dims=((1,), (1,)),
        )
        physical_wi = casted_wi.data.reshape(experts, hidden, 2 * intermediate).transpose(0, 2, 1)
        combined, swiglu_row, swiglu_col, scale_row, scale_col = grouped_gemm_swiglu_mxfp8(
            casted_x.data.reshape(rows, hidden, 1),
            physical_wi,
            casted_x.scale_inv,
            casted_wi.scale_inv,
            jnp.cumsum(group_sizes),
            jnp.ones((rows, 1, 1), dtype=jnp.float32),
            compute_dtype=jnp.bfloat16,
            output_dtype=jnp.float8_e4m3fn,
        )
        return (
            reference,
            combined.reshape(rows, 2 * intermediate),
            swiglu_row,
            swiglu_col,
            scale_row,
            scale_col,
        )

    reference, combined, swiglu_row, swiglu_col, scale_row, scale_col = run(x, wi)
    jax.block_until_ready((reference, combined, swiglu_row, swiglu_col))
    np.testing.assert_array_equal(combined, reference)
    assert swiglu_row.shape == swiglu_col.shape == (rows, intermediate, 1)
    assert swiglu_row.dtype == swiglu_col.dtype == jnp.float8_e4m3fn
    assert scale_row.dtype == scale_col.dtype == jnp.float8_e8m0fnu

    gate, up = unpack_swiglu_pair(combined)
    swiglu_reference = np.asarray(jax.nn.silu(gate) * up, dtype=np.float32)
    for is_colwise, payload, scale in (
        (False, swiglu_row, scale_row),
        (True, swiglu_col, scale_col),
    ):
        scaled = ScaledTensorFactory.create_1x(
            payload.reshape(-1),
            scale,
            scaling_mode=ScalingMode.MXFP8_1D_SCALING,
            dq_dtype=jnp.bfloat16,
            is_colwise=is_colwise,
            data_layout="N",
            flatten_axis=1,
            first_dims=group_sizes,
            original_shape=(rows, intermediate),
            pre_swizzled=True,
        )
        dequantized = np.asarray(scaled.dequantize()[0], dtype=np.float32)
        assert np.all(np.isfinite(dequantized))
        relative_error = np.linalg.norm(dequantized - swiglu_reference) / np.linalg.norm(
            swiglu_reference
        )
        assert relative_error < 0.05


def test_dswiglu_backward_quantized_output_parity():
    """The backward fused call emits the expected packed MXFP8 VJP."""
    try:
        from transformer_engine_jax import get_device_compute_capability

        if get_device_compute_capability(0) != 100:
            pytest.skip("cuDNN frontend grouped GEMM dSwiGLU requires SM100")
        load_grouped_gemm_swiglu_kernel()
        from transformer_engine.jax.cutedsl_extensions.moe import load_grouped_gemm_dswiglu_kernel

        load_grouped_gemm_dswiglu_kernel()
        import cutlass.jax  # noqa: F401  # pylint: disable=unused-import,import-outside-toplevel
    except (ImportError, RuntimeError) as exc:
        pytest.skip(f"CuTeDSL JAX dependencies are unavailable: {exc}")

    experts, rows, hidden, intermediate = 1, 256, 128, 128
    key = jax.random.PRNGKey(321)
    d_eo = jax.random.normal(key, (rows, hidden), dtype=jnp.bfloat16)
    wo = jax.random.normal(
        jax.random.fold_in(key, 1), (experts, intermediate, hidden), dtype=jnp.bfloat16
    )
    gate = jax.random.normal(jax.random.fold_in(key, 2), (rows, intermediate), dtype=jnp.bfloat16)
    up = jax.random.normal(jax.random.fold_in(key, 3), (rows, intermediate), dtype=jnp.bfloat16)
    packed_forward = pack_swiglu_pair(gate, up)
    group_sizes = jnp.asarray([rows], dtype=jnp.int32)
    quantizers = QuantizerFactory.create_set(
        scaling_mode=ScalingMode.MXFP8_1D_SCALING,
        fwd_dtype=jnp.float8_e4m3fn,
        bwd_dtype=jnp.float8_e4m3fn,
        is_2x2x=True,
        n_groups=experts,
    )

    @jax.jit
    def run(d_eo_arg, wo_arg, packed_arg):
        casted_d_eo = tex.grouped_quantize(
            d_eo_arg, quantizers.dgrad, group_sizes, flatten_axis=-1
        ).get_tensor(TensorUsage.LHS)
        casted_wo = tex.grouped_quantize(wo_arg, quantizers.kernel, flatten_axis=-1).get_tensor(
            TensorUsage.RHS_TRANS
        )
        reference_intermediate = tex.grouped_gemm(
            casted_d_eo,
            casted_wo,
            contracting_dims=((1,), (2,)),
        )
        d_row, d_col, scale_row, scale_col, dprob = grouped_gemm_dswiglu_mxfp8(
            casted_d_eo.data.reshape(rows, hidden, 1),
            casted_wo.data.reshape(experts, intermediate, hidden).transpose(1, 2, 0),
            packed_arg.reshape(rows, 2 * intermediate, 1),
            casted_d_eo.scale_inv,
            casted_wo.scale_inv,
            jnp.cumsum(group_sizes),
            jnp.ones((rows, 1, 1), dtype=jnp.float32),
            output_dtype=quantizers.dgrad.q_dtype,
        )
        return reference_intermediate, d_row, d_col, scale_row, scale_col, dprob

    reference_intermediate, d_row, d_col, scale_row, scale_col, dprob = run(
        d_eo, wo, packed_forward
    )
    jax.block_until_ready((reference_intermediate, d_row, d_col, dprob))
    assert d_row.shape == d_col.shape == (rows, 2 * intermediate, 1)
    assert d_row.dtype == d_col.dtype == jnp.float8_e4m3fn
    assert scale_row.dtype == scale_col.dtype == jnp.float8_e8m0fnu
    assert dprob.shape == (rows, 1, 1)

    sigmoid = jax.nn.sigmoid(gate.astype(jnp.float32))
    swish = gate.astype(jnp.float32) * sigmoid
    ref = reference_intermediate.astype(jnp.float32)
    d_up = ref * swish
    d_gate = ref * up.astype(jnp.float32) * sigmoid * (1 + gate.astype(jnp.float32) * (1 - sigmoid))
    packed_reference = np.asarray(pack_swiglu_pair(d_gate, d_up), dtype=np.float32)

    for is_colwise, payload, scale in (
        (False, d_row, scale_row),
        (True, d_col, scale_col),
    ):
        scaled = ScaledTensorFactory.create_1x(
            payload.reshape(-1),
            scale,
            scaling_mode=ScalingMode.MXFP8_1D_SCALING,
            dq_dtype=jnp.bfloat16,
            is_colwise=is_colwise,
            data_layout="N",
            flatten_axis=1,
            first_dims=group_sizes,
            original_shape=(rows, 2 * intermediate),
            pre_swizzled=True,
        )
        dequantized = np.asarray(scaled.dequantize()[0], dtype=np.float32)
        assert np.all(np.isfinite(dequantized))
        relative_error = np.linalg.norm(dequantized - packed_reference) / np.linalg.norm(
            packed_reference
        )
        assert relative_error < 0.08
