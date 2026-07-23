# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Multiprocess partitioning and ragged-parity experiments for grouped SwiGLU."""

from __future__ import annotations

import sys

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import multihost_utils
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

_RAGGED_EIGHT_EXPERT_GROUP_SIZES = (512, 256, 1024, 512, 768, 256, 256, 512)


def _scale_size(rows: int, cols: int, *, colwise: bool = False) -> int:
    if colwise:
        rows, cols = cols, rows
    ceil_div = lambda value, divisor: (value + divisor - 1) // divisor
    return 32 * 4 * ceil_div(rows, 128) * 4 * ceil_div(ceil_div(cols, 32), 4)


def _global_array(local_array, mesh, spec):
    mesh_devices = tuple(np.asarray(mesh.devices).reshape(-1))
    if all(device.process_index == jax.process_index() for device in mesh_devices):
        return jax.device_put(local_array, NamedSharding(mesh, spec))
    return multihost_utils.host_local_array_to_global_array(local_array, mesh, spec)


def _run_matrix_cell(mesh: Mesh, label: str) -> None:
    from transformer_engine.jax import cpp_extensions as tex

    local_rows, hidden, intermediate, experts = 256, 128, 128, 1
    combined = 2 * intermediate
    process_key = jax.random.fold_in(jax.random.PRNGKey(2026), jax.process_index())
    x_local = jax.random.uniform(
        process_key,
        (local_rows, hidden, 1),
        minval=-0.5,
        maxval=0.5,
        dtype=jnp.float32,
    ).astype(jnp.float8_e4m3fn)
    weight_local = jax.random.uniform(
        jax.random.PRNGKey(17),
        (experts, combined, hidden),
        minval=-0.5,
        maxval=0.5,
        dtype=jnp.float32,
    ).astype(jnp.float8_e4m3fn)
    sfa_local = jnp.ones(
        (_scale_size(local_rows, hidden),), dtype=jnp.float8_e8m0fnu
    )
    sfb_local = jnp.ones(
        (_scale_size(combined, hidden) * experts,), dtype=jnp.float8_e8m0fnu
    )
    prob_local = jnp.ones((local_rows, 1, 1), dtype=jnp.float32)

    data_axis = mesh.axis_names[0]
    x = _global_array(x_local, mesh, P(data_axis, None, None))
    weight = _global_array(weight_local, mesh, P())
    sfa = _global_array(sfa_local, mesh, P(data_axis))
    sfb = _global_array(sfb_local, mesh, P())
    prob = _global_array(prob_local, mesh, P(data_axis, None, None))
    padded_offsets = _global_array(
        jnp.asarray([x.shape[0]], dtype=jnp.int32), mesh, P()
    )

    def fused(a, b, a_scale, b_scale, offsets, probabilities):
        return tex.grouped_gemm_swiglu(
            a,
            b,
            a_scale,
            b_scale,
            offsets,
            probabilities,
            compute_dtype=jnp.bfloat16,
            output_dtype=jnp.float8_e4m3fn,
        )

    custom_outputs = jax.jit(fused)(x, weight, sfa, sfb, padded_offsets, prob)

    def local_fused(a, b, a_scale, b_scale, _offsets, probabilities):
        local_offsets = jnp.asarray([a.shape[0]], dtype=jnp.int32)
        return fused(a, b, a_scale, b_scale, local_offsets, probabilities)

    mapped_fused = shard_map(
        local_fused,
        mesh=mesh,
        in_specs=(
            P(data_axis, None, None),
            P(),
            P(data_axis),
            P(),
            P(),
            P(data_axis, None, None),
        ),
        out_specs=[
            P(data_axis, None, None),
            P(data_axis, None, None),
            P(data_axis, None, None),
            P(data_axis),
            P(data_axis),
        ],
        check_rep=False,
    )
    shard_map_outputs = jax.jit(mapped_fused)(x, weight, sfa, sfb, padded_offsets, prob)
    jax.block_until_ready((custom_outputs, shard_map_outputs))

    def metrics(custom, mapped, a, b):
        reference = jnp.matmul(
            a.reshape(a.shape[0], a.shape[1]).astype(jnp.bfloat16),
            b[0].astype(jnp.bfloat16).T,
        ).reshape(a.shape[0], b.shape[1], 1)
        custom_c = custom[0].astype(jnp.float32)
        reference = reference.astype(jnp.float32)
        custom_vs_mapped = jnp.max(
            jnp.stack(
                [
                    jnp.max(jnp.abs(lhs.astype(jnp.float32) - rhs.astype(jnp.float32)))
                    for lhs, rhs in zip(custom, mapped)
                ]
            )
        )
        relative_reference = jnp.linalg.norm(custom_c - reference) / jnp.maximum(
            jnp.linalg.norm(reference), 1e-6
        )
        finite = jnp.all(
            jnp.stack(
                [jnp.all(jnp.isfinite(value.astype(jnp.float32))) for value in custom]
            )
        )
        return custom_vs_mapped, relative_reference, finite

    custom_vs_mapped, relative_reference, finite = jax.jit(metrics)(
        custom_outputs, shard_map_outputs, x, weight
    )
    custom_vs_mapped = float(custom_vs_mapped)
    relative_reference = float(relative_reference)
    finite = bool(finite)
    if custom_vs_mapped > 1e-3:
        raise AssertionError(f"{label}: custom_partitioning vs shard_map max diff {custom_vs_mapped}")
    if relative_reference > 0.05:
        raise AssertionError(f"{label}: fused vs matmul relative error {relative_reference}")
    if not finite:
        raise AssertionError(f"{label}: non-finite fused output")
    if jax.process_index() == 0:
        print(
            f"PASSED {label}: custom_vs_shard_map={custom_vs_mapped:.3e}, "
            f"relative_reference={relative_reference:.3e}",
            flush=True,
        )


def _run_ragged_multiprocess_cell(mesh: Mesh) -> None:
    """Exercise eight unequal experts through a global multiprocess shard_map."""
    from transformer_engine.jax import cpp_extensions as tex
    from transformer_engine.jax.cutedsl_extensions.moe import unpack_swiglu_pair
    from transformer_engine.jax.quantize import (
        QuantizerFactory,
        ScaledTensorFactory,
        ScalingMode,
        TensorUsage,
    )

    group_sizes_tuple = _RAGGED_EIGHT_EXPERT_GROUP_SIZES
    group_sizes = jnp.asarray(group_sizes_tuple, dtype=jnp.int32)
    local_rows = sum(group_sizes_tuple)
    experts, hidden, intermediate = len(group_sizes_tuple), 128, 128
    combined = 2 * intermediate
    assert experts >= 8
    assert len(set(group_sizes_tuple)) > 1
    assert all(offset % 256 == 0 for offset in np.cumsum(group_sizes_tuple))

    process_key = jax.random.fold_in(jax.random.PRNGKey(20260723), jax.process_index())
    x_local = jax.random.uniform(
        process_key,
        (local_rows, hidden, 1),
        minval=-0.5,
        maxval=0.5,
        dtype=jnp.float32,
    ).astype(jnp.float8_e4m3fn)
    weight_local = jax.random.uniform(
        jax.random.PRNGKey(20260724),
        (experts, combined, hidden),
        minval=-0.5,
        maxval=0.5,
        dtype=jnp.float32,
    ).astype(jnp.float8_e4m3fn)
    sfa_local = jnp.ones(
        (_scale_size(local_rows, hidden),), dtype=jnp.float8_e8m0fnu
    )
    sfb_local = jnp.ones(
        (_scale_size(combined, hidden) * experts,), dtype=jnp.float8_e8m0fnu
    )
    prob_local = jnp.ones((local_rows, 1, 1), dtype=jnp.float32)

    data_axis = mesh.axis_names[0]
    x = _global_array(x_local, mesh, P(data_axis, None, None))
    weight = _global_array(weight_local, mesh, P())
    sfa = _global_array(sfa_local, mesh, P(data_axis))
    sfb = _global_array(sfb_local, mesh, P())
    prob = _global_array(prob_local, mesh, P(data_axis, None, None))

    def local_fused(a, b, a_scale, b_scale, probabilities):
        return tex.grouped_gemm_swiglu(
            a,
            b,
            a_scale,
            b_scale,
            jnp.cumsum(group_sizes),
            probabilities,
            compute_dtype=jnp.bfloat16,
            output_dtype=jnp.float8_e4m3fn,
        )

    mapped_fused = shard_map(
        local_fused,
        mesh=mesh,
        in_specs=(
            P(data_axis, None, None),
            P(),
            P(data_axis),
            P(),
            P(data_axis, None, None),
        ),
        out_specs=[
            P(data_axis, None, None),
            P(data_axis, None, None),
            P(data_axis, None, None),
            P(data_axis),
            P(data_axis),
        ],
        check_rep=False,
    )
    outputs = jax.jit(mapped_fused)(x, weight, sfa, sfb, prob)
    jax.block_until_ready(outputs)
    combined_local, row_local, col_local, row_scale_local, col_scale_local = (
        output.addressable_data(0) for output in outputs
    )

    boundaries = np.cumsum((0, *group_sizes_tuple))
    reference_parts = [
        jnp.matmul(
            x_local[boundaries[i] : boundaries[i + 1], :, 0].astype(jnp.bfloat16),
            weight_local[i].astype(jnp.bfloat16).T,
        )
        for i in range(experts)
    ]
    combined_reference = jnp.concatenate(reference_parts, axis=0)
    np.testing.assert_array_equal(
        np.asarray(combined_local[:, :, 0]),
        np.asarray(combined_reference),
    )
    gate, up = unpack_swiglu_pair(combined_reference)
    swiglu_reference = jax.nn.silu(gate) * up

    quantizers = QuantizerFactory.create_set(
        scaling_mode=ScalingMode.MXFP8_1D_SCALING,
        fwd_dtype=jnp.float8_e4m3fn,
        bwd_dtype=jnp.float8_e5m2,
        is_2x2x=True,
        n_groups=experts,
    )
    quantized_reference = tex.grouped_quantize(
        swiglu_reference,
        quantizers.x,
        group_sizes,
        flatten_axis=-1,
    )
    reference_row = quantized_reference.get_tensor(TensorUsage.LHS)
    reference_col = quantized_reference.get_tensor(TensorUsage.LHS_TRANS)

    errors = []
    for is_colwise, payload, scale, reference_tensor in (
        (False, row_local, row_scale_local, reference_row),
        (True, col_local, col_scale_local, reference_col),
    ):
        expected_scale_size = ScalingMode.MXFP8_1D_SCALING.get_grouped_scale_shape(
            (local_rows, intermediate),
            experts,
            is_colwise,
            is_padded=True,
            flatten_axis=1,
        )[0]
        scale = jnp.pad(scale, (0, expected_scale_size - scale.size))
        scaled = ScaledTensorFactory.create_1x(
            payload.reshape(-1),
            scale,
            scaling_mode=ScalingMode.MXFP8_1D_SCALING,
            dq_dtype=jnp.bfloat16,
            is_colwise=is_colwise,
            data_layout="N",
            flatten_axis=1,
            first_dims=group_sizes,
            original_shape=(local_rows, intermediate),
            pre_swizzled=True,
        )
        dequantized = jnp.concatenate(scaled.dequantize(), axis=0).astype(jnp.float32)
        reference_dequantized = jnp.concatenate(
            reference_tensor.dequantize(), axis=0
        ).astype(jnp.float32)
        swiglu_error = jnp.linalg.norm(
            dequantized - swiglu_reference.astype(jnp.float32)
        ) / jnp.linalg.norm(swiglu_reference.astype(jnp.float32))
        quantization_error = jnp.linalg.norm(
            dequantized - reference_dequantized
        ) / jnp.linalg.norm(reference_dequantized)
        errors.append((float(swiglu_error), float(quantization_error)))
        if errors[-1][0] >= 0.05 or errors[-1][1] >= 0.05:
            raise AssertionError(
                "eight-expert ragged fused quantization parity failed: "
                f"is_colwise={is_colwise}, swiglu_error={errors[-1][0]:.4e}, "
                f"quantization_error={errors[-1][1]:.4e}"
            )

    if jax.process_index() == 0:
        print(
            "PASSED multiprocess eight-expert ragged grouped GEMM/SwiGLU/MXFP8 parity: "
            f"group_sizes={group_sizes_tuple}, row_errors={errors[0]}, col_errors={errors[1]}",
            flush=True,
        )


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit(f"usage: {sys.argv[0]} COORDINATOR PROCESS_ID NUM_PROCESSES")
    coordinator, process_id, num_processes = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    jax.distributed.initialize(
        coordinator_address=coordinator,
        num_processes=num_processes,
        process_id=process_id,
        local_device_ids=process_id,
    )
    import transformer_engine  # Load the TE core library before the raw JAX extension.
    from transformer_engine_jax import get_device_compute_capability

    if get_device_compute_capability(0) != 100:
        raise RuntimeError("TVM-FFI grouped SwiGLU experiment requires SM100")
    local_mesh = Mesh(np.asarray(jax.local_devices()), ("data",))
    global_mesh = Mesh(np.asarray(jax.devices()), ("data",))
    with local_mesh:
        _run_matrix_cell(local_mesh, "single-device shard_map/custom_partitioning")
    multihost_utils.sync_global_devices("single-device-cell")
    with global_mesh:
        _run_matrix_cell(global_mesh, "multiprocess shard_map/custom_partitioning")
    multihost_utils.sync_global_devices("multiprocess-cell")
    with global_mesh:
        _run_ragged_multiprocess_cell(global_mesh)
    multihost_utils.sync_global_devices("multiprocess-eight-expert-ragged-cell")
    if jax.process_index() == 0:
        print("PASSED TVM-FFI grouped SwiGLU partitioning matrix", flush=True)


if __name__ == "__main__":
    main()
