# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Shape-only reproducer for nvbug 6432162 and CuTeDSL MoE outputs.

This intentionally does not call the kernel. It isolates Shardy propagation
over the exact five output shapes used by the production EP2/FSDP2 slice.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.custom_partitioning import (
    CompoundFactor,
    SdyShardingRule,
    custom_partitioning,
)
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


EXPERTS = 16
M = 266240
HIDDEN = 1792
INTERMEDIATE = 2048
PAIR = 2


def _ceil_div(x, y):
    return (x + y - 1) // y


ROW_SCALE_SIZE = 32 * 4 * _ceil_div(M, 128) * 4 * _ceil_div(_ceil_div(INTERMEDIATE, 32), 4)
COL_SCALE_SIZE = 32 * 4 * _ceil_div(INTERMEDIATE, 128) * 4 * _ceil_div(_ceil_div(M, 32), 4)


def _shape_only_impl(a, b, sfa, sfb, padded_offsets, prob):
    del a, b, sfa, sfb, padded_offsets, prob
    return (
        jnp.zeros((M, PAIR * INTERMEDIATE, 1), dtype=jnp.bfloat16),
        jnp.zeros((M, INTERMEDIATE, 1), dtype=jnp.float8_e4m3fn),
        jnp.zeros((M, INTERMEDIATE, 1), dtype=jnp.float8_e4m3fn),
        jnp.zeros((ROW_SCALE_SIZE,), dtype=jnp.float8_e8m0fnu),
        jnp.zeros((COL_SCALE_SIZE,), dtype=jnp.float8_e8m0fnu),
    )


_shape_only = custom_partitioning(_shape_only_impl)


def _partition(mesh, arg_infos, result_infos):
    arg_shardings = tuple(info.sharding for info in arg_infos)
    result_shardings = tuple(info.sharding for info in result_infos)
    return mesh, _shape_only_impl, result_shardings, arg_shardings


def _shardy_rule(mesh, value_types, result_types):
    del mesh, value_types, result_types
    combined = CompoundFactor("intermediate", "swiglu_pair")
    return SdyShardingRule(
        (
            ("tokens", "hidden", "a_l"),
            ("experts", combined, "hidden"),
            ("sfa_flat",),
            ("sfb_flat",),
            ("experts",),
            ("tokens", "prob_n", "prob_l"),
        ),
        (
            ("tokens", combined, "combined_l"),
            ("tokens", "intermediate", "row_l"),
            ("tokens", "intermediate", "col_l"),
            ("row_scale_flat",),
            ("col_scale_flat",),
        ),
        swiglu_pair=PAIR,
    )


_shape_only.def_partition(partition=_partition, sharding_rule=_shardy_rule)


def main() -> None:
    jax.config.update("jax_use_shardy_partitioner", True)
    devices = np.asarray(jax.devices())
    if devices.size < 2:
        raise RuntimeError("This Shardy reproducer requires at least two JAX devices")
    ep = 2
    dp = devices.size // ep
    devices = devices[: dp * ep].reshape(dp, ep)
    mesh = Mesh(devices, ("data", "expert"))
    token_sharding = NamedSharding(mesh, P(("data", "expert"), None, None))
    expert_sharding = NamedSharding(mesh, P("expert", None, None))
    replicated = NamedSharding(mesh, P())

    args = (
        jax.ShapeDtypeStruct((M, HIDDEN, 1), jnp.float8_e4m3fn, sharding=token_sharding),
        jax.ShapeDtypeStruct(
            (EXPERTS, PAIR * INTERMEDIATE, HIDDEN),
            jnp.float8_e4m3fn,
            sharding=expert_sharding,
        ),
        jax.ShapeDtypeStruct((1,), jnp.float8_e8m0fnu, sharding=replicated),
        jax.ShapeDtypeStruct((1,), jnp.float8_e8m0fnu, sharding=replicated),
        jax.ShapeDtypeStruct(
            (EXPERTS,),
            jnp.int32,
            sharding=NamedSharding(mesh, P("expert")),
        ),
        jax.ShapeDtypeStruct((M, 1, 1), jnp.float32, sharding=token_sharding),
    )
    with jax.set_mesh(mesh):
        lowered = jax.jit(_shape_only).lower(*args)
    print(
        "Shardy lowering passed for CuTeDSL MoE outputs: "
        f"M={M}, I={INTERMEDIATE}, row_scale={ROW_SCALE_SIZE}, "
        f"col_scale={COL_SCALE_SIZE}, devices={math.prod(mesh.devices.shape)}"
    )
    print(lowered.compiler_ir(dialect="stablehlo"))


if __name__ == "__main__":
    main()
