#!/usr/bin/env python3
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Standalone equivalent of ``test_fwd_and_bwd_smoke[triton]``.

This script runs *literally* the same code body as the pytest test but with
no pytest, no conftest.py, no autouse fixtures, no plugins (jaxtyping,
typeguard, forked, anyio). Run with:

    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 \
    NVTE_TRITON_PERMUTATION_BLOCK_SIZES=128 \
    python3 tests/jax/standalone_smoke_triton.py 2>&1 | tee /tmp/standalone.log

If this **passes** while
``pytest -k 'test_fwd_and_bwd_smoke[triton]' tests/jax/test_distributed_moe_vjp.py``
hangs, the bug is in pytest's plugin / conftest layer (likely the
``import transformer_engine.jax`` at conftest module-level race, the
autouse ``clear_live_arrays`` fixture, the ``NVTE_FUSED_ATTN=1`` flip in
``enable_fused_attn_after_hopper``, or a typeguard/jaxtyping wrapper).

If this **also hangs**, the bug is in our application code; the
distributed_triton_hang.py repro is missing whatever the actual triggering
sequence is and we should add it here as a starting point.
"""

import os
import sys
import time
import faulthandler
import signal


_WATCHDOG_SECS = int(os.environ.get("MOE_VJP_WATCHDOG_SECS", "60") or "0")
faulthandler.enable()
try:
    faulthandler.register(signal.SIGUSR1, all_threads=True, chain=False)
except (AttributeError, ValueError):
    pass
if _WATCHDOG_SECS > 0:
    faulthandler.dump_traceback_later(_WATCHDOG_SECS, repeat=True)


import jax
import jax.numpy as jnp
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from flax.linen import partitioning as nn_partitioning


# Identical config to TestMoeVjpDistributedSmoke.test_fwd_and_bwd_smoke[triton]
EP_AXIS = "ep"
FSDP_AXIS = "fsdp"
EP_SIZE = 2
FSDP_SIZE = 2
LOGICAL_AXIS_RULES = (
    ("exp", EP_AXIS),
    ("embed", FSDP_AXIS),
    ("mlp", None),
    ("batch", (EP_AXIS, FSDP_AXIS)),
)
SMOKE_BATCH = 4
SMOKE_SEQ = 16
SMOKE_HIDDEN = 32
SMOKE_INTER = 64
SMOKE_NUM_EXPERTS = 4
SMOKE_TOPK = 2


def _hb(msg: str) -> None:
    sys.stdout.write(f"  [{time.strftime('%H:%M:%S')}] {msg}\n")
    sys.stdout.flush()


def main() -> int:
    if jax.device_count() < EP_SIZE * FSDP_SIZE:
        _hb(f"FATAL: need {EP_SIZE*FSDP_SIZE} devices, have {jax.device_count()}")
        return 1

    _hb("import transformer_engine.jax")
    from transformer_engine.jax.flax import _MoEBlock as MoEBlock
    from transformer_engine.jax.moe import PermutationBackend
    from transformer_engine.jax.sharding import MeshResource, global_shard_guard

    _hb("building mesh")
    devices = mesh_utils.create_device_mesh((EP_SIZE, FSDP_SIZE))
    mesh = Mesh(devices, axis_names=(EP_AXIS, FSDP_AXIS))

    _hb("building block")
    block = MoEBlock(
        num_experts=SMOKE_NUM_EXPERTS,
        num_experts_per_tok=SMOKE_TOPK,
        intermediate_size=SMOKE_INTER,
        permutation_backend=PermutationBackend.TRITON,
        data_parallelism_axes=(FSDP_AXIS,),
        aux_loss_coeff=0.0,
        dtype=jnp.bfloat16,
        _align_size=0,
    )

    x = jax.random.normal(
        jax.random.PRNGKey(0),
        (SMOKE_BATCH, SMOKE_SEQ, SMOKE_HIDDEN),
        dtype=jnp.bfloat16,
    )

    _hb("entering mesh + global_shard_guard + axis_rules")
    with mesh, global_shard_guard(
        MeshResource(ep_resource=EP_AXIS, fsdp_resource=FSDP_AXIS)
    ), nn_partitioning.axis_rules(LOGICAL_AXIS_RULES):
        x_sh = jax.lax.with_sharding_constraint(
            x, NamedSharding(mesh, P((EP_AXIS, FSDP_AXIS), None, None))
        )
        _hb("  -> jit(block.init)")
        variables = jax.jit(block.init)(jax.random.PRNGKey(1), x_sh)
        jax.block_until_ready(jax.tree_util.tree_leaves(variables)[0])
        _hb("  -> jit(block.apply)")
        output, aux = jax.jit(block.apply)(variables, x_sh)
        jax.block_until_ready(output)
    _hb(f"apply done -- output.shape={output.shape}, aux={aux}")

    # Grad step (matches _grad_step in the test)
    _hb("entering mesh ctx for grad")
    with mesh, global_shard_guard(
        MeshResource(ep_resource=EP_AXIS, fsdp_resource=FSDP_AXIS)
    ), nn_partitioning.axis_rules(LOGICAL_AXIS_RULES):
        x_sh = jax.lax.with_sharding_constraint(
            x, NamedSharding(mesh, P((EP_AXIS, FSDP_AXIS), None, None))
        )

        def loss_fn(variables, x):
            out, a = block.apply(variables, x)
            main = jnp.mean(out.astype(jnp.float32) ** 2)
            return main + (a.astype(jnp.float32) if a is not None else 0.0)

        _hb("  -> jit(grad(loss_fn))")
        grads = jax.jit(jax.grad(loss_fn))(variables, x_sh)
        jax.block_until_ready(jax.tree_util.tree_leaves(grads)[0])
    _hb("grad done")

    _hb("SUCCESS: standalone triton smoke completed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
