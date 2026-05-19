# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Distributed (2x2 ep,fsdp) bisection repro for the TRITON moe hang.

Run on a >=4-GPU node:

    TRITON_PRINT_AUTOTUNING=1 \
    JAX_LOG_COMPILES=1 \
    python tests/jax/repro_distributed_triton_hang.py 2>&1 \
        | tee dist_triton_repro.log

Each phase logs ``[t=...]`` BEFORE executing, so the last printed line
identifies the hang's call site. Phase ordering:

  1. mesh / axis_rules sanity check (no kernels)
  2. PURE_JAX block forward (should pass -- confirms wiring)
  3. TRITON dispatch helpers in isolation under shard_map
       3a. make_row_id_map only
       3b. permute_with_mask_map only
       3c. ragged_all_to_all only
       3d. sort_chunks_by_map only
       3e. unpermute_with_mask_map only
  4. TRITON full forward, eager
  5. TRITON full forward, jit'd
  6. TRITON full forward, jit'd  with rerun (cache hit -- should be fast)

A hang in 3a/3b/3e isolates the bug to a single triton kernel; in 4/5
it's a higher-level orchestration (shard_map spec mismatch, recv buffer
sizing, etc.).
"""

import os
import sys
import time

import jax
import jax.numpy as jnp
from jax.experimental import mesh_utils
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from flax.linen import partitioning as nn_partitioning


def _log(msg: str) -> None:
    sys.stdout.write(f"[t={time.monotonic():.2f}s] {msg}\n")
    sys.stdout.flush()


def main():
    EP_AXIS, FSDP_AXIS = "ep", "fsdp"
    EP, FSDP = 2, 2

    if jax.device_count() < EP * FSDP:
        _log(f"FATAL: need >={EP*FSDP} devices, have {jax.device_count()}")
        return

    devices = mesh_utils.create_device_mesh((EP, FSDP))
    mesh = Mesh(devices, axis_names=(EP_AXIS, FSDP_AXIS))
    _log(f"mesh built: {mesh}")

    DTYPE = jnp.bfloat16
    # Shapes are configurable via env vars so this script can run against
    # either the original 8x32x64 "medium" repro shape or against the
    # ``test_distributed_moe_vjp.py`` smoke shape (4x16x32) that exposes
    # the May-2026 in-process triton hang.
    #
    # Defaults match the smoke test exactly so a one-line ``python3
    # tests/jax/repro_distributed_triton_hang.py`` invocation reproduces
    # the hang. Set REPRO_SHAPE=medium to revert to the original.
    _shape = os.environ.get("REPRO_SHAPE", "smoke").lower()
    if _shape == "medium":
        BATCH = EP * FSDP * 2  # 8 -- two micro-batches per device
        SEQ, HIDDEN, INTER = 32, 64, 128
        E, K = 8, 2
    elif _shape == "smoke":
        BATCH = EP * FSDP  # 4 -- one micro-batch per device (smoke test)
        SEQ, HIDDEN, INTER = 16, 32, 64
        E, K = 4, 2
    else:
        raise ValueError(
            f"REPRO_SHAPE={_shape!r}; expected one of 'smoke' (smoke-test"
            " parity, exposes hang) or 'medium' (legacy)."
        )
    _log(
        f"shape config: REPRO_SHAPE={_shape} BATCH={BATCH} SEQ={SEQ}"
        f" HIDDEN={HIDDEN} INTER={INTER} E={E} K={K}"
    )
    LOGICAL_AXIS_RULES = (
        ("exp", EP_AXIS),
        ("embed", FSDP_AXIS),
        ("mlp", None),
        ("batch", (EP_AXIS, FSDP_AXIS)),
    )

    from transformer_engine.jax.flax import _MoEBlock as MoEBlock
    from transformer_engine.jax.moe import PermutationBackend
    from transformer_engine.jax.sharding import MeshResource, global_shard_guard
    from transformer_engine.jax.triton_extensions.permutation import (
        make_row_id_map,
        permute_with_mask_map,
        unpermute_with_mask_map,
        sort_chunks_by_map,
        make_chunk_sort_map,
    )
    _log("imports done")

    def _make_block(backend):
        return MoEBlock(
            num_experts=E, num_experts_per_tok=K,
            intermediate_size=INTER,
            permutation_backend=backend,
            data_parallelism_axes=(FSDP_AXIS,),
            dtype=DTYPE,
            _align_size=0,
        )

    x = jax.random.normal(jax.random.PRNGKey(0), (BATCH, SEQ, HIDDEN), dtype=DTYPE)

    # -----------------------------------------------------------------
    # Phase 1: open mesh / axis_rules context. No kernels.
    # -----------------------------------------------------------------
    _log("phase 1: open mesh + axis_rules + MeshResource (no kernels)")
    with mesh, global_shard_guard(
        MeshResource(ep_resource=EP_AXIS, fsdp_resource=FSDP_AXIS)
    ), nn_partitioning.axis_rules(LOGICAL_AXIS_RULES):
        pass
    _log("phase 1: done")

    # -----------------------------------------------------------------
    # Phase 2: PURE_JAX block forward (sanity).
    # -----------------------------------------------------------------
    _log("phase 2: PURE_JAX block forward (jit'd)")
    block_pj = _make_block(PermutationBackend.PURE_JAX)
    with mesh, global_shard_guard(
        MeshResource(ep_resource=EP_AXIS, fsdp_resource=FSDP_AXIS)
    ), nn_partitioning.axis_rules(LOGICAL_AXIS_RULES):
        x_sh = jax.lax.with_sharding_constraint(
            x, NamedSharding(mesh, P((EP_AXIS, FSDP_AXIS), None, None))
        )
        variables = jax.jit(block_pj.init)(jax.random.PRNGKey(1), x_sh)
        out_pj, _ = jax.jit(block_pj.apply)(variables, x_sh)
        out_pj.block_until_ready()
    _log(f"phase 2: done -- out_pj.shape={out_pj.shape}")

    # -----------------------------------------------------------------
    # Phase 3: TRITON dispatch primitives in isolation under shard_map.
    # Shapes per shard: each shard owns BATCH/(EP*FSDP)=2 batches of SEQ
    # tokens, so num_tokens_per_shard = 2*32 = 64, num_out_tokens = 128.
    # -----------------------------------------------------------------
    T_per_shard = (BATCH // (EP * FSDP)) * SEQ  # 2*32 = 64
    NUM_OUT_PER_SHARD = T_per_shard * K  # 128
    _log(
        f"phase 3 prep: per-shard T={T_per_shard} num_out={NUM_OUT_PER_SHARD} "
        f"H={HIDDEN} E={E}"
    )

    rng = jax.random.PRNGKey(42)
    rng_r, rng_x = jax.random.split(rng)
    # Build a fake routing map sharded over batch.
    routing_map_full = jax.random.bernoulli(
        rng_r, p=K / E, shape=(BATCH * SEQ, E)
    )
    x_2d_full = jax.random.normal(rng_x, (BATCH * SEQ, HIDDEN), dtype=DTYPE)

    spec_batch = P((EP_AXIS, FSDP_AXIS), None)

    with mesh, global_shard_guard(
        MeshResource(ep_resource=EP_AXIS, fsdp_resource=FSDP_AXIS)
    ), nn_partitioning.axis_rules(LOGICAL_AXIS_RULES):
        routing_map_full = jax.lax.with_sharding_constraint(
            routing_map_full, NamedSharding(mesh, spec_batch)
        )
        x_2d_full = jax.lax.with_sharding_constraint(
            x_2d_full, NamedSharding(mesh, spec_batch)
        )

        # --- 3a: make_row_id_map per shard ---
        _log("phase 3a: make_row_id_map under shard_map (jit'd)")

        @jax.jit
        def _fn_3a(rmap):
            def body(rmap_local):
                return make_row_id_map(rmap_local, T_per_shard, E)
            return shard_map(
                body, mesh=mesh,
                in_specs=(spec_batch,),
                out_specs=spec_batch, check_rep=False,
            )(rmap)

        row_id_map_full = _fn_3a(routing_map_full)
        row_id_map_full.block_until_ready()
        _log(f"phase 3a: done -- row_id_map_full.shape={row_id_map_full.shape}")

        # --- 3b: permute_with_mask_map per shard ---
        _log("phase 3b: permute_with_mask_map under shard_map (jit'd)")

        @jax.jit
        def _fn_3b(x2d, rmap_ids):
            def body(x2d_l, rmap_ids_l):
                sorted_x, _ = permute_with_mask_map(
                    x2d_l, rmap_ids_l, None,
                    T_per_shard, E, NUM_OUT_PER_SHARD, HIDDEN,
                )
                return sorted_x
            return shard_map(
                body, mesh=mesh,
                in_specs=(spec_batch, spec_batch),
                out_specs=spec_batch, check_rep=False,
            )(x2d, rmap_ids)

        sorted_x_full = _fn_3b(x_2d_full, row_id_map_full)
        sorted_x_full.block_until_ready()
        _log(f"phase 3b: done -- sorted_x_full.shape={sorted_x_full.shape}")

        # --- 3c: ragged_all_to_all on a tiny payload ---
        _log("phase 3c: ragged_all_to_all under shard_map (jit'd)")
        recv_rows = NUM_OUT_PER_SHARD * EP  # worst case
        send_sizes = jnp.full((EP,), NUM_OUT_PER_SHARD // EP, dtype=jnp.int32)
        send_offsets = jnp.cumsum(
            jnp.concatenate([jnp.array([0], dtype=jnp.int32), send_sizes[:-1]])
        )
        recv_sizes = send_sizes.copy()
        recv_offsets = jnp.cumsum(
            jnp.concatenate([jnp.array([0], dtype=jnp.int32), recv_sizes[:-1]])
        )

        @jax.jit
        def _fn_3c(sx):
            def body(sx_l):
                recv = jnp.zeros((recv_rows, HIDDEN), dtype=sx_l.dtype)
                return jax.lax.ragged_all_to_all(
                    sx_l, recv,
                    send_offsets, send_sizes, recv_offsets, recv_sizes,
                    axis_name=EP_AXIS,
                )
            return shard_map(
                body, mesh=mesh,
                in_specs=spec_batch,
                out_specs=P((EP_AXIS, FSDP_AXIS), None),
                check_rep=False,
            )(sx)

        sx_recv_full = _fn_3c(sorted_x_full)
        sx_recv_full.block_until_ready()
        _log(f"phase 3c: done -- sx_recv_full.shape={sx_recv_full.shape}")

    # -----------------------------------------------------------------
    # Phase 4: TRITON block forward, eager (no jit). Reuse pure_jax's
    # variables for shape parity.
    # -----------------------------------------------------------------
    _log("phase 4: TRITON block forward (eager, reuses PURE_JAX init)")
    block_tr = _make_block(PermutationBackend.TRITON)
    with mesh, global_shard_guard(
        MeshResource(ep_resource=EP_AXIS, fsdp_resource=FSDP_AXIS)
    ), nn_partitioning.axis_rules(LOGICAL_AXIS_RULES):
        x_sh = jax.lax.with_sharding_constraint(
            x, NamedSharding(mesh, P((EP_AXIS, FSDP_AXIS), None, None))
        )
        out_tr, _ = block_tr.apply(variables, x_sh)
        out_tr.block_until_ready()
    _log(f"phase 4: done -- out_tr.shape={out_tr.shape}")

    # -----------------------------------------------------------------
    # Phase 4b: TRITON block forward with TRITON init (this is what
    # the actual failing test does -- jit(block.init) traces the moe
    # forward with permutation_backend=TRITON and may compile a
    # different graph than jit(block.apply) does later).
    # -----------------------------------------------------------------
    _log("phase 4b: jit(block_tr.init) -- this is the first thing the failing test does")
    with mesh, global_shard_guard(
        MeshResource(ep_resource=EP_AXIS, fsdp_resource=FSDP_AXIS)
    ), nn_partitioning.axis_rules(LOGICAL_AXIS_RULES):
        x_sh = jax.lax.with_sharding_constraint(
            x, NamedSharding(mesh, P((EP_AXIS, FSDP_AXIS), None, None))
        )
        variables_tr = jax.jit(block_tr.init)(jax.random.PRNGKey(1), x_sh)
        jax.tree.map(lambda v: v.value.block_until_ready() if hasattr(v, "value") else v.block_until_ready(), variables_tr)
    _log("phase 4b: done")

    # -----------------------------------------------------------------
    # Phase 4c: TRITON block apply using TRITON-initialised variables
    # (i.e. the exact second call the failing test makes).
    # -----------------------------------------------------------------
    _log("phase 4c: jit(block_tr.apply)(variables_tr, x) -- the failing test's apply call")
    with mesh, global_shard_guard(
        MeshResource(ep_resource=EP_AXIS, fsdp_resource=FSDP_AXIS)
    ), nn_partitioning.axis_rules(LOGICAL_AXIS_RULES):
        x_sh = jax.lax.with_sharding_constraint(
            x, NamedSharding(mesh, P((EP_AXIS, FSDP_AXIS), None, None))
        )
        out_tr_init, _ = jax.jit(block_tr.apply)(variables_tr, x_sh)
        out_tr_init.block_until_ready()
    _log(f"phase 4c: done -- out_tr_init.shape={out_tr_init.shape}")

    # -----------------------------------------------------------------
    # Phase 4d: EXACTLY mirror what the failing test does -- init +
    # apply inside the same `with` block, then do assertion-style
    # access (jnp.isfinite + .item()) OUTSIDE the mesh/axis_rules
    # context. The .item() forces compute under no active mesh.
    # -----------------------------------------------------------------
    _log("phase 4d: init+apply inside `with`, .item() OUTSIDE the with-block")
    block_tr2 = _make_block(PermutationBackend.TRITON)
    with mesh, global_shard_guard(
        MeshResource(ep_resource=EP_AXIS, fsdp_resource=FSDP_AXIS)
    ), nn_partitioning.axis_rules(LOGICAL_AXIS_RULES):
        x_sh = jax.lax.with_sharding_constraint(
            x, NamedSharding(mesh, P((EP_AXIS, FSDP_AXIS), None, None))
        )
        variables_tr2 = jax.jit(block_tr2.init)(jax.random.PRNGKey(11), x_sh)
        out_4d, aux_4d = jax.jit(block_tr2.apply)(variables_tr2, x_sh)
    _log("phase 4d: with-block exited; now doing .item() on isfinite outside ctx")
    finite = jnp.all(jnp.isfinite(out_4d)).item()
    _log(f"phase 4d: done -- finite={finite} aux_4d_is_none={aux_4d is None}")

    # -----------------------------------------------------------------
    # Phase 5: TRITON block forward, jit'd.
    # -----------------------------------------------------------------
    _log("phase 5: TRITON block forward (jit'd)")
    with mesh, global_shard_guard(
        MeshResource(ep_resource=EP_AXIS, fsdp_resource=FSDP_AXIS)
    ), nn_partitioning.axis_rules(LOGICAL_AXIS_RULES):
        x_sh = jax.lax.with_sharding_constraint(
            x, NamedSharding(mesh, P((EP_AXIS, FSDP_AXIS), None, None))
        )
        out_tr2, _ = jax.jit(block_tr.apply)(variables, x_sh)
        out_tr2.block_until_ready()
    _log(f"phase 5: done -- out_tr2.shape={out_tr2.shape}")

    # -----------------------------------------------------------------
    # Phase 6: TRITON block forward, jit'd, rerun (cache hit).
    # -----------------------------------------------------------------
    _log("phase 6: TRITON block forward (rerun, expect cache hit)")
    with mesh, global_shard_guard(
        MeshResource(ep_resource=EP_AXIS, fsdp_resource=FSDP_AXIS)
    ), nn_partitioning.axis_rules(LOGICAL_AXIS_RULES):
        x_sh = jax.lax.with_sharding_constraint(
            x, NamedSharding(mesh, P((EP_AXIS, FSDP_AXIS), None, None))
        )
        out_tr3, _ = jax.jit(block_tr.apply)(variables, x_sh)
        out_tr3.block_until_ready()
    _log(f"phase 6: done -- out_tr3.shape={out_tr3.shape}")

    _log("ALL PHASES DONE")


if __name__ == "__main__":
    main()
