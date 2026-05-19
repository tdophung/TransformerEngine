# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Minimal standalone repro to bisect where the TRITON moe path hangs.

Run with:

    TRITON_PRINT_AUTOTUNING=1 \
    JAX_LOG_COMPILES=1 \
    python tests/jax/repro_triton_hang.py 2>&1 | tee triton_repro.log

Each phase prints its name BEFORE running so a hang's culprit is
obvious from the last printed line.
"""

import os
import sys
import time

import jax
import jax.numpy as jnp


def _log(msg: str) -> None:
    sys.stdout.write(f"[t={time.monotonic():.2f}s] {msg}\n")
    sys.stdout.flush()


def main():
    DTYPE = jnp.float32
    BATCH, SEQ, H, M = 2, 16, 32, 64
    E, K = 8, 2
    T = BATCH * SEQ

    _log("imports: starting")
    from transformer_engine.jax.moe import PermutationBackend, moe
    from transformer_engine.jax.triton_extensions.permutation import (
        make_row_id_map,
        permute_with_mask_map,
        unpermute_with_mask_map,
    )
    from transformer_engine.jax import cpp_extensions as tex

    _log("imports: done")

    key = jax.random.PRNGKey(0)
    kp, kx = jax.random.split(key)
    x = jax.random.normal(kx, (BATCH, SEQ, H), dtype=DTYPE)
    init = jax.nn.initializers.variance_scaling(1.0, "fan_in", "truncated_normal")
    kg, kw0, kw1, kwo = jax.random.split(kp, 4)
    gate_kernel = init(kg, (H, E), DTYPE)
    wi_0 = init(kw0, (E, H, M), DTYPE)
    wi_1 = init(kw1, (E, H, M), DTYPE)
    wo = init(kwo, (E, M, H), DTYPE)
    _log(f"shapes: x={x.shape} gate={gate_kernel.shape} wi0={wi_0.shape}")

    # -----------------------------------------------------------------
    # Phase 1: make_row_id_map only.
    # -----------------------------------------------------------------
    _log("phase 1: make_row_id_map (eager, no jit)")
    routing_map = jnp.zeros((T, E), dtype=jnp.bool_)
    routing_map = routing_map.at[jnp.arange(T), jnp.arange(T) % E].set(True)
    routing_map = routing_map.at[jnp.arange(T), (jnp.arange(T) + 1) % E].set(True)
    row_id_map = make_row_id_map(routing_map, T, E)
    row_id_map.block_until_ready()
    _log(f"phase 1: done -- row_id_map.shape={row_id_map.shape}")

    # -----------------------------------------------------------------
    # Phase 2: permute_with_mask_map only.
    # -----------------------------------------------------------------
    _log("phase 2: permute_with_mask_map (eager, no jit)")
    x_2d = x.reshape(T, H)
    sorted_x, _ = permute_with_mask_map(x_2d, row_id_map, None, T, E, T * K, H)
    sorted_x.block_until_ready()
    _log(f"phase 2: done -- sorted_x.shape={sorted_x.shape}")

    # -----------------------------------------------------------------
    # Phase 3: unpermute_with_mask_map only.
    # -----------------------------------------------------------------
    _log("phase 3: unpermute_with_mask_map (eager, no jit)")
    merging = jnp.ones((T, E), dtype=DTYPE) * (1.0 / K)
    out_2d, _ = unpermute_with_mask_map(sorted_x, row_id_map, merging, None, T, E, H)
    out_2d.block_until_ready()
    _log(f"phase 3: done -- out_2d.shape={out_2d.shape}")

    # -----------------------------------------------------------------
    # Phase 4: grouped_quantize + grouped_gemm only (FFN building blocks
    # -- these are shared with the pure_jax path so they should be fine,
    # but worth measuring in isolation).
    # -----------------------------------------------------------------
    _log("phase 4: grouped_quantize + grouped_gemm (eager)")
    from transformer_engine.jax.quantize import noop_quantizer_set, TensorUsage

    group_sizes = jnp.full((E,), T * K // E, dtype=jnp.int32)
    cs = tex.grouped_quantize(sorted_x, noop_quantizer_set.x, group_sizes, flatten_axis=-1)
    cw = tex.grouped_quantize(wi_0, noop_quantizer_set.kernel, flatten_axis=-1)
    out = tex.grouped_gemm(
        cs.get_tensor(usage=TensorUsage.LHS),
        cw.get_tensor(usage=TensorUsage.RHS),
        contracting_dims=((1,), (1,)),
    )
    out.block_until_ready()
    _log(f"phase 4: done -- out.shape={out.shape}")

    # -----------------------------------------------------------------
    # Phase 5: full TRITON forward, eager (no jit, no grad).
    # -----------------------------------------------------------------
    _log("phase 5: full triton forward (eager, no jit, no grad)")
    out_te, _ = moe(
        x,
        gate_kernel,
        wi_0,
        wi_1,
        wo,
        num_experts=E,
        num_experts_per_tok=K,
        activation_type="silu",
        score_function="softmax",
        use_pre_softmax=False,
        scaling_factor=1.0,
        aux_loss_coeff=0.0,
        permutation_backend=PermutationBackend.TRITON,
        align_size=0,
        dtype=DTYPE,
    )
    out_te.block_until_ready()
    _log(f"phase 5: done -- out_te.shape={out_te.shape}")

    # -----------------------------------------------------------------
    # Phase 6: jit'd forward.
    # -----------------------------------------------------------------
    _log("phase 6: full triton forward (jit'd)")

    @jax.jit
    def _fwd(x, gate_kernel, wi_0, wi_1, wo):
        return moe(
            x,
            gate_kernel,
            wi_0,
            wi_1,
            wo,
            num_experts=E,
            num_experts_per_tok=K,
            activation_type="silu",
            score_function="softmax",
            use_pre_softmax=False,
            scaling_factor=1.0,
            aux_loss_coeff=0.0,
            permutation_backend=PermutationBackend.TRITON,
            align_size=0,
            dtype=DTYPE,
        )

    out_te2, _ = _fwd(x, gate_kernel, wi_0, wi_1, wo)
    out_te2.block_until_ready()
    _log(f"phase 6: done -- out_te2.shape={out_te2.shape}")

    # -----------------------------------------------------------------
    # Phase 7: jit'd grad (this is what test_grads_finite_and_nonzero hits).
    # -----------------------------------------------------------------
    _log("phase 7: jit'd grad of mean(out**2)")

    @jax.jit
    def _grad_loss(x, gate_kernel, wi_0, wi_1, wo):
        def loss(*args):
            o, _ = moe(
                *args,
                num_experts=E,
                num_experts_per_tok=K,
                activation_type="silu",
                score_function="softmax",
                use_pre_softmax=False,
                scaling_factor=1.0,
                aux_loss_coeff=0.0,
                permutation_backend=PermutationBackend.TRITON,
                align_size=0,
                dtype=DTYPE,
            )
            return jnp.mean(o**2)

        return jax.grad(loss, argnums=(1, 2, 3, 4))(x, gate_kernel, wi_0, wi_1, wo)

    g_gate, g_wi0, g_wi1, g_wo = _grad_loss(x, gate_kernel, wi_0, wi_1, wo)
    g_gate.block_until_ready()
    _log(f"phase 7: done -- g_gate.shape={g_gate.shape}")

    _log("ALL PHASES DONE")


if __name__ == "__main__":
    main()
