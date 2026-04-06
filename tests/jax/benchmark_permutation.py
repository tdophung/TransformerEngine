# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Benchmark token_dispatch / token_combine / sort_chunks_by_index.

Compares Triton autotuning ON against each individual BLOCK_SIZE candidate.

Env vars that control the run
------------------------------
NVTE_DISABLE_TRITON_AUTOTUNING=1   disable autotuning, use single config
NVTE_TRITON_BLOCK_SIZE=N           when autotuning disabled, pin BLOCK_SIZE to N
                                   (valid: 64 128 256 512 1024 2048 4096)

Usage examples
--------------
# autotuning ON (default)
python3 benchmark_permutation.py

# fixed BLOCK_SIZE=256
NVTE_DISABLE_TRITON_AUTOTUNING=1 NVTE_TRITON_BLOCK_SIZE=256 python3 benchmark_permutation.py
"""

import os
import time

import jax
import jax.numpy as jnp
import numpy as np

# Disable JAX persistent compilation cache so every run measures real compile time.
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", "")

# ── constants ──────────────────────────────────────────────────────────────────

N_WARMUP = 5
N_RUNS = 30

# Realistic MoE cases: (num_tokens, num_experts, hidden_size, topk)
DISPATCH_CASES = [
    (4096, 32, 1280, 2),
    (4096, 256, 4096, 6),
    (16384, 64, 2048, 4),  # large-batch / many-expert case
]

# sort_chunks cases: (num_splits, total_tokens, hidden_size)
SORT_CASES = [
    (32, 4096, 1280),
    (64, 4096, 4096),
    (64, 16384, 2048),
]

# ── helpers ────────────────────────────────────────────────────────────────────


def _generate_routing_map(num_tokens, num_experts, topk, key):
    routing_map = jnp.zeros((num_tokens, num_experts), dtype=jnp.int32)
    for i in range(num_tokens):
        key, sub = jax.random.split(key)
        idx = jax.random.choice(sub, num_experts, shape=(topk,), replace=False)
        routing_map = routing_map.at[i, idx].set(1)
    return routing_map


def _generate_sort_inputs(num_splits, total_tokens, hidden_size, key):
    key, sk, ik = jax.random.split(key, 3)
    # Generate valid split sizes that sum to total_tokens
    split_sizes = jax.random.randint(sk, (num_splits,), 10, total_tokens // num_splits)
    split_sizes = split_sizes.at[-1].set(total_tokens - jnp.sum(split_sizes[:-1]))
    sorted_indices = jax.random.permutation(ik, num_splits)
    inp = jax.random.uniform(key, (total_tokens, hidden_size), dtype=jnp.bfloat16, minval=-1, maxval=1)
    return inp, split_sizes, sorted_indices


def _time_fn(fn, *args):
    """Return (t_jit_s, steady_ms_array).

    First call (t_jit) includes XLA lowering + Triton PTX compilation.
    """
    t0 = time.perf_counter()
    out = fn(*args)
    jax.block_until_ready(out)
    t_jit = time.perf_counter() - t0

    for _ in range(N_WARMUP):
        jax.block_until_ready(fn(*args))

    times = []
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        times.append((time.perf_counter() - t0) * 1e3)

    return t_jit, np.array(times)


def _fmt(t_jit, t_arr):
    return (
        f"jit={t_jit:6.1f}s  "
        f"mean={t_arr.mean():6.3f}ms  "
        f"min={t_arr.min():6.3f}ms  "
        f"std={t_arr.std():.3f}ms"
    )


def _mode_label():
    disabled = os.environ.get("NVTE_DISABLE_TRITON_AUTOTUNING", "0") == "1"
    bs = os.environ.get("NVTE_TRITON_BLOCK_SIZE", "")
    if not disabled:
        return "AUTOTUNE_ON"
    if bs:
        return f"BLOCK_SIZE={bs}"
    return "AUTOTUNE_OFF(first_config)"


# ── benchmarks ─────────────────────────────────────────────────────────────────


def bench_dispatch_combine():
    from transformer_engine.jax.permutation import token_dispatch, token_combine  # noqa: PLC0415

    print("\n── token_dispatch / token_combine ─────────────────────────────────────")

    for num_tokens, num_experts, hidden_size, topk in DISPATCH_CASES:
        dtype = jnp.bfloat16
        key = jax.random.PRNGKey(42)
        key, ik, pk = jax.random.split(key, 3)
        inp = jax.random.uniform(ik, (num_tokens, hidden_size), dtype=dtype, minval=-1, maxval=1)
        routing_map = _generate_routing_map(num_tokens, num_experts, topk, key)
        probs = jax.random.uniform(pk, (num_tokens, num_experts), dtype=dtype, minval=0.1, maxval=1)
        num_out_tokens = num_tokens * topk
        uniform_merging = routing_map.astype(dtype) / jnp.maximum(
            jnp.sum(routing_map, axis=1, keepdims=True), 1.0
        )

        hdr = f"T={num_tokens:5d}  E={num_experts:3d}  H={hidden_size:4d}  topk={topk}"
        print(f"\n  {hdr}")

        @jax.jit
        def dispatch_fwd(x, rm, p):
            out, pp, rid, _, _ = token_dispatch(x, rm, num_out_tokens, probs=p)
            return out, pp, rid

        t, ts = _time_fn(dispatch_fwd, inp, routing_map, probs)
        print(f"    dispatch_fwd    {_fmt(t, ts)}")

        dispatched, _, rid_map = dispatch_fwd(inp, routing_map, probs)
        jax.block_until_ready((dispatched, rid_map))

        @jax.jit
        def combine_fwd(d, r, mp):
            return token_combine(d, r, mp)

        t, ts = _time_fn(combine_fwd, dispatched, rid_map, uniform_merging)
        print(f"    combine_fwd     {_fmt(t, ts)}")

        @jax.jit
        def roundtrip_fwd(x, rm, mp):
            d, _, rid, _, _ = token_dispatch(x, rm, num_out_tokens)
            return token_combine(d, rid, mp)

        t, ts = _time_fn(roundtrip_fwd, inp, routing_map, uniform_merging)
        print(f"    roundtrip_fwd   {_fmt(t, ts)}")

        def roundtrip_loss(x, rm, mp):
            d, _, rid, _, _ = token_dispatch(x, rm, num_out_tokens)
            return jnp.sum(token_combine(d, rid, mp) ** 2)

        grad_fn = jax.jit(jax.grad(roundtrip_loss, argnums=0))
        t, ts = _time_fn(grad_fn, inp, routing_map, uniform_merging)
        print(f"    roundtrip_bwd   {_fmt(t, ts)}")


def bench_sort_chunks():
    from transformer_engine.jax.permutation import sort_chunks_by_index  # noqa: PLC0415

    print("\n── sort_chunks_by_index ───────────────────────────────────────────────")

    for num_splits, total_tokens, hidden_size in SORT_CASES:
        key = jax.random.PRNGKey(42)
        inp, split_sizes, sorted_indices = _generate_sort_inputs(
            num_splits, total_tokens, hidden_size, key
        )

        hdr = f"splits={num_splits:3d}  T={total_tokens:5d}  H={hidden_size:4d}"
        print(f"\n  {hdr}")

        @jax.jit
        def sort_fwd(x, ss, si):
            out, _ = sort_chunks_by_index(x, ss, si)
            return out

        t, ts = _time_fn(sort_fwd, inp, split_sizes, sorted_indices)
        print(f"    sort_fwd        {_fmt(t, ts)}")

        def sort_loss(x, ss, si):
            out, _ = sort_chunks_by_index(x, ss, si)
            return jnp.sum(out**2)

        grad_fn = jax.jit(jax.grad(sort_loss, argnums=0))
        t, ts = _time_fn(grad_fn, inp, split_sizes, sorted_indices)
        print(f"    sort_bwd        {_fmt(t, ts)}")


def run_benchmarks():
    mode = _mode_label()

    print(f"\n{'='*72}")
    print(f" Triton Permutation Benchmark  [{mode}]")
    print(f" JAX {jax.__version__}   devices: {jax.devices()}")
    print(f"{'='*72}")

    bench_dispatch_combine()
    bench_sort_chunks()

    print(f"\n{'='*72}")
    print(f" Done [{mode}]")
    print(f"{'='*72}\n")


if __name__ == "__main__":
    run_benchmarks()
