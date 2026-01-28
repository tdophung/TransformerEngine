# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""
JAX TransformerLayer Benchmarking Suite
========================================

Benchmarks transformer_engine.jax.flax.TransformerLayer with:
- Forward-only (inference) timing
- Forward+backward (training) timing
- Optional Perfetto profiler integration for kernel breakdown

Usage:
    python benchmark_transformer_layer_jax.py --config large --warmup 10 --timing_iters 50
    python benchmark_transformer_layer_jax.py --hidden_size 4096 --seq_length 2048 --batch_size 8 --profile
"""

import argparse
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn

import transformer_engine.jax as te
import transformer_engine.jax.flax as te_flax
from transformer_engine.jax.sharding import MeshResource


# =============================================================================
# Model Configurations
# =============================================================================

MODEL_CONFIGS = {
    "small": {
        "hidden_size": 1024,
        "num_heads": 8,
        "mlp_hidden_size": 4096,
        "seq_length": 512,
    },
    "medium": {
        "hidden_size": 2048,
        "num_heads": 16,
        "mlp_hidden_size": 8192,
        "seq_length": 1024,
    },
    "large": {
        "hidden_size": 4096,
        "num_heads": 32,
        "mlp_hidden_size": 16384,
        "seq_length": 2048,
    },
}


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run."""

    hidden_size: int
    num_heads: int
    mlp_hidden_size: int
    seq_length: int
    batch_size: int
    warmup_iters: int
    timing_iters: int
    num_rounds: int
    dtype: jnp.dtype = jnp.bfloat16


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""

    times_ms: List[float]
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float


# =============================================================================
# Benchmark Functions
# =============================================================================


def create_transformer_layer(config: BenchmarkConfig) -> te_flax.TransformerLayer:
    """Create a TransformerLayer with the given configuration."""
    return te_flax.TransformerLayer(
        hidden_size=config.hidden_size,
        mlp_hidden_size=config.mlp_hidden_size,
        num_attention_heads=config.num_heads,
        mlp_activations=("gelu",),
        self_attn_mask_type="causal",
        layernorm_epsilon=1e-5,
        use_bias=True,
        attention_dropout=0.0,
        intermediate_dropout=0.0,
        hidden_dropout=0.0,
        enable_relative_embedding=False,
        self_attn_bias_type="no_bias",
        dtype=config.dtype,
        transpose_batch_sequence=False,
    )


def benchmark_forward(
    model: nn.Module,
    params: Any,
    x: jnp.ndarray,
    config: BenchmarkConfig,
    mesh_resource: MeshResource,
) -> BenchmarkResult:
    """Benchmark forward-only (inference) pass.

    Args:
        model: The TransformerLayer model
        params: Model parameters
        x: Input tensor
        config: Benchmark configuration
        mesh_resource: TE mesh resource

    Returns:
        BenchmarkResult with timing statistics
    """

    @jax.jit
    def forward_fn(params, x):
        return model.apply(params, x, deterministic=True)

    # Warmup runs
    for _ in range(config.warmup_iters):
        output = forward_fn(params, x)
        jax.block_until_ready(output)

    # Timing runs across multiple rounds
    all_times = []
    for round_idx in range(config.num_rounds):
        round_times = []
        for _ in range(config.timing_iters):
            start = time.perf_counter()
            output = forward_fn(params, x)
            jax.block_until_ready(output)
            elapsed = (time.perf_counter() - start) * 1000  # Convert to ms
            round_times.append(elapsed)

        avg_time = sum(round_times) / len(round_times)
        all_times.append(avg_time)
        print(f"  Round {round_idx + 1}/{config.num_rounds}: {avg_time:.2f} ms")

    return BenchmarkResult(
        times_ms=all_times,
        mean_ms=np.mean(all_times),
        std_ms=np.std(all_times),
        min_ms=np.min(all_times),
        max_ms=np.max(all_times),
    )


def benchmark_forward_backward(
    model: nn.Module,
    params: Any,
    x: jnp.ndarray,
    config: BenchmarkConfig,
    mesh_resource: MeshResource,
) -> BenchmarkResult:
    """Benchmark forward + backward (training) pass.

    Args:
        model: The TransformerLayer model
        params: Model parameters
        x: Input tensor
        config: Benchmark configuration
        mesh_resource: TE mesh resource

    Returns:
        BenchmarkResult with timing statistics
    """

    def loss_fn(params, x):
        y = model.apply(params, x, deterministic=True)
        return jnp.sum(y)

    grad_fn = jax.jit(jax.value_and_grad(loss_fn))

    # Warmup runs
    for _ in range(config.warmup_iters):
        loss, grads = grad_fn(params, x)
        jax.block_until_ready((loss, grads))

    # Timing runs across multiple rounds
    all_times = []
    for round_idx in range(config.num_rounds):
        round_times = []
        for _ in range(config.timing_iters):
            start = time.perf_counter()
            loss, grads = grad_fn(params, x)
            jax.block_until_ready((loss, grads))
            elapsed = (time.perf_counter() - start) * 1000  # Convert to ms
            round_times.append(elapsed)

        avg_time = sum(round_times) / len(round_times)
        all_times.append(avg_time)
        print(f"  Round {round_idx + 1}/{config.num_rounds}: {avg_time:.2f} ms")

    return BenchmarkResult(
        times_ms=all_times,
        mean_ms=np.mean(all_times),
        std_ms=np.std(all_times),
        min_ms=np.min(all_times),
        max_ms=np.max(all_times),
    )


def run_with_profiler(
    model: nn.Module,
    params: Any,
    x: jnp.ndarray,
    output_dir: str,
    profile_iters: int = 5,
) -> None:
    """Run a few iterations with JAX profiler to generate Perfetto traces.

    Args:
        model: The TransformerLayer model
        params: Model parameters
        x: Input tensor
        output_dir: Directory to save profiler output
        profile_iters: Number of iterations to profile
    """
    os.makedirs(output_dir, exist_ok=True)

    @jax.jit
    def forward_fn(params, x):
        return model.apply(params, x, deterministic=True)

    def loss_fn(params, x):
        y = model.apply(params, x, deterministic=True)
        return jnp.sum(y)

    grad_fn = jax.jit(jax.value_and_grad(loss_fn))

    # Warmup before profiling
    for _ in range(3):
        output = forward_fn(params, x)
        jax.block_until_ready(output)
        loss, grads = grad_fn(params, x)
        jax.block_until_ready((loss, grads))

    print(f"\nCapturing profiler trace ({profile_iters} iterations)...")

    # Profile with JAX profiler
    with jax.profiler.trace(output_dir, create_perfetto_link=False):
        for _ in range(profile_iters):
            # Forward only
            output = forward_fn(params, x)
            jax.block_until_ready(output)

            # Forward + backward
            loss, grads = grad_fn(params, x)
            jax.block_until_ready((loss, grads))

    print(f"\nProfiler trace saved to: {output_dir}/")
    print("  View at: https://ui.perfetto.dev (upload perfetto_trace.json.gz)")


def print_result(name: str, result: BenchmarkResult) -> None:
    """Print formatted benchmark results."""
    print(f"  Mean: {result.mean_ms:.2f} ms, Std: {result.std_ms:.2f} ms, "
          f"Min: {result.min_ms:.2f} ms, Max: {result.max_ms:.2f} ms")


# =============================================================================
# Main
# =============================================================================


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark TransformerLayer JAX performance",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model configuration
    parser.add_argument(
        "--config",
        type=str,
        choices=list(MODEL_CONFIGS.keys()),
        default=None,
        help="Use a predefined model configuration",
    )
    parser.add_argument("--hidden_size", type=int, default=4096, help="Hidden dimension size")
    parser.add_argument("--seq_length", type=int, default=2048, help="Sequence length")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--num_heads", type=int, default=32, help="Number of attention heads")
    parser.add_argument(
        "--mlp_hidden_size",
        type=int,
        default=None,
        help="MLP hidden size (default: 4 * hidden_size)",
    )

    # Benchmark parameters
    parser.add_argument("--warmup", type=int, default=10, help="Number of warmup iterations")
    parser.add_argument("--timing_iters", type=int, default=50, help="Number of timing iterations per round")
    parser.add_argument("--num_rounds", type=int, default=3, help="Number of benchmark rounds")

    # Profiler
    parser.add_argument("--profile", action="store_true", help="Enable Perfetto profiling")
    parser.add_argument(
        "--profile_dir",
        type=str,
        default="./jax_trace",
        help="Directory to save profiler traces",
    )
    parser.add_argument(
        "--profile_iters",
        type=int,
        default=5,
        help="Number of iterations to capture in profile",
    )

    # Benchmark mode
    parser.add_argument(
        "--forward_only",
        action="store_true",
        help="Only run forward (inference) benchmark",
    )
    parser.add_argument(
        "--backward_only",
        action="store_true",
        help="Only run forward+backward (training) benchmark",
    )

    return parser.parse_args()


def main() -> None:
    """Main benchmark entry point."""
    args = parse_args()

    # Determine configuration
    if args.config:
        model_config = MODEL_CONFIGS[args.config]
        hidden_size = model_config["hidden_size"]
        num_heads = model_config["num_heads"]
        mlp_hidden_size = model_config["mlp_hidden_size"]
        seq_length = model_config["seq_length"]
    else:
        hidden_size = args.hidden_size
        num_heads = args.num_heads
        mlp_hidden_size = args.mlp_hidden_size or (4 * hidden_size)
        seq_length = args.seq_length

    config = BenchmarkConfig(
        hidden_size=hidden_size,
        num_heads=num_heads,
        mlp_hidden_size=mlp_hidden_size,
        seq_length=seq_length,
        batch_size=args.batch_size,
        warmup_iters=args.warmup,
        timing_iters=args.timing_iters,
        num_rounds=args.num_rounds,
    )

    # Print header
    print("=" * 60)
    print("TransformerLayer JAX Benchmark")
    print("=" * 60)
    print(f"Config: hidden={config.hidden_size}, seq={config.seq_length}, "
          f"batch={config.batch_size}, heads={config.num_heads}, dtype=bfloat16")
    print(f"MLP hidden: {config.mlp_hidden_size}")
    print(f"Warmup: {config.warmup_iters}, Timing iters: {config.timing_iters}, "
          f"Rounds: {config.num_rounds}")
    print()

    # Print device info
    devices = jax.devices()
    print(f"JAX devices: {len(devices)} x {devices[0].device_kind}")
    print()

    # Create synthetic data
    key = jax.random.PRNGKey(42)
    x = jax.random.normal(
        key, (config.batch_size, config.seq_length, config.hidden_size)
    ).astype(config.dtype)

    mesh_resource = MeshResource()

    # Create and initialize model
    model = create_transformer_layer(config)

    with te.autocast(enabled=False, mesh_resource=mesh_resource):
        params = model.init(key, x, deterministic=False)

    # Determine which benchmarks to run
    run_forward = not args.backward_only
    run_backward = not args.forward_only

    # Run benchmarks
    if run_forward:
        print("Forward Only (Inference):")
        print("-" * 40)
        forward_result = benchmark_forward(model, params, x, config, mesh_resource)
        print_result("Forward", forward_result)
        print()

    if run_backward:
        print("Forward + Backward (Training):")
        print("-" * 40)
        backward_result = benchmark_forward_backward(model, params, x, config, mesh_resource)
        print_result("Forward+Backward", backward_result)
        print()

    # Run profiler if requested
    if args.profile:
        run_with_profiler(model, params, x, args.profile_dir, args.profile_iters)

    print("=" * 60)
    print("Benchmark complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
