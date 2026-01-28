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
from typing import Any, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn

import transformer_engine.jax as te
import transformer_engine.jax.flax as te_flax
from transformer_engine.jax.sharding import MeshResource


# =============================================================================
# Model Configurations (matching jeremy_bench.py for apples-to-apples comparison)
# =============================================================================

MODEL_CONFIGS = {
    "small": {
        "batch_size": 8,
        "seq_length": 512,
        "hidden_size": 1024,
        "mlp_hidden_size": 4096,
        "num_heads": 16,
        "num_gqa_groups": 16,
        "num_layers": 4,
    },
    "medium": {
        "batch_size": 4,
        "seq_length": 1024,
        "hidden_size": 2048,
        "mlp_hidden_size": 8192,
        "num_heads": 32,
        "num_gqa_groups": 32,
        "num_layers": 4,
    },
    "large": {
        "batch_size": 2,
        "seq_length": 2048,
        "hidden_size": 4096,
        "mlp_hidden_size": 16384,
        "num_heads": 32,
        "num_gqa_groups": 32,
        "num_layers": 4,
    },
    "llama-7b": {
        "batch_size": 1,
        "seq_length": 2048,
        "hidden_size": 4096,
        "mlp_hidden_size": 11008,
        "num_heads": 32,
        "num_gqa_groups": 32,
        "num_layers": 4,
    },
    "llama-70b": {
        "batch_size": 1,
        "seq_length": 2048,
        "hidden_size": 8192,
        "mlp_hidden_size": 28672,
        "num_heads": 64,
        "num_gqa_groups": 8,
        "num_layers": 4,
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
    num_gqa_groups: int = None  # Defaults to num_heads if not specified
    num_layers: int = 1  # Number of stacked layers
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


class StackedTransformerLayers(nn.Module):
    """A stack of TransformerLayer decoder layers for benchmarking.

    Matches jeremy_bench.py StackedDecoderLayers for apples-to-apples comparison.
    """

    num_layers: int
    hidden_size: int
    mlp_hidden_size: int
    num_attention_heads: int
    num_gqa_groups: int
    layernorm_type: str = "rmsnorm"
    hidden_dropout: float = 0.0
    attention_dropout: float = 0.0
    intermediate_dropout: float = 0.0
    mlp_activations: Tuple[str, ...] = ("silu", "linear")
    transpose_batch_sequence: bool = False
    self_attn_mask_type: str = "causal"
    enable_rotary_pos_emb: bool = True
    rotary_pos_emb_group_method: str = "consecutive"

    @nn.compact
    def __call__(
        self,
        inputs: jnp.ndarray,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        """Forward pass through all decoder layers."""
        from transformer_engine.jax.flax import TransformerLayerType

        hidden_states = inputs
        for i in range(self.num_layers):
            hidden_states = te_flax.TransformerLayer(
                layer_type=TransformerLayerType.DECODER,
                hidden_size=self.hidden_size,
                mlp_hidden_size=self.mlp_hidden_size,
                num_attention_heads=self.num_attention_heads,
                num_gqa_groups=self.num_gqa_groups,
                layernorm_type=self.layernorm_type,
                hidden_dropout=self.hidden_dropout,
                attention_dropout=self.attention_dropout,
                intermediate_dropout=self.intermediate_dropout,
                mlp_activations=self.mlp_activations,
                transpose_batch_sequence=self.transpose_batch_sequence,
                self_attn_mask_type=self.self_attn_mask_type,
                enable_rotary_pos_emb=self.enable_rotary_pos_emb,
                rotary_pos_emb_group_method=self.rotary_pos_emb_group_method,
                enable_relative_embedding=False,
                name=f"decoder_layer_{i}",
            )(
                hidden_states,
                deterministic=deterministic,
            )
        return hidden_states


def create_transformer_layer(config: BenchmarkConfig) -> nn.Module:
    """Create a TransformerLayer (or stack) with the given configuration."""
    num_gqa_groups = config.num_gqa_groups or config.num_heads

    if config.num_layers == 1:
        # Single layer for backward compatibility
        return te_flax.TransformerLayer(
            hidden_size=config.hidden_size,
            mlp_hidden_size=config.mlp_hidden_size,
            num_attention_heads=config.num_heads,
            num_gqa_groups=num_gqa_groups,
            mlp_activations=("silu", "linear"),
            self_attn_mask_type="causal",
            layernorm_type="rmsnorm",
            layernorm_epsilon=1e-5,
            use_bias=True,
            attention_dropout=0.0,
            intermediate_dropout=0.0,
            hidden_dropout=0.0,
            enable_relative_embedding=False,
            enable_rotary_pos_emb=True,
            rotary_pos_emb_group_method="consecutive",
            self_attn_bias_type="no_bias",
            dtype=config.dtype,
            transpose_batch_sequence=False,
        )
    else:
        # Stacked layers matching jeremy_bench.py
        return StackedTransformerLayers(
            num_layers=config.num_layers,
            hidden_size=config.hidden_size,
            mlp_hidden_size=config.mlp_hidden_size,
            num_attention_heads=config.num_heads,
            num_gqa_groups=num_gqa_groups,
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
        help="Use a predefined model configuration (small, medium, large, llama-7b, llama-70b)",
    )
    parser.add_argument("--hidden_size", type=int, default=None, help="Hidden dimension size")
    parser.add_argument("--seq_length", type=int, default=None, help="Sequence length")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size")
    parser.add_argument("--num_heads", type=int, default=None, help="Number of attention heads")
    parser.add_argument(
        "--mlp_hidden_size",
        type=int,
        default=None,
        help="MLP hidden size (default: 4 * hidden_size)",
    )
    parser.add_argument(
        "--num_gqa_groups",
        type=int,
        default=None,
        help="Number of GQA groups (default: same as num_heads)",
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=None,
        help="Number of stacked transformer layers (default: 4 for configs, 1 otherwise)",
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

    # Determine configuration - start with defaults or predefined config
    if args.config:
        model_config = MODEL_CONFIGS[args.config]
        hidden_size = model_config["hidden_size"]
        num_heads = model_config["num_heads"]
        mlp_hidden_size = model_config["mlp_hidden_size"]
        seq_length = model_config["seq_length"]
        batch_size = model_config["batch_size"]
        num_gqa_groups = model_config["num_gqa_groups"]
        num_layers = model_config["num_layers"]
    else:
        # Custom config - require essential parameters
        hidden_size = args.hidden_size or 4096
        num_heads = args.num_heads or 32
        mlp_hidden_size = args.mlp_hidden_size or (4 * hidden_size)
        seq_length = args.seq_length or 2048
        batch_size = args.batch_size or 8
        num_gqa_groups = args.num_gqa_groups or num_heads
        num_layers = args.num_layers or 1

    # Allow CLI args to override predefined config values
    if args.hidden_size is not None:
        hidden_size = args.hidden_size
    if args.num_heads is not None:
        num_heads = args.num_heads
    if args.mlp_hidden_size is not None:
        mlp_hidden_size = args.mlp_hidden_size
    if args.seq_length is not None:
        seq_length = args.seq_length
    if args.batch_size is not None:
        batch_size = args.batch_size
    if args.num_gqa_groups is not None:
        num_gqa_groups = args.num_gqa_groups
    if args.num_layers is not None:
        num_layers = args.num_layers

    config = BenchmarkConfig(
        hidden_size=hidden_size,
        num_heads=num_heads,
        mlp_hidden_size=mlp_hidden_size,
        seq_length=seq_length,
        batch_size=batch_size,
        warmup_iters=args.warmup,
        timing_iters=args.timing_iters,
        num_rounds=args.num_rounds,
        num_gqa_groups=num_gqa_groups,
        num_layers=num_layers,
    )

    # Print header
    print("=" * 60)
    print("TransformerLayer JAX Benchmark")
    print("=" * 60)
    print(f"Config: hidden={config.hidden_size}, seq={config.seq_length}, "
          f"batch={config.batch_size}, heads={config.num_heads}, dtype=bfloat16")
    print(f"MLP hidden: {config.mlp_hidden_size}, GQA groups: {config.num_gqa_groups}, "
          f"Layers: {config.num_layers}")
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
