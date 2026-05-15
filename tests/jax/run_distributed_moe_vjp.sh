#!/usr/bin/env bash
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
#
# Driver script for the multi-GPU MoE VJP tests on a single host.
#
# Layout:
#   * "Level 2" -- :class:`TestMoeVjpDistributedSmoke`. Small shapes, fast,
#     verifies shard_map ctx specs match, gradients are finite, and the
#     two permutation backends agree.
#   * "Level 3" -- :class:`TestMoeVjpDistributedPerf`. Mixtral-ish-shape
#     throughput. Reports tokens/sec and steps/sec.
#
# Usage from the TransformerEngine repo root (or any cwd; this script
# resolves its own path):
#
#   # Both levels (default; requires 4 GPUs):
#   bash tests/jax/run_distributed_moe_vjp.sh
#
#   # Just Level 2 (correctness; smaller and faster):
#   bash tests/jax/run_distributed_moe_vjp.sh smoke
#
#   # Just Level 3 (perf):
#   bash tests/jax/run_distributed_moe_vjp.sh perf
#
#   # A single test by name pattern (passed through to pytest -k):
#   bash tests/jax/run_distributed_moe_vjp.sh "test_pure_jax_triton_parity"
#
# Environment variables:
#   CUDA_VISIBLE_DEVICES -- defaults to "0,1,2,3". Override to use
#                           different GPUs.
#   PYTEST_EXTRA_ARGS    -- appended verbatim to the pytest invocation,
#                           e.g. PYTEST_EXTRA_ARGS="--maxfail=1 -x"
#
# Notes:
#   * Single-host multi-device. No SLURM, no jax.distributed.initialize
#     -- a single Python process drives all 4 GPUs via JAX's default
#     device discovery + a 2x2 (ep, fsdp) Mesh built inside the test.
#   * The tests are gated on the ``triton`` pytest marker so this script
#     is a no-op in environments where TE was built without the
#     fused-router CUDA kernel / Triton permutation backend.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TE_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TEST_FILE="tests/jax/test_distributed_moe_vjp.py"

mode="${1:-all}"

case "$mode" in
    smoke|level2)
        marker_args=("-m" "triton and not slow")
        kfilter=()
        ;;
    perf|level3)
        marker_args=("-m" "triton and slow")
        kfilter=()
        ;;
    all)
        marker_args=("-m" "triton")
        kfilter=()
        ;;
    *)
        # Treat anything else as a -k filter.
        marker_args=("-m" "triton")
        kfilter=("-k" "$mode")
        ;;
esac

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

cd "$TE_ROOT"

echo "============================================================"
echo "MoE VJP distributed tests"
echo "  mode                : $mode"
echo "  marker filter       : ${marker_args[*]}"
echo "  -k filter           : ${kfilter[*]:-<none>}"
echo "  CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "  test file           : $TEST_FILE"
echo "  PYTEST_EXTRA_ARGS   : ${PYTEST_EXTRA_ARGS:-<unset>}"
echo "============================================================"

# -s so the perf line in TestMoeVjpDistributedPerf is not captured.
# -v for one line per test result.
exec python3 -m pytest \
    "$TEST_FILE" \
    "${marker_args[@]}" \
    "${kfilter[@]}" \
    -v -s \
    ${PYTEST_EXTRA_ARGS:-}
