#!/usr/bin/env bash
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
#
# Dev-loop convenience wrapper for the distributed MoE VJP tests.
#
# This is NOT the CI invocation -- CI uses
# ``qa/L0_jax_distributed_unittest/test.sh`` which calls pytest with the
# exact same flags as this script. Keep the two in sync: any flag added
# here for correctness (``-p no:typeguard``, env vars, etc.) MUST also
# appear in the QA script and vice versa.
#
# Usage from the TransformerEngine repo root (or any cwd; this script
# resolves its own path):
#
#   # All tests (smoke + perf):
#   bash tests/jax/run_distributed_moe_vjp.sh
#
#   # Just smoke (Level 2 correctness):
#   bash tests/jax/run_distributed_moe_vjp.sh smoke
#
#   # Just perf (Level 3 throughput):
#   bash tests/jax/run_distributed_moe_vjp.sh perf
#
#   # A single test by name pattern (passed through to pytest -k):
#   bash tests/jax/run_distributed_moe_vjp.sh test_fwd_and_bwd_smoke
#
# Required environment / flags (mirrored from
# qa/L0_jax_distributed_unittest/test.sh):
#
#   * XLA_PYTHON_CLIENT_PREALLOCATE=false / MEM_FRACTION=0.5 -- prevents
#     NCCL OOM during EP all-to-all setup. JAX's default 90% HBM
#     preallocation leaves no room for the communicator.
#   * ``-p no:typeguard`` -- jaxtyping's pytest plugin auto-loads
#     typeguard, whose @typechecked import hook materialises JAX tracers
#     via isinstance() checks and deadlocks the first ``block.apply`` of
#     the triton backend inside shard_map + ragged_all_to_all. See
#     CLAUDE.md and the test module docstring for the bisection record.
#
# Optional environment knobs (dev-only; CI does not need these):
#
#   CUDA_VISIBLE_DEVICES -- defaults to "0,1,2,3".
#   PYTEST_EXTRA_ARGS    -- appended verbatim to the pytest invocation,
#                           e.g. PYTEST_EXTRA_ARGS="--maxfail=1 -x" or
#                           PYTEST_EXTRA_ARGS="-k 'fwd_and_bwd_smoke[triton]'".

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TE_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TEST_FILE="$TE_ROOT/tests/jax/test_distributed_moe_vjp.py"
PYTEST_INI="$TE_ROOT/tests/jax/pytest.ini"

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
        marker_args=("-m" "triton")
        kfilter=("-k" "$mode")
        ;;
esac

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.5}"

echo "============================================================"
echo "MoE VJP distributed tests (dev wrapper; CI: qa/L0_jax_distributed_unittest/test.sh)"
echo "  mode                : $mode"
echo "  marker filter       : ${marker_args[*]}"
echo "  -k filter           : ${kfilter[*]:-<none>}"
echo "  CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "  test file           : $TEST_FILE"
echo "  XLA_PYTHON_CLIENT_PREALLOCATE: $XLA_PYTHON_CLIENT_PREALLOCATE"
echo "  XLA_PYTHON_CLIENT_MEM_FRACTION: $XLA_PYTHON_CLIENT_MEM_FRACTION"
echo "  PYTEST_EXTRA_ARGS   : ${PYTEST_EXTRA_ARGS:-<unset>}"
echo "============================================================"

# IMPORTANT: keep the pytest invocation in lock-step with
# qa/L0_jax_distributed_unittest/test.sh. The two scripts must call
# pytest with identical flags so a dev-loop pass guarantees a CI pass.
exec python3 -m pytest \
    -c "$PYTEST_INI" \
    "$TEST_FILE" \
    "${marker_args[@]}" \
    "${kfilter[@]}" \
    -p no:typeguard \
    -v -s \
    ${PYTEST_EXTRA_ARGS:-}
