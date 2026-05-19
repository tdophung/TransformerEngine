#!/usr/bin/env bash
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
#
# Multiprocess (one-GPU-per-process) launcher for the unified MoE VJP
# smoke suite. See tests/jax/test_multiprocess_moe_vjp.py for *why* we
# need this instead of -- or in addition to -- the single-process file.
#
# Pattern mirrors examples/jax/encoder/run_test_multiprocessing_encoder.sh:
# fork one pytest invocation per visible GPU, pass each its own
# --num-process=N --process-id=i, and wait for all of them. Each child
# calls jax.distributed.initialize(..., local_device_ids=process_id) so
# each Python process only sees its one GPU as a local device, the four
# processes form a global 4-device mesh, and the JAX/XLA lazy-Triton-
# load + active-NCCL deadlock (past_JAX_XLA_deadlock.txt, nvbug/5564750)
# CANNOT occur: every process has its own CUDA driver context, so the
# global module-load lock is not shared across the threads driving
# different GPUs.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TE_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TEST_FILE="$TE_ROOT/tests/jax/test_multiprocess_moe_vjp.py"
PYTEST_INI="$TE_ROOT/tests/jax/pytest.ini"

NUM_GPUS="${NUM_GPUS:-$(nvidia-smi -L | wc -l)}"
if [ "$NUM_GPUS" -lt 4 ]; then
    echo "[run_multiprocess_moe_vjp.sh] need >=4 GPUs (got $NUM_GPUS); aborting" >&2
    exit 1
fi

mode="${1:-smoke}"
case "$mode" in
    smoke|level2) marker_args=("-m" "triton and not slow") ;;
    perf|level3)  marker_args=("-m" "triton and slow") ;;
    all)          marker_args=("-m" "triton") ;;
    *)            marker_args=("-m" "triton") ;;
esac

export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.5}"
export MOE_VJP_COORDINATOR_ADDRESS="${MOE_VJP_COORDINATOR_ADDRESS:-127.0.0.1:13456}"

# We do NOT set CUDA_LAUNCH_BLOCKING=1 here. The whole point of this
# launcher is that one-GPU-per-process makes the deadlock window
# impossible without needing that workaround.

echo "============================================================"
echo "MoE VJP MULTIPROCESS smoke (one process per GPU, ${NUM_GPUS} GPUs)"
echo "  mode               : $mode"
echo "  marker filter      : ${marker_args[*]}"
echo "  test file          : $TEST_FILE"
echo "  coordinator        : $MOE_VJP_COORDINATOR_ADDRESS"
echo "  XLA_PYTHON_CLIENT_PREALLOCATE: $XLA_PYTHON_CLIENT_PREALLOCATE"
echo "  XLA_PYTHON_CLIENT_MEM_FRACTION: $XLA_PYTHON_CLIENT_MEM_FRACTION"
echo "============================================================"

LOG_DIR=$(mktemp -d -t moe_vjp_mp_XXXXXX)
echo "Per-process logs: $LOG_DIR"

PIDS=()

cleanup() {
    for pid in "${PIDS[@]:-}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill -TERM "$pid" 2>/dev/null || true
        fi
    done
    sleep 1
    for pid in "${PIDS[@]:-}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill -KILL "$pid" 2>/dev/null || true
        fi
    done
}
trap cleanup EXIT INT TERM

# Launch one pytest per GPU. Process 0 streams to stdout; others log
# only to file so the live output isn't a mosaic.
for i in $(seq 0 $((NUM_GPUS - 1))); do
    LOG_FILE="$LOG_DIR/proc_${i}.log"
    PYTEST_CMD=(
        python3 -m pytest -c "$PYTEST_INI"
        "$TEST_FILE"
        "${marker_args[@]}"
        -p no:typeguard
        -v -s
        --num-process="$NUM_GPUS"
        --process-id="$i"
    )
    if [ "$i" -eq 0 ]; then
        echo "=== Live output from process 0 ==="
        "${PYTEST_CMD[@]}" 2>&1 | tee "$LOG_FILE" &
    else
        "${PYTEST_CMD[@]}" > "$LOG_FILE" 2>&1 &
    fi
    PIDS+=("$!")
done

# Wait for all and collect exit codes.
EXITS=()
for pid in "${PIDS[@]}"; do
    if wait "$pid"; then
        EXITS+=("0")
    else
        EXITS+=("$?")
    fi
done

# Summary.
echo
echo "============================================================"
echo "Per-process exit codes:"
for i in "${!EXITS[@]}"; do
    echo "  proc $i -> ${EXITS[$i]}"
done

# Final pass/fail. Any non-zero in any process fails the suite, but
# we tolerate non-zero on the non-zero processes only if proc 0
# reports PASS (this matches the encoder launcher's logic). Simplest
# strict rule: any non-zero is a failure.
FAILED=0
for e in "${EXITS[@]}"; do
    if [ "$e" != "0" ]; then
        FAILED=1
        break
    fi
done

echo
if [ "$FAILED" -eq 0 ]; then
    echo "[run_multiprocess_moe_vjp.sh] all processes PASSED"
    rm -rf "$LOG_DIR"
    exit 0
fi

echo "[run_multiprocess_moe_vjp.sh] at least one process FAILED"
echo "  retaining logs at $LOG_DIR for diagnosis"
echo "  process 0 tail:"
tail -20 "$LOG_DIR/proc_0.log" 2>/dev/null || true
exit 1
