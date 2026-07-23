#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_FILE="$SCRIPT_DIR/tvm_ffi_grouped_mlp_multiprocess.py"
NUM_GPUS="${NUM_GPUS:-2}"
COORDINATOR="${TVM_FFI_COORDINATOR_ADDRESS:-127.0.0.1:13461}"
TIMEOUT_SECONDS="${TEST_TIMEOUT_S:-600}"
LOG_DIR="${TVM_FFI_MP_LOG_DIR:-$(mktemp -d -t te_tvm_ffi_mp_XXXXXX)}"

if [ "$(nvidia-smi -L | wc -l)" -lt "$NUM_GPUS" ]; then
    echo "Need at least $NUM_GPUS GPUs for the TVM-FFI multiprocess test" >&2
    exit 1
fi

export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
mkdir -p "$LOG_DIR"
PIDS=()

cleanup() {
    for pid in "${PIDS[@]:-}"; do
        kill -TERM "$pid" 2>/dev/null || true
    done
}
trap cleanup EXIT INT TERM

for rank in $(seq 0 $((NUM_GPUS - 1))); do
    log_file="$LOG_DIR/rank_${rank}.log"
    if [ "$rank" -eq 0 ]; then
        timeout --foreground --signal=KILL "$TIMEOUT_SECONDS" \
            python3 "$TEST_FILE" "$COORDINATOR" "$rank" "$NUM_GPUS" 2>&1 \
            | tee "$log_file" &
    else
        timeout --foreground --signal=KILL "$TIMEOUT_SECONDS" \
            python3 "$TEST_FILE" "$COORDINATOR" "$rank" "$NUM_GPUS" \
            >"$log_file" 2>&1 &
    fi
    PIDS+=("$!")
done

failed=0
for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
        failed=1
    fi
done
PIDS=()

if [ "$failed" -ne 0 ]; then
    echo "TVM-FFI multiprocess test failed; logs: $LOG_DIR" >&2
    for log_file in "$LOG_DIR"/*.log; do
        echo "--- $log_file ---" >&2
        tail -80 "$log_file" >&2 || true
    done
    exit 1
fi

echo "TVM-FFI multiprocess test passed; logs: $LOG_DIR"
