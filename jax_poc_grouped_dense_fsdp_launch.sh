#!/bin/bash
# Multi-process launcher for jax_poc_grouped_dense_fsdp.py
#
# Lightly modified copy of TransformerEngine/tests/jax/multi_process_launch.sh.
# Differences:
#   - Defaults SCRIPT_NAME to this script's sibling Python file.
#   - DOES NOT redirect background-rank stderr to /dev/null (TE's launcher
#     swallows stderr from non-zero ranks, which makes debugging painful).
#   - Echoes the rank command lines so it's obvious what's running.
#   - Cleans up children on Ctrl-C.
#
# Why this launcher exists:
#   TE's ``grouped_gemm`` caches CUDA streams in a process-static array
#   (common/util/multi_stream.cpp:22-50). Those streams are device-bound,
#   so a single multi-GPU process produces ``CUDA Error: invalid resource
#   handle`` from cuBLAS the first time the kernel runs on any device
#   other than the one current at TE's first call. The fix (and what TE
#   itself does for grouped_gemm tests) is to launch one process per GPU
#   so each process has its own static cache pinned to its own device.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_NAME="${SCRIPT_NAME:-$HERE/jax_poc_grouped_dense_fsdp.py}"
COORD="${COORD:-127.0.0.1:12345}"

# Mirror TE's xla flags from multi_process_launch.sh -- they tune scheduler
# overlap, which is helpful here because we expect comm/compute overlap on
# the FSDP all-gather of the kernel.
XLA_BASE_FLAGS="--xla_gpu_enable_latency_hiding_scheduler=true \
                --xla_gpu_enable_command_buffer=''"
export XLA_FLAGS="${XLA_FLAGS:-} ${XLA_BASE_FLAGS}"

NUM_RUNS="${NUM_RUNS:-$(nvidia-smi -L | wc -l)}"

if [ "$NUM_RUNS" -lt 2 ]; then
  echo "ERROR: need at least 2 visible GPUs to run this POC. Got $NUM_RUNS."
  exit 1
fi

echo "Launching $NUM_RUNS-process run of $SCRIPT_NAME (coord=$COORD)"
echo

# Track child PIDs so we can clean up on Ctrl-C.
PIDS=()
trap 'echo; echo "Caught signal -- killing rank PIDs: ${PIDS[*]}"; kill "${PIDS[@]}" 2>/dev/null || true; exit 130' INT TERM

# Background ranks 1..NUM_RUNS-1 (each pinned to its own GPU).
for ((i=1; i<NUM_RUNS; i++)); do
  echo "  rank $i: CUDA_VISIBLE_DEVICES=$i python $SCRIPT_NAME $COORD $i $NUM_RUNS  -> rank${i}.log"
  CUDA_VISIBLE_DEVICES=$i python "$SCRIPT_NAME" "$COORD" "$i" "$NUM_RUNS" \
      > "$HERE/rank${i}.log" 2>&1 &
  PIDS+=($!)
done

# Run rank 0 in the foreground so the user sees its output live.
echo "  rank 0: CUDA_VISIBLE_DEVICES=0 python $SCRIPT_NAME $COORD 0 $NUM_RUNS  (foreground)"
echo
CUDA_VISIBLE_DEVICES=0 python "$SCRIPT_NAME" "$COORD" "0" "$NUM_RUNS"
RANK0_RC=$?

# Wait for all background ranks to finish.
RC=$RANK0_RC
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then
    RC=1
  fi
done

if [ "$RC" -ne 0 ]; then
  echo
  echo "Run failed (rank 0 RC=$RANK0_RC). Background-rank logs are in:"
  for ((i=1; i<NUM_RUNS; i++)); do
    echo "  $HERE/rank${i}.log"
  done
fi

exit "$RC"

