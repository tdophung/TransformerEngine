#!/bin/bash
# Launch the permutation benchmark suite on dlcluster.
#
# Workflow:
#   1. Submit one "build + autotune" job (full TE build + autotuning benchmark).
#   2. Submit 7 fixed-BLOCK_SIZE jobs that depend on job 1 (--dependency=afterok).
#      These jobs skip the C++ rebuild; they only re-register the editable package
#      (cmake detects no changes → fast) and run their benchmark.
#
# Usage (run this on the dlcluster login node):
#   bash /home/scratch.tdophung_sw_1/Repos/launch_bench.sh

set -e
cd /home/scratch.tdophung_sw_1/Repos

echo "=== Submitting build + autotune job ==="
BUILD_JOB_LINE=$(sbatch jax_bench_permutation.sh build_autotune)
echo "$BUILD_JOB_LINE"
BUILD_JOB_ID=$(echo "$BUILD_JOB_LINE" | awk '{print $NF}')
echo "Build job ID: $BUILD_JOB_ID"

echo ""
echo "=== Submitting 7 fixed-BLOCK_SIZE jobs (depend on $BUILD_JOB_ID) ==="
BLOCK_SIZES="64 128 256 512 1024 2048 4096"
JOB_IDS="$BUILD_JOB_ID"

for BS in $BLOCK_SIZES; do
    LINE=$(sbatch --dependency=afterok:$BUILD_JOB_ID jax_bench_permutation.sh $BS)
    JID=$(echo "$LINE" | awk '{print $NF}')
    echo "  BLOCK_SIZE=$BS  job=$JID"
    JOB_IDS="$JOB_IDS,$JID"
done

echo ""
echo "=== All jobs submitted ==="
echo "Job IDs: $JOB_IDS"
echo ""
squeue -j "$JOB_IDS" -o '%.8i %.2t %.10M %R' 2>/dev/null || true
echo ""
echo "Logs: /home/scratch.tdophung_sw_1/Repos/logs/jax_bisect/bench_perm_<JOBID>.log"
