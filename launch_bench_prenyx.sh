#!/bin/bash
# Launch the permutation benchmark suite on prenyx (GB300).
#
# Workflow:
#   1. Submit one "build + autotune" job (full TE build + autotuning benchmark).
#   2. Submit 7 fixed-BLOCK_SIZE jobs that depend on job 1 (--dependency=afterok).
#      Each benchmark job rsyncs the TE source to its own workspace and symlinks
#      the compiled build_jax from the build job — no concurrent C++ builds.
#
# Usage (run this on the prenyx login node):
#   cd /lustre/fsw/coreai_dlfw_dev/tdophung/TransformerEngine
#   bash launch_bench_prenyx.sh

set -e
cd /lustre/fsw/coreai_dlfw_dev/tdophung/TransformerEngine
mkdir -p bug_repro_logs

echo "=== Submitting build + autotune job ==="
BUILD_JOB_LINE=$(sbatch jax_bench_permutation_prenyx.sh build_autotune)
echo "$BUILD_JOB_LINE"
BUILD_JOB_ID=$(echo "$BUILD_JOB_LINE" | awk '{print $NF}')
echo "Build job ID: $BUILD_JOB_ID"

echo ""
echo "=== Submitting 7 fixed-BLOCK_SIZE jobs (depend on $BUILD_JOB_ID) ==="
BLOCK_SIZES="64 128 256 512 1024 2048 4096"
JOB_IDS="$BUILD_JOB_ID"

for BS in $BLOCK_SIZES; do
    LINE=$(sbatch --dependency=afterok:$BUILD_JOB_ID \
                  jax_bench_permutation_prenyx.sh $BS $BUILD_JOB_ID)
    JID=$(echo "$LINE" | awk '{print $NF}')
    echo "  BLOCK_SIZE=$BS  job=$JID"
    JOB_IDS="$JOB_IDS,$JID"
done

echo ""
echo "=== All jobs submitted ==="
echo "Job IDs: $JOB_IDS"
echo ""
squeue -j "$JOB_IDS" -o '%.8i %.2t %.10M %N %R' 2>/dev/null || true
echo ""
echo "Logs: /lustre/fsw/coreai_dlfw_dev/tdophung/TransformerEngine/bug_repro_logs/bench_perm_<JOBID>.log"
