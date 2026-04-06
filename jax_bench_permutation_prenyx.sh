#!/bin/bash
#SBATCH -A coreai_dlfw_dev
#SBATCH -p batch
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=01:30:00
#SBATCH --job-name=te_bench_perm
#SBATCH --output=/lustre/fsw/coreai_dlfw_dev/tdophung/TransformerEngine/bug_repro_logs/bench_perm_%j.log

# Benchmark Triton permutation kernels on prenyx (GB300, Pyxis/Enroot).
#
# Usage:
#   sbatch jax_bench_permutation_prenyx.sh build_autotune
#       Rsyncs TE repo to a per-job workspace, does a full TE build, then runs
#       the autotune benchmark.  Writes <WORKSPACE>/build_jax/.built_ok on success.
#
#   sbatch --dependency=afterok:<BUILD_JOB_ID> \
#          jax_bench_permutation_prenyx.sh 64|128|...|4096 <BUILD_JOB_ID>
#       Rsyncs TE repo to its own workspace but reuses the C++ build from the
#       build job's workspace (bind-mounted into the container as /code/build_jax).
#       Runs the fixed-BLOCK_SIZE benchmark only.
#
# Typical launch sequence (from prenyx login node):
#   cd /lustre/fsw/coreai_dlfw_dev/tdophung/TransformerEngine
#   bash launch_bench_prenyx.sh

set +e

MODE="${1:-build_autotune}"
BUILD_JOB_ID="${2:-}"

LUSTRE_HOME="/lustre/fsw/coreai_dlfw_dev/tdophung"
TE_SOURCE="$LUSTRE_HOME/TransformerEngine"
CCACHE_DIR="/lustre/fsw/coreai_dlfw_dev/te_ccache"
LOG_DIR="$TE_SOURCE/bug_repro_logs"

# Per-job isolated workspace so concurrent jobs never race on build_jax/
WORKSPACE="$LUSTRE_HOME/te_work_${SLURM_JOB_ID}"
TE_WORK="$WORKSPACE/TransformerEngine"

IMAGE="gitlab-master.nvidia.com/dl/dgx/jax:jax"

echo "=== Permutation Benchmark [mode=$MODE] ==="
echo "Image: $IMAGE   Start: $(date)   Host: $(hostname)"
echo "Job ID: $SLURM_JOB_ID   Workspace: $WORKSPACE"

mkdir -p "$WORKSPACE" "$LOG_DIR"

# Rsync TE source into per-job workspace (exclude build dirs to keep it fast)
echo "=== Syncing TE repo → $TE_WORK ==="
rsync -a \
    --exclude='build/' \
    --exclude='build_jax*/' \
    --exclude='*.core' \
    --exclude='bug_repro_logs/' \
    "$TE_SOURCE/" "$TE_WORK/"

# For benchmark jobs: also copy the compiled build_jax from the build job's workspace
if [ "$MODE" != "build_autotune" ] && [ -n "$BUILD_JOB_ID" ]; then
    BUILD_WORKSPACE="$LUSTRE_HOME/te_work_${BUILD_JOB_ID}"
    BUILD_JAX_SRC="$BUILD_WORKSPACE/TransformerEngine/build_jax"
    if [ -d "$BUILD_JAX_SRC" ]; then
        echo "=== Linking build_jax from build job $BUILD_JOB_ID ==="
        ln -sfn "$BUILD_JAX_SRC" "$TE_WORK/build_jax"
    else
        echo "WARNING: build_jax not found at $BUILD_JAX_SRC; will fall back to full build"
    fi
fi

# Write inner script to lustre (NOT /tmp — /tmp is separate inside the container)
INNER_SCRIPT="$WORKSPACE/te_bench_inner_${SLURM_JOB_ID}.sh"
BENCH_MODE="$MODE"

cat > "$INNER_SCRIPT" << INNER_EOF
set +e
export PATH=\$HOME/.local/bin:\$PATH
export CCACHE_DIR=/mnt/ccache

echo "=== Container info ==="
python3 -c "import jax; print('JAX:', jax.__version__)" 2>&1 \
    || python3 -c "import jaxlib; print('jaxlib:', jaxlib.version.version)" 2>&1
python3 --version
nvcc --version 2>/dev/null | grep release

echo "=== Installing pybind11 / triton / pytest ==="
python3 -m pip install --user -q pybind11 triton pytest

# cmake: avoid 3.21.0 (C++ compiler detection bug on GCC-12 containers)
#        and avoid 4.x (CUDA 13.1 compiler-id bug).
python3 -m pip install --user -q "cmake>=3.28,<4.0"

# Ensure g++ / gcc are on PATH (some nightly containers name them gcc-12 etc.)
if ! command -v g++ &>/dev/null; then
    GXX=\$(ls /usr/bin/g++-* 2>/dev/null | sort -V | tail -1)
    GCC=\$(ls /usr/bin/gcc-* 2>/dev/null | sort -V | tail -1)
    [ -n "\$GXX" ] && export CXX="\$GXX" && echo "Using CXX=\$CXX"
    [ -n "\$GCC" ] && export CC="\$GCC"  && echo "Using  CC=\$CC"
fi
echo "cmake \$(cmake --version | head -1)  CC=\${CC:-gcc}  CXX=\${CXX:-g++}"

cd /mnt/tdophung/prenyx_lustre_home/te_work_${SLURM_JOB_ID}/TransformerEngine

if [ "${BENCH_MODE}" = "build_autotune" ]; then
    # ── Full build ─────────────────────────────────────────────────────────
    echo "=== Full TE build ==="
    rm -rf build_jax
    NVTE_CMAKE_BUILD_DIR=build_jax NVTE_FRAMEWORK=jax NVTE_CUDA_ARCHS="100a" \\
        pip3 install --no-build-isolation -e . 2>&1 | tee /mnt/tdophung/prenyx_lustre_home/te_work_${SLURM_JOB_ID}/te_build.log
    BUILD_EXIT=\${PIPESTATUS[0]}
    echo "TE build exit: \$BUILD_EXIT"
    if [ \$BUILD_EXIT -ne 0 ]; then
        echo "RESULT: TE_BUILD_FAILED"
        tail -40 /mnt/tdophung/prenyx_lustre_home/te_work_${SLURM_JOB_ID}/te_build.log
        exit \$BUILD_EXIT
    fi
    touch build_jax/.built_ok
    echo "=== Running benchmark [AUTOTUNE_ON] ==="
    NVTE_DISABLE_TRITON_AUTOTUNING=0 python3 tests/jax/benchmark_permutation.py
    BENCH_EXIT=\$?
else
    # ── Fast re-registration (build_jax symlinked/copied from build job) ──
    echo "=== Fast editable re-install (skipping C++ rebuild) ==="
    if [ ! -f build_jax/.built_ok ]; then
        echo "WARNING: build_jax/.built_ok not found; doing full build as fallback"
        rm -rf build_jax
        NVTE_CMAKE_BUILD_DIR=build_jax NVTE_FRAMEWORK=jax NVTE_CUDA_ARCHS="100a" \\
            pip3 install --no-build-isolation -e . 2>&1 | tee /mnt/tdophung/prenyx_lustre_home/te_work_${SLURM_JOB_ID}/te_build.log
        BUILD_EXIT=\${PIPESTATUS[0]}
        if [ \$BUILD_EXIT -ne 0 ]; then
            echo "RESULT: TE_BUILD_FAILED (fallback)"
            tail -20 /mnt/tdophung/prenyx_lustre_home/te_work_${SLURM_JOB_ID}/te_build.log
            exit \$BUILD_EXIT
        fi
        touch build_jax/.built_ok
    else
        # build_jax present → cmake detects no changes → fast re-register only
        NVTE_CMAKE_BUILD_DIR=build_jax NVTE_FRAMEWORK=jax NVTE_CUDA_ARCHS="100a" \\
            pip3 install --no-build-isolation -e . 2>&1 | tee /mnt/tdophung/prenyx_lustre_home/te_work_${SLURM_JOB_ID}/te_build.log
        BUILD_EXIT=\${PIPESTATUS[0]}
        if [ \$BUILD_EXIT -ne 0 ]; then
            echo "RESULT: TE_BUILD_FAILED (incremental)"
            tail -20 /mnt/tdophung/prenyx_lustre_home/te_work_${SLURM_JOB_ID}/te_build.log
            exit \$BUILD_EXIT
        fi
    fi
    echo "=== Running benchmark [BLOCK_SIZE=${BENCH_MODE}] ==="
    NVTE_DISABLE_TRITON_AUTOTUNING=1 \\
    NVTE_TRITON_BLOCK_SIZE=${BENCH_MODE} \\
        python3 tests/jax/benchmark_permutation.py
    BENCH_EXIT=\$?
fi

echo "Benchmark exit: \$BENCH_EXIT"
exit \$BENCH_EXIT
INNER_EOF

chmod +x "$INNER_SCRIPT"

# Launch via Pyxis/srun (NOT docker run — prenyx uses Enroot)
srun \
    --container-image="$IMAGE" \
    --container-mounts "$LUSTRE_HOME:/mnt/tdophung/prenyx_lustre_home,$CCACHE_DIR:/mnt/ccache" \
    bash "$INNER_SCRIPT"

EXIT_CODE=$?
rm -f "$INNER_SCRIPT"

echo ""
echo "Container exit: $EXIT_CODE    End: $(date)"
[ $EXIT_CODE -eq 0 ] \
    && echo "FINAL RESULT: PASS" \
    || echo "FINAL RESULT: FAIL (exit=$EXIT_CODE)"
