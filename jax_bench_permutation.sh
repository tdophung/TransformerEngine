#!/bin/bash
#SBATCH -p gb200nvl4
#SBATCH --time=01:30:00
#SBATCH --gpus=1
#SBATCH --exclusive
#SBATCH --job-name=jax_bench_perm
#SBATCH --output=/home/scratch.tdophung_sw_1/Repos/logs/jax_bisect/bench_perm_%j.log

# Benchmark Triton permutation kernels.  Two modes:
#
#   sbatch jax_bench_permutation.sh build_autotune
#       Removes build_jax/, does a full TE build, then runs the autotune benchmark.
#       On success writes build_jax/.built_ok so subsequent jobs can skip the build.
#
#   sbatch --dependency=afterok:<JOB_ID> jax_bench_permutation.sh 64|128|...|4096
#       Skips the C++ rebuild (build_jax/ already present from the first job) and
#       just registers the editable package, then runs the fixed-BLOCK_SIZE benchmark.
#
# Typical launch sequence (from the login node):
#   cd /home/scratch.tdophung_sw_1/Repos
#   bash launch_bench.sh

set +e

MODE="${1:-build_autotune}"
REPOS="/home/scratch.tdophung_sw_1/Repos"
DOCKER_HOME="$REPOS/docker_home"
DOCKER_USER="$(whoami)"

IMAGE="gitlab-master.nvidia.com/dl/dgx/jax:jax"
CONTAINER="te-bench-perm-dgx-nightly"

echo "=== Permutation Benchmark [mode=$MODE] ==="
echo "Image: $IMAGE   Start: $(date)   Host: $(hostname)"

docker pull "$IMAGE" || { echo "FATAL: image pull failed"; exit 2; }

docker build \
    --build-arg NEW_USER="$DOCKER_USER" \
    --build-arg NEW_UID="$(id -u)" \
    --build-arg NEW_GID="$(id -g)" \
    --build-arg IMAGE="$IMAGE" \
    -t "$CONTAINER" \
    -f "$DOCKER_HOME/te.Dockerfile" \
    "$DOCKER_HOME" || { echo "FATAL: docker build failed"; exit 2; }

INNER_SCRIPT=$(mktemp /tmp/te_bench_XXXXXX.sh)
BENCH_MODE="$MODE"

cat > "$INNER_SCRIPT" << INNER_EOF
set +e
export PATH=\$HOME/.local/bin:\$PATH

echo "=== Container info ==="
python3 -c "import jax; print('JAX:', jax.__version__)" 2>&1 \
    || python3 -c "import jaxlib; print('jaxlib:', jaxlib.version.version)" 2>&1
python3 --version
nvcc --version 2>/dev/null | grep release

echo "=== Installing pybind11 / triton / pytest ==="
python3 -m pip install --user -q pybind11 triton pytest

# cmake: 3.21.0 fails (can't find g++ on GCC-12 containers);
#         4.x fails (CUDA 13.1 compiler-id detection bug).
# Use cmake 3.28-3.31 which handles GCC 12+ and CUDA 12/13.
python3 -m pip install --user -q "cmake>=3.28,<4.0"

# Ensure g++ / gcc are on PATH (some nightly containers name them gcc-12 etc.)
if ! command -v g++ &>/dev/null; then
    GXX=\$(ls /usr/bin/g++-* 2>/dev/null | sort -V | tail -1)
    GCC=\$(ls /usr/bin/gcc-* 2>/dev/null | sort -V | tail -1)
    [ -n "\$GXX" ] && export CXX="\$GXX" && echo "Using CXX=\$CXX"
    [ -n "\$GCC" ] && export CC="\$GCC"  && echo "Using  CC=\$CC"
fi
echo "cmake \$(cmake --version | head -1)  CC=\${CC:-gcc}  CXX=\${CXX:-g++}"

cd /code/TransformerEngine

if [ "${BENCH_MODE}" = "build_autotune" ]; then
    # ── Full build ──────────────────────────────────────────────────────────
    echo "=== Full TE build ==="
    rm -rf build_jax
    NVTE_CMAKE_BUILD_DIR=build_jax NVTE_FRAMEWORK=jax NVTE_CUDA_ARCHS="100a" \\
        pip3 install --no-build-isolation -e . 2>&1 | tee /tmp/te_build.log
    BUILD_EXIT=\${PIPESTATUS[0]}
    echo "TE build exit: \$BUILD_EXIT"
    if [ \$BUILD_EXIT -ne 0 ]; then
        echo "RESULT: TE_BUILD_FAILED"
        tail -40 /tmp/te_build.log
        exit \$BUILD_EXIT
    fi
    touch build_jax/.built_ok
    echo "=== Running benchmark [AUTOTUNE_ON] ==="
    NVTE_DISABLE_TRITON_AUTOTUNING=0 python3 tests/jax/benchmark_permutation.py
    BENCH_EXIT=\$?
else
    # ── Fast re-registration (no recompile) ────────────────────────────────
    echo "=== Fast editable re-install (skipping C++ rebuild) ==="
    if [ ! -f build_jax/.built_ok ]; then
        echo "WARNING: build_jax/.built_ok not found; doing full build as fallback"
        rm -rf build_jax
        NVTE_CMAKE_BUILD_DIR=build_jax NVTE_FRAMEWORK=jax NVTE_CUDA_ARCHS="100a" \\
            pip3 install --no-build-isolation -e . 2>&1 | tee /tmp/te_build.log
        BUILD_EXIT=\${PIPESTATUS[0]}
        if [ \$BUILD_EXIT -ne 0 ]; then
            echo "RESULT: TE_BUILD_FAILED"
            tail -40 /tmp/te_build.log
            exit \$BUILD_EXIT
        fi
        touch build_jax/.built_ok
    else
        # build_jax exists → cmake will detect no changes and skip recompilation
        NVTE_CMAKE_BUILD_DIR=build_jax NVTE_FRAMEWORK=jax NVTE_CUDA_ARCHS="100a" \\
            pip3 install --no-build-isolation -e . 2>&1 | tee /tmp/te_build.log
        BUILD_EXIT=\${PIPESTATUS[0]}
        if [ \$BUILD_EXIT -ne 0 ]; then
            echo "RESULT: TE_BUILD_FAILED (incremental)"
            tail -20 /tmp/te_build.log
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

docker run --rm -i \
    --gpus all \
    --runtime=nvidia \
    --net=host \
    -v /mnt/nvdl/:/data \
    -v "$REPOS:/code" \
    -v /raid:/raid \
    --ipc=host \
    --privileged \
    -v "$INNER_SCRIPT:/tmp/te_bench.sh" \
    "$CONTAINER" /bin/bash /tmp/te_bench.sh

EXIT_CODE=$?
rm -f "$INNER_SCRIPT"

echo ""
echo "Container exit: $EXIT_CODE    End: $(date)"
[ $EXIT_CODE -eq 0 ] \
    && echo "FINAL RESULT: PASS" \
    || echo "FINAL RESULT: FAIL (exit=$EXIT_CODE)"
