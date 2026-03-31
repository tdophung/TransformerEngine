#!/bin/bash
#SBATCH -A coreai_dlfw_dev
#SBATCH -p batch
#SBATCH --nodes=1
# prenyx: nodes are fully allocated so --exclusive is rarely satisfiable; use per-resource
# limits instead — CPU/mem isolation via cgroups. GPUs are not tracked GRES on this cluster
# (sinfo shows null); the container runtime exposes all node GPUs, set CUDA_VISIBLE_DEVICES
# in the inner script if finer GPU pinning is needed.
# dlcluster: replace the two lines below with a single #SBATCH --exclusive instead.
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --job-name=te_sort_chunks_repro
#SBATCH --output=/lustre/fsw/coreai_dlfw_dev/tdophung/TransformerEngine/bug_repro_logs/slurm_%j.log

set +e

REPOS="/lustre/fsw/coreai_dlfw_dev/tdophung"
TE_SOURCE="$REPOS/TransformerEngine"
WORK_DIR="$REPOS/te_work_${SLURM_JOB_ID}"
TE_COPY="$WORK_DIR/TransformerEngine"
BUG_LOG_DIR="$TE_SOURCE/bug_repro_logs"
LOG_FILE="$BUG_LOG_DIR/full_suite_${SLURM_JOB_ID}.log"
IMAGE="ghcr.io/nvidia/jax:jax-2026-01-13"
CUDA_ARCH="100a"   # Blackwell B200

mkdir -p "$BUG_LOG_DIR"

echo "=== TE sort_chunks_by_map bug repro ===" | tee "$LOG_FILE"
echo "Image: $IMAGE" | tee -a "$LOG_FILE"
echo "Job: $SLURM_JOB_ID  Host: $(hostname)  Start: $(date)" | tee -a "$LOG_FILE"

# --- Step 1: per-job isolated copy (avoids build_jax/ contention between concurrent jobs) ---
echo "--- Copying TE repo to $WORK_DIR ---" | tee -a "$LOG_FILE"
mkdir -p "$WORK_DIR"
rsync -a --exclude='build/' --exclude='build_jax*/' --exclude='*.core' \
    "$TE_SOURCE/" "$TE_COPY/" >> "$LOG_FILE" 2>&1
echo "Copy done: $(du -sh "$TE_COPY" | cut -f1)" | tee -a "$LOG_FILE"

# --- Step 2: write the container-side inner script to lustre so it's visible inside the container ---
# Inside the container, $REPOS maps to /mnt/tdophung/prenyx_lustre_home
INNER_HOST="$WORK_DIR/te_inner.sh"
INNER_CONTAINER="/mnt/tdophung/prenyx_lustre_home/te_work_${SLURM_JOB_ID}/te_inner.sh"
WORK_IN_CONTAINER="/mnt/tdophung/prenyx_lustre_home/te_work_${SLURM_JOB_ID}"

cat > "$INNER_HOST" << INNER_EOF
#!/bin/bash
set +e
export PATH=\$HOME/.local/bin:\$PATH

echo "=== Container: JAX/jaxlib version ==="
python3 -c "import jaxlib; print('jaxlib:', jaxlib.__version__)"

echo "=== Installing build tools ==="
pip3 install --user -q "cmake==3.21.0" pybind11 triton pytest 2>&1

echo "=== Building TransformerEngine ==="
cd ${WORK_IN_CONTAINER}/TransformerEngine
rm -rf build_jax
NVTE_CMAKE_BUILD_DIR=build_jax NVTE_FRAMEWORK=jax NVTE_CUDA_ARCHS="${CUDA_ARCH}" \\
    pip3 install --no-build-isolation -e . 2>&1 | tee /tmp/te_build.log
BUILD_EXIT=\${PIPESTATUS[0]}
echo "TE build exit: \${BUILD_EXIT}"
[ \${BUILD_EXIT} -ne 0 ] && { echo "RESULT: TE_BUILD_FAILED"; exit \${BUILD_EXIT}; }

echo "=== Running full L0 JAX test suite ==="
export TE_PATH=${WORK_IN_CONTAINER}/TransformerEngine
export XML_LOG_DIR=/tmp/jax_xml
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
mkdir -p \${XML_LOG_DIR}
bash qa/L0_jax_unittest/test.sh 2>&1
TEST_EXIT=\$?
echo "Full suite exit: \${TEST_EXIT}"
exit \${TEST_EXIT}
INNER_EOF

chmod +x "$INNER_HOST"

# --- Step 3: run inside container via Pyxis/srun ---
echo "--- Launching container ---" | tee -a "$LOG_FILE"
srun \
    --container-image="$IMAGE" \
    --container-mounts "/lustre/fsw/coreai_dlfw_dev/tdophung/:/mnt/tdophung/prenyx_lustre_home,/lustre/fsw/coreai_dlfw_dev/te_ccache:/mnt/ccache" \
    bash "$INNER_CONTAINER" >> "$LOG_FILE" 2>&1

EXIT_CODE=$?
rm -rf "$WORK_DIR"   # clean up per-job copy

echo "Container exit: $EXIT_CODE  End: $(date)" | tee -a "$LOG_FILE"

# Classify result
if grep -qiE "computed_grad != ref_grad|sort_chunks_by_map BACKWARD" "$LOG_FILE"; then
    RESULT="BUG_REPRODUCED"
elif [ $EXIT_CODE -eq 0 ]; then
    RESULT="PASS"
else
    RESULT="FAIL_OTHER_exit${EXIT_CODE}"
fi

echo "FINAL RESULT: $RESULT" | tee -a "$LOG_FILE"
cp "$LOG_FILE" "$BUG_LOG_DIR/full_suite_${SLURM_JOB_ID}_${RESULT}.log"
