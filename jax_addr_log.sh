#!/bin/bash
#SBATCH -p b100_preprod
#SBATCH --time=08:00:00
#SBATCH --gpus=1
#SBATCH --exclusive
#SBATCH --job-name=jax_addr_log
#SBATCH --output=/home/scratch.tdophung_sw_1/Repos/logs/jax_bisect/addr_log_%j.log

set +e

TEST_DATE="${1:-2026-01-13}"
REPOS="/home/scratch.tdophung_sw_1/Repos"
DOCKER_HOME="$REPOS/docker_home"
LOG_DIR="$REPOS/logs/jax_bisect"
LOG_FILE="$LOG_DIR/addr_log_${SLURM_JOB_ID}_${TEST_DATE}.log"
TE_LOG_DIR="$REPOS/TransformerEngine/bug_repro_logs"
DOCKER_USER="$(whoami)"

WORK_DIR="$REPOS/te_work_${SLURM_JOB_ID}"
TE_COPY="$WORK_DIR/TransformerEngine"

mkdir -p "$LOG_DIR" "$TE_LOG_DIR"

IMAGE="ghcr.io/nvidia/jax:jax-${TEST_DATE}"
CONTAINER="te-addr-log-${TEST_DATE}-${SLURM_JOB_ID}"

echo "=== Approach 1 addr-log test: $IMAGE ===" | tee "$LOG_FILE"
echo "Start time: $(date)" | tee -a "$LOG_FILE"
echo "Host: $(hostname)" | tee -a "$LOG_FILE"
echo "Job ID: $SLURM_JOB_ID" | tee -a "$LOG_FILE"

AVAIL_GB=$(df --output=avail -BG "$REPOS" | tail -1 | tr -d 'G ')
echo "Disk available: ${AVAIL_GB}G" | tee -a "$LOG_FILE"
if [ "$AVAIL_GB" -lt 100 ]; then
    echo "FATAL: less than 100GB free. Aborting." | tee -a "$LOG_FILE"
    exit 2
fi

echo "--- Copying TE repo ---" | tee -a "$LOG_FILE"
mkdir -p "$WORK_DIR"
cp -r "$REPOS/TransformerEngine" "$TE_COPY" >> "$LOG_FILE" 2>&1

# Copy the inner container script to WORK_DIR so it's accessible via the Docker volume mount
cp "$REPOS/TransformerEngine/te_addr_log_inner.sh" "$WORK_DIR/te_addr_log_inner.sh"

# --- Build cache: keyed on hostname + C++/CMake source hash ---
# Python-only changes leave the hash unchanged => cache hit => skip C++ compile
BUILD_CACHE_DIR="$REPOS/te_build_cache"
NODE_NAME=$(hostname -s)
CPP_HASH=$(find "$TE_COPY" \( -name "*.cpp" -o -name "*.cu" -o -name "*.h" -o -name "*.cuh" -o -name "CMakeLists.txt" \) \
    | sort | xargs md5sum 2>/dev/null | md5sum | cut -d' ' -f1)
CACHE_KEY="${NODE_NAME}_${CPP_HASH}"
CACHE_PATH="${BUILD_CACHE_DIR}/${CACHE_KEY}/build_jax"
CACHE_HIT=0
if [ -d "$CACHE_PATH" ]; then
    echo "--- Build cache HIT ($CACHE_KEY) --- copying artifacts ---" | tee -a "$LOG_FILE"
    rsync -a "$CACHE_PATH/" "$TE_COPY/build_jax/" >> "$LOG_FILE" 2>&1
    CACHE_HIT=1
else
    echo "--- Build cache MISS ($CACHE_KEY) --- full compile needed ---" | tee -a "$LOG_FILE"
fi

echo "--- Pulling image ---" | tee -a "$LOG_FILE"
docker pull "$IMAGE" >> "$LOG_FILE" 2>&1 \
    || { echo "FATAL: image not found" | tee -a "$LOG_FILE"; rm -rf "$WORK_DIR"; exit 2; }

echo "--- Building container image ---" | tee -a "$LOG_FILE"
docker build \
    --build-arg NEW_USER="$DOCKER_USER" \
    --build-arg NEW_UID="$(id -u)" \
    --build-arg NEW_GID="$(id -g)" \
    --build-arg IMAGE="$IMAGE" \
    -t "$CONTAINER" \
    -f "$DOCKER_HOME/te.Dockerfile" \
    "$DOCKER_HOME" >> "$LOG_FILE" 2>&1 \
    || { echo "FATAL: docker build failed" | tee -a "$LOG_FILE"; rm -rf "$WORK_DIR"; exit 2; }

echo "--- Running container ---" | tee -a "$LOG_FILE"
docker run --rm -i \
    --gpus all \
    --runtime=nvidia \
    --net=host \
    -v /mnt/nvdl/:/data \
    -v "$WORK_DIR:/code" \
    -v /raid:/raid \
    --ipc=host \
    --privileged \
    --env CACHE_HIT="$CACHE_HIT" \
    --env SLURM_JOB_ID="$SLURM_JOB_ID" \
    "$CONTAINER" /bin/bash /code/te_addr_log_inner.sh >> "$LOG_FILE" 2>&1

EXIT_CODE=$?

# Save build artifacts to cache if this was a fresh build that succeeded
if [ "$CACHE_HIT" = "0" ] && grep -q "TE build exit: 0" "$LOG_FILE" 2>/dev/null; then
    echo "--- Caching build artifacts to $CACHE_PATH ---" | tee -a "$LOG_FILE"
    mkdir -p "$CACHE_PATH"
    rsync -a "$TE_COPY/build_jax/" "$CACHE_PATH/" >> "$LOG_FILE" 2>&1
fi

# Rescue XLA dump from workdir to bug_repro_logs before cleanup
XLA_DUMP_SRC="$WORK_DIR/xla_dump_addr_log_${SLURM_JOB_ID}"
if [ -d "$XLA_DUMP_SRC" ]; then
    XLA_DUMP_DEST="$TE_LOG_DIR/xla_dump_addr_log_${SLURM_JOB_ID}"
    echo "--- Rescuing XLA dump to $XLA_DUMP_DEST ---" | tee -a "$LOG_FILE"
    cp -r "$XLA_DUMP_SRC" "$XLA_DUMP_DEST" >> "$LOG_FILE" 2>&1
    echo "XLA dump rescued: $(ls $XLA_DUMP_DEST | wc -l) files" | tee -a "$LOG_FILE"
fi

rm -rf "$WORK_DIR"

echo "Container exit: $EXIT_CODE" | tee -a "$LOG_FILE"
echo "End time: $(date)" | tee -a "$LOG_FILE"

# Classify result
if grep -qE 'TE_BUG_DBG.*YES_PHYSICAL_ALIAS' "$LOG_FILE"; then
    ALIAS_FOUND="PHYSICAL_ALIAS_DETECTED"
else
    ALIAS_FOUND="no_alias_detected"
fi

if grep -qiE 'FAILED.*test_sort_chunks_by_index|AssertionError.*ref_grad' "$LOG_FILE"; then
    BUG="BUG_REPRODUCED"
elif [ "$EXIT_CODE" -eq 0 ]; then
    BUG="PASS"
else
    BUG="FAIL_OTHER_exit${EXIT_CODE}"
fi

RESULT="${BUG}__${ALIAS_FOUND}"
echo "FINAL RESULT: $RESULT" | tee -a "$LOG_FILE"
cp "$LOG_FILE" "$TE_LOG_DIR/addr_log_${SLURM_JOB_ID}_${RESULT}.log"
echo "Saved: $TE_LOG_DIR/addr_log_${SLURM_JOB_ID}_${RESULT}.log" | tee -a "$LOG_FILE"
