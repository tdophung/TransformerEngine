#!/bin/bash
# Inner container script for Approach 1 addr-log experiment.
# Patches triton_kernels.cc with buffer address logging, rebuilds jaxlib,
# builds TE, and runs the full L0 JAX test suite.
set +e
export PATH=$HOME/.local/bin:$PATH

echo "=== Container info ==="
python3 -c "import jax; print('JAX:', jax.__version__)" 2>&1
python3 -c "import jaxlib; print('jaxlib:', jaxlib.__version__)" 2>&1
python3 --version
nvcc --version 2>/dev/null | grep release || true

# =========================================================
# Step 1: Find triton_kernels.cc (exclude mock/test fixtures)
# =========================================================
echo "=== Locating triton_kernels.cc ==="
TCC_FILE=$(find /opt -name triton_kernels.cc 2>/dev/null \
    | grep -vE 'mock|triage|nsys-jax.*test' | head -1)
if [ -z "$TCC_FILE" ]; then
    TCC_FILE=$(find /usr -name triton_kernels.cc 2>/dev/null \
        | grep -vE 'mock|triage|nsys-jax.*test' | head -1)
fi
echo "triton_kernels.cc: $TCC_FILE"

if [ -z "$TCC_FILE" ]; then
    echo "FATAL: triton_kernels.cc not found"
    echo "Listing /opt:"
    ls /opt/ 2>/dev/null
    find /opt -name '*.cc' -path '*/triton*' 2>/dev/null | head -10
    exit 3
fi

# =========================================================
# Step 2: Write Python patch script to /tmp
# =========================================================
cat > /tmp/patch_tcc.py << 'PYEOF'
import sys

fname = sys.argv[1]
outfname = sys.argv[2]
with open(fname, 'r') as f:
    content = f.read()

ANCHOR = '// If an input aliases with an output'
if ANCHOR not in content:
    print(f'ERROR: anchor not found in {fname}', file=sys.stderr)
    sys.exit(1)

idx = content.find(ANCHOR)
insert_pos = content.rfind('\n', 0, idx) + 1

DEBUG_BLOCK = '''  // [TE_BUG_DBG] Inserted for physical aliasing investigation (Approach 1)
  // Filters to sort_chunks_by_map only; uses fprintf so output appears regardless
  // of ABSL/glog log-level settings in newer JAX nightly builds.
  // Buffer layout for sort_chunks_by_map backward (6 buffers, indices 0-5):
  //   [0]=output_grad  [1]=row_id_map  [2]=probs
  //   [3]=output_buf   [4]=output      [5]=permuted_probs
  if (kernel_call.name_.find("sort_chunks_by_map") != std::string::npos) {
    // Hardcode 6: input_output_aliases_ is empty for this kernel so _dbg_max
    // computed from it would stay 0 (only logging buffer[0]).
    static constexpr size_t _DBG_N = 6;
    fprintf(stderr, "[TE_BUG_DBG] Autotune() kernel=%s input_output_aliases_.size()=%zu\\n",
            kernel_call.name_.c_str(), kernel_call.input_output_aliases_.size());
    for (size_t _i = 0; _i < _DBG_N; _i++) {
      fprintf(stderr, "[TE_BUG_DBG]   buffers[%zu]=%p\\n", _i, buffers[_i]);
    }
    // All-pairs alias check (since alias list is empty, check manually).
    for (size_t _i = 0; _i < _DBG_N; _i++) {
      for (size_t _j = _i + 1; _j < _DBG_N; _j++) {
        if (buffers[_i] == buffers[_j]) {
          fprintf(stderr, "[TE_BUG_DBG] YES_PHYSICAL_ALIAS buffers[%zu]=%p == buffers[%zu]=%p\\n",
                  _i, buffers[_i], _j, buffers[_j]);
        }
      }
    }
    fflush(stderr);
  }
'''

content = content[:insert_pos] + DEBUG_BLOCK + content[insert_pos:]
with open(outfname, 'w') as f:
    f.write(content)
print(f'Patch written to {outfname}')
PYEOF

# =========================================================
# Step 3: Apply patch and copy back with sudo
# =========================================================
echo "=== Patching triton_kernels.cc ==="
TCC_TMP=/tmp/tcc_patched.cc
python3 /tmp/patch_tcc.py "$TCC_FILE" "$TCC_TMP"
PATCH_EXIT=$?
if [ $PATCH_EXIT -ne 0 ]; then
    echo "FATAL: Python patch script failed (exit $PATCH_EXIT)"
    exit 3
fi

sudo cp "$TCC_TMP" "$TCC_FILE"
echo "Patched file written back to $TCC_FILE via sudo"

echo "--- Verifying patch in source ---"
grep -n 'TE_BUG_DBG' "$TCC_FILE" | head -5

# =========================================================
# Step 4: Find build-jax.sh (exclude mock/test fixtures)
# =========================================================
echo "=== Finding build-jax.sh ==="
BUILD_SCRIPT=""
# Check PATH entries one by one, skipping mock paths
while IFS= read -r CANDIDATE; do
    if [ -n "$CANDIDATE" ] && echo "$CANDIDATE" | grep -qvE 'mock|triage|nsys-jax.*test'; then
        BUILD_SCRIPT="$CANDIDATE"
        break
    fi
done < <(which -a build-jax.sh 2>/dev/null)

# Fall back to find if not in PATH
if [ -z "$BUILD_SCRIPT" ]; then
    BUILD_SCRIPT=$(find /opt /usr/local/bin -maxdepth 5 -name 'build-jax.sh' \
        -not -path '*/mock*' -not -path '*/triage*' -not -path '*nsys-jax*' 2>/dev/null | head -1)
fi
echo "Build script: $BUILD_SCRIPT"

if [ -z "$BUILD_SCRIPT" ]; then
    echo "FATAL: No real build-jax.sh found (excluding mocks)"
    echo "All build-jax.sh on this system:"
    find / -name 'build-jax.sh' 2>/dev/null | grep -v proc
    exit 3
fi

# =========================================================
# Step 5: Rebuild jaxlib from patched source
# =========================================================
echo "=== Fixing permissions for Bazel build ==="
sudo chown -R "$(whoami)" /opt/jax /opt/xla /opt/jaxlibs 2>&1 || true
# Remove ALL stale editable jax install artifacts owned by root so pip can reinstall cleanly
sudo rm -f  /usr/local/lib/python3.12/dist-packages/__editable__*jax*.pth \
            /usr/local/lib/python3.12/dist-packages/__editable___jax*.py 2>/dev/null || true
sudo rm -rf /usr/local/lib/python3.12/dist-packages/jax*.dist-info 2>/dev/null || true

echo "=== Running jaxlib build: $BUILD_SCRIPT ==="
bash "$BUILD_SCRIPT" 2>&1
BUILD_EXIT=$?
echo "jaxlib build exit: $BUILD_EXIT"

if [ $BUILD_EXIT -ne 0 ]; then
    echo "FATAL: jaxlib rebuild failed (exit $BUILD_EXIT) — aborting (unpatched binary is useless)"
    echo "RESULT: JAXLIB_BUILD_FAILED"
    exit 3
fi

# =========================================================
# Step 6: Verify TE_BUG_DBG string in compiled binary
# =========================================================
echo "=== Verifying TE_BUG_DBG in compiled binary ==="
PATCH_SO=$(find /opt/jaxlibs -name '*.so' 2>/dev/null | while read SO; do
    if strings "$SO" 2>/dev/null | grep -q 'TE_BUG_DBG'; then
        echo "$SO"
        break
    fi
done)
if [ -n "$PATCH_SO" ]; then
    echo "BINARY_PATCH_VERIFIED: TE_BUG_DBG found in $PATCH_SO"
else
    echo "BINARY_PATCH_MISSING: TE_BUG_DBG not found in any .so under /opt/jaxlibs"
    exit 3
fi

# =========================================================
# Step 7: Build TransformerEngine
# =========================================================
echo "=== Installing build tools ==="
python3 -m pip install --user -q "cmake==3.21.0" pybind11 triton pytest 2>&1

echo "=== Building TransformerEngine ==="
cd /code/TransformerEngine
if [ "${CACHE_HIT:-0}" = "1" ]; then
    echo "Cache hit — skipping C++ compile, re-registering editable install"
    NVTE_CMAKE_BUILD_DIR=build_jax NVTE_FRAMEWORK=jax NVTE_CUDA_ARCHS="100a" \
        pip3 install --no-build-isolation -e . 2>&1 | tail -5
    TE_BUILD_EXIT=${PIPESTATUS[0]}
else
    echo "Cache miss — full C++ compile"
    rm -rf build_jax
    NVTE_CMAKE_BUILD_DIR=build_jax NVTE_FRAMEWORK=jax NVTE_CUDA_ARCHS="100a" \
        pip3 install --no-build-isolation -e . 2>&1 | tee /tmp/te_build.log
    TE_BUILD_EXIT=${PIPESTATUS[0]}
fi
echo "TE build exit: $TE_BUILD_EXIT"
if [ $TE_BUILD_EXIT -ne 0 ]; then
    echo "RESULT: TE_BUILD_FAILED"
    exit $TE_BUILD_EXIT
fi

# =========================================================
# Step 8: Run full test suite (TF INFO logs capture TE_BUG_DBG output)
# =========================================================
echo "=== Running full L0 JAX test suite (addr-log mode) ==="
export TF_CPP_MIN_LOG_LEVEL=0
export TF_CPP_VMODULE=triton_kernels=1
export TE_PATH=/code/TransformerEngine
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas

# Fix /logs permission: test.sh defaults XML_LOG_DIR=/logs (not writable in container);
# redirect to /tmp so TestFusedAttnWithDeterminism, mnist, encoder all run and
# contribute the correct BFC allocator history needed to reproduce the bug.
export XML_LOG_DIR=/tmp/jax_xml_logs
mkdir -p "$XML_LOG_DIR"

# Dump XLA HLO for both sort_chunks (aliasing investigation) and loss_fn (top-level
# buffer assignment showing the aliasing context).
XLA_DUMP_DIR=/code/xla_dump_addr_log_${SLURM_JOB_ID}
mkdir -p "$XLA_DUMP_DIR"
export XLA_FLAGS="--xla_gpu_enable_latency_hiding_scheduler=true \
    --xla_dump_to=${XLA_DUMP_DIR} \
    --xla_dump_hlo_module_re=sort_chunks|loss_fn"

bash qa/L0_jax_unittest/test.sh 2>&1
TEST_EXIT=$?
echo "Full suite exit: $TEST_EXIT"

echo "=== XLA dump summary ==="
ls "$XLA_DUMP_DIR" | wc -l | xargs -I{} echo "XLA dump files: {}"
echo "XLA dump dir: $XLA_DUMP_DIR"

echo "=== TE_BUG_DBG address log summary ==="
echo "(Address log lines appear inline in the output above — see TE_BUG_DBG markers)"

exit $TEST_EXIT
