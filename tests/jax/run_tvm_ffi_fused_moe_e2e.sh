#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

# End-to-end remote SM100 validation for the cuDNN-FE compile-only API,
# JAX TVM-FFI forward primitive, fused backward, partitioning, and MoEBlock.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TE_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CUDNN_FE_ROOT="${CUDNN_FE_ROOT:-$(cd "$TE_ROOT/.." && pwd)/cudnn-frontend}"
CUDNN_FE_REPOSITORY="${CUDNN_FE_REPOSITORY:-https://github.com/NVIDIA/cudnn-frontend.git}"
CUDNN_FE_BASE_REV="${CUDNN_FE_BASE_REV:-fbc713624ac403800ae286e422b5243d694abab2}"
CUDNN_FE_PATCH="$SCRIPT_DIR/patches/0001-Add-framework-neutral-grouped-SwiGLU-compiler.patch"
INSTALL_DEPS="${INSTALL_DEPS:-1}"
NUM_GPUS="${NUM_GPUS:-$(nvidia-smi -L | wc -l)}"
ARTIFACT_DIR="${E2E_ARTIFACT_DIR:-$TE_ROOT/tvm_ffi_e2e_$(date +%Y%m%d_%H%M%S)}"

if [ ! -f "$CUDNN_FE_ROOT/pyproject.toml" ]; then
    if [ ! -f "$CUDNN_FE_PATCH" ]; then
        echo "Missing bundled cuDNN-FE patch: $CUDNN_FE_PATCH" >&2
        exit 1
    fi
    mkdir -p "$(dirname "$CUDNN_FE_ROOT")"
    git clone --filter=blob:none "$CUDNN_FE_REPOSITORY" "$CUDNN_FE_ROOT"
    git -C "$CUDNN_FE_ROOT" checkout --detach "$CUDNN_FE_BASE_REV"
fi

CUDNN_FE_COMPILE_API="$CUDNN_FE_ROOT/python/cudnn/grouped_gemm/grouped_gemm_swiglu/compile.py"
if [ ! -f "$CUDNN_FE_COMPILE_API" ]; then
    if ! git -C "$CUDNN_FE_ROOT" apply --check "$CUDNN_FE_PATCH"; then
        echo "The bundled compile-only patch does not apply cleanly to $CUDNN_FE_ROOT." >&2
        echo "Use the pinned base $CUDNN_FE_BASE_REV or provide an already-patched checkout." >&2
        exit 1
    fi
    git -C "$CUDNN_FE_ROOT" apply "$CUDNN_FE_PATCH"
fi
if [ "$NUM_GPUS" -lt 2 ]; then
    echo "The E2E suite requires at least two SM100 GPUs" >&2
    exit 1
fi

mkdir -p "$ARTIFACT_DIR"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.5}"
export NVTE_CUDA_ARCHS="${NVTE_CUDA_ARCHS:-100a}"
export NVTE_CMAKE_BUILD_DIR="${NVTE_CMAKE_BUILD_DIR:-$TE_ROOT/build_jax}"

if [ "$INSTALL_DEPS" = "1" ]; then
    python3 -m pip install -U pip setuptools wheel cmake ninja pybind11 pytest
    python3 -m pip install -r "$TE_ROOT/tests/jax/requirements_cutedsl.txt"
    python3 -m pip install "$CUDNN_FE_ROOT[cutedsl-compile]"
    NVTE_FRAMEWORK=jax python3 -m pip install --no-build-isolation -e "$TE_ROOT"
fi

python3 - <<'PY' | tee "$ARTIFACT_DIR/preflight.log"
import inspect
import jax
import jax_tvm_ffi
import tvm_ffi
from cudnn import compile_grouped_gemm_swiglu

print("jax", jax.__version__)
print("devices", jax.devices())
print("cudnn_frontend", inspect.getfile(compile_grouped_gemm_swiglu))
print("jax_tvm_ffi", inspect.getfile(jax_tvm_ffi))
print("tvm_ffi", inspect.getfile(tvm_ffi))
assert all(device.platform == "gpu" for device in jax.devices())
PY

python3 -m pytest -q "$TE_ROOT/tests/jax/test_cutedsl_moe.py" \
    -k "compile_only_api or swiglu_forward_fused_output_parity or dswiglu_backward_quantized_output_parity" \
    2>&1 | tee "$ARTIFACT_DIR/standalone.log"

NUM_GPUS=2 TVM_FFI_MP_LOG_DIR="$ARTIFACT_DIR/partitioning" \
    bash "$TE_ROOT/tests/jax/run_tvm_ffi_grouped_mlp_multiprocess.sh" \
    2>&1 | tee "$ARTIFACT_DIR/partitioning.log"

if [ "$NUM_GPUS" -ge 4 ]; then
    NUM_GPUS=4 \
    TE_EP_MOE_MP_LOG_DIR="$ARTIFACT_DIR/moe" \
    NVTE_JAX_MOE_USE_CUDNN_CUTEDSL_FUSION=1 \
        bash "$TE_ROOT/tests/jax/run_te_ep_moe.sh" -k TestTeEpMoeCudnnCutedslFusion \
        2>&1 | tee "$ARTIFACT_DIR/moe.log"
else
    echo "SKIPPED integrated MoEBlock test: four GPUs are required" \
        | tee "$ARTIFACT_DIR/moe.log"
fi

echo "TVM-FFI fused MoE E2E validation passed. Artifacts: $ARTIFACT_DIR"
