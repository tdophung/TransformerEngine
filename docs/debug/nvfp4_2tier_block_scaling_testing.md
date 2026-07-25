# NVFP4 2-Tier Block Scaling Testing Guide

This guide covers the correctness-first C/C++ prototype for
`NVTE_NVFP4_2TIER_BLOCK_SCALING`. The implementation is rowwise-only, uses
separate unswizzled E4M3 S1 and S2 buffers, and requires contiguous `[M, K]`
input with `K % 256 == 0`.

## Test environment

Use a system with:

- a Blackwell-family GPU with compute capability 10.0 or newer;
- CUDA Toolkit 12.8 or newer, including NVCC and the FP4 headers;
- cuDNN, CMake, Ninja, a C++17 compiler, and Python development tools;
- initialized repository submodules.

Record the environment before testing:

```bash
nvidia-smi
nvcc --version
python3 --version
git rev-parse HEAD
git status --short
```

The implementation has a runtime compute-capability check. Testing on Hopper
or an older architecture should fail with a clear unsupported-architecture
error rather than produce results.

## Build Transformer Engine

Run these commands from the Transformer Engine repository root in a clean
Python environment:

```bash
git submodule update --init --recursive

NVTE_FRAMEWORK=none \
NVTE_CUDA_ARCHS=100 \
MAX_JOBS=8 \
python3 -m pip install -v --no-build-isolation .
```

Use `NVTE_CUDA_ARCHS=100`, not `100a`, at the package-build interface. The
common library CMake configuration derives the required `sm_100a`
architecture-specific target for `cast/cast.cu`, where the FP4 PTX is
instantiated.

Confirm that Python resolves the newly built installation:

```bash
python3 - <<'PY'
from pathlib import Path
import transformer_engine

print(Path(transformer_engine.__file__).resolve())
PY
```

## Build the C++ operator tests

The standalone C++ test project links against the installed
`libtransformer_engine`:

```bash
cmake \
  -S tests/cpp \
  -B tests/cpp/build-nvfp4-2tier \
  -GNinja \
  -DCMAKE_CUDA_ARCHITECTURES=100

cmake \
  --build tests/cpp/build-nvfp4-2tier \
  --target test_operator \
  -j8
```

If CMake finds an older Transformer Engine installation, provide the library
package directory explicitly:

```bash
TE_LIB_PATH="$(
  python3 - <<'PY'
from pathlib import Path
import transformer_engine

print(Path(transformer_engine.__file__).resolve().parent)
PY
)"

cmake \
  -S tests/cpp \
  -B tests/cpp/build-nvfp4-2tier \
  -GNinja \
  -DTE_LIB_PATH="${TE_LIB_PATH}" \
  -DCMAKE_CUDA_ARCHITECTURES=100
```

## Run the focused tests

Run the test binary directly for the clearest failures:

```bash
tests/cpp/build-nvfp4-2tier/operator/test_operator \
  --gtest_filter='NVFP4TwoTierBlockQuantize.*:NVFP4TwoTierBlockDequantize.*'
```

Five tests should run:

```text
NVFP4TwoTierBlockQuantize.FP32PartialMTileAndMultipleOuterBlocks
NVFP4TwoTierBlockQuantize.BF16PartialMTileAndMultipleOuterBlocks
NVFP4TwoTierBlockQuantize.ZeroAmaxUsesBenignOuterScale
NVFP4TwoTierBlockDequantize.FP32PartialMTileAndMultipleOuterBlocks
NVFP4TwoTierBlockDequantize.BF16PartialMTileAndMultipleOuterBlocks
```

The quantization tests compare all of the following bitwise against a host
reference:

- packed E2M1 bytes and nibble ordering;
- E4M3 S1 values at one scale per 16 elements;
- E4M3 S2 values at one scale per 256 elements;
- post-cast-S2 use when computing S1;
- zero-amax behavior;
- scale saturation and non-finite input behavior;
- zeroed padding in the physical `128 x 4`-aligned scale allocations;
- partial 16-row CTA tiles and multiple outer blocks per row.

The dequantization tests reconstruct a host reference from the actual packed
bytes, S1, and S2 buffers and verify:

```text
x_hat[r,k] = e2m1[r,k] * float(S1[r,k/16]) * float(S2[r,k/256])
```

They also exercise both public routes: the named 2-tier APIs and generic
`nvte_quantize`/`nvte_dequantize` dispatch.

The same tests can be run through CTest:

```bash
ctest \
  --test-dir tests/cpp/build-nvfp4-2tier \
  -R 'NVFP4TwoTierBlock' \
  --output-on-failure
```

## Regression testing

After the focused tests pass, run the complete operator test binary:

```bash
tests/cpp/build-nvfp4-2tier/operator/test_operator
```

Then run the complete C++ test project:

```bash
ctest \
  --test-dir tests/cpp/build-nvfp4-2tier \
  -j4 \
  --output-on-failure
```

Pay particular attention to existing one-tier NVFP4 and row-scaled NVFP4
tests. The new mode must not change their scale allocation, amax semantics,
packing, dequantization, or dispatch.

## Failure triage

### FP4 headers or instructions are unavailable

Verify CUDA 12.8 or newer is first on `PATH` and that the library was rebuilt
after changing CUDA installations:

```bash
command -v nvcc
nvcc --version
```

### Unsupported compute capability

Verify the test is running on a Blackwell-family device:

```bash
nvidia-smi --query-gpu=name,compute_cap --format=csv
```

### New API symbol is missing

The test binary is probably loading an older `libtransformer_engine`. Inspect
the resolved dependency and rebuild the C++ test directory with the intended
`TE_LIB_PATH`:

```bash
ldd tests/cpp/build-nvfp4-2tier/operator/test_operator \
  | grep transformer_engine
```

### Bitwise quantization mismatch

Rerun one failing case without parallel CTest output:

```bash
tests/cpp/build-nvfp4-2tier/operator/test_operator \
  --gtest_filter='NVFP4TwoTierBlockQuantize.FP32PartialMTileAndMultipleOuterBlocks' \
  --gtest_break_on_failure
```

Check the mismatch in this order:

1. S2 raw byte and physical row stride;
2. S1 raw byte and confirmation that it uses post-cast `float(S2)`;
3. the reciprocal of `float(S1) * float(S2)`;
4. E2M1 pair ordering in the packed output;
5. whether scale padding was zero-initialized.

## Acceptance checklist

- All five focused tests pass on BF16 and FP32 cases.
- Focused tests pass when run alone and through CTest discovery.
- Existing NVFP4 quantize and dequantize tests still pass.
- The complete `test_operator` binary passes.
- The complete C++ CTest suite passes.
- S1 and S2 padding remains zero.
- No test requires an `amax` allocation for the 2-tier mode.
- No columnwise, transpose, swizzle, stochastic-rounding, 4over6, or GEMM
  behavior is accidentally enabled.

## Current scope boundary

This branch provides the standalone C/C++ tensor metadata, quantize and
dequantize kernels, public C APIs, generic cast dispatch, and operator tests.
It does not provide a PyTorch/JAX quantizer, GEMM consumption, scale
swizzling, columnwise output, transpose fusion, or TMA optimization.
