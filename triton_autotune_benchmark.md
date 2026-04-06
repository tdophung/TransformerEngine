# Triton Permutation Kernel Autotuning Benchmark

Measures steady-state kernel latency (mean over 30 runs, ms) for every BLOCK_SIZE
candidate vs. the `TritonAutotunedKernelCall` path.

**Key question**: how much do we lose by disabling autotuning (falling back to the
default BLOCK_SIZE=64 that the primitive passes via `constexprs`)?

## Setup

| Item | Value |
|------|-------|
| Container | `gitlab-master.nvidia.com/dl/dgx/jax:jax` (latest nightly) |
| JAX fix | `jax-ml/jax#35218` — input_output_alias bug in `TritonAutotunedKernelCall` |
| `version_utils.py` guard | `TRITON_AUTOTUNED_INPUT_OUTPUT_ALIAS_MIN_JAX_VERSION = "0.9.2.dev0"` |
| cmake | 3.31.10 (pip, `>=3.28,<4.0`) |
| Dtype | bfloat16 |
| Warmup / runs | 5 / 30 |

**Benchmark cases**

| # | num\_tokens | num\_experts | hidden\_size | topk |
|---|-------------|--------------|--------------|------|
| 1 | 4 096 | 32 | 1 280 | 2 |
| 2 | 4 096 | 256 | 4 096 | 6 |
| 3 | 16 384 | 64 | 2 048 | 4 |

**sort\_chunks cases** (num\_splits, total\_tokens, hidden\_size)

| # | splits | tokens | hidden |
|---|--------|--------|--------|
| S1 | 32 | 4 096 | 1 280 |
| S2 | 64 | 4 096 | 4 096 |
| S3 | 64 | 16 384 | 2 048 |

---

## GB200 NVL4 (dlcluster, gb200nvl4 partition)

Each fixed-BLOCK_SIZE job ran on its own exclusive node to avoid OOM from concurrent
JAX processes.  The `AUTOTUNE_ON` job ran on a separate node; node-to-node GB200
variance makes it unsuitable for direct latency comparison, so the "no-autotuning
loss" column is derived from the fixed-BS runs only (BS=64 vs best-BS).

### dispatch\_fwd (token\_dispatch forward pass)

| Case (H) | BS=64 | BS=128 | BS=256 | BS=512 | BS=1024 | BS=2048 | BS=4096 | **Best BS** | **BS=64 loss** |
|----------|-------|--------|--------|--------|---------|---------|---------|-------------|----------------|
| H=1280   | 0.226 | 0.269  | **0.225** | 0.268 | 0.231  | 0.260   | 0.265   | 256         | +0.4%          |
| H=4096   | **0.589** | 0.639 | 0.596 | 0.646 | 0.604 | 0.632  | 0.640   | 64          | 0%             |
| H=2048   | **0.761** | 0.824 | 0.802 | 0.883 | 0.768 | 0.792  | 0.834   | 64          | 0%             |

### combine\_fwd (token\_combine forward pass)

| Case (H) | BS=64 | BS=128 | BS=256 | BS=512 | BS=1024 | BS=2048 | BS=4096 | **Best BS** | **BS=64 loss** |
|----------|-------|--------|--------|--------|---------|---------|---------|-------------|----------------|
| H=1280   | **0.165** | 0.188 | 0.166 | 0.194 | 0.171  | 0.198   | 0.191   | 64          | 0%             |
| H=4096   | **0.356** | 0.383 | 0.366 | 0.387 | 0.365  | 0.395   | 0.386   | 64          | 0%             |
| H=2048   | 0.457 | 0.487  | 0.479  | 0.543  | **0.444** | 0.476 | 0.502   | 1024        | +3%            |

### roundtrip\_fwd (dispatch → combine forward)

| Case (H) | BS=64 | BS=128 | BS=256 | BS=512 | BS=1024 | BS=2048 | BS=4096 | **Best BS** | **BS=64 loss** |
|----------|-------|--------|--------|--------|---------|---------|---------|-------------|----------------|
| H=1280   | **0.265** | 0.281 | 0.265 | 0.291 | 0.270  | 0.298   | 0.284   | 64/256 tie  | 0%             |
| H=4096   | **0.739** | 0.762 | 0.750 | 0.767 | 0.730  | 0.775   | 0.764   | 1024        | +1%            |
| H=2048   | **1.051** | 1.082 | 1.077 | 1.136 | 1.043 | 1.067   | 1.094   | 1024        | +1%            |

### roundtrip\_bwd (grad of dispatch → combine)

| Case (H) | BS=64 | BS=128 | BS=256 | BS=512 | BS=1024 | BS=2048 | BS=4096 | **Best BS** | **BS=64 loss** |
|----------|-------|--------|--------|--------|---------|---------|---------|-------------|----------------|
| H=1280   | **0.333** | 0.353 | 0.340 | 0.364 | 0.342  | 0.371   | 0.360   | 64          | 0%             |
| H=4096   | 1.280 | 1.309  | 1.284  | 1.313  | **1.266** | 1.318 | 1.308   | 1024        | +1%            |
| H=2048   | 1.716 | 1.744  | 1.733  | 1.780  | **1.677** | 1.725 | 1.758   | 1024        | +2%            |

### sort\_chunks\_by\_index forward

| Case (H) | BS=64 | BS=128 | BS=256 | BS=512 | BS=1024 | BS=2048 | BS=4096 | **Best BS** | **BS=64 loss** |
|----------|-------|--------|--------|--------|---------|---------|---------|-------------|----------------|
| S1 H=1280 | 0.172 | 0.196 | 0.181 | 0.249 | **0.162** | 0.181 | 0.209  | 1024        | **+6%**        |
| S2 H=4096 | 0.286 | 0.313 | 0.300 | 0.362 | **0.284** | 0.293 | 0.323  | 1024        | **+1%**        |
| S3 H=2048 | **0.497** | 0.536 | 0.521 | 0.584 | **0.495** | 0.520 | 0.542 | 1024        | **+0.4%**      |

### sort\_chunks\_by\_index backward

| Case (H) | BS=64 | BS=128 | BS=256 | BS=512 | BS=1024 | BS=2048 | BS=4096 | **Best BS** | **BS=64 loss** |
|----------|-------|--------|--------|--------|---------|---------|---------|-------------|----------------|
| S1 H=1280 | 0.226 | 0.252 | 0.234 | 0.285 | **0.220** | 0.234 | 0.265  | 1024        | **+3%**        |
| S2 H=4096 | 0.446 | 0.473 | 0.455 | 0.507 | **0.440** | 0.461 | 0.484  | 1024        | **+1%**        |
| S3 H=2048 | 0.829 | 0.857 | 0.843 | 0.907 | **0.825** | (noisy)| 0.869  | 1024        | **+0.5%**      |

### AUTOTUNE\_ON reference (job 694220, node ts2-105 — different node, for reference only)

| Case | dispatch\_fwd | combine\_fwd | roundtrip\_fwd | roundtrip\_bwd |
|------|--------------|-------------|----------------|----------------|
| H=1280 | 0.276 | 0.195 | 0.293 | 0.359 |
| H=4096 | 0.648 | 0.384 | 0.771 | 1.322 |
| H=2048 | 0.907 | 0.559 | 1.149 | 1.827 |

| Sort case | sort\_fwd | sort\_bwd |
|-----------|-----------|-----------|
| S1 H=1280 | 0.283 | 0.332 |
| S2 H=4096 | 0.386 | 0.555 |
| S3 H=2048 | 0.598 | 0.898 |

### Summary: GB200 NVL4

| Kernel group | BS=64 loss vs best available BS |
|---|---|
| `_permute_kernel` (dispatch) | **~0%** — BS=64 is optimal |
| `_unpermute_kernel` (combine) | **~0–3%** — BS=64 is optimal or near-optimal |
| `_unpermute_bwd_*` (roundtrip\_bwd) | **~1–2%** |
| `_sort_chunks_by_map_kernel` | **~1–6%** — BS=1024 consistently wins |

**Conclusion**: disabling autotuning (falling back to BS=64) has negligible impact on
dispatch/combine/roundtrip kernels on GB200.  For `sort_chunks_by_index` the
autotuner earns its keep: BS=1024 is 6% faster than BS=64 on the S1 case (small H).

---

## GB300 (prenyx, batch partition)

*(results pending — jobs submitted)*

<!-- GB300 results will be appended here -->
