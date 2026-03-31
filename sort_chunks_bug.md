# Permutation CI Intermittent Failure: XLA Buffer Aliasing Bug

## Problem

**Test**: `tests/jax/test_permutation.py::test_sort_chunks_by_index`
**Project**: TransformerEngine (JAX backend)

### Symptom

- ~50% element mismatch in **backward pass gradients** (`computed_grad != ref_grad`)
- Forward pass and `row_id_map` are correct
- Fails ~30% overall across systems, but **deterministically** when running the full L0 test suite via `stress_test.sh` (all `tests/jax/` in one pytest process)
- **Never** fails when running `pytest test_permutation.py` in isolation

### Diagnostic output in test

```
BUG: sort_chunks_by_map BACKWARD kernel (fwd+map OK, grad wrong)
Theory: XLA buffer aliasing during autotuning of FORWARD=False kernel
```

---

## Approach 3 WAR — Current Implementation Status

### First half (APPLIED in `utils.py`)

`output_to_operand_aliasing={{0}: (3, {})}` IS passed to `jax.ffi.ffi_lowering(operand_output_aliases=...)`.
This declares to XLA that the custom-call's output[0] lives in the same buffer as input[3] (`output_buf`).

**Confirmed present** in the inner module's HLO (from XLA VLOG in job 514131):
```
%te_sort_chunks_by_map_triton.3 = custom-call(...), output_to_operand_aliasing={{0}: (3, {})}
```

### Second half (BLOCKED, NOT applied)

Passing `input_output_aliases_with_sizes` to `TritonAutotunedKernelCall` is blocked by a **jaxlib bug**:
the restore phase in `triton_kernels.cc` unconditionally accesses `input_copies[input_idx]` even when XLA
did NOT actually alias the buffers (the save phase only saves when `buffers[input_idx] == buffers[output_idx]`).
Accessing an entry that was never saved causes `CUDA_ERROR_INVALID_VALUE`.

**Current workaround in `utils.py`** (line ~493):
```python
kernel_call = gpu_triton.TritonAutotunedKernelCall(
    f"{actual_kernel_fn.__name__}_autotuned",
    kernel_calls,
    (),  # Empty — avoids buggy save/restore in jaxlib/gpu/triton_kernels.cc
)
```

**Fix needed in jaxlib**: add `if (input_copies.count(input_idx) == 0) continue;` guard in the restore loop
before accessing `input_copies[input_idx]`.

---

## Root Cause Analysis

Three interacting factors combine to produce the corruption:

### 1. XLA implicit buffer aliasing — CONFIRMED (buffers[0] == buffers[5])

XLA's BFC allocator assigns the **same GPU memory** to `output_grad` (input[0]) and `permuted_probs`
(output[1]) of the backward `sort_chunks_by_map` custom call. Both have the same shape (`f32[4096,1280]`).
XLA determines `output_grad` is dead after the call and reuses its physical memory for the second output.

**Confirmed by runtime address logging** (jobs 547975/547976):
```
buffers[0] = 0x7f1344410e00  ← output_grad (input[0])
buffers[5] = 0x7f1344410e00  ← permuted_probs (output[1])  SAME ADDRESS
```

Note: The VLOG analysis of job 514131 looked at `jit_loss_fn` allocations and found SEPARATE buffers for
`output_grad` and `output_buf` (param 3). That was the wrong pair — the actual aliasing is between
`output_grad` (param 0) and `permuted_probs` (second output), not `output_buf` (param 3).

### 2. Permutation kernels cannot operate in-place

`_sort_chunks_by_map_kernel` (in `transformer_engine/common/triton/permutation.py`) reads from `src_row`
and writes to `dst_row` where `src_row != dst_row`. When input and output share the same buffer, GPU thread
blocks execute in waves — early blocks overwrite data that later blocks haven't read yet, causing corruption.

### 3. Triton autotuning amplifies corruption

`TritonAutotunedKernelCall` (in `jaxlib/gpu/triton_kernels.cc`) runs the kernel multiple times with
different configs. It normally saves/restores aliased buffers between runs using
`input_output_aliases_with_sizes`. But currently `()` (empty) is passed, so no save/restore occurs.
If buffers ARE aliased at runtime, each autotuning trial corrupts the shared buffer further.

### Why only `sort_chunks_by_map`?

Other permutation kernels in TE (e.g., `_permute_with_mask_map_kernel`) already declare explicit
`input_output_aliases`, which **claims** the output buffer slot and prevents XLA from implicitly aliasing
a different input to that output. `_sort_chunks_by_map_kernel` lacked this explicit alias.

### Why only in the full test suite?

Running all tests in a single pytest process changes XLA's buffer assignment decisions compared to an
isolated run — likely due to different memory pressure, compilation caches, or HLO graph differences.

---

## VLOG Analysis (job 514131, BUG_REPRODUCED run)

### Setup

`TF_CPP_VMODULE=buffer_assignment=3` is set in the inner test script before running `test.sh`. The XLA
C++ VLOG output appears in pytest's **"Captured stderr call"** section for the failing test
`test_sort_chunks_by_index[dtype_float32-8-4096-1280]`. All 37 module compilations that occur during
that test are captured there.

### Outer JIT module discovery

The outer JIT is **`jit_loss_fn`**, not `jit_sort_chunks_by_index`. The test wraps the call inside:

```python
@jax.jit
def loss_fn(x):
    output, _ = sort_chunks_by_index(x, split_sizes, sorted_indices)
    return jnp.sum(output**2)

loss_val, computed_grad = jax.value_and_grad(loss_fn)(inp)
```

Two `jit_loss_fn` compilations occur:

| # | Layout | Role |
|---|--------|------|
| 1 | `(f32[4096,1280]) → (f32[], s32[4096], f32[4096,1280])` | Forward pass + residuals for VJP |
| 2 | `(s32[4096], f32[4096,1280], f32[]) → f32[4096,1280]` | Backward pass (VJP) |

### Forward `jit_loss_fn` — custom-call parameters

```
%te_sort_chunks_by_map_triton.3 = custom-call(
  %Arg_0.1,              ← param 0: input x           → HloBuffer 16
  %te_make_chunk_sort_map_triton.1,  ← param 1: row_id_map
  %constant_7_0,         ← param 2: probs (f32[0])
  %loop_broadcast_fusion ← param 3: output_buf        → HloBuffer 21
)
output_to_operand_aliasing={{0}: (3, {})}
```

- `Arg_0.1` (input `x`) → **HloBuffer 16** (separate from output)
- `loop_broadcast_fusion` (output_buf) → **HloBuffer 21**, also aliased to output[0]
- **Separate buffers in the forward pass** — consistent with expectations.

### Backward `jit_loss_fn` — custom-call parameters

```
%te_sort_chunks_by_map_triton.3 = custom-call(
  %loop_multiply_fusion, ← param 0: output_grad = 2 * Arg_1.1 * broadcast(Arg_2.1)
  %Arg_0.1,              ← param 1: row_id_map residual (s32[4096])
  %constant_2_0,         ← param 2: probs (f32[0])
  %loop_broadcast_fusion ← param 3: output_buf = broadcast(0.0)
)
output_to_operand_aliasing={{0}: (3, {})}
```

Where:
- `Arg_0.1`: `s32[4096]` = row_id_map residual from forward
- `Arg_1.1`: `f32[4096,1280]` = saved forward output (residual)
- `Arg_2.1`: `f32[]` = grad cotangent (= 1.0)
- `loop_multiply_fusion` = `2 × Arg_1.1 × broadcast(Arg_2.1)` = output_grad

### Backward `jit_loss_fn` — physical allocations (BUG run)

```
allocation 0: size 20971520, maybe-live-out (OUTPUT)
  loop_broadcast_fusion  @offset 0  ← output_buf (param 3) = inp_grad return value

allocation 1: size 20971520, parameter 1
  Arg_1.1                           ← saved forward output

allocation 2: size 16384,   parameter 0
  Arg_0.1                           ← row_id_map

allocation 4: size 4,       parameter 2
  Arg_2.1                           ← grad cotangent (scalar)

allocation 5: size 20971776, preallocated-temp
  te_sort_chunks_by_map_triton.3{}  @offset 0   (16 bytes) ← status tuple
  loop_multiply_fusion              @offset 256             ← output_grad (param 0)
  te_sort_chunks_by_map_triton.3{1} @offset 256 (0 bytes)  ← empty probs
```

**Key finding**: `output_grad` (param 0 of custom-call) is in **allocation 5, offset 256** (temp buffer).
`output_buf` (param 3 of custom-call) is in **allocation 0** (output buffer). These are **different
physical allocations** — they cannot share the same GPU memory address.

### Implication

The original aliasing hypothesis (outer module assigns `output_grad` and `output_buf` to the same GPU
buffer) is **NOT confirmed** by the VLOG data from the BUG run. The XLA buffer_assignment.cc analysis
(see below) also confirms this should not happen — two values with overlapping live ranges are always
placed at non-overlapping offsets by the heap simulator.

**The exact corruption mechanism is still unknown.** Possible remaining explanations:
- Aliasing happens in a context not captured by the VLOG (e.g., an even outer JIT, or during tracing)
- The corruption comes from a different execution path than what was analyzed
- A bug in the Triton autotuning itself (independent of aliasing)

### Modules NOT in the VLOG (compiled before the failing test)

The backward `jit_te_sort_chunks_by_map_triton` does NOT appear as a standalone module in the VLOG —
it is compiled **inline** as a `triton_kernel_call` custom-call inside `jit_loss_fn`. Only the **forward**
`jit_te_sort_chunks_by_map_triton` appears standalone (triggered by the eager
`sort_chunks_by_index(inp, ...)` call before `jax.value_and_grad`).

### Inner module allocation (both PASS and BUG — identical)

From `xla_dump_514129_pass/` and `xla_dump_514131_bug_reproduced/` (captured via
`--xla_dump_hlo_module_re=sort_chunks`):

```
allocation 0: size 20971520, maybe-live-out   ← output (aliased to copy.4)
  copy.4, te_sort_chunks_by_map_triton.3{0}

allocation 1: size 20971520, parameter 0      ← output_grad / input x (SEPARATE)
  args_0_.1

allocation 2: size 20971520, parameter 3      ← output_buf (SEPARATE)
  args_3_.1
```

The inner module's buffer assignment is **identical between PASS and BUG** — the inner module itself
is not the source of aliasing. The aliasing (if any) would have to come from the caller.

---

## XLA Code Analysis

### `CanShareOperandBufferWithUser`

**File**: `xla/hlo/analysis/hlo_dataflow_analysis.cc`

- For `kCustomCall` **without** explicit `output_operand_aliasing`, returns `false` — direct operand
  sharing path does NOT allow aliasing between `output_grad` and `inp_grad`.
- With `output_to_operand_aliasing={{0}: (3, {})}`, returns `true` only for input[3] (the declared alias),
  NOT for input[0]. So input[0] cannot share with output[0] via this path.

### `MaybeAssignBuffer` / `LiveRangeInterferes`

**File**: `xla/service/buffer_assignment.cc`

- The `can_share_as_operand` lambda calls `CanShareOperandBufferWithUser` and only permits sharing when
  live ranges touch (not overlap) AND the dataflow analysis approves. For our custom-call, this blocks
  input[0] from sharing with output[0].
- Entry computation **parameters** (`kParameter`) always get `NewAllocation()` called immediately and
  are **never** passed to `MaybeAssignBuffer` or the heap simulator.

### Heap Simulator (`GlobalDecreasingSizeBestFitHeap`)

**File**: `xla/service/heap_simulator/heap_simulator.cc`

- Two non-parameter buffers with **overlapping live ranges** are placed at **non-overlapping**
  `[offset, offset+size)` intervals by `MakeFreeChunks` + `CommitChunk`.
- In the backward `jit_loss_fn`: `loop_multiply_fusion` (output_grad) and `loop_broadcast_fusion`
  (output_buf) are BOTH live during the custom-call, so they CANNOT share a heap offset.

### Conclusion

The VLOG analysis looked at the wrong buffer pair (`output_grad` vs `output_buf`). The actual aliasing
is between `output_grad` (param 0) and `permuted_probs` (output[1]) — both `f32[4096,1280]`. XLA
correctly identifies `output_grad` as dead after the custom call and reuses its memory for the second
output. This aliasing is **not prevented** because no `output_to_operand_aliasing` is declared for
output[1] ↔ input[0]. The declared alias `{{0}: (3, {})}` only protects output[0] ↔ input[3].

**Root cause confirmed by runtime address logging**: `buffers[0] == buffers[5]` in BUG_REPRODUCED runs
(jobs 547975, 547976). Not observed in PASS runs (different BFC allocator history → different addresses).

---

## WAR: Approach 3 — Pre-allocated Output Buffer

**Status**: **First half in place** (`output_to_operand_aliasing` declared via `ffi_lowering`).
**Second half blocked** pending jaxlib fix. Bug still occurs because autotuning save/restore is disabled.

**Pattern**: Mirror what `PermuteWithMaskMapPrimitive` already does — pass a pre-allocated output buffer
as an additional input and declare an explicit `input_output_alias`.

### Changes across 3 files

#### File 1: `transformer_engine/common/triton/permutation.py`

Add `output_buf_ptr` as a new input pointer to `_sort_chunks_by_map_kernel`:

```python
@triton.jit
def _sort_chunks_by_map_kernel(
    input_ptr,
    row_id_map_ptr,
    probs_ptr,
    output_buf_ptr,       # NEW: pre-allocated output buffer (aliased to output)
    stride_input_token,
    stride_input_hidden,
    stride_output_token,
    stride_output_hidden,
    stride_probs_token,
    stride_permuted_probs_token,
    output_ptr,
    permuted_probs_ptr,
    hidden_size: tl.constexpr,
    PERMUTE_PROBS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    FORWARD: tl.constexpr,
):
```

The kernel body does NOT use `output_buf_ptr` — it only occupies a parameter slot for the JAX lowering
alias declaration.

#### File 2: `transformer_engine/jax/triton_extensions/permutation.py` — `SortChunksByMapPrimitive`

- **`abstract()`**: Accept `output_buf_aval` as a new input.
- **`impl()`**: Allocate a dummy output buffer (`jnp.empty(...)`) with the same shape/dtype as the
  expected output and pass it to `inner_primitive.bind()`.
- **`lowering()`**: Accept `output_buf` as a new operand. Declare `input_output_aliases={3: 0}` (input
  index 3 = `output_buf` aliased to output index 0). Pass `output_buf` to `triton_call_lowering()`.
- **`partition()`**: Accept and discard `output_buf` in `sharded_impl()`.
- **`shardy_sharding_rule()`**: Add `output_buf_spec` to the input specs.

#### File 3: `transformer_engine/jax/triton_extensions/utils.py`

The `triton_call_lowering` function already passes `ffi_operand_output_aliases` to
`jax.ffi.ffi_lowering(operand_output_aliases=...)`. **No further changes needed here** for the first half.

For the second half (once jaxlib is fixed): change line ~493 from `()` to `input_output_aliases_with_sizes`
in the `TritonAutotunedKernelCall(...)` call.

### Why this works

By declaring `input_output_aliases={3: 0}`, XLA knows output slot 0 is claimed by input 3 (`output_buf`).
XLA will not implicitly assign any other input (like `output_grad`) to that output buffer. Input and
output are guaranteed separate memory.

---

## Other Approaches Explored (Not Pursued)

### Approach 1: Remove autotuning for `sort_chunks_by_map`

Would fix the bug but loses performance.

### Approach 2: D2D de-aliasing on jaxlib side (`triton_kernels.cc`)

Add a `DeAliasState` struct in `TritonAutotunedKernelCall` that detects implicit aliasing at runtime
and copies aliased inputs to temporary buffers before autotuning. Was partially implemented then
**reverted** in favor of Approach 3. More complex, requires jaxlib changes.

### Initial diff applied to `triton_kernels.cc`

An early attempt to pass `input_output_aliases_with_sizes` to `TritonAutotunedKernelCall` but NOT to
`jax.ffi.ffi_lowering`. Made the problem **worse** (deterministic forward failure too), because without
the FFI alias hint XLA still aliased the buffers, but now the autotuner's D2H/H2D save/restore was
active and interfered.

---

## Debugging Setup

### VLOG approach

Set `TF_CPP_VMODULE=buffer_assignment=3` before running the test suite (done in the inner container script
in `jax_test_full_suite.sh`). The C++ output appears in pytest's **"Captured stderr call"** sections in
the main log file. Only FAILED tests show their captured stderr, so VLOG from the failing test is visible
but PASS tests' VLOG is silenced by pytest.

**Note**: `TF_CPP_VMODULE` controls absl/glog verbosity in the XLA C++ layer. Modern JAX nightly
containers (2026-01-13) do include this logging. The VLOG DOES work — confirmed by job 514131.

### XLA HLO dump approach

`--xla_dump_to=DIR --xla_dump_hlo_pass_re=buffer-assignment` dumps buffer-assignment pass HLO for all
(or filtered) modules. Module filter: `--xla_dump_hlo_module_re=<pattern>`.

**Key finding**: `--xla_dump_hlo_module_re=sort_chunks` only captures the **inner primitive**
(`jit_te_sort_chunks_by_map_triton`). The outer `jit_loss_fn` module is NOT captured by this filter.
To capture it, use `--xla_dump_hlo_module_re=loss_fn` or drop the module regex entirely.

### Discovery job 515903 (in progress)

Runs only `test_sort_chunks_by_index` in isolation, dumps ALL XLA modules (no module regex filter).
Goal: identify the outer module name and full buffer layout in isolation. Results will show whether
`jit_loss_fn` appears and what its buffer assignment looks like without memory pressure from the full suite.

---

## Key Files

| File | Purpose |
|------|---------|
| `transformer_engine/common/triton/permutation.py` | Triton kernel definitions (`_sort_chunks_by_map_kernel`) |
| `transformer_engine/jax/triton_extensions/permutation.py` | JAX primitives, lowering, sharding (`SortChunksByMapPrimitive`) |
| `transformer_engine/jax/triton_extensions/utils.py` | `triton_call_lowering` helper (FFI lowering + alias passing) |
| `transformer_engine/jax/permutation.py` | High-level API (`sort_chunks_by_index` with `@custom_vjp`) |
| `tests/jax/test_permutation.py` | Test file (line ~752: `test_sort_chunks_by_index`) |
| `qa/L0_jax_unittest/stress_test.sh` | Stress test runner |
| `jaxlib/gpu/triton_kernels.cc` | Triton autotuning + kernel dispatch in jaxlib |
| `xla/service/buffer_assignment.cc` | XLA buffer assignment logic |
| `xla/hlo/analysis/hlo_dataflow_analysis.cc` | `CanShareOperandBufferWithUser` |
| `xla/service/heap_simulator/heap_simulator.cc` | Heap-based buffer offset assignment |

### dlcluster artifacts

| Location | Contents |
|----------|---------|
| `bug_repro_logs/xla_dump_514129_pass/` | XLA buffer-assignment dumps, PASS run (sort_chunks modules only) |
| `bug_repro_logs/xla_dump_514130_pass/` | XLA buffer-assignment dumps, PASS run (sort_chunks modules only) |
| `bug_repro_logs/xla_dump_514131_bug_reproduced/` | XLA buffer-assignment dumps, BUG run (23,248 files, sort_chunks modules only) |
| `bug_repro_logs/full_suite_514131_BUG_REPRODUCED.log` | Full log with VLOG embedded — primary analysis source |
| `bug_repro_logs/outer_module_discovery_515903.log` | (pending) Discovery job: all modules in isolation |

---

## `triton_kernels.cc` Autotune Flow Analysis

**File**: `/Users/tdophung/Repos/jax/jaxlib/gpu/triton_kernels.cc`

### Full `Autotune()` + `Launch()` execution path

```
TritonKernelCall(stream, buffers, opaque)
  └─ GetKernelCall(opaque, stream, buffers)
       ├─ [first call]: AutotunedKernelCall::Autotune(move(call), stream, buffers)
       │    ├─ Save: for each alias (input_idx, output_idx, size):
       │    │    if buffers[input_idx] == buffers[output_idx]: copy to host
       │    ├─ Benchmark warmup + timed runs for each config (all write to buffers[output_idx])
       │    ├─ Restore: copy host saves back to buffers[input_idx]
       │    ├─ gpuStreamSynchronize(stream)
       │    └─ return configs_[0].kernel_call   ← winning KernelCall (NOT a re-run of winner)
       └─ cache winning KernelCall*, return it
  └─ kernel_call->Launch(stream, buffers)       ← ONE final execution of winning config
```

**Key facts confirmed from source**:
1. `Launch()` IS always called after `Autotune()` — no code path skips it.
2. `Autotune()` does NOT re-run the winner before returning — it just swaps it to `configs_[0]`.
   The winner's output after autotuning is whatever the last benchmark trial left in `buffers[output_idx]`.
3. After the restore + `gpuStreamSynchronize`, the final `Launch()` runs. Since this is a new
   kernel dispatch on the same stream (asynchronous), XLA's subsequent operations see its result.
4. `KernelCall::Launch()` has a `bytes_to_zero` per-parameter field. All parameters created in
   `triton_call_lowering` use `bytes_to_zero=0` — no pre-zeroing by `Launch()` itself.
5. `TritonAutotuneAliasRestoreFixEnabled()` (env: `JAX_TRITON_AUTOTUNE_ALIAS_RESTORE_FIX`, default ON)
   prevents the nullptr CUDA error in the restore phase when XLA didn't actually alias buffers.

### `_sort_chunks_by_map_kernel` read-write pattern

Examining the kernel source (`transformer_engine/common/triton/permutation.py:596`):

```python
inp = tl.load(input_ptr + input_offsets, mask=mask)   # reads from input_ptr[src_row, :]
tl.store(output_ptr + output_offsets, inp, mask=mask)  # writes to output_ptr[dst_row, :]
# output_buf_ptr is UNUSED (# pylint: disable=unused-argument)
```

The kernel **reads from `input_ptr` and writes to `output_ptr`**. It does NOT read from `output_ptr`.
It is a gather-scatter: `output_ptr[dst_row, :] = input_ptr[src_row, :]`.

**Critically**: if `input_ptr == output_ptr` (same physical buffer), different GPU thread blocks write
to `output[A]` while reading from `input[B]` where `B` may equal `A` from a different block's
perspective. Blocks execute in waves — early blocks overwrite positions that later blocks still need
to read, causing corruption. This is the core reason permutation cannot operate in-place.

### All positions are written per launch

Grid: `(num_tokens, cdiv(hidden_size, min_BLOCK_SIZE=64))` = `(4096, 20)` — fixed for all configs.

For `hidden_size=1280`, `FORWARD=False`:
- `dst_row = pid_t` (0..4095 covering all tokens exactly once)
- `mask = current_offset < 1280` (ensures only valid elements are stored)
- For each `(pid_t, pid_h)` in the grid, ALL valid positions of `output_ptr[pid_t, :]` are written

Every one of the `4096 × 1280 = 20,971,520` elements is written exactly once. Wasted thread blocks
(pid_h beyond `cdiv(1280, BLOCK_SIZE)`) do nothing due to masking — no spurious writes or reads.

### Why save/restore is expected to have no effect on a write-only kernel

Given the above, the expected behavior WITHOUT save/restore is identical to WITH save/restore:

| Step | Without save/restore | With save/restore |
|------|---------------------|-------------------|
| Before Autotune | alloc0 = zeros (loop_broadcast_fusion) | alloc0 = zeros |
| Autotune trials | alloc0 = last trial's correct result | alloc0 = last trial's correct result |
| Restore | — | alloc0 = zeros (original state) |
| Launch() | alloc0 = winning config's correct result | alloc0 = winning config's correct result |

Both paths produce the same `alloc0 = correct result` after `Launch()`.

### The open question: why does the WAR empirically fix the corruption?

The static analysis above cannot explain why save/restore would fix a purely write-only kernel.
The user has confirmed empirically that applying BOTH halves (including aliases to
`TritonAutotunedKernelCall`) fixes the bug. Possible explanations not ruled out by static analysis:

**H1 — GPU L2 cache flushing effect of H2D/D2H copies**:
The `gpuMemcpyDtoHAsync` + `gpuMemcpyHtoDAsync` calls issued during save/restore force GPU L2 cache
flush and refill. Without these copies, alloc0 might contain stale values from a previous benchmark
iteration that were in an L2 "dirty" state and not yet visible to a subsequent read-back by XLA (e.g.,
if XLA verifies the output on the CPU side). This is the most plausible GPU-microarchitecture explanation.

**H2 — The bug is elsewhere, WAR is a coincidence**:
The corruption might be in a DIFFERENT kernel or operation (not `_sort_chunks_by_map_kernel`), and
the aliases/save/restore machinery changes something unrelated (scheduling, stream serialization, etc.)
that happens to prevent the other corruption.

**H3 — `Benchmark()` timed_iters calculation interacts with XLA memory pressure**:
Under high memory pressure, `kBenchmarkTimeMillis=10ms` benchmarking launches up to 707 kernel
invocations (7 configs × (1 warmup + 100 timed)). This may interact with CUDA's async memory reclaim
or cuMemPoolTrimTo in a way that corrupts alloc0's physical pages. The D2H copy during save would
force synchronization before alloc0 pages are reclaimed.

**H4 — Cluster jaxlib has a different `Autotune()` implementation**:
The cluster's nightly jaxlib (`/opt/jaxlibs/jaxlib/`, updated daily) may differ from the local JAX
repo. If the cluster's version has different save/restore logic, the empty-`()` path might diverge.

### Targeted investigation results

1. **Disable autotuning entirely** (single-config / no-autotune experiment):
   Gated with `NVTE_DISABLE_SORT_CHUNKS_AUTOTUNE=1` env var. This disables both `@triton.autotune`
   (fixes to single fixed config) AND the `input_output_aliases={3: 0}` FFI hint in
   `SortChunksByMapPrimitive.lowering()` — testing autotuning removal in isolation.

   **Result: 7/7 passes** (3 user-manual + 4 dlcluster jobs 517314–517317, all `FAIL_OTHER_exit1`
   = unrelated test failures, `test_sort_chunks_by_index` itself PASSED in all). **Autotuning is
   confirmed as the root cause.**

2. **Approach 1 — Runtime buffer address logging** (in progress):
   Patch `triton_kernels.cc::Autotune()` to emit `LOG(INFO)` for all buffer GPU addresses.
   Key check: `buffers[0]==buffers[4]? YES_PHYSICAL_ALIAS` would confirm physical aliasing hypothesis.
   Jobs submitted: **535330, 535531–535534** (5 jobs for PASS/BUG statistical coverage).
   The inner script discovers `/opt/jax/jaxlib/gpu/triton_kernels.cc`, patches it with a Python script,
   runs `build-jax.sh` to rebuild jaxlib, then runs the full suite with `TF_CPP_MIN_LOG_LEVEL=0`.

3. **Minimal standalone repro script** (validation in progress):
   `tests/jax/minimal_sort_chunks_repro.py` — creates memory pressure, runs backward trials, checks
   for gradient corruption. Jobs: **535332, 535333, 535335** running. Not yet confirmed to reproduce
   the bug — will update once results are in.

---

## Current State (updated 2026-03-17)

- **Approach 3 first half is in place**: `output_to_operand_aliasing={{0}: (3, {})}` declared in `utils.py`
  via `ffi_lowering`. Inner module HLO confirms it is present.
- **Approach 3 second half confirmed to fix the bug**: User manually applied both halves and confirmed
  the bug is resolved. The diff is in `utils_diff`. **The exact mechanism remains unknown** — static
  analysis shows the kernel reads input/writes output and `Launch()` always runs.
- **Bug still reproduces WITHOUT both halves** (confirmed jobs 509116, 514131 on dlcluster `b100_preprod`
  and addr_log jobs 546885, 546886 on `b100_preprod`).
- **VLOG analysis complete**: outer `jit_loss_fn` backward module shows SEPARATE allocations for
  output_grad (alloc 5+256) and output_buf (alloc 0). Original aliasing hypothesis is NOT confirmed
  at the XLA buffer-assignment level.
- **Autotuning confirmed as root cause**: 7/7 no-autotune runs passed (3 user-manual + 4 dlcluster
  jobs 517314–517317). Disabling autotuning alone (without the FFI alias hint) is sufficient to prevent
  the corruption.
- **Active experiments**: None — root cause now confirmed.
- **Minimal repro**: CONCLUDED — all 7 jobs PASS. Bug cannot be reproduced without the full L0 test
  suite's BFC allocator history (needed to drive `output_grad` and `permuted_probs` to the same address).
- **Root cause CONFIRMED** (jobs 547975/547976): `buffers[0] == buffers[5]` — XLA's BFC allocator
  reuses `output_grad`'s GPU memory for `permuted_probs` output. During 100+ autotune iterations, the
  kernel reads `output_grad` and writes `permuted_probs` at the same GPU address, corrupting the buffer.
- **XLA dumps rescued**: `bug_repro_logs/xla_dump_addr_log_547973/` (23,460 files) — contains full
  `jit_loss_fn` buffer-assignment artifacts with `sort_chunks|loss_fn` regex.

---

## Approach 1 Findings (addr_log, jobs 546885/546886 — BUG_REPRODUCED runs)

### Infrastructure fixes required (discovered during addr_log experiments)

**`/logs` permission bug**: `qa/L0_jax_unittest/test.sh` defaults `XML_LOG_DIR=/logs`. In the Docker
container `/logs` is not writable, causing `TestFusedAttnWithDeterminism`, `mnist`, and `encoder` tests
to abort before executing any GPU ops. These tests run BEFORE `test_sort_chunks_by_index` and their
GPU memory activity is part of the BFC allocator state needed to trigger the bug.

**Fix**: Set `XML_LOG_DIR=/tmp/jax_xml_logs` in the inner container script before calling `test.sh`.
Evidence: addr_log jobs 545402/403/405 all FAIL_OTHER without reproducing bug (missing `/logs` mount).
Jobs 546885/886 — first with the fix — BOTH reproduced the bug (BUG_REPRODUCED).

**XLA dump persistence bug**: Setting `--xla_dump_to=/tmp/xla_dump_...` writes inside the container's
ephemeral `/tmp`. Dump is lost when container exits (`--rm`). Fix: use `/code/xla_dump_...` (host-mounted
volume) and rescue to `bug_repro_logs/` in the outer script before `rm -rf "$WORK_DIR"`.

### `input_output_aliases_` is empty

The debug block in `Autotune()` revealed:
```
[TE_BUG_DBG] Autotune() kernel=_sort_chunks_by_map_kernel_autotuned input_output_aliases_.size()=0
[TE_BUG_DBG]   buffers[0]=0x769852000900
```
`input_output_aliases_` has **zero entries** for this kernel. The original debug block only logged
`buffers[0]` because `_dbg_max` was computed from the (empty) alias list. This was fixed to hardcode
6 buffers and do an all-pairs comparison.

**Implication**: The save/restore loop in `Autotune()` is a no-op for this kernel (no aliases declared).
The Triton-level aliasing mechanism (`input_output_aliases_with_sizes`) does not apply here.

### Physical aliasing CONFIRMED at Autotune() time (jobs 547975/547976)

**Buffer layout**: `[0]=output_grad  [1]=row_id_map  [2]=probs  [3]=output_buf  [4]=output  [5]=permuted_probs`

From BUG_REPRODUCED run 547975 (first Autotune() call):
```
[TE_BUG_DBG]   buffers[0]=0x7f133e000900
[TE_BUG_DBG]   buffers[1]=0x7f133f400900
[TE_BUG_DBG]   buffers[2]=0x7f11e7e3df00
[TE_BUG_DBG]   buffers[3]=0x7f1340810d00
[TE_BUG_DBG]   buffers[4]=0x7f1340810d00
[TE_BUG_DBG]   buffers[5]=(nil)
[TE_BUG_DBG] YES_PHYSICAL_ALIAS buffers[3]=0x7f1340810d00 == buffers[4]=0x7f1340810d00
```

Second Autotune() call (the failing backward config):
```
[TE_BUG_DBG]   buffers[0]=0x7f1344410e00
[TE_BUG_DBG]   buffers[1]=0x7f133f400900
[TE_BUG_DBG]   buffers[2]=0x7f11e7e3eb00
[TE_BUG_DBG]   buffers[3]=0x7f1341c10d00
[TE_BUG_DBG]   buffers[4]=0x7f1341c10d00
[TE_BUG_DBG]   buffers[5]=0x7f1344410e00
[TE_BUG_DBG] YES_PHYSICAL_ALIAS buffers[0]=0x7f1344410e00 == buffers[5]=0x7f1344410e00
[TE_BUG_DBG] YES_PHYSICAL_ALIAS buffers[3]=0x7f1341c10d00 == buffers[4]=0x7f1341c10d00
```

**Two distinct aliases observed:**

| Alias | Buffers | Type |
|-------|---------|------|
| `buffers[3] == buffers[4]` | `output_buf == output` | **Expected** — declared via `output_to_operand_aliasing={{0}: (3, {})}` |
| `buffers[0] == buffers[5]` | `output_grad == permuted_probs` | **ROOT CAUSE** — unexpected XLA reuse |

**Root cause confirmed**: XLA's BFC allocator reuses `output_grad`'s GPU memory for `permuted_probs`
(second output of the custom call). Because `output_grad` is dead after the call and `permuted_probs`
has the same shape/dtype, XLA assigns them the same physical GPU address. During the 100+ autotune
iterations, the kernel reads from `output_grad` (buffers[0]) while writing to `permuted_probs`
(buffers[5] = same address) — corrupting the shared buffer across thread blocks.

**PASS runs**: `PHYSICAL_ALIAS_DETECTED` in the filename is a **false positive** — the outer script's
grep matches the C++ source code line (`fprintf(stderr, "[TE_BUG_DBG] YES_PHYSICAL_ALIAS ...")`) printed
during patch verification, not a runtime address comparison. Runtime TE_BUG_DBG output for passing tests
is suppressed by pytest's `--capture=fd`. In PASS runs, `buffers[0] != buffers[5]` — the BFC allocator
happened to assign different memory, so no corruption occurs despite 100+ autotune iterations.

However, even if Autotune() shows no aliasing, the corruption may occur in **Execute()** — the cached
kernel path used for all subsequent calls. Execute() is called with fresh BFC-allocated buffers each time,
and those allocations may change as the test runs.

### MemStats from BUG_REPRODUCED run (546885)

```
[MemStats:test_start]    bytes_in_use=0.000 GB  peak=14.110 GB  limit=143.636 GB
[MemStats:before_backward] bytes_in_use=0.063 GB  peak=14.110 GB  limit=143.636 GB
[MemStats:test_end]      bytes_in_use=0.105 GB  peak=14.110 GB  limit=143.636 GB
```

BFC pool was carved to 14.1 GB by prior tests (44 min of fused_attn + custom_call_compute +
not_distributed tests). At test start, nearly all is freed (0.000 GB in use) — but the BFC pool
retains its fragmented chunk history, creating specific offset patterns that lead to aliasing.

### XLA dump status

- **sort_chunks XLA dumps** (BUG=514131, PASS=514129): Available, have full compilation artifacts.
  Regex was `sort_chunks` — captures `jit_te_sort_chunks_by_map_triton` only.
- **jit_loss_fn XLA dumps**: NOT captured in 514131 (regex excluded it). NOT captured in 546885/886
  (dump went to container `/tmp`, lost on exit).
- **Next**: Jobs 547973–547976 use regex `sort_chunks|loss_fn` and write dump to `/code/` (persists).
  These will be the first runs with full jit_loss_fn XLA artifacts.

### Local package

`~/Downloads/sort_chunks_xla_bug_report_v2.zip` contains:
- `logs/addr_log_546885_BUG_REPRODUCED__no_alias_detected.log` — BUG run with MemStats + TE_BUG_DBG
- `logs/addr_log_546886_BUG_REPRODUCED__no_alias_detected.log` — second BUG run
- `logs/addr_log_546887_PASS__no_alias_detected.log` — PASS run for comparison
- `xla_dump_514131_bug_reproduced/sort_chunks_inner/` — full sort_chunks XLA artifacts (BUG run)
- `xla_dump_514131_bug_reproduced/loss_fn/` — jit_loss_fn stablehlo + config only (no buffer-assignment)
- `xla_dump_514129_pass/sort_chunks_inner/` — full sort_chunks XLA artifacts (PASS run)
- `xla_dump_514129_pass/loss_fn/` — jit_loss_fn stablehlo + config only (no buffer-assignment)
