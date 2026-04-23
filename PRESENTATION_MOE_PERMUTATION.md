# MoE Permutation in TransformerEngine (JAX)
## Permutation as a First-Class Op, Triton from JAX, and Distributed MoE

---

## Agenda

1. **MoE in 3 minutes** — why we need routing, permutation, and grouped GEMM
2. **The Router** — score functions, top-K, aux loss
3. **Permutation: the math** — row-ID map, permute, unpermute
4. **Permutation as a JAX op** — `token_dispatch`, `token_combine`, `custom_vjp`
5. **Triton kernels from JAX** — BasePrimitive, JAX FFI, lowering
6. **Grouped GEMM** — how expert computation works after permutation
7. **Distributed MoE** — FSDP, ring-of-experts, ragged all-to-all
8. **Performance numbers** *(separate slides)*

---

# Part 1: MoE in 3 Minutes

---

## What is Mixture of Experts?

Instead of one big dense FFN per layer, MoE has **E specialized experts** and a router that picks the best **K** per token.

```
Dense Transformer FFN:
  token → [FFN] → output         (all tokens, same weights)

MoE FFN:
  token → [Router] → top-K experts → weighted merge → output
                          ↑
                  only K/E of parameters active per token
```

**Why?**
- More parameters → better quality at same compute cost (sparse activation)
- Key insight: different tokens benefit from different specializations

---

## The Full MoE Forward Pass

```
Input tokens [N, H]
      │
      ▼
┌─────────────────────────────┐
│  ROUTER                     │
│  logits = x @ W_gate [N,E]  │
│  softmax/sigmoid → top-K    │
│  routing_map  [N, E]        │  ← which token goes where
│  probs        [N, E]        │  ← how much weight
└─────────────────────────────┘
      │
      ▼
┌─────────────────────────────┐
│  PERMUTE  (token_dispatch)  │
│  scatter tokens by expert   │
│  [N, H] → [N·K, H]         │  ← tokens grouped by expert
│  expert 0: [T3]             │
│  expert 1: [T0, T2, T3]     │
│  expert 2: [T1]             │
│  expert 3: [T0, T4]         │
└─────────────────────────────┘
      │
      ▼
┌─────────────────────────────┐
│  GROUPED GEMM  (x2)         │
│  for each expert i:         │
│    up   = tokens_i @ W1_i   │
│    act  = activation(up)    │
│    down = act @ W2_i        │
└─────────────────────────────┘
      │
      ▼
┌─────────────────────────────┐
│  UNPERMUTE  (token_combine) │
│  gather & weighted-merge    │
│  [N·K, H] → [N, H]         │
│  T0 = p(E1)*E1(T0)          │
│       + p(E3)*E3(T0)        │
└─────────────────────────────┘
      │
      ▼
Output tokens [N, H]
```

---

## Why Permutation is a Bottleneck

Without permutation you'd need to process each token separately (no batching).

With permutation:
- All tokens for expert $i$ are **contiguous** in memory
- Grouped GEMM can run one cuBLAS call for all experts
- Memory access is coalesced (full cache lines used)

The permutation is a **scatter** (forward) and a **gather** (backward). It touches every token × hidden-dim element exactly once — but in a data-dependent, non-contiguous order.

```
Before permute:           After permute:
token order = sequence    token order = by expert

[T0] ──────────────────► [T3]  ← Expert 0
[T1]        ┌──────────► [T0]  ← Expert 1
[T2]  ───┐  │  ┌───────► [T2]
[T3] ─┐  │  │  │    ┌──► [T3]
[T4]  │  └──┼──┼──► │    [T1]  ← Expert 2
      │     │  │    │    [T0]  ← Expert 3
      └─────┘  └────┘    [T4]
```

---

# Part 2: The Router

---

## Score Functions

Two modes, selected at runtime:

### Softmax (score_function = 1)

$$\text{softmax}(x_i) = \frac{e^{x_i - \max(x)}}{\sum_j e^{x_j - \max(x)}}$$

Applied either **pre-topK** (over all E experts) or **post-topK** (over the K selected).

### Sigmoid (score_function = 0)

$$\sigma(x_i) = \frac{1}{1+e^{-x_i}}, \quad \text{then normalize: } p_i = \frac{\sigma(x_i)}{\sum_{j \in \text{topK}} \sigma(x_j) + \epsilon}$$

Sigmoid allows independent expert scores (no competition). Normalization ensures weights sum to 1.

---

## Top-K Selection: How It Works in CUDA

One warp (32 threads) handles one token. Repeat K times:

```
for k = 0 .. K-1:
  Each thread: scan experts [lane, lane+32, lane+64, ...]
               find local max
  Warp shuffle: 5 rounds of __shfl_xor_sync → global max
  Lane 0 writes: topk_indices[k], topk_scores[k]
  Mask winner out
```

```
Thread 0:  experts [0, 32, 64, ...]
Thread 1:  experts [1, 33, 65, ...]
...
Thread 31: experts [31, 63, 95, ...]
```

Warp shuffle reduction: O(log₂ 32) = 5 rounds. No shared memory needed for the reduction itself.

---

## Fused Router Pipeline (Forward)

```
logits [N, E]
    │
    ▼  ─────────────────────────────────────────────────────────────────────────
    │  fused_topk_with_score_function_forward_kernel
    │
    │  Per warp (1 token):
    │    1. Load logits → shmem
    │    2. score function (softmax/sigmoid) → intermediate_output [N,E]
    │    3. [optional] add expert_bias
    │    4. top-K (or grouped top-K)
    │    5. [optional] revert bias, post-softmax/normalize
    │    6. write probs[topk_indices] = scaling * score
    │       write routing_map[topk_indices] = True
    │  ─────────────────────────────────────────────────────────────────────────
    │
    ├──► probs        [N, E]   sparse (only K nonzero per row)
    ├──► routing_map  [N, E]   bool
    └──► intermediate_output   saved for backward
```

---

## Auxiliary Load-Balancing Loss

Prevents expert collapse (all tokens going to one expert):

$$L_{\text{aux}} = C \cdot \sum_{i=1}^{E} \underbrace{\left(\sum_{t=1}^{N} p_{t,i}\right)}_{\text{avg prob for expert } i} \cdot \underbrace{f_i}_{\text{tokens routed to expert } i}$$

where $C = \frac{E \cdot \text{coeff}}{K \cdot T^2}$.

**Two separate kernels** for cleanliness:
- `fused_score_for_moe_aux_loss` — computes **dense** scores (all E, no bias/groups)
- `fused_moe_aux_loss` — reduction kernel; uses cooperative groups on SM 90+

---

# Part 3: Permutation — The Math

---

## The Permutation Problem

Tokens arrive in sequence order. We need them **grouped by expert**.

```
Initial:                   Target:
token 0 → experts [1, 3]   Expert 0: [T3]
token 1 → experts [2]      Expert 1: [T0, T2, T3]
token 2 → experts [1]      Expert 2: [T1]
token 3 → experts [0, 1]   Expert 3: [T0, T4]
token 4 → experts [3]
```

Mathematical formulation: find permutation $\pi$ s.t. all copies of tokens routed to expert $e$ are contiguous at a known offset.

Key insight: the router output is a **binary routing map** $M \in \{0,1\}^{N \times E}$.

---

## The Routing Map

```
Routing map M [N=5 tokens, E=4 experts]:

         E₀   E₁   E₂   E₃
  T₀  [[ 0,   1,   0,   1],    T₀ → Experts 1, 3
  T₁   [ 0,   0,   1,   0],    T₁ → Expert 2
  T₂   [ 0,   1,   0,   0],    T₂ → Expert 1
  T₃   [ 1,   1,   0,   0],    T₃ → Experts 0, 1
  T₄   [ 0,   0,   0,   1]]    T₄ → Expert 3

Each row sums to topK (or less if a token is dropped).

Total output tokens = sum of all 1s = N_out = 7 (topK=2, 3 tokens sent to 1 expert)
```

---

## The Row-ID Map: 3-Pass Construction

The key data structure: $R \in \mathbb{Z}^{N \times (2E+1)}$

```
For each token i:
  R[i, 0 .. n-1]       = destination row indices (where each expert copy goes)
  R[i, E .. E+n-1]     = corresponding expert IDs
  R[i, 2E]             = n (number of active experts for this token)
```

This encodes the full scatter plan in a single matrix, enabling parallel kernel access.

---

## Pass 1: Block-Level Cumulative Sum

```
Goal: for each expert column, compute cumsum(mask) * mask within each BLOCK_SIZE block.

Expert 1 column of M:  [1, 0, 1, 1, 0]   (block size = 3)

Block 0 [T0,T1,T2]:   mask = [1, 0, 1]
  cumsum:              [1, 1, 2]
  cumsum * mask:       [1, 0, 2]   ← local row IDs within block
  workspace[0] = 2    (block total)

Block 1 [T3,T4]:      mask = [1, 0]
  cumsum:              [1, 1]
  cumsum * mask:       [1, 0]
  workspace[1] = 1

Triton kernel grid: (num_experts, ceil(num_tokens/BLOCK_SIZE))
```

---

## Pass 2: Global Offset Correction

```
Goal: convert block-local positions to global positions.

workspace = [2, 1]              (block totals from Pass 1)
prefix_sum = [0, 2]             (exclusive scan → global offsets)

Block 0: [1, 0, 2] + 0 - 1 → [0, -1, 1]   (0-indexed)
Block 1: [1, 0]    + 2 - 1 → [2, -1]

After Pass 2, R[:,1] (expert 1 column):
  T0:  0
  T1: -1   (not routed)
  T2:  1
  T3:  2
  T4: -1   (not routed)
```

Full R after all expert columns processed (Pass 2 output):

```
       E₀   E₁   E₂   E₃
  T₀  [-1,   0,  -1,   0]
  T₁  [-1,  -1,   0,  -1]
  T₂  [-1,   1,  -1,  -1]
  T₃  [ 0,   2,  -1,  -1]
  T₄  [-1,  -1,  -1,   1]
```

---

## Pass 3: Densification (Bitonic Sort Per Token)

```
Goal: for each token, pack valid entries into the front of R.

Before (sparse, 4 expert cols each):   After (dense 2E+1 format):
  T₀: [-1, 0, -1, 0]   experts [_,1,_,3]   →  [0, 0 | 1, 3 | 2]
                                                 ↑dsts↑  ↑exps↑  ↑n↑
  T₃: [0, 2, -1, -1]   experts [0,1,_,_]   →  [0, 2 | 0, 1 | 2]
  T₁: [-1,-1, 0, -1]   experts [_,_,2,_]   →  [0   | 2       | 1]
```

Triton kernel grid: `(num_tokens,)` — one program per token, in-place bitonic sort.

---

## Permutation: Scatter

With the row-ID map built, permutation is a scatter:

$$Y[R_{i,k}, :] = X[i, :] \quad \text{for each active expert } k$$

```
Input X [N=5, H=3]:             Permuted Y [N_out=7, H=3]:
T0: [1, 2, 3]                   Row 0: [2,3,4]  ← T3 for Expert 0
T1: [4, 5, 6]   R[T0]=[0,0]    Row 1: [1,2,3]  ← T0 for Expert 1
T2: [7, 8, 9]   R[T1]=[0]  →   Row 2: [7,8,9]  ← T2 for Expert 1
T3: [2, 3, 4]   R[T2]=[1]      Row 3: [2,3,4]  ← T3 for Expert 1
T4: [5, 6, 7]   R[T3]=[0,2]    Row 4: [4,5,6]  ← T1 for Expert 2
                R[T4]=[1]       Row 5: [1,2,3]  ← T0 for Expert 3
                                Row 6: [5,6,7]  ← T4 for Expert 3
```

Triton kernel grid: `(num_tokens, ceil(hidden_size/BLOCK_SIZE))` — one program per (token, chunk).

---

## Unpermutation: Gather + Weighted Merge

After experts process their tokens, we gather back:

$$X_{\text{out}}[i, :] = \sum_{k=0}^{n_i-1} W_{i, e_k} \cdot Y_{\text{expert}}[R_{i,k}, :]$$

```
Expert outputs (after FFN):        Restored tokens (weighted):
Row 0: E0 processed T3             T0 = p(E1)·Row1 + p(E3)·Row5
Row 1: E1 processed T0             T1 = p(E2)·Row4
Row 2: E1 processed T2             T2 = p(E1)·Row2
Row 3: E1 processed T3        →    T3 = p(E0)·Row0 + p(E1)·Row3
Row 4: E2 processed T1             T4 = p(E3)·Row6
Row 5: E3 processed T0
Row 6: E3 processed T4
```

The same `row_id_map` is reused — no need to recompute.

---

## Alignment Padding (for cuBLAS efficiency)

cuBLAS GEMMs run fastest when rows are aligned (e.g., 16-element boundaries for BF16).

TE supports fused padding in the permute kernel:

```
Without padding:               With align_size=16:
Expert 0: 3 tokens             Expert 0: 3 tokens + 13 pad  (→ 16 rows)
Expert 1: 7 tokens        →    Expert 1: 7 tokens + 9 pad   (→ 16 rows)
Expert 2: 11 tokens            Expert 2: 11 tokens + 5 pad  (→ 16 rows)
Expert 3: 5 tokens             Expert 3: 5 tokens + 11 pad  (→ 16 rows)

Total: 26 tokens               Total: 64 tokens (all aligned)
```

`pad_offsets[E]` stored per expert. The unpermute kernel uses it to skip padding during gather.

---

# Part 4: Permutation as a JAX Op

---

## The JAX API

Three high-level functions in `transformer_engine/jax/permutation.py`:

```python
# Forward dispatch: scatter tokens to experts
output, permuted_probs, row_id_map, pad_offsets, tokens_per_expert = \
    token_dispatch(
        inp,           # [num_tokens, hidden]
        routing_map,   # [num_tokens, num_experts]  bool
        num_out_tokens,
        probs=None,    # [num_tokens, num_experts]  optional
        align_size=0,  # padding alignment (0 = disabled)
    )

# Backward gather: combine expert outputs
output = token_combine(
    inp,          # [num_out_tokens, hidden]  expert outputs
    row_id_map,   # saved from token_dispatch
    merging_probs=None,  # [num_tokens, num_experts] optional
    pad_offsets=None,
)

# For EP: reorder chunks within a buffer
output, row_id_map = sort_chunks_by_index(
    inp,            # [total_tokens, hidden]
    split_sizes,    # [num_chunks]  token count per chunk
    sorted_indices, # [num_chunks]  target order
)
```

---

## Custom VJP: Making Permutation Differentiable

JAX's AD doesn't know how to differentiate scatter/gather. TE uses `jax.custom_vjp`:

```
FORWARD:
  token_dispatch → { output, residuals }
  residuals = (row_id_map, pad_offsets, probs, routing_map)

BACKWARD (fwd saved residuals → use them directly):
  grad_output [num_out_tokens, hidden]
       ↓  unpermute (gather with row_id_map)
  grad_input  [num_tokens, hidden]

  Also: if probs were permuted, compute grad_probs too.
```

The backward of `token_dispatch` is exactly `token_combine` (a gather). The backward of `token_combine` is `token_dispatch` (a scatter).

```python
@partial(jax.custom_vjp, nondiff_argnums=(3, 4, 5, 6))
def _token_dispatch(inp, routing_map, num_out_tokens, ...):
    ...

def _token_dispatch_fwd_rule(inp, routing_map, num_out_tokens, ...):
    out = _token_dispatch(inp, routing_map, num_out_tokens, ...)
    residuals = (row_id_map, pad_offsets, ...)
    return out, residuals

def _token_dispatch_bwd_rule(residuals, g):
    row_id_map, pad_offsets, ... = residuals
    grad_inp = unpermute_with_mask_map(g, row_id_map, ...)
    return grad_inp, ...
```

---

## What Flows Through the Op

```
token_dispatch:
  Inputs:                        Outputs:
  ┌──────────────────┐           ┌─────────────────────────────────┐
  │ inp [N, H]       │   →   →   │ output   [N_out, H]  permuted   │
  │ routing_map [N,E]│           │ row_id_map [N, 2E+1]  saved     │
  │ probs [N, E]     │           │ pad_offsets [E]       saved     │
  └──────────────────┘           │ tokens_per_expert [E]           │
                                 │ permuted_probs [N_out, 1]       │
                                 └─────────────────────────────────┘

  Saved residuals for VJP:
  (row_id_map, pad_offsets, probs, routing_map, align_size, num_out_tokens)

token_combine (backward of dispatch):
  Inputs:                        Outputs:
  ┌──────────────────────────┐   ┌─────────────────────────────────┐
  │ inp [N_out, H]           │   │ output [N, H]   token order     │
  │ row_id_map [N, 2E+1]     │ → │ grad_merging_probs [N, E]       │
  │ merging_probs [N, E]     │   └─────────────────────────────────┘
  └──────────────────────────┘
```

---

## The Row-ID Map in JAX: Shape and Semantics

```
row_id_map: int32 [num_tokens, 2*num_experts + 1]

For token i routed to experts [e₁, e₂] at destination rows [r₁, r₂]:

  row_id_map[i, 0]            = r₁   (dest row for expert e₁)
  row_id_map[i, 1]            = r₂   (dest row for expert e₂)
  row_id_map[i, 2 .. E-1]     = -1   (unused slots)
  row_id_map[i, E]            = e₁   (expert ID)
  row_id_map[i, E+1]          = e₂   (expert ID)
  row_id_map[i, E+2 .. 2E-1]  = -1
  row_id_map[i, 2E]           = 2    (n_routed = 2 for this token)

This compact format lets the permute/unpermute Triton kernels do:
  for k in range(row_id_map[i, 2E]):          # iterate active experts
      dst = row_id_map[i, k]                   # destination row
      exp = row_id_map[i, num_experts + k]     # expert ID
      Y[dst, :] = X[i, :]
```

---

# Part 5: Triton Kernels from JAX

---

## The Challenge

JAX's default path: XLA → cuBLAS / cuDNN / cuSolver.

But permutation's scatter/gather pattern doesn't map to any cuBLAS primitive. We need custom CUDA code.

Options:
1. **CUDA C++ XLA custom call** (traditional TE approach for dense ops)
2. **Triton kernel via JAX FFI** ← TE's choice for permutation

Why Triton?
- Permutation is memory-bandwidth bound, not compute-bound
- Triton makes it easy to write and tune tiled memory-access patterns
- Autotuning over BLOCK_SIZE without separate CUDA compilation per config
- Readable Python source, still native PTX on GPU

---

## TE's Triton-in-JAX Architecture

```
transformer_engine/jax/
├── permutation.py                    ← public API (custom_vjp wrappers)
└── triton_extensions/
    ├── permutation.py                ← BasePrimitive subclasses (8 ops)
    └── utils.py                      ← compile_triton(), triton_call_lowering()

transformer_engine/common/triton/
└── permutation.py                    ← actual Triton @triton.jit kernels

transformer_engine/jax/cpp_extensions/
└── base.py                           ← register_primitive() machinery
```

**Data flow:**
```
JAX user code
    │ calls
    ▼
permutation.token_dispatch()           Python
    │ calls
    ▼
triton_extensions.permute_with_mask_map()   Python
    │ calls primitive's abstract_eval + lowering
    ▼
BasePrimitive.lowering()               Python → MLIR
    │ calls
    ▼
triton_call_lowering()                 Python → MLIR stablehlo custom_call
    │ dispatches to
    ▼
Triton PTX kernel                      GPU
```

---

## BasePrimitive: The Bridge

Every Triton op is wrapped as a `BasePrimitive` subclass. Example: `PermuteWithMaskMapPrimitive`.

```python
class PermuteWithMaskMapPrimitive(BasePrimitive):
    name = "te_permute_with_mask_map"
    multiple_results = True
    impl_static_args = (2, 3, 4, 5)   # non-array args

    @staticmethod
    def abstract(inp, row_id_map, num_out_tokens, hidden_size, ...):
        # shape inference — runs at trace time, not GPU time
        return (
            ShapedArray([num_out_tokens, hidden_size], inp.dtype),  # output
            ShapedArray([num_out_tokens], jnp.float32),             # permuted_probs
        )

    @staticmethod
    def lowering(ctx, inp, row_id_map, ...):
        # called by XLA during compilation; emits MLIR custom_call
        return triton_call_lowering(
            ctx,
            kernel=_permute_kernel,           # Triton @jit function
            grid=lambda meta: (num_tokens, cdiv(hidden_size, meta['BLOCK_SIZE'])),
            in_specs=[inp_aval, row_id_map_aval, ...],
            out_specs=[out_aval, probs_aval],
            input_output_aliases={0: 0},      # output[0] aliases input[0]
            ...
        )

    @staticmethod
    def partition(mesh, arg_shapes, result_shape):
        # how to shard across devices
        ...

register_primitive(PermuteWithMaskMapPrimitive)
```

---

## Registration: Inner vs. Outer Primitive

`register_primitive()` creates **two** JAX primitives from one class:

```
BasePrimitive subclass
    │
    ├── INNER primitive  (single-device)
    │     name:     "te_permute_with_mask_map"
    │     impl:     xla.apply_primitive (eager)
    │     abstract: cls.abstract()
    │     lowering: cls.lowering()  →  MLIR custom_call
    │     batching: vmap support
    │
    └── OUTER primitive  (multi-device)
          name:     "te_permute_with_mask_map_wrapper"
          impl:     cls.outer_impl()
          sharding: custom_partitioning(cls.partition, ...)
                    → tells XLA how to split arrays across devices
          shardy:   cls.shardy_sharding_rule()
                    → mesh axis annotations
```

The outer primitive wraps the inner with sharding annotations. XLA sees only the outer when compiling distributed programs.

---

## compile_triton(): Kernel Compilation Pipeline

```python
def compile_triton(kernel, grid, scalar_args, avals, ...):
    # 1. Hash: (kernel source + args + grid) → MD5
    cache_key = md5(kernel.fn, scalar_args, avals, ...)
    if cache_key in _kernel_cache:
        return _kernel_cache[cache_key]  # reuse compiled kernel

    # 2. Specialization: instantiate kernel with concrete shapes
    specialized = kernel.specialize(
        num_warps=4,
        BLOCK_SIZE=config.BLOCK_SIZE,   # from autotuner or env var
        ...
    )

    # 3. Compile to PTX via Triton compiler
    ptx = triton.compile(specialized, target="ptx", ...)

    # 4. Wrap in JAX's TritonKernel object
    triton_kernel = gpu_triton.TritonKernel(
        ptx, kernel_name, num_warps, shared_mem_bytes, ...
    )

    _kernel_cache[cache_key] = triton_kernel
    return triton_kernel
```

Autotuning: if `NVTE_DISABLE_TRITON_AUTOTUNING=0`, TE tests all `@triton.autotune` configs and picks the fastest. Can be disabled with `NVTE_TRITON_BLOCK_SIZE=<N>`.

---

## triton_call_lowering(): MLIR Emission

```python
def triton_call_lowering(ctx, kernel, grid, in_specs, out_specs, ...):
    # 1. Compute launch grid from shapes
    grid_vals = grid(meta)    # e.g., (num_tokens, ceil(H/BLOCK_SIZE))

    # 2. Build TritonKernelCall proto
    call = gpu_triton.TritonKernelCall(
        kernel=compiled_kernel,
        grid=grid_vals,
        operands=[...],
        output_operands=[...],
    )

    # 3. Serialize + compress (zlib) → base64 string in MLIR attribute
    serialized = zlib.compress(call.SerializeToString())

    # 4. Emit stablehlo.custom_call
    return stablehlo.custom_call(
        call_target_name="__gpu$xla.gpu.triton",
        inputs=mlir_inputs,
        result_types=mlir_output_types,
        backend_config=serialized,           # kernel + grid embedded here
        input_output_aliases=aliases,
    )
```

XLA's GPU backend sees a `custom_call` node and dispatches it via the registered Triton runtime.

---

## The Permutation Triton Kernels

Five kernels in `transformer_engine/common/triton/permutation.py`:

```
Kernel                            Grid                          Purpose
─────────────────────────────────────────────────────────────────────────────────
_row_id_map_pass_1_kernel         (E, ⌈N/BLOCK⌉)               block cumsum × mask
_row_id_map_pass_2_kernel         (E, ⌈N/BLOCK⌉)               add global offsets
_row_id_map_pass_3_kernel         (N,)                          bitonic sort, densify
_permute_kernel (autotuned)       (N, ⌈H/BLOCK⌉)               scatter: X[src] → Y[dst]
_unpermute_kernel (autotuned)     (N, ⌈H/BLOCK⌉)               gather: Y[dst] → X[src]
_unpermute_bwd_with_merging_probs (N,)                          backward of weighted merge
_make_chunk_sort_map_kernel       (⌈chunks/BLOCK⌉,)             build chunk sort index
_sort_chunks_by_map_kernel        (N, ⌈H/BLOCK⌉)               chunk reorder (for EP)
```

**Autotuning** for `_permute_kernel` and `_unpermute_kernel`:
- `BLOCK_SIZE ∈ {64, 128, 256, 512, 1024, 2048, 4096}`
- Best choice depends on hidden dimension and GPU memory bandwidth

---

## Version Guard: Triton Extensions on Older JAX

```python
# transformer_engine/jax/triton_extensions/utils.py

TRITON_EXTENSION_MIN_JAX_VERSION = "0.8.0"

def is_triton_extension_supported() -> bool:
    return jax_version_meet_requirement(TRITON_EXTENSION_MIN_JAX_VERSION)

# guard placed BEFORE gpu_triton import:
if not is_triton_extension_supported():
    raise RuntimeError(
        f"JAX >= {TRITON_EXTENSION_MIN_JAX_VERSION} required for Triton extensions. "
        f"Current: {jax.__version__}"
    )

import jax._src.interpreters.mlir as mlir
from jax._src.lib import gpu_triton   # only imported if guard passes
```

Tests skip `@pytest.mark.triton` tests automatically on old JAX via `conftest.py`.

---

# Part 6: Grouped GEMM

---

## Why Grouped GEMM?

After permutation, each expert has a different number of tokens. Launching E separate GEMM calls is wasteful.

Grouped GEMM: one cuBLASLt call computes all E GEMMs simultaneously.

```
Standard GEMM:     D [M,N] = A [M,K] @ B [K,N]

Grouped GEMM:      for i = 0..E-1:
                     D_i [M_i, N] = A_i [M_i, K] @ B_i [K, N]
                   All packed in contiguous buffers.
```

---

## GroupedTensor: One Buffer, Multiple Logical Matrices

```
GroupedTensor {
    data:           void*         ← single contiguous GPU buffer
    num_tensors:    size_t        ← E (number of experts)
    logical_shape:  [rows, cols]  ← [sum(M_i), K]
    first_dims:     int64[E]      ← M_i per expert   (on GPU)
    last_dims:      int64[E]      ← K_i per expert   (empty if uniform)
    tensor_offsets: int64[E]      ← element offset for expert i
}
```

**In MoE (typical case: M varies, K uniform):**

```
After permutation:

  Expert 0: M_0=3 tokens    first_dims = [3, 5, 2]
  Expert 1: M_1=5 tokens →  last_dims  = (empty, K=1024 uniform)
  Expert 2: M_2=2 tokens    offsets:   [0, 3*1024, 8*1024]

  Contiguous buffer:
  ┌─────────────────────┬──────────────────────────┬──────────────┐
  │ Expert 0 [3×1024]   │ Expert 1 [5×1024]        │ Expert 2 ... │
  └─────────────────────┴──────────────────────────┴──────────────┘
```

---

## Two Grouped GEMMs: The Expert MLP

```
GROUPED GEMM #1 (Gate/Up projection)
  A: permuted tokens  [sum(M_i), K]   first_dims=[M_0..M_E]
  B: W1 per expert    [E × K × N']    (uniform, each expert K×N')
  D: intermediate     [sum(M_i), N']

ACTIVATION (element-wise SiLU / GeLU)

GROUPED GEMM #2 (Down projection)
  A: intermediate     [sum(M_i), N']  first_dims=[M_0..M_E]
  B: W2 per expert    [E × N' × K]
  D: expert outputs   [sum(M_i), K]

UNPERMUTE
  [sum(M_i), K] → [N, K]   (token_combine)
```

The same `first_dims` array (token distribution) is reused for both GEMMs — the expert assignment doesn't change between layers.

---

# Part 7: Distributed MoE — FSDP and Expert Parallelism

---

## Two Parallelism Dimensions in MoE

```
                    Tensor/FSDP axis
                    (weight sharding)
                         │
                    ┌────┴────┐
                    │         │
         ───────────┼─────────┼─────────── Expert axis
    EP shard 0      │  shard  │  EP shard 1
    experts 0..E/2  │ overlap │  experts E/2..E
                    └─────────┘

FSDP: weights sharded across devices → all-gather before GEMM
EP:   experts assigned to device subsets → tokens must travel to expert's device
```

**Key tension:** FSDP needs full weights on one device, EP needs tokens on the right device.

---

## MaxText Integration: How End Users Call TE Permutation

From `maxtext/src/MaxText/layers/te_permutation.py`:

```python
from transformer_engine.jax.permutation import (
    token_dispatch,
    token_combine,
    sort_chunks_by_index,
)

def te_permute(x, logits, ...):
    # Step 1: route
    sparse_probs, routing_map, lb_loss, bias_updates = te_route(x, logits, ...)

    # Step 2: dispatch tokens to experts
    output, permuted_probs, row_id_map, pad_offsets, tokens_per_expert = \
        token_dispatch(
            x,
            routing_map,
            num_out_tokens=num_tokens * topk,
            probs=sparse_probs,
            align_size=config.te_permutation_align_size,
        )
    return output, perm_state

def te_unpermute(x, perm_state, ...):
    return token_combine(
        x,
        perm_state.row_id_map,
        merging_probs=perm_state.probs,
        pad_offsets=perm_state.pad_offsets,
    )
```

---

## Ring-of-Experts: All-Gather Then Compute

All tokens are sent to every expert shard. Each shard processes its local experts.

```
4 tokens, 4 experts, 2 EP shards:

  Shard 0 (experts 0,1)     Shard 1 (experts 2,3)
  ┌──────────────────┐       ┌──────────────────┐
  │ T0, T1, T2, T3   │       │ T0, T1, T2, T3   │ ← all-gather (tokens replicated)
  └────────┬─────────┘       └─────────┬────────┘
           │  local route              │  local route
           ▼                           ▼
  Permute to Expert 0,1       Permute to Expert 2,3
           │                           │
    E0[T3] E1[T0,T2,T3]       E2[T1]  E3[T0,T4]
           │                           │
     Grouped GEMM                Grouped GEMM
           │                           │
           ▼                           ▼
   E0_out, E1_out              E2_out, E3_out
           │                           │
           └─────────┬─────────────────┘
                  psum_scatter   ← each shard gets its token slice
                     │
              Final output
```

**Code (MaxText `moe.py`):**
```python
# replicate all tokens to all EP shards
x = jax.lax.all_gather(x, axis_name=ep_axis, tiled=True)

# route + permute (each shard sees all tokens, routes to local experts)
x, perm_state = self.permute(x, ..., roll_to_expert_id=shard_id*n_local_experts)

# experts compute (only local experts processed)
x = expert_mlp(x, local_weights)

# reduce: each token's contribution summed across shards
x = jax.lax.psum_scatter(x, ep_axis, scatter_dimension=0, tiled=True)
```

**Tradeoff:** Simple to implement. Communication cost = all-gather (O(N×H×(E_shards-1))).

---

## Ragged All-to-All: Send Only What's Needed

Each shard sends only the tokens actually routed to the remote shard's experts.

```
4 tokens, 4 experts, 2 EP shards:

  Shard 0 (experts 0,1)          Shard 1 (experts 2,3)
  ┌────────────────────┐          ┌────────────────────┐
  │ T0, T1, T2, T3     │          │ T0, T1, T2, T3     │
  └──────────┬─────────┘          └──────────┬─────────┘
             │ route                         │ route
             ▼                               ▼
  local: T3→E0, T0,T2,T3→E1        local: T1→E2, T0,T4→E3
  remote: T1→E2 (shard 1)          remote: T3→E1 (shard 0)
             │                               │
             │◄──── ragged_all_to_all ───────►│
             │                               │
  Shard 0 receives: T1 from shard 1          │
  Shard 1 receives: T3 from shard 0          │
             │                               │
    sort_chunks_by_index                sort_chunks_by_index
   (reorder: shard×expert → expert×shard)
             │                               │
  E0[T3]  E1[T0,T2,T3,T1]        E2[T1,T3]  E3[T0,T4]
             │                               │
       Grouped GEMM                   Grouped GEMM
             │                               │
   sort_chunks (inverse) + ragged_all_to_all (reverse)
             │                               │
         psum/combine
             │
         Final output
```

---

## Ragged All-to-All: The Role of sort_chunks_by_index

After `ragged_all_to_all`, tokens arrive grouped by **(source shard, expert)**. We need them grouped by **(expert, source shard)** for batch processing.

```
After ragged_all_to_all on Shard 0:
  [shard0_E0_tokens | shard0_E1_tokens | shard1_E0_tokens | shard1_E1_tokens]
     ↑──── (shard, expert) order ─────────────────────────────────────────────

Need for Grouped GEMM:
  [shard0_E0_tokens | shard1_E0_tokens | shard0_E1_tokens | shard1_E1_tokens]
     ↑──── (expert, shard) order ─────────────────────────────────────────────
```

`sort_chunks_by_index` does this reordering **without materializing indices** — it uses the same Triton scatter kernel with a chunk-level permutation map.

```python
# Conceptually: transpose the (num_shards, num_local_experts) matrix
indices_matrix = jnp.arange(num_shards * n_local_experts)\
                    .reshape(num_shards, n_local_experts)
sorted_chunk_indices = indices_matrix.T.reshape(-1)   # transpose = (expert, shard)

# TE kernel handles the actual data movement
output, sort_map = sort_chunks_by_index(x, split_sizes, sorted_chunk_indices)
```

---

## FSDP Weight Sharding for MoE

Expert weights are typically **large** and need to be sharded across devices.

```
Weight sharding strategies in MaxText:

Standard FSDP:
  w0 [E, K, N'] sharded on (exp_with_fsdp, embed, mlp)
  → all-gather on fsdp axis before each GEMM
  → reduce-scatter after

2D FSDP (shard_exp_on_fsdp=True):
  w0 [E, K, N'] sharded on (exp+fsdp, embed_T, mlp_no_fsdp)
  → weights sharded on both expert and fsdp dimensions
  → better for very large expert counts

Two-Stage All-Gather (moe_fsdp_use_two_stage_all_gather=True):
  Stage 1: all-gather on fsdp axis only
  Stage 2: optimization_barrier   ← prevent kernel fusion across stages
  Stage 3: all-gather on fsdp_transpose axis
  → more fine-grained pipeline overlap with compute
```

**Practical rule**: FSDP sharding happens at the weight level; token/activation sharding (EP) happens at the permutation/routing level. They compose independently.

---

## End-to-End: Putting It All Together

```
Tokens [batch, seq_len, H]
     │  shard_map (data-parallel over EP shards)
     ▼
Gate logits = tokens @ W_gate [N, E]
     │
     ▼ te_route() → routing_map, sparse_probs
     │
     ├── [Ring-of-Experts?] all_gather tokens across EP shards
     │
     ▼ token_dispatch() ← Triton permute kernels
     │   build row_id_map (3 passes)
     │   scatter X → Y[N_out, H]
     │
     ├── [Ragged A2A?] ragged_all_to_all + sort_chunks_by_index
     │
     ▼ Grouped GEMM #1 (cuBLASLt)
     │   [after FSDP all-gather of W1]
     │   Y_up [N_out, N']
     ▼ activation (SiLU/GeLU)
     ▼ Grouped GEMM #2 (cuBLASLt)
     │   [after FSDP all-gather of W2]
     │   Y_down [N_out, H]
     │
     ├── [Ragged A2A?] sort_chunks (inverse) + ragged_all_to_all reverse
     │
     ▼ token_combine() ← Triton unpermute kernels
     │   gather + weighted merge
     │   Z[N, H]
     │
     ├── [Ring-of-Experts?] psum_scatter
     │
     ▼ output [batch, seq_len, H]
     + lb_loss (aux load-balancing loss)
```

---

## Key Takeaways

| Concept | Key point |
|---------|-----------|
| Permutation | Scatter (forward) + Gather (backward); enables efficient batched expert compute |
| Row-ID map | 3-pass Triton construction; compact `[N, 2E+1]` format stores full scatter plan |
| JAX op design | `custom_vjp` for differentiability; inner/outer primitive for single/multi-device |
| Triton in JAX | BasePrimitive → `triton_call_lowering` → MLIR `custom_call` → PTX |
| Alignment padding | Fused into permute kernel; improves cuBLAS efficiency; unpadded at gather |
| Ring-of-experts | All-gather tokens, compute locally, psum_scatter; simple but O(tokens) communication |
| Ragged A2A | Send only routed tokens; `sort_chunks_by_index` reorders (shard×expert)→(expert×shard) |
| FSDP | Independent of EP; operates on weight tensors; two-stage all-gather for pipeline overlap |

---

## File Map

```
transformer_engine/
├── common/triton/
│   └── permutation.py              Triton @jit kernels (permute, unpermute, row_id_map passes)
├── jax/
│   ├── permutation.py              Public API: token_dispatch, token_combine, sort_chunks_by_index
│   ├── router.py                   Public API: fused_topk_with_score_function, fused_moe_aux_loss
│   ├── triton_extensions/
│   │   ├── permutation.py          BasePrimitive subclasses (8 ops)
│   │   └── utils.py                compile_triton(), triton_call_lowering(), version guard
│   └── cpp_extensions/
│       └── base.py                 register_primitive() → inner + outer JAX primitives
└── common/fused_router/
    ├── fused_topk_with_score_function.cu
    ├── fused_score_for_moe_aux_loss.cu
    └── fused_moe_aux_loss.cu

maxtext/src/MaxText/layers/
├── moe.py                          Full MoE layer: ring-of-experts, ragged A2A, grouped GEMM
├── te_router.py                    TE router wrapper
└── te_permutation.py               TE permutation wrapper + ragged A2A param computation
```

---

*[Performance numbers — separate slides to be added]*
