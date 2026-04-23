# Fused Router: CUDA Kernels for MoE Gating

This document provides a detailed explanation of the fused router CUDA kernels in TransformerEngine, covering the score functions, top-k selection, grouped routing, auxiliary load-balancing loss, and the GPU parallelism strategies used in each kernel.

## Table of Contents
1. [Overview and Pipeline](#overview-and-pipeline)
2. [Score Functions](#score-functions)
3. [Top-K Selection: naive_topk_and_mask](#top-k-selection-naive_topk_and_mask)
4. [Fused Top-K with Score Function (Forward)](#fused-top-k-with-score-function-forward)
5. [Fused Top-K with Score Function (Backward)](#fused-top-k-with-score-function-backward)
6. [Grouped Top-K](#grouped-top-k)
7. [Fused Score for MoE Aux Loss](#fused-score-for-moe-aux-loss)
8. [Fused MoE Auxiliary Loss (Forward)](#fused-moe-auxiliary-loss-forward)
9. [Fused MoE Auxiliary Loss (Backward)](#fused-moe-auxiliary-loss-backward)
10. [Utility Functions](#utility-functions)
11. [GPU Parallelism and Memory Layout](#gpu-parallelism-and-memory-layout)
12. [Numerical Precision: Internal float64](#numerical-precision-internal-float64)
13. [File Map](#file-map)

---

## Overview and Pipeline

The fused router implements MoE gating in three composable kernels. In a typical MoE layer, the full pipeline is:

```
                       ┌──────────────────────────────────────────┐
                       │         MAIN ROUTING PATH                │
                       │                                          │
  logits ──────────────┼──► fused_topk_with_score_function ──────►│──► probs (sparse)
  [N, E]               │     score → [optional bias] → topk      │    [N, E]
                       │     → [optional post-softmax] → scale   │
                       │                                          │──► routing_map (bool)
                       │                                          │    [N, E]
                       └──────────────────────────────────────────┘
                                                                     │
                       ┌─────────────────────────────────────────┐   │
                       │       AUX LOSS PATH                     │   │
                       │                                         │   │
  logits ──────────────┼──► fused_score_for_moe_aux_loss ───────►│   │
  [N, E]               │     score → topk (clean, no bias/groups)│   │
                       │                                         │   │
                       │   outputs:                              │   │
                       │     scores       [N, E]  (dense probs)  │   │
                       │     routing_map  [N, E]  (bool)         │   │
                       └─────────────────────────────────────────┘   │
                                        │                            │
                                        ▼                            │
                       tokens_per_expert = routing_map.sum(dim=0)    │
                                        │                            │
                       ┌────────────────┼────────────────────────┐   │
                       │                ▼                        │   │
                       │  fused_moe_aux_loss                    │   │
                       │    (scores, tokens_per_expert)         │   │
                       │         → scalar loss                  │   │
                       └────────────────────────────────────────┘   │
                                                                     │
                                                                     ▼
                                                              Token dispatch
                                                              (permutation)
```

**Three kernels, two purposes:**

| Kernel | Purpose | Key difference |
|--------|---------|----------------|
| `fused_topk_with_score_function` | **Main routing** — produces sparse probs and routing_map for token dispatch | Supports expert_bias, grouped top-k, scaling_factor, pre/post softmax |
| `fused_score_for_moe_aux_loss` | **Aux loss input** — produces clean scores and routing_map for loss computation | No bias, no groups, no scaling. Always pre-softmax/sigmoid, writes *all* expert scores (dense) |
| `fused_moe_aux_loss` | **Aux loss computation** — takes scores + token counts and produces a scalar loss | Reduction kernel using cooperative groups on SM90+ |

---

## Score Functions

Two score functions are supported, selected by the `score_function` parameter:

### Softmax (score_function = 1)

$$\text{softmax}(x_i) = \frac{e^{x_i - \max(x)}}{\sum_j e^{x_j - \max(x)}}$$

Implemented in `apply_softmax_on_float`:
1. Compute max via warp reduction
2. Subtract max, exponentiate in-place
3. Compute sum via warp reduction (of the calculated 2.)
4. Divide 2. by sum 3. in-place

Can be applied **pre-topk** (over all E experts) or **post-topk** (over just the K selected experts), controlled by `use_pre_softmax`.

### Sigmoid (score_function = 0)

$$\sigma(x_i) = \frac{1}{1 + e^{-x_i}}$$

Always applied pre-topk over all experts. When `topk > 1`, a normalization step is applied to the selected scores after top-k:

$$\text{prob}_i = \frac{\sigma(x_i)}{\sum_{j \in \text{topk}} \sigma(x_j) + \epsilon}$$

where $\epsilon = 10^{-20}$.

### Backward Passes

**Softmax backward** (standard Jacobian):

$$\frac{\partial L}{\partial x_i} = s_i \left(\frac{\partial L}{\partial s_i} - \sum_j s_j \frac{\partial L}{\partial s_j}\right)$$

where $s_i = \text{softmax}(x_i)$.

**Sigmoid backward:**

$$\frac{\partial L}{\partial x_i} = \frac{\partial L}{\partial \sigma_i} \cdot \sigma_i \cdot (1 - \sigma_i)$$

---

## Top-K Selection: naive_topk_and_mask

The core top-k algorithm is `naive_topk_and_mask` in `utils.h`. It uses a simple iterative approach: repeat K times, finding the argmax and masking the winner.

### Algorithm

```
Input:  scores[E], topk=K
Output: topk_indices[K], topk_scores[K]

for k = 0 to K-1:
    1. Each thread loads scores[lane_id], scores[lane_id+32], scores[lane_id+64], ...
       (skipping already-selected indices via is_masked)
    2. Local reduction: each thread finds max across its strided elements
    3. Warp shuffle reduction: 5 rounds of __shfl_xor_sync to find global max
    4. Lane 0 writes topk_indices[k] = winning index, topk_scores[k] = winning value
    5. __syncwarp()
```

### Parallelism: 1 Warp per Token

Each warp (32 threads) processes one token independently. The 32 threads cooperatively scan all E experts using a stride of 32:

```
Thread 0:  experts [0, 32, 64, 96, ...]
Thread 1:  experts [1, 33, 65, 97, ...]
...
Thread 31: experts [31, 63, 95, 127, ...]
```

### Expert Count Limitation

The warp shuffle reduction itself is not the bottleneck — the strided loop `for (int i = lane_id + 32; i < data_size; i += 32)` handles arbitrarily many experts. However, all scores must reside in shared memory that one warp can access. The practical limit comes from shared memory capacity, not the 32-thread warp size.

### Warp Shuffle Reduction Detail

After each thread has its local max, the warp reduces across all 32 threads in 5 steps:

```
Round 1: __shfl_xor_sync(mask, val, 16)  →  threads 0-15 exchange with 16-31
Round 2: __shfl_xor_sync(mask, val, 8)   →  threads 0-7 exchange with 8-15, etc.
Round 3: __shfl_xor_sync(mask, val, 4)
Round 4: __shfl_xor_sync(mask, val, 2)
Round 5: __shfl_xor_sync(mask, val, 1)
```

After 5 rounds, all threads hold the global max. Thread 0 writes the result.

---

## Fused Top-K with Score Function (Forward)

**Kernel:** `fused_topk_with_score_function_forward_kernel`
**File:** `fused_topk_with_score_function.cu`

This is the main routing kernel. It fuses score computation, top-k selection, and post-processing into a single kernel launch.

### Per-Token Pipeline (one warp)

```
1. Load logits[token, :] → shared memory (scores)
2. Clear probs[token, :] = 0, routing_map[token, :] = false

3. PREPROCESS (in-place on scores):
   ├── If softmax + pre_softmax: apply_softmax_on_float → save to intermediate_output
   └── If sigmoid:              apply_sigmoid_on_float → save to intermediate_output
                                If expert_bias: scores[i] += expert_bias[i]

4. TOP-K:
   ├── If group_topk > 0: grouped_topk (see Grouped Top-K section)
   └── Else:              naive_topk_and_mask(scores, E, K, ...)

5. POSTPROCESS:
   ├── If expert_bias (sigmoid): revert bias from topk_scores
   ├── If softmax + !pre_softmax: apply_softmax_on_float(topk_scores, K)
   ├── If sigmoid + topk > 1:    normalize: topk_scores[i] / sum(topk_scores)
   └── Write: probs[topk_indices[i]] = scaling_factor * topk_scores[i]
              routing_map[topk_indices[i]] = true
```

### Shared Memory Layout

Per block (4 warps, 4 tokens):

```
┌─────────────────────────────────────────────────────────┐
│ scores          [4 × E]     DataType                    │
├─────────────────────────────────────────────────────────┤
│ topk_scores     [4 × K]     DataType                    │
├─────────────────────────────────────────────────────────┤
│ (if group_topk > 0):                                    │
│   masked_scores [4 × E]     DataType                    │
│   group_scores  [4 × G]     DataType                    │
├─────────────────────────────────────────────────────────┤
│ topk_indices    [4 × K]     int                         │
└─────────────────────────────────────────────────────────┘
```

Each warp addresses its own slice: `scores + warp_id * E`, etc.

### Launch Configuration

```
Block size:  128 threads (4 warps × 32 threads)
Grid size:   ceil(num_tokens / 4)
Shared mem:  E*4*sizeof(DataType) + K*4*sizeof(DataType) + K*4*sizeof(int)
             + (if group_topk: G*4*sizeof(DataType) + E*4*sizeof(DataType))
```

### Outputs

- `probs[N, E]`: Sparse — only topk positions per row are nonzero, containing scaled scores
- `routing_map[N, E]`: Boolean — true at the topk positions
- `intermediate_output[N, E]`: Saved softmax/sigmoid values for backward pass

---

## Fused Top-K with Score Function (Backward)

**Kernel:** `fused_topk_with_score_function_backward_kernel`

The backward pass reverses each forward step in reverse order, using `routing_map` to identify which positions were selected and `intermediate_output` for the saved activations.

### Per-Token Backward Pipeline (one warp)

```
1. Load grad_probs, intermediate_output, routing_map → shared memory

2. REVERSE POSTPROCESS:
   ├── Scale: grad[i] *= scaling_factor  (where routing_map[i] = true)
   ├── If sigmoid + topk > 1: backward of normalization
   │     S = sum(act[selected])
   │     grad_out_x_act = sum(grad[i] * act[i], selected)
   │     grad[i] = grad[i]/(S+ε) − grad_out_x_act/(S+ε)²
   └── If softmax + !pre_softmax: apply_softmax_bwd_on_float(grad, act, K, mask)

3. REVERSE TOPK:
   └── Zero out grad[i] where routing_map[i] = false

4. REVERSE PREPROCESS:
   ├── If softmax + pre_softmax: apply_softmax_bwd_on_float(grad, act, E, no mask)
   └── If sigmoid:              apply_sigmoid_bwd_on_float(grad, act, E)

5. Write grad_logits[token, :] = grad[:]
```

The explanation for the reverse post process above is in fact just the backward of normalization.

To recap, fwd normalization looks like:
$$y_i = \frac{\sigma_i}{S + \epsilon}, \quad S = \sum_{j \in \text{topk}} \sigma_j$$
The derivative of $y_i = \frac{\sigma_i}{S+\epsilon}$ with respect to $\sigma_k$ (for selected experts) uses the quotient rule:

$$\frac{\partial y_i}{\partial \sigma_k} = \begin{cases} \frac{1}{S+\epsilon} - \frac{\sigma_i}{(S+\epsilon)^2} & \text{if } i = k \\ -\frac{\sigma_i}{(S+\epsilon)^2} & \text{if } i \neq k \end{cases}$$

Applying the chain rule with upstream gradient $g_i = \frac{\partial L}{\partial y_i}$:

$$\frac{\partial L}{\partial \sigma_i} = \sum_k g_k \frac{\partial y_k}{\partial \sigma_i} = \frac{g_i}{S+\epsilon} - \frac{\sum_k g_k \sigma_k}{(S+\epsilon)^2}$$

### Shared Memory Layout (Backward)

```
┌─────────────────────────────────────────────────────────┐
│ grad_probs      [4 × E]     DataType                    │
│ act_from_fwd    [4 × E]     DataType                    │
│ comp_buf        [4 × E]     DataType   (scratch)        │
│ routing_map     [4 × E]     bool                        │
└─────────────────────────────────────────────────────────┘
```

---

## Grouped Top-K

When `group_topk > 0`, experts are partitioned into `num_groups` groups of equal size `group_size = E / num_groups`. The algorithm selects the best groups first, then picks experts within those groups.

### Algorithm

```
Given: E experts, G groups, group_size = E/G
       topk = K, group_topk = G_K
       experts_per_group = K / G_K

1. For each group g = 0..G-1:
   a. Run naive_topk_and_mask on scores[g*group_size .. (g+1)*group_size]
      with topk = experts_per_group
   b. group_scores[g] = sum of topk scores within group g

2. Run naive_topk_and_mask on group_scores[0..G-1]
   with topk = G_K
   → selects the best G_K groups

3. Copy scores from the selected groups into masked_scores
   (all other positions remain at -inf)

4. Run naive_topk_and_mask on masked_scores[0..E-1]
   with topk = K
   → final topk selection constrained to the winning groups
```

### Example

```
E=8 experts, G=4 groups, group_size=2, topk=4, group_topk=2

Groups: [E0,E1], [E2,E3], [E4,E5], [E6,E7]
                                             
Step 1: Best expert per group:               
  Group 0: best is E1 (score 0.8)            
  Group 1: best is E2 (score 0.9)            
  Group 2: best is E4 (score 0.3)            
  Group 3: best is E7 (score 0.7)            
  group_scores = [0.8, 0.9, 0.3, 0.7]       
                                             
Step 2: Top-2 groups: Group 1 (0.9), Group 0 (0.8)
                                             
Step 3: masked_scores = [-inf, -inf, scores[2], scores[3],
                         -inf, -inf, -inf, -inf]
        Plus group 0:   [scores[0], scores[1], ...]
                                             
Step 4: Top-4 from masked_scores → E2, E0, E1, E3
```

---

## Fused Score for MoE Aux Loss

**Kernel:** `fused_score_for_moe_aux_loss_forward_kernel`
**File:** `fused_score_for_moe_aux_loss.cu`

A simplified version of the main routing kernel, purpose-built for computing aux loss inputs. Key differences from `fused_topk_with_score_function`:

| Feature | fused_topk_with_score_function | fused_score_for_moe_aux_loss |
|---------|-------------------------------|------------------------------|
| Expert bias | Supported | Not supported |
| Grouped top-k | Supported | Not supported |
| Scaling factor | Applied | Not applied |
| Pre/post softmax | Configurable | Always pre (full distribution) |
| Output scores | Sparse (only topk nonzero) | Dense (all experts) |
| Normalization (sigmoid) | Over topk only | Over all experts |

### Forward Pipeline

```
1. Load logits → shared memory
2. Apply score function (softmax or sigmoid) → save to intermediate_output
3. If sigmoid + topk > 1: normalize scores over ALL experts (not just topk)
4. Run naive_topk_and_mask → routing_map[topk_indices] = true
5. Write all scores to output (dense — all E experts, not just selected)
```

### Why a Separate Kernel?

The aux loss requires:
- **Dense scores** over all experts (not just the topk selected ones)
- **Clean routing** — no bias influence, no group constraints, no scaling

These differ from the main routing path enough to justify a separate kernel rather than overloading the main one with more conditional branches.

### Backward

The backward kernel reverses the sigmoid normalization and score function derivatives. Since there is no topk masking of the gradient (all expert scores participate in the loss), no routing_map is needed in the backward — the gradient flows through all expert positions.

---

## Fused MoE Auxiliary Loss (Forward)

**Kernel:** `fused_moe_aux_loss_forward_kernel`
**File:** `fused_moe_aux_loss.cu`

### Mathematical Formulation

The auxiliary load-balancing loss encourages even distribution of tokens across experts:

$$L_{\text{aux}} = C \cdot \sum_{i=1}^{E} \left(\sum_{t=1}^{N} p_{t,i}\right) \cdot f_i$$

where:
- $p_{t,i}$ = probability that token $t$ assigns to expert $i$ (from `fused_score_for_moe_aux_loss`)
- $f_i$ = number of tokens routed to expert $i$ (`tokens_per_expert[i]`, derived from `routing_map.sum(dim=0)`)
- $C = \frac{E \cdot \text{coeff}}{K \cdot T^2}$ where $T$ = `total_num_tokens`, $K$ = topk

### Breakdown

Step by step:

1. **Column reduction** — sum probs along the token dimension:
   $$\bar{p}_i = \sum_{t=1}^{N} p_{t,i} \quad \text{(aggregated\_probs\_per\_expert)}$$

2. **Element-wise multiply** with token counts:
   $$\bar{p}_i \leftarrow \bar{p}_i \times f_i$$

3. **Horizontal reduction** — sum across experts:
   $$S = \sum_{i=1}^{E} \bar{p}_i \cdot f_i$$

4. **Apply coefficient:**
   $$L_{\text{aux}} = S \times C$$

### SM 90+ Implementation (Cooperative Groups)

On Hopper (SM 90+), the kernel uses **CUDA Cooperative Groups** with **Thread Block Clusters** to distribute the column reduction across multiple SMs:

```
Cluster of 8 blocks, each with 1024 threads
Each block: allocates shared memory [E × sizeof(double)]
                                                        
Phase 1: Each block reduces its share of rows         
  block 0: rows [0, 8, 16, ...]                       
  block 1: rows [1, 9, 17, ...]                       
  ...                                                  
  Each warp within a block handles strided rows        
  Result: partial sums in each block's shared memory   
                                                        
Phase 2: cluster.sync() — barrier across all 8 blocks  
                                                        
Phase 3: Block 0 reads other blocks' shared memory     
  via cluster.map_shared_rank(shmem, block_id)         
  Adds all partial sums into block 0's shared memory   
                                                        
Phase 4: cluster.sync()                                
                                                        
Phase 5: Block 0 multiplies by tokens_per_expert       
  Then warp 0 of block 0 does final reduction          
  Thread 0 computes C_coeff and writes aux_loss[0]     
```

### Pre-SM 90 Implementation

Falls back to a single block of 1024 threads. All rows are reduced by a single block, which limits parallelism but avoids the need for inter-block synchronization:

```
1 block × 1024 threads
All warps reduce rows in a strided pattern
__syncthreads() for intra-block synchronization
Warp 0, lane 0 writes final result
```

### Const_buf

The forward pass saves `C_coeff` to a GPU-side buffer (`Const_buf[0]`). This avoids recomputing it in the backward pass and ensures the backward uses the exact same coefficient.

---

## Fused MoE Auxiliary Loss (Backward)

**Kernel:** `fused_moe_aux_loss_backward_kernel`

### Mathematical Derivation

Given the forward:

$$L = C \cdot \sum_i \bar{p}_i \cdot f_i = C \cdot \sum_i f_i \cdot \sum_t p_{t,i}$$

The gradient with respect to each probability element $p_{t,i}$:

$$\frac{\partial L}{\partial p_{t,i}} = C \cdot f_i \cdot \frac{\partial L_{\text{downstream}}}{\partial L}$$

In the kernel:

```
grad_probs[t, i] = C_coeff * tokens_per_expert[i] * grad_aux_loss[0]
```

This is a simple broadcast: the gradient for every token-expert pair $(t, i)$ is the same — it depends only on the expert index $i$ (via $f_i$) and the upstream scalar gradient. No shared memory is needed.

### Launch Configuration

```
Block size: 256 threads
Grid size:  ceil(num_rows / 256)
Shared mem: 0
```

Each warp handles a strided set of rows, each thread handles a strided set of columns.

---

## Utility Functions

All defined in `utils.h`.

### warp_reduce_on_shmem

Reduces an array in shared memory to a single scalar using a warp:

```
1. Each thread accumulates its strided elements: data[lane_id], data[lane_id+32], ...
2. 5 rounds of __shfl_xor_sync to reduce across the warp
3. Returns the result (available in all threads)
```

Supports SUM and MAX reduction types. Uses `double` internally for precision.

### masked_warp_reduce_on_shmem

Same as `warp_reduce_on_shmem` but only includes elements where `mask[i] = true`. Used in the backward pass for sigmoid normalization.

### apply_softmax_on_float

In-place softmax over shared memory array:

```
1. max_val = warp_reduce(scores, MAX)
2. scores[i] = exp(scores[i] - max_val)       // numerically stable
3. sum_val = warp_reduce(scores, SUM)
4. scores[i] /= sum_val
```

### apply_softmax_bwd_on_float

In-place softmax backward. Supports an optional mask for post-topk softmax backward:

```
1. comp_buf[i] = grad[i] * fwd_output[i]      (masked if applicable)
2. sum_grad_output = warp_reduce(comp_buf, SUM)
3. grad[i] = fwd_output[i] * (grad[i] - sum_grad_output)
```

### apply_sigmoid_on_float / apply_sigmoid_bwd_on_float

Straightforward element-wise sigmoid and its derivative.

---

## GPU Parallelism and Memory Layout

### Thread Hierarchy (Top-K and Score Kernels)

```
Grid
 └── Block (128 threads = 4 warps)
      ├── Warp 0 → Token 0  (32 threads scan all E experts)
      ├── Warp 1 → Token 1
      ├── Warp 2 → Token 2
      └── Warp 3 → Token 3
```

- **1 warp = 1 token**: each warp independently processes one token
- **Warp-level sync only**: `__syncwarp()` instead of `__syncthreads()` (except shared memory init)
- **No inter-warp communication**: tokens are fully independent

### Thread Hierarchy (Aux Loss Kernel, SM 90+)

```
Cluster (8 blocks)
 └── Block 0..7 (1024 threads each)
      └── All warps cooperatively reduce rows
```

- **Inter-block communication** via `cluster.map_shared_rank()` (distributed shared memory)
- **cluster.sync()** for barriers across the cluster

### I/O Tensor Shapes

| Tensor | Shape | Type | Description |
|--------|-------|------|-------------|
| logits | [N, E] | float32/bf16/fp16 | Raw gating network output |
| probs | [N, E] | same as logits | Sparse output — topk scores per row |
| routing_map | [N, E] | bool | Binary mask of selected experts |
| intermediate_output | [N, E] | same as logits | Saved softmax/sigmoid for backward |
| expert_bias | [E] | float32/bf16/fp16 | Optional bias for sigmoid routing |
| tokens_per_expert | [E] | int32/int64 | Token counts per expert |
| aux_loss | [1] | same as probs | Scalar loss output |
| Const_buf | [1] | float32 | Cached coefficient for backward |

---

## Numerical Precision: Internal float64

The CUDA kernels use `double` (float64) extensively as an **internal computation type**, while all I/O tensors remain float32/bf16/fp16.

### Where float64 is Used

1. **`fused_moe_aux_loss.cu`**: `using CompType = double;` — the entire aggregation buffer in shared memory is double-precision
2. **`utils.h`**: `warp_reduce_on_shmem` casts to `volatile double` for all reductions and shuffle operations
3. **`naive_topk_and_mask`**: comparisons use `volatile double` to avoid precision-related misranking
4. **Score function kernels**: expert bias addition and normalization use `static_cast<double>(...)` for intermediate arithmetic

### Why This is Safe for JAX (float32-only)

JAX does not support float64 on GPU by default, but this is irrelevant: the float64 arithmetic happens entirely in CUDA registers and shared memory, invisible to the host-side type system. The kernel's template parameter `DataType` controls what goes in and comes out — always float32/bf16/fp16. The double precision serves the same role as XLA's internal float32 accumulation for bf16 matmuls: it prevents catastrophic cancellation during reductions over many tokens.

---

## File Map

```
transformer_engine/common/
├── fused_router/
│   ├── utils.h                                 # Shared device functions:
│   │                                           #   naive_topk_and_mask, warp_reduce_on_shmem,
│   │                                           #   apply_softmax/sigmoid + bwd, type macros
│   ├── fused_topk_with_score_function.cu       # Main routing kernel (fwd + bwd)
│   ├── fused_score_for_moe_aux_loss.cu         # Clean scoring for aux loss (fwd + bwd)
│   └── fused_moe_aux_loss.cu                   # Aux loss reduction (fwd + bwd, cooperative groups)
└── include/transformer_engine/
    └── fused_router.h                          # C API declarations (6 functions)

transformer_engine/pytorch/
├── router.py                                   # PyTorch torch.autograd.Function wrappers
└── csrc/extensions.h                           # Pybind11 declarations for PyTorch

transformer_engine/jax/
├── csrc/extensions/
│   └── router.cpp                              # XLA FFI handlers (6 handlers)
├── csrc/extensions.h                           # FFI handler declarations
├── cpp_extensions/
│   └── router.py                               # JAX BasePrimitive subclasses (6 primitives)
├── router.py                                   # High-level JAX API with jax.custom_vjp
└── csrc/extensions/pybind.cpp                  # FFI registration

tests/
├── pytorch/test_fused_router.py                # PyTorch reference tests
└── jax/test_fused_router.py                    # JAX reference tests
```
