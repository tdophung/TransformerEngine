# Mathematical Foundation of MoE Permutation Operations

This document provides a detailed mathematical explanation of the permutation operations used in Mixture-of-Experts (MoE) models in TransformerEngine, including the routing, permutation, and unpermutation algorithms.

## Table of Contents
1. [MoE Background](#moe-background)
2. [The Permutation Problem](#the-permutation-problem)
3. [Routing Map Representations](#routing-map-representations)
4. [Row ID Map Construction](#row-id-map-construction)
5. [Permutation Algorithm](#permutation-algorithm)
6. [Unpermutation Algorithm](#unpermutation-algorithm)
7. [Probability Weighting and Merging](#probability-weighting-and-merging)
8. [FP8 Quantization Support](#fp8-quantization-support)
9. [Chunk Sorting](#chunk-sorting)
10. [Implementation Details](#implementation-details)

---

## MoE Background

### What is Mixture-of-Experts?

**Mixture-of-Experts (MoE)** is a neural network architecture that uses multiple specialized sub-networks (experts) and dynamically routes inputs to different experts based on a learned routing function.

### Basic MoE Architecture

```
Input Tokens: [T₁, T₂, T₃, ..., Tₙ]
           ↓
      Router Network
           ↓
    Routing Decisions
           ↓
    ┌──────┴──────┬──────┬──────┐
    ↓      ↓      ↓      ↓      ↓
  Expert₁ Expert₂ Expert₃ ... Expertₘ
    ↓      ↓      ↓      ↓      ↓
    └──────┬──────┴──────┴──────┘
           ↓
     Merge Results
           ↓
    Output Tokens
```

### Key Concepts

- **Token**: A single unit of input (e.g., word embedding)
- **Expert**: A specialized feed-forward network
- **Router**: Network that decides which tokens go to which experts
- **TopK**: Number of experts each token can be routed to
- **Capacity**: Maximum number of tokens an expert can process

---

## The Permutation Problem

### Why Permutation is Needed

In MoE, tokens are initially arranged by sequence position but need to be **grouped by expert** for efficient batch processing.

### Example Scenario

```
Initial State (Tokens by Position):
Token 0 → Experts [1, 3]  
Token 1 → Experts [2]     
Token 2 → Experts [1]     
Token 3 → Experts [0, 1]  
Token 4 → Experts [3]     

Goal: Group by Expert for Batch Processing
Expert 0: [Token 3]
Expert 1: [Token 0, Token 2, Token 3]
Expert 2: [Token 1]
Expert 3: [Token 0, Token 4]
```

### Mathematical Formulation

Given:
- $N$ tokens: $\{T_0, T_1, ..., T_{N-1}\}$
- $E$ experts: $\{E_0, E_1, ..., E_{E-1}\}$
- Routing function: $R: T \rightarrow \{E_i\}_{i=1}^K$ (returns top-K experts)

Find a permutation $\pi$ such that tokens routed to the same expert are contiguous.

---

## Routing Map Representations

### Type 1: Mask Map (Binary Matrix)

A binary matrix $M \in \{0, 1\}^{N \times E}$ where:

$$M_{i,j} = \begin{cases}
1 & \text{if token } i \text{ is routed to expert } j \\
0 & \text{otherwise}
\end{cases}$$

**Example:**
```
Tokens: 5
Experts: 4
TopK: 2

Mask Map M:
     E₀  E₁  E₂  E₃
T₀ [[ 0,  1,  0,  1],    Token 0 → Experts 1, 3
T₁  [ 0,  0,  1,  0],    Token 1 → Expert 2
T₂  [ 0,  1,  0,  0],    Token 2 → Expert 1
T₃  [ 1,  1,  0,  0],    Token 3 → Experts 0, 1
T₄  [ 0,  0,  0,  1]]    Token 4 → Expert 3

Each row sums to TopK (or less with dropped tokens)
```

### Type 2: Index Map (Expert Indices)

A matrix $I \in \mathbb{Z}^{N \times K}$ where:

$$I_{i,k} = \text{index of k-th expert for token } i$$

**Example:**
```
Tokens: 5
TopK: 2

Index Map I:
     Expert₁ Expert₂
T₀ [[ 1,      3],       Token 0 → Experts [1, 3]
T₁  [ 2,     -1],       Token 1 → Expert [2] (padded with -1)
T₂  [ 1,     -1],       Token 2 → Expert [1]
T₃  [ 0,      1],       Token 3 → Experts [0, 1]
T₄  [ 3,     -1]]       Token 4 → Expert [3]

-1 indicates padding (< TopK experts chosen)
```

### Relationship Between Representations

Converting from Index Map to Mask Map:

$$M_{i,j} = \sum_{k=1}^{K} \mathbb{1}_{I_{i,k} = j}$$

where $\mathbb{1}$ is the indicator function.

---

## Row ID Map Construction

### Purpose

The **Row ID Map** is a data structure that enables efficient permutation and unpermutation without materializing the full permuted array indices.

### Structure

For mask-based routing, the Row ID Map is a matrix $R \in \mathbb{Z}^{N \times (2E + 1)}$ where:

**For each token $i$:**
- $R_{i, 0 \ldots n-1}$: Destination row indices (where token copies will go)
- $R_{i, E \ldots E+n-1}$: Corresponding expert indices
- $R_{i, 2E}$: Number of experts routed to ($n$)

### Three-Pass Algorithm

The construction happens in three passes over the data:

#### Pass 1: Block-Level Cumulative Sum

**Goal:** Compute cumulative sum within each block of tokens for each expert.

The key operation is: `cumsum(mask) * mask`, which computes the cumulative sum and then zeros out positions where the mask is 0.

$$C_{b,e} = \sum_{i \in \text{block}_b} M_{i,e}$$

**Pseudocode:**
```python
for each expert e:
    for each block b (size = BLOCK_SIZE):
        cumsum = 0
        for each token i in block b:
            if M[i,e] == 1:
                cumsum += 1
                R[i,e] = cumsum  # Local cumsum within block
            else:
                R[i,e] = 0  # Zero out non-routed positions
        workspace[b,e] = cumsum  # Save block total
```

**Example:**
```
Block size = 3
Mask Map M (Expert 1 column):
Block 0: [1, 1, 0]  → cumsum: [1, 2, 2] → cumsum*mask: [1, 2, 0]  → workspace[0] = 2
Block 1: [1, 0, 1]  → cumsum: [1, 1, 2] → cumsum*mask: [1, 0, 2]  → workspace[1] = 2

After Pass 1, R[:,1] has local cumsums (cumsum * mask):
[1, 2, 0, 1, 0, 2]
```

#### Pass 2: Global Cumulative Sum

**Goal:** Convert block-level cumsums to global positions.

$$\text{global\_offset}_e = \sum_{b'=0}^{b-1} C_{b',e}$$

Then for each token:
$$R_{i,e} = \begin{cases}
R_{i,e} + \text{global\_offset}_e - 1 & \text{if } M_{i,e} = 1 \\
-1 & \text{otherwise}
\end{cases}$$

**Pseudocode:**
```python
for each expert e:
    # Compute prefix sum of block totals
    global_offset = exclusive_scan(workspace[:,e])
    
    for each block b:
        for each token i in block b:
            if R[i,e] > 0:  # Token was routed to this expert
                R[i,e] = R[i,e] + global_offset[b] - 1
            else:
                R[i,e] = -1
```

**Example (continuing):**
```
Workspace: [2, 2]
Prefix sum: [0, 2]  (exclusive scan)

Converting local to global:
Block 0: [1, 2, 0] + 0 - 1 = [0, 1, -1]  (only non-zero values are converted)
Block 1: [1, 0, 2] + 2 - 1 = [2, -1, 3]  (zero values become -1)

After Pass 2, R[:,1] has global positions:
[0, 1, -1, 2, -1, 3]
```

#### Pass 3: Densification

**Goal:** Convert sparse structure to dense, storing only valid routing information.

**Pseudocode:**
```python
for each token i:
    count = 0
    for each expert e:
        if R[i,e] >= 0:  # Valid routing
            R[i, count] = R[i,e]           # Store destination row
            R[i, E + count] = e            # Store expert index
            count += 1
    R[i, 2*E] = count  # Store total count
```

**Example:**
```
Token 0: routed to experts [1, 3] with destinations [0, 4]
  R[0, :] = [0, 4, ?, ..., ?, 1, 3, ?, ..., ?, 2]
            ↑dst rows↑      ↑expert ids↑   ↑count↑

Token 1: routed to expert [2] with destination [7]
  R[1, :] = [7, ?, ?, ..., ?, 2, ?, ?, ..., ?, 1]
```

### Visualization

```
Input: 5 tokens, 4 experts, TopK=2

Mask Map:
     E₀  E₁  E₂  E₃
T₀ [[ 0,  1,  0,  1],
T₁  [ 0,  0,  1,  0],
T₂  [ 0,  1,  0,  0],
T₃  [ 1,  1,  0,  0],
T₄  [ 0,  0,  0,  1]]

After Pass 1 (block cumsum, block_size=3):
     E₀  E₁  E₂  E₃
T₀ [[ 0,  1,  0,  1],     Block 0
T₁  [ 0,  0,  1,  0],     (cumsum * mask zeros out non-routed positions)
T₂  [ 0,  2,  0,  0],
T₃  [ 1,  1,  0,  0],     Block 1
T₄  [ 0,  0,  0,  1]]

Workspace (block totals):
     E₀  E₁  E₂  E₃
B₀ [[ 0,  2,  1,  1],
B₁  [ 1,  1,  0,  1]]

After Pass 2 (global positions):
     E₀  E₁  E₂  E₃
T₀ [[-1,  0, -1,  0],
T₁  [-1, -1,  0, -1],
T₂  [-1,  1, -1, -1],
T₃  [ 0,  2, -1, -1],
T₄  [-1, -1, -1,  1]]

After Pass 3 (densified):
     Dest₀ Dest₁ Dest₂ Dest₃ | Exp₀ Exp₁ Exp₂ Exp₃ | Count
T₀ [[  0,    0,    -,    -,  |  1,   3,   -,   -,  |  2  ],
T₁   [  0,    -,    -,    -,  |  2,   -,   -,   -,  |  1  ],
T₂   [  1,    -,    -,    -,  |  1,   -,   -,   -,  |  1  ],
T₃   [  0,    2,    -,    -,  |  0,   1,   -,   -,  |  2  ],
T₄   [  1,    -,    -,    -,  |  3,   -,   -,   -,  |  1  ]]
```

---

## Permutation Algorithm

### Goal

Rearrange tokens so that all tokens going to the same expert are contiguous.

### Mathematical Definition

Given:
- Input: $X \in \mathbb{R}^{N \times H}$ (N tokens, H hidden dim)
- Row ID Map: $R$
- Number of output tokens: $N_{\text{out}}$

Compute:
- Output: $Y \in \mathbb{R}^{N_{\text{out}} \times H}$

where each token is copied to its designated positions.

### Algorithm

For each token $i$ with $n_i$ routed experts:

$$Y_{R_{i,k}, :} = X_{i,:} \quad \text{for } k = 0, 1, ..., n_i - 1$$

This means token $i$ is copied to rows $R_{i,0}, R_{i,1}, ..., R_{i,n_i-1}$.

### Example

```
Input tokens X:
T₀: [1.0, 2.0, 3.0]  → Experts [1, 3]
T₁: [4.0, 5.0, 6.0]  → Expert [2]
T₂: [7.0, 8.0, 9.0]  → Expert [1]
T₃: [2.0, 3.0, 4.0]  → Experts [0, 1]
T₄: [5.0, 6.0, 7.0]  → Expert [3]

Row ID Map R (destinations):
T₀: [0, 0]  (goes to rows 0 and 0 for experts 1 and 3)
T₁: [0]     (goes to row 0 for expert 2)
T₂: [1]     (goes to row 1 for expert 1)
T₃: [0, 2]  (goes to rows 0 and 2 for experts 0 and 1)
T₄: [2]     (goes to row 2 for expert 3)

After grouping by expert:
Expert 0: starts at row 0
Expert 1: starts at row 1
Expert 2: starts at row 4
Expert 3: starts at row 5

Permuted output Y:
Row 0: [2.0, 3.0, 4.0]  ← T₃ (Expert 0)
Row 1: [1.0, 2.0, 3.0]  ← T₀ (Expert 1, first)
Row 2: [7.0, 8.0, 9.0]  ← T₂ (Expert 1, second)
Row 3: [2.0, 3.0, 4.0]  ← T₃ (Expert 1, third)
Row 4: [4.0, 5.0, 6.0]  ← T₁ (Expert 2)
Row 5: [1.0, 2.0, 3.0]  ← T₀ (Expert 3, first)
Row 6: [5.0, 6.0, 7.0]  ← T₄ (Expert 3, second)
```

### Memory Layout After Permutation

```
Permuted Array (grouped by expert):
┌─────────────────────────────────────────┐
│ Expert 0 tokens: [T₃]                   │
├─────────────────────────────────────────┤
│ Expert 1 tokens: [T₀, T₂, T₃]          │
├─────────────────────────────────────────┤
│ Expert 2 tokens: [T₁]                   │
├─────────────────────────────────────────┤
│ Expert 3 tokens: [T₀, T₄]              │
└─────────────────────────────────────────┘

Expert can now process its tokens in a batch!
```

### Pseudocode

```python
def permute(X, row_id_map, num_out_tokens, hidden_size):
    """
    X: [num_tokens, hidden_size]
    row_id_map: [num_tokens, 2*num_experts+1]
    """
    Y = zeros(num_out_tokens, hidden_size)
    
    for token_idx in range(num_tokens):
        n_routed = row_id_map[token_idx, 2*num_experts]  # Count
        
        for k in range(n_routed):
            dst_row = row_id_map[token_idx, k]
            Y[dst_row, :] = X[token_idx, :]
    
    return Y
```

---

## Unpermutation Algorithm

### Goal

Restore tokens from expert-grouped layout back to original sequence order, optionally merging multiple expert outputs per token.

### Mathematical Definition

Given:
- Permuted input: $Y \in \mathbb{R}^{N_{\text{out}} \times H}$
- Row ID Map: $R$
- Optional merging weights: $W \in \mathbb{R}^{N \times E}$

Compute:
- Restored output: $X \in \mathbb{R}^{N \times H}$

### Two Modes

#### Mode 1: Simple Accumulation (No Merging Weights)

$$X_{i,:} = \sum_{k=0}^{n_i - 1} Y_{R_{i,k}, :}$$

Each token accumulates outputs from all experts it was routed to.

#### Mode 2: Weighted Merging (With Probabilities)

$$X_{i,:} = \sum_{k=0}^{n_i - 1} W_{i, e_k} \cdot Y_{R_{i,k}, :}$$

where $e_k = R_{i, E+k}$ is the expert index.

Outputs are weighted by routing probabilities before merging.

### Example: Simple Accumulation

```
Permuted input Y (after expert processing):
Row 0: [2.1, 3.1, 4.1]  ← Expert 0 processed T₃
Row 1: [1.5, 2.5, 3.5]  ← Expert 1 processed T₀
Row 2: [7.5, 8.5, 9.5]  ← Expert 1 processed T₂
Row 3: [2.5, 3.5, 4.5]  ← Expert 1 processed T₃
Row 4: [4.5, 5.5, 6.5]  ← Expert 2 processed T₁
Row 5: [1.8, 2.8, 3.8]  ← Expert 3 processed T₀
Row 6: [5.5, 6.5, 7.5]  ← Expert 3 processed T₄

Row ID Map (which rows belong to which original tokens):
T₀: destinations [1, 5]  experts [1, 3]
T₁: destinations [4]     experts [2]
T₂: destinations [2]     experts [1]
T₃: destinations [0, 3]  experts [0, 1]
T₄: destinations [6]     experts [3]

Unpermuted output X (accumulated):
T₀: Y[1] + Y[5] = [1.5+1.8, 2.5+2.8, 3.5+3.8] = [3.3, 5.3, 7.3]
T₁: Y[4] = [4.5, 5.5, 6.5]
T₂: Y[2] = [7.5, 8.5, 9.5]
T₃: Y[0] + Y[3] = [2.1+2.5, 3.1+3.5, 4.1+4.5] = [4.6, 6.6, 8.6]
T₄: Y[6] = [5.5, 6.5, 7.5]
```

### Example: Weighted Merging

```
Same permuted input Y, but now with routing probabilities:

Routing probabilities W:
       E₀   E₁   E₂   E₃
T₀: [ 0.0, 0.6, 0.0, 0.4]  (60% to E₁, 40% to E₃)
T₁: [ 0.0, 0.0, 1.0, 0.0]  (100% to E₂)
T₂: [ 0.0, 1.0, 0.0, 0.0]  (100% to E₁)
T₃: [ 0.7, 0.3, 0.0, 0.0]  (70% to E₀, 30% to E₁)
T₄: [ 0.0, 0.0, 0.0, 1.0]  (100% to E₃)

Unpermuted output X (weighted):
T₀: 0.6*Y[1] + 0.4*Y[5] 
  = 0.6*[1.5,2.5,3.5] + 0.4*[1.8,2.8,3.8]
  = [0.9+0.72, 1.5+1.12, 2.1+1.52]
  = [1.62, 2.62, 3.62]

T₁: 1.0*Y[4] = [4.5, 5.5, 6.5]

T₂: 1.0*Y[2] = [7.5, 8.5, 9.5]

T₃: 0.7*Y[0] + 0.3*Y[3]
  = 0.7*[2.1,3.1,4.1] + 0.3*[2.5,3.5,4.5]
  = [1.47+0.75, 2.17+1.05, 2.87+1.35]
  = [2.22, 3.22, 4.22]

T₄: 1.0*Y[6] = [5.5, 6.5, 7.5]
```

### Pseudocode

```python
def unpermute(Y, row_id_map, merging_probs, num_tokens, hidden_size):
    """
    Y: [num_out_tokens, hidden_size]
    row_id_map: [num_tokens, 2*num_experts+1]
    merging_probs: [num_tokens, num_experts] or None
    """
    X = zeros(num_tokens, hidden_size)
    
    for token_idx in range(num_tokens):
        n_routed = row_id_map[token_idx, 2*num_experts]
        
        for k in range(n_routed):
            src_row = row_id_map[token_idx, k]
            expert_id = row_id_map[token_idx, num_experts + k]
            
            if merging_probs is not None:
                weight = merging_probs[token_idx, expert_id]
                X[token_idx, :] += weight * Y[src_row, :]
            else:
                X[token_idx, :] += Y[src_row, :]
    
    return X
```

---

## Probability Weighting and Merging

### Router Output Probabilities

The router network outputs probabilities for each expert:

$$p_{i,e} = \text{softmax}(\text{router}(T_i))_e$$

For TopK routing, we select K experts with highest probabilities:

$$\text{TopK}(p_i) = \{e_{i,1}, e_{i,2}, ..., e_{i,K}\}$$

where $p_{i,e_{i,1}} \geq p_{i,e_{i,2}} \geq ... \geq p_{i,e_{i,K}}$

### Probability Permutation

When tokens are permuted, we also need to permute their routing probabilities:

```
Original probabilities:
T₀ → E₁: 0.6, E₃: 0.4
T₁ → E₂: 1.0
T₂ → E₁: 1.0
T₃ → E₀: 0.7, E₁: 0.3
T₄ → E₃: 1.0

Permuted probabilities (same order as permuted tokens):
Row 0 (T₃→E₀): 0.7
Row 1 (T₀→E₁): 0.6
Row 2 (T₂→E₁): 1.0
Row 3 (T₃→E₁): 0.3
Row 4 (T₁→E₂): 1.0
Row 5 (T₀→E₃): 0.4
Row 6 (T₄→E₃): 1.0
```

### Load Balancing

To ensure experts are used evenly, an auxiliary loss is often added:

$$\mathcal{L}_{\text{aux}} = \alpha \cdot \text{CV}(f_1, f_2, ..., f_E)^2$$

where:
- $f_e = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}_{e \in \text{TopK}(p_i)}$ (fraction of tokens routed to expert $e$)
- $\text{CV}$ is the coefficient of variation
- $\alpha$ is a hyperparameter

---

## FP8 Quantization Support

### Quantization Schemes

TransformerEngine supports three FP8 quantization schemes:

#### 1. Per-Tensor Scaling

$$X_{\text{fp8}} = \text{quantize}(X \cdot s)$$

where $s$ is a single scalar for the entire tensor.

**During permutation:** Only data is permuted; scale stays with original tensor.

#### 2. Blockwise (Row-wise) Scaling

$$X_{\text{fp8}}[i,:] = \text{quantize}(X[i,:] \cdot s_i)$$

where $s_i$ is a scale per row (token).

**During permutation:** Both data and scales must be permuted together!

```
Input:
  X_data: [T₀, T₁, T₂, T₃, T₄]  (FP8)
  X_scale: [s₀, s₁, s₂, s₃, s₄] (FP32)

After permutation:
  Y_data: [T₃, T₀, T₂, T₃, T₁, T₀, T₄]  (reordered)
  Y_scale: [s₃, s₀, s₂, s₃, s₁, s₀, s₄] (reordered same way)
```

#### 3. MXFP8 Scaling

Similar to blockwise but with different scale representation.

### Permutation with FP8

```python
def permute_fp8(X_data, X_scale, row_id_map, num_out_tokens):
    """
    X_data: [num_tokens, hidden_size] in FP8
    X_scale: [num_tokens, scale_hidden_dim] in FP32
    """
    Y_data = zeros(num_out_tokens, hidden_size, dtype=fp8)
    Y_scale = zeros(num_out_tokens, scale_hidden_dim, dtype=fp32)
    
    for token_idx in range(num_tokens):
        n_routed = row_id_map[token_idx, 2*num_experts]
        
        for k in range(n_routed):
            dst_row = row_id_map[token_idx, k]
            # Copy both data and scale
            Y_data[dst_row, :] = X_data[token_idx, :]
            Y_scale[dst_row, :] = X_scale[token_idx, :]
    
    return Y_data, Y_scale
```

---

## Chunk Sorting

### Purpose

Some MoE implementations use chunk-based routing where the input is divided into chunks and entire chunks are assigned to experts.

### Chunk Sort Problem

Given:
- Split sizes: $[s_0, s_1, ..., s_{C-1}]$ (size of each chunk)
- Sorted indices: $[\sigma_0, \sigma_1, ..., \sigma_{C-1}]$ (permutation of chunks)

Rearrange chunks according to sorted indices.

### Example

```
Input chunks:
Chunk 0 (size 2): [T₀, T₁]  
Chunk 1 (size 3): [T₂, T₃, T₄]
Chunk 2 (size 1): [T₅]
Chunk 3 (size 2): [T₆, T₇]

Split sizes: [2, 3, 1, 2]
Sorted indices: [2, 0, 3, 1]  (reorder: 2nd, then 0th, then 3rd, then 1st)

Output:
Chunk 2: [T₅]
Chunk 0: [T₀, T₁]
Chunk 3: [T₆, T₇]
Chunk 1: [T₂, T₃, T₄]
```

### Row ID Map for Chunks

$$\text{row\_id\_map}[i] = \sum_{c=0}^{k-1} s_{\sigma_c} + j$$

where token $i$ is the $j$-th token in chunk $k$.

### Pseudocode

```python
def make_chunk_sort_map(split_sizes, sorted_indices):
    """
    split_sizes: [num_chunks]
    sorted_indices: [num_chunks]
    """
    row_id_map = zeros(num_tokens, dtype=int32)
    
    # Compute prefix sums
    prefix_sum_original = exclusive_scan(split_sizes)
    prefix_sum_sorted = exclusive_scan(split_sizes[sorted_indices])
    
    token_idx = 0
    for chunk_id in range(num_chunks):
        size = split_sizes[chunk_id]
        sorted_chunk_id = find(sorted_indices == chunk_id)
        
        for j in range(size):
            row_id_map[token_idx] = prefix_sum_sorted[sorted_chunk_id] + j
            token_idx += 1
    
    return row_id_map
```

---

## Implementation Details

### Triton Kernel Structure

#### Kernel 1: `row_id_map_pass_1_kernel`

**Grid:** `(num_experts, num_blocks)`  
**Block size:** Typically 1024 threads

**Per thread block:**
- Process one block of tokens for one expert
- Compute local cumulative sum
- Store block total in workspace

#### Kernel 2: `row_id_map_pass_2_kernel`

**Grid:** `(num_experts, num_blocks)`

**Per thread block:**
- Load workspace (block totals)
- Compute prefix sum to get global offsets
- Convert local cumsums to global row indices
- Mark non-routed entries with -1

#### Kernel 3: `row_id_map_pass_3_kernel`

**Grid:** `(num_tokens,)`

**Per thread:**
- Process one token
- Scan across all experts
- Pack valid routings into dense format
- Store destination rows, expert IDs, and count

#### Kernel 4: `permute_kernel`

**Grid:** `(num_tokens, num_hidden_blocks)`  
**Block size:** Tunable (e.g., 256)

**Per thread block:**
- Process hidden dimension in blocks
- For each token's routing:
  - Load source token data
  - Store to destination row(s)
  - Optionally permute probabilities and scales

#### Kernel 5: `unpermute_kernel`

**Grid:** `(num_tokens, num_hidden_blocks)`

**Per thread block:**
- For each destination token:
  - Load from multiple source rows (expert outputs)
  - Apply optional probability weighting
  - Accumulate/merge results
  - Store to original token position

### Memory Access Patterns

```
Permutation (Scatter):
┌─────────────────────┐
│ Token 0 │───┐       │
│ Token 1 │─┐ │       │
│ Token 2 │ │ │       │  Coalesced read,
│ Token 3 │ │ │       │  potentially
│ Token 4 │ │ └───────┼─→ scattered write
│   ...   │ └─────────┼─→ (depends on
│         │           └─→  routing pattern)
└─────────────────────┘

Unpermutation (Gather):
┌─────────────────────┐
│       ┌─→ Token 0   │  Potentially
│   ┌───┼─→ Token 1   │  scattered read,
│   │ ┌─┘   Token 2   │  coalesced write
│   │ │     Token 3   │
│───┘ │     Token 4   │
│     │       ...     │
└─────┴───────────────┘
```

### Optimization Techniques

1. **Coalesced Memory Access**
   - Process hidden dimension in blocks
   - Threads in a warp access consecutive elements

2. **Atomic-Free Accumulation**
   - Use row_id_map to avoid write conflicts
   - Each output position written by exactly one thread

3. **FP8 Optimization**
   - Separate kernels for data and scale
   - Minimize dtype conversions

4. **Load Balancing**
   - Distribute work evenly across thread blocks
   - Handle variable routing counts efficiently

---

## Complete Example Walkthrough

### Setup

```
Tokens: 5 (T₀, T₁, T₂, T₃, T₄)
Experts: 3 (E₀, E₁, E₂)
TopK: 2
Hidden size: 4

Routing decisions (router network output):
T₀ → E₁ (0.7), E₂ (0.3)
T₁ → E₀ (0.6), E₁ (0.4)
T₂ → E₁ (1.0)
T₃ → E₀ (0.5), E₂ (0.5)
T₄ → E₂ (1.0)
```

### Step 1: Construct Mask Map

```
Mask Map M:
     E₀  E₁  E₂
T₀ [[ 0,  1,  1],
T₁  [ 1,  1,  0],
T₂  [ 0,  1,  0],
T₃  [ 1,  0,  1],
T₄  [ 0,  0,  1]]
```

### Step 2: Build Row ID Map (Simplified)

```
Expert grouping (conceptual):
E₀: T₁, T₃        (2 tokens)
E₁: T₀, T₁, T₂    (3 tokens)
E₂: T₀, T₃, T₄    (3 tokens)

Destination row indices:
E₀ starts at row 0
E₁ starts at row 2
E₂ starts at row 5

Row ID Map R:
     Dst₀ Dst₁ | Exp₀ Exp₁ | Count
T₀ [[  2,   5, |  1,   2,  |  2  ],   E₁→row2, E₂→row5
T₁  [  0,   3, |  0,   1,  |  2  ],   E₀→row0, E₁→row3
T₂  [  4,  -1, |  1,  -1,  |  1  ],   E₁→row4
T₃  [  1,   6, |  0,   2,  |  2  ],   E₀→row1, E₂→row6
T₄  [  7,  -1, |  2,  -1,  |  1  ]]   E₂→row7
```

### Step 3: Permute Tokens

```
Input X:
T₀: [1.0, 2.0, 3.0, 4.0]
T₁: [5.0, 6.0, 7.0, 8.0]
T₂: [9.0, 1.0, 2.0, 3.0]
T₃: [4.0, 5.0, 6.0, 7.0]
T₄: [8.0, 9.0, 1.0, 2.0]

Permuted Y (grouped by expert):
Row 0 (T₁→E₀): [5.0, 6.0, 7.0, 8.0]  ┐ Expert 0
Row 1 (T₃→E₀): [4.0, 5.0, 6.0, 7.0]  ┘
Row 2 (T₀→E₁): [1.0, 2.0, 3.0, 4.0]  ┐
Row 3 (T₁→E₁): [5.0, 6.0, 7.0, 8.0]  ├ Expert 1
Row 4 (T₂→E₁): [9.0, 1.0, 2.0, 3.0]  ┘
Row 5 (T₀→E₂): [1.0, 2.0, 3.0, 4.0]  ┐
Row 6 (T₃→E₂): [4.0, 5.0, 6.0, 7.0]  ├ Expert 2
Row 7 (T₄→E₂): [8.0, 9.0, 1.0, 2.0]  ┘
```

### Step 4: Expert Processing

```
Each expert processes its batch:

Expert 0 processes rows 0-1:
  Output: some transformation of inputs

Expert 1 processes rows 2-4:
  Output: some transformation of inputs

Expert 2 processes rows 5-7:
  Output: some transformation of inputs

Processed Y' (assume experts add 0.1 to each element):
Row 0: [5.1, 6.1, 7.1, 8.1]
Row 1: [4.1, 5.1, 6.1, 7.1]
Row 2: [1.1, 2.1, 3.1, 4.1]
Row 3: [5.1, 6.1, 7.1, 8.1]
Row 4: [9.1, 1.1, 2.1, 3.1]
Row 5: [1.1, 2.1, 3.1, 4.1]
Row 6: [4.1, 5.1, 6.1, 7.1]
Row 7: [8.1, 9.1, 1.1, 2.1]
```

### Step 5: Unpermute with Probability Weighting

```
Routing probabilities W:
     E₀   E₁   E₂
T₀ [ 0.0, 0.7, 0.3]
T₁ [ 0.6, 0.4, 0.0]
T₂ [ 0.0, 1.0, 0.0]
T₃ [ 0.5, 0.0, 0.5]
T₄ [ 0.0, 0.0, 1.0]

Unpermuted output X':
T₀: 0.7*Y'[2] + 0.3*Y'[5]
  = 0.7*[1.1,2.1,3.1,4.1] + 0.3*[1.1,2.1,3.1,4.1]
  = [1.1, 2.1, 3.1, 4.1]

T₁: 0.6*Y'[0] + 0.4*Y'[3]
  = 0.6*[5.1,6.1,7.1,8.1] + 0.4*[5.1,6.1,7.1,8.1]
  = [5.1, 6.1, 7.1, 8.1]

T₂: 1.0*Y'[4]
  = [9.1, 1.1, 2.1, 3.1]

T₃: 0.5*Y'[1] + 0.5*Y'[6]
  = 0.5*[4.1,5.1,6.1,7.1] + 0.5*[4.1,5.1,6.1,7.1]
  = [4.1, 5.1, 6.1, 7.1]

T₄: 1.0*Y'[7]
  = [8.1, 9.1, 1.1, 2.1]
```

---

## Summary

### Key Operations

| Operation | Input | Output | Purpose |
|-----------|-------|--------|---------|
| **Row ID Map** | Routing mask | Mapping structure | Efficient permutation lookup |
| **Permute** | Tokens + map | Grouped tokens | Batch processing by expert |
| **Unpermute** | Expert outputs + map | Original order | Restore sequence structure |

### Complexity

For $N$ tokens, $E$ experts, $H$ hidden dimension:

| Operation | Time | Space |
|-----------|------|-------|
| Row ID Map | $O(NE)$ | $O(NE)$ |
| Permute | $O(NH \cdot K)$ | $O(NK \cdot H)$ |
| Unpermute | $O(NH \cdot K)$ | $O(NH)$ |

where $K$ is TopK (typically $K \ll E$).

### Advantages

1. **Memory Efficient**: Row ID map is compact
2. **Batched Processing**: Experts process tokens in batches
3. **FP8 Support**: Handles quantized tensors efficiently
4. **Flexible**: Supports various routing strategies

---

## References

1. [Switch Transformers: Scaling to Trillion Parameter Models](https://arxiv.org/abs/2101.03961)
2. [GShard: Scaling Giant Models with Conditional Computation](https://arxiv.org/abs/2006.16668)
3. [Mixtral of Experts](https://arxiv.org/abs/2401.04088)
4. [ST-MoE: Designing Stable and Transferable Sparse Expert Models](https://arxiv.org/abs/2202.08906)
