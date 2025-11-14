# JAX-Triton Kernels for Transformer Engine

This directory contains JAX wrappers for high-performance Triton kernels used in Transformer Engine.

## Overview

The JAX-Triton integration allows you to use efficient Triton kernels directly in JAX code, with full support for:
- JAX's automatic differentiation (`jax.grad`, `jax.value_and_grad`)
- JIT compilation (`jax.jit`)
- Distributed training with `pmap` and tensor parallelism
- GPU acceleration

## Dependencies

To use these kernels, you need:
- `jax` - JAX framework
- `jax-triton` - JAX's Triton integration library
- `triton` - Triton GPU programming language

These dependencies are automatically installed when you install `transformer_engine_jax`.

## Available Kernels

### Cross Entropy Loss

The `cross_entropy_loss` function provides a memory-efficient implementation of cross entropy loss using Triton kernels.

**Features:**
- Fused softmax and cross entropy computation
- Online softmax algorithm for numerical stability
- Label smoothing support
- Tensor parallelism support for large vocabulary sizes
- Configurable loss reduction (mean or per-token)
- Ignore index support

**Usage:**

```python
import jax.numpy as jnp
from transformer_engine.jax.triton import cross_entropy_loss

# Input logits: (batch_size, seq_length, vocab_size)
logits = jnp.zeros((2, 10, 50000))

# Target indices: (batch_size, seq_length)
targets = jnp.zeros((2, 10), dtype=jnp.int32)

# Compute loss
loss = cross_entropy_loss(
    logits,
    targets,
    label_smoothing=0.1,      # Optional: label smoothing factor
    reduce_loss=True,         # True: return scalar, False: return per-token loss
    ignore_idx=-100,          # Index to ignore in loss computation
    axis_name=None,           # For tensor parallelism with pmap
)
```

**Tensor Parallelism Example:**

```python
import jax
from transformer_engine.jax.triton import cross_entropy_loss

@jax.pmap
def compute_loss_parallel(logits, targets):
    # Each device has a shard of the vocabulary
    return cross_entropy_loss(
        logits,
        targets,
        axis_name="batch",  # Collective axis for all-gather
    )

# logits shape: (num_devices, batch_size, seq_length, vocab_size_per_device)
loss_per_device = compute_loss_parallel(logits_sharded, targets_sharded)
```

## Implementation Details

### Memory Efficiency

The cross entropy kernel uses an online softmax algorithm that:
1. Computes softmax statistics (max and sum) in a single pass
2. Computes gradients on-the-fly without materializing the full softmax output
3. Reduces memory usage for large vocabulary sizes

### Tensor Parallelism

When using tensor parallelism (vocabulary sharding):
1. Each device computes softmax statistics for its vocabulary shard
2. Statistics are gathered across devices using `jax.lax.all_gather`
3. Final loss and gradients are computed using the global softmax statistics

### Custom VJP

The implementation uses JAX's custom VJP (vector-Jacobian product) to define efficient gradients. The forward pass computes both the loss and the gradients with respect to the input, which are then used in the backward pass.

## Performance Tips

1. **Use JIT compilation:** Wrap your loss computation in `jax.jit` for best performance
2. **Contiguous arrays:** Ensure input tensors are C-contiguous for optimal memory access
3. **Batch size:** Larger batch sizes typically achieve better GPU utilization
4. **Vocabulary sharding:** For very large vocabularies (>100k), consider using tensor parallelism

## Example

See `examples/jax/triton_cross_entropy_example.py` for complete usage examples including:
- Basic usage
- Gradient computation
- Tensor parallelism with pmap

## References

- [Triton: GPU Programming Language](https://github.com/openai/triton)
- [JAX-Triton](https://github.com/jax-ml/jax-triton)
- [Online Softmax Algorithm](https://arxiv.org/abs/1805.02867)

