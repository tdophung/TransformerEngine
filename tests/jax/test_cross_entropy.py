# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Tests for cross entropy Triton kernels"""

# Patch jax-triton for Triton 3.5.1 compatibility - MUST BE FIRST!
# import jax_triton_patch  # Apply compatibility patch before importing JAX/Triton

import jax
import jax.numpy as jnp
import pytest
from jax import value_and_grad, jit

from transformer_engine.jax.triton.cross_entropy import cross_entropy_loss
from utils import assert_allclose, dtype_tols


def reference_cross_entropy(
    logits: jnp.ndarray,
    target: jnp.ndarray,
    label_smoothing: float = 0.0,
    reduce_loss: bool = True,
    ignore_idx: int = -100,
) -> jnp.ndarray:
    """
    Reference cross entropy implementation using JAX primitives.
    
    Parameters
    ----------
    logits : jnp.ndarray
        Input logits of shape (B, SQ, V) or (B * SQ, V)
    target : jnp.ndarray
        Target indices of shape (B, SQ) or (B * SQ,)
    label_smoothing : float
        Amount of label smoothing to apply
    reduce_loss : bool
        Whether to reduce the loss to a scalar (mean) or return per-token loss
    ignore_idx : int
        Index to ignore in the loss computation
        
    Returns
    -------
    jnp.ndarray
        The computed cross entropy loss
    """
    # Reshape logits if needed
    original_shape = logits.shape
    if logits.ndim == 3:
        B, SQ, V = logits.shape
        logits = logits.reshape(-1, V)
        target = target.reshape(-1)
    else:
        V = logits.shape[-1]
    
    n_tokens = logits.shape[0]
    
    # Compute log softmax
    log_probs = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)
    
    # Create one-hot encoding with label smoothing
    if label_smoothing > 0:
        # Smooth labels. Teh advance indexing [token id, target] is ensuring i=c
        smoothed_labels = jnp.ones_like(log_probs) * (label_smoothing / V)
        smoothed_labels = smoothed_labels.at[jnp.arange(n_tokens), target].add(
            1.0 - label_smoothing
        )
        loss = -jnp.sum(log_probs * smoothed_labels, axis=-1)
    else:
        # Standard cross entropy
        loss = -log_probs[jnp.arange(n_tokens), target]
    
    # Handle ignore index
    mask = (target != ignore_idx).astype(jnp.float32)
    loss = loss * mask
    
    # Count valid tokens for normalization
    n_valid = jnp.maximum(jnp.sum(mask), 1.0)  # Avoid division by zero
    
    # Reduce loss
    if reduce_loss:
        loss = jnp.sum(loss) / n_valid
    elif original_shape[0] != n_tokens:
        # Reshape back to (B, SQ) if input was 3D
        loss = loss.reshape(original_shape[0], original_shape[1])
    
    return loss


class TestCrossEntropy:
    """Test cross entropy loss implementation"""
    
    @staticmethod
    def generate_input(
        batch: int,
        seq_len: int,
        vocab_size: int,
        dtype: jnp.dtype,
        swap_dim: bool = False,
        ignore_idx_positions: int = 0,
        ignore_idx: int = -100,
        key: jax.Array = None,
    ):
        """Generate random test inputs"""
        if key is None:
            key = jax.random.PRNGKey(0)
        
        logits_key, target_key = jax.random.split(key, 2)
        
        # Generate logits and target
        if swap_dim:
            logits_shape = (seq_len, batch, vocab_size)
            target_shape = (seq_len, batch)
        else:
            logits_shape = (batch, seq_len, vocab_size)
            target_shape = (batch, seq_len)
        
        logits = jax.random.uniform(logits_key, logits_shape, dtype, -1.0, 1.0)
        target = jax.random.randint(target_key, target_shape, 0, vocab_size)
        
        # Add ignore indices if requested
        if ignore_idx_positions > 0:
            ignore_positions = jax.random.randint(
                key, (ignore_idx_positions,), 0, seq_len
            )
            if swap_dim:
                target = target.at[ignore_positions, 0].set(ignore_idx)
            else:
                target = target.at[0, ignore_positions].set(ignore_idx)
        
        return logits, target
    
    @staticmethod
    def run_forward_test(
        batch: int,
        seq_len: int,
        vocab_size: int,
        dtype: jnp.dtype,
        label_smoothing: float = 0.0,
        reduce_loss: bool = True,
        swap_dim: bool = False,
        ignore_idx_positions: int = 0,
        ignore_idx: int = -100,
    ):
        """Run forward pass test"""
        key = jax.random.PRNGKey(42)
        
        # Generate inputs
        logits, target = TestCrossEntropy.generate_input(
            batch, seq_len, vocab_size, dtype, swap_dim, ignore_idx_positions, ignore_idx, key
        )
        
        # Test implementation
        test_loss = cross_entropy_loss(
            logits,
            target,
            label_smoothing=label_smoothing,
            reduce_loss=reduce_loss,
            ignore_idx=ignore_idx,
            axis_name=None,  # Single GPU
        )
        
        # Reference implementation
        ref_loss = reference_cross_entropy(
            logits, target, label_smoothing, reduce_loss, ignore_idx
        )
        
        # Compare results
        tols = dtype_tols(dtype)
        assert_allclose(test_loss, ref_loss, **tols)
    
    @staticmethod
    def run_backward_test(
        batch: int,
        seq_len: int,
        vocab_size: int,
        dtype: jnp.dtype,
        label_smoothing: float = 0.0,
        reduce_loss: bool = True,
        swap_dim: bool = False,
        ignore_idx_positions: int = 0,
        ignore_idx: int = -100,
    ):
        """Run backward pass test"""
        key = jax.random.PRNGKey(42)
        
        # Generate inputs
        logits, target = TestCrossEntropy.generate_input(
            batch, seq_len, vocab_size, dtype, swap_dim, ignore_idx_positions, ignore_idx, key
        )
        
        def loss_fn(logits_input, target_input, use_test_impl=True):
            """Wrapper for computing loss"""
            if use_test_impl:
                loss = cross_entropy_loss(
                    logits_input,
                    target_input,
                    label_smoothing=label_smoothing,
                    reduce_loss=reduce_loss,
                    ignore_idx=ignore_idx,
                    axis_name=None,
                )
            else:
                loss = reference_cross_entropy(
                    logits_input, target_input, label_smoothing, reduce_loss, ignore_idx
                )
            
            # Return scalar for gradient computation
            if reduce_loss:
                return loss
            else:
                # Square to avoid trivial backward pass (matching PyTorch test)
                return jnp.sum(jnp.square(loss))
        
        # Compute gradients for test implementation
        test_loss, test_grad = jit(value_and_grad(
            lambda x: loss_fn(x, target, use_test_impl=True)
        ))(logits)
        
        # Compute gradients for reference implementation
        ref_loss, ref_grad = jit(value_and_grad(
            lambda x: loss_fn(x, target, use_test_impl=False)
        ))(logits)
        
        # Compare results
        tols = dtype_tols(dtype)
        # Use slightly relaxed tolerances for gradients
        grad_tols = dtype_tols(dtype, rtol=tols["rtol"] * 5, atol=tols["atol"] * 5)
        
        assert_allclose(test_loss, ref_loss, **tols)
        assert_allclose(test_grad, ref_grad, **grad_tols)
    
    # Test float32 input
    @pytest.mark.parametrize("batch,seq_len,vocab_size", [
        (1, 64, 64000),
        (2, 128, 128000),
        (1, 64, 32000),
    ])
    def test_float32_forward(self, batch, seq_len, vocab_size):
        """Test forward pass with float32"""
        self.run_forward_test(
            batch=batch,
            seq_len=seq_len,
            vocab_size=vocab_size,
            dtype=jnp.float32,
            label_smoothing=0.0,
            reduce_loss=True,
        )
    
    @pytest.mark.parametrize("batch,seq_len,vocab_size", [
        (1, 64, 64000),
        (2, 128, 128000),
    ])
    def test_float32_backward(self, batch, seq_len, vocab_size):
        """Test backward pass with float32"""
        self.run_backward_test(
            batch=batch,
            seq_len=seq_len,
            vocab_size=vocab_size,
            dtype=jnp.float32,
            label_smoothing=0.0,
            reduce_loss=True,
        )
    
    # Test bfloat16 input
    @pytest.mark.parametrize("batch,seq_len,vocab_size", [
        (1, 64, 64000),
        (2, 128, 128000),
    ])
    def test_bfloat16_forward(self, batch, seq_len, vocab_size):
        """Test forward pass with bfloat16"""
        self.run_forward_test(
            batch=batch,
            seq_len=seq_len,
            vocab_size=vocab_size,
            dtype=jnp.bfloat16,
            label_smoothing=0.0,
            reduce_loss=True,
        )
    
    @pytest.mark.parametrize("batch,seq_len,vocab_size", [
        (1, 64, 64000),
        (2, 128, 128000),
    ])
    def test_bfloat16_backward(self, batch, seq_len, vocab_size):
        """Test backward pass with bfloat16"""
        self.run_backward_test(
            batch=batch,
            seq_len=seq_len,
            vocab_size=vocab_size,
            dtype=jnp.bfloat16,
            label_smoothing=0.0,
            reduce_loss=True,
        )
    
    # Test swapped dimensions
    @pytest.mark.parametrize("batch,seq_len,vocab_size", [
        (1, 64, 64000),
        (2, 128, 32000),
    ])
    def test_swapped_input_forward(self, batch, seq_len, vocab_size):
        """Test forward pass with swapped dimensions (SQ, B, V)"""
        self.run_forward_test(
            batch=batch,
            seq_len=seq_len,
            vocab_size=vocab_size,
            dtype=jnp.float32,
            label_smoothing=0.0,
            reduce_loss=True,
            swap_dim=True,
        )
    
    @pytest.mark.parametrize("batch,seq_len,vocab_size", [
        (1, 64, 64000),
    ])
    def test_swapped_input_backward(self, batch, seq_len, vocab_size):
        """Test backward pass with swapped dimensions (SQ, B, V)"""
        self.run_backward_test(
            batch=batch,
            seq_len=seq_len,
            vocab_size=vocab_size,
            dtype=jnp.float32,
            label_smoothing=0.0,
            reduce_loss=True,
            swap_dim=True,
        )
    
    # Test label smoothing
    @pytest.mark.parametrize("label_smoothing", [0.1, 0.2])
    @pytest.mark.parametrize("batch,seq_len,vocab_size", [
        (1, 64, 32000),
    ])
    def test_label_smoothing_forward(self, label_smoothing, batch, seq_len, vocab_size):
        """Test forward pass with label smoothing"""
        self.run_forward_test(
            batch=batch,
            seq_len=seq_len,
            vocab_size=vocab_size,
            dtype=jnp.float32,
            label_smoothing=label_smoothing,
            reduce_loss=True,
        )
    
    @pytest.mark.parametrize("label_smoothing", [0.1])
    @pytest.mark.parametrize("batch,seq_len,vocab_size", [
        (1, 64, 32000),
    ])
    def test_label_smoothing_backward(self, label_smoothing, batch, seq_len, vocab_size):
        """Test backward pass with label smoothing"""
        self.run_backward_test(
            batch=batch,
            seq_len=seq_len,
            vocab_size=vocab_size,
            dtype=jnp.float32,
            label_smoothing=label_smoothing,
            reduce_loss=True,
        )
    
    # Test non-reduced loss
    @pytest.mark.parametrize("batch,seq_len,vocab_size", [
        (1, 64, 32000),
        (2, 64, 64000),
    ])
    def test_non_reduced_loss_forward(self, batch, seq_len, vocab_size):
        """Test forward pass with non-reduced loss"""
        self.run_forward_test(
            batch=batch,
            seq_len=seq_len,
            vocab_size=vocab_size,
            dtype=jnp.float32,
            label_smoothing=0.0,
            reduce_loss=False,
        )
    
    @pytest.mark.parametrize("batch,seq_len,vocab_size", [
        (1, 64, 32000),
    ])
    def test_non_reduced_loss_backward(self, batch, seq_len, vocab_size):
        """Test backward pass with non-reduced loss"""
        self.run_backward_test(
            batch=batch,
            seq_len=seq_len,
            vocab_size=vocab_size,
            dtype=jnp.float32,
            label_smoothing=0.0,
            reduce_loss=False,
        )
    
    # Test ignore index
    @pytest.mark.parametrize("batch,seq_len,vocab_size", [
        (1, 64, 32000),
    ])
    @pytest.mark.parametrize("ignore_idx_positions", [5, 10])
    def test_ignore_idx_forward(self, batch, seq_len, vocab_size, ignore_idx_positions):
        """Test forward pass with ignore index"""
        self.run_forward_test(
            batch=batch,
            seq_len=seq_len,
            vocab_size=vocab_size,
            dtype=jnp.float32,
            label_smoothing=0.0,
            reduce_loss=False,
            ignore_idx_positions=ignore_idx_positions,
            ignore_idx=-100,
        )
    
    @pytest.mark.parametrize("batch,seq_len,vocab_size", [
        (1, 64, 32000),
    ])
    @pytest.mark.parametrize("ignore_idx_positions", [5])
    def test_ignore_idx_backward(self, batch, seq_len, vocab_size, ignore_idx_positions):
        """Test backward pass with ignore index"""
        self.run_backward_test(
            batch=batch,
            seq_len=seq_len,
            vocab_size=vocab_size,
            dtype=jnp.float32,
            label_smoothing=0.0,
            reduce_loss=False,
            ignore_idx_positions=ignore_idx_positions,
            ignore_idx=-100,
        )

