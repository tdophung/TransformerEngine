# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Standalone CuTeDSL/JAX toolchain smoke test.

Run this before the TE tests. It verifies CUTLASS FFI runtime discovery and
executes a trivial CuTeDSL kernel from inside ``jax.jit``.
"""

from importlib.metadata import version

import cuda.bindings.driver as cuda
import cutlass.cute as cute
import cutlass.jax as cjax
import jax
import jax.numpy as jnp
import numpy as np


BLOCK = 256


@cute.kernel
def _vector_add_kernel(a: cute.Tensor, b: cute.Tensor, c: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()

    frag_a = cute.make_rmem_tensor(cute.size(a, mode=[0]), a.element_type)
    frag_b = cute.make_rmem_tensor(cute.size(b, mode=[0]), b.element_type)
    frag_c = cute.make_rmem_tensor(cute.size(c, mode=[0]), c.element_type)
    cute.autovec_copy(a[None, tidx, bidx], frag_a)
    cute.autovec_copy(b[None, tidx, bidx], frag_b)
    frag_c.store(frag_a.load() + frag_b.load())
    cute.autovec_copy(frag_c, c[None, tidx, bidx])


@cute.jit
def _launch_vector_add(
    stream: cuda.CUstream,
    a: cute.Tensor,
    b: cute.Tensor,
    c: cute.Tensor,
):
    _vector_add_kernel(a, b, c).launch(
        grid=[a.shape[-1], 1, 1],
        block=[a.shape[-2], 1, 1],
        stream=stream,
    )


@jax.jit
def _cutlass_add(a, b):
    size = a.shape[0]
    padded_size = ((size + BLOCK - 1) // BLOCK) * BLOCK
    a_3d = jnp.pad(a, (0, padded_size - size)).reshape(1, BLOCK, -1)
    b_3d = jnp.pad(b, (0, padded_size - size)).reshape(1, BLOCK, -1)
    call = cjax.cutlass_call(
        _launch_vector_add,
        output_shape_dtype=jax.ShapeDtypeStruct.like(a_3d),
        use_static_tensors=True,
    )
    return call(a_3d, b_3d).reshape(-1)[:size]


def main() -> None:
    if not cjax.is_available():
        raise RuntimeError(
            "cutlass.jax.is_available() is false; verify cute_dsl_runtime.so discovery"
        )
    devices = jax.devices("gpu")
    if not devices:
        raise RuntimeError("No JAX GPU device is available")

    a = jnp.arange(1024, dtype=jnp.float32)
    b = jnp.arange(1024, dtype=jnp.float32) * 2
    actual = _cutlass_add(a, b)
    actual.block_until_ready()
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(a + b))
    print(
        "CuTeDSL JAX smoke passed: "
        f"jax={jax.__version__}, cutlass={version('nvidia-cutlass-dsl')}, device={devices[0]}"
    )


if __name__ == "__main__":
    main()
