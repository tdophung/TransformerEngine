# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""JAX binding for cuDNN frontend's MXFP8 grouped-GEMM SwiGLU kernel.

The kernel is intentionally imported from the installed
``nvidia-cudnn-frontend`` distribution.  The public cuDNN frontend package
initializers import the Torch API wrappers eagerly, so this module loads the
kernel source directly under a private namespace.  No kernel implementation is
vendored into Transformer Engine.
"""

from __future__ import annotations

from functools import lru_cache
import importlib.metadata
import importlib.util
import sys
import threading
import types
from typing import Any

import jax
import jax.numpy as jnp


_CUDNN_FRONTEND_DISTRIBUTION = "nvidia-cudnn-frontend"
_PRIVATE_PACKAGE = "_transformer_engine_cudnn_grouped_gemm"
_LOAD_LOCK = threading.Lock()


def _namespace_package(name: str, path) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__package__ = name
    module.__path__ = [str(path)]
    sys.modules[name] = module
    return module


def _load_source_module(name: str, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create an import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


@lru_cache(maxsize=1)
def load_grouped_gemm_swiglu_kernel():
    """Load the cuDNN frontend kernel class without importing its Torch API.

    cuDNN frontend 1.25's shared grouped-GEMM utility module imports ``torch``
    solely for a type annotation and an FP4 API helper.  The MXFP8 SwiGLU
    kernel uses neither.  When Torch is absent, a typing-only ``Tensor``
    sentinel lets that unmodified pip source execute. It remains registered
    because CuTeDSL's AST preprocessor replays source imports during deferred
    kernel compilation; no Torch operation is executed.
    """
    with _LOAD_LOCK:
        distribution = importlib.metadata.distribution(_CUDNN_FRONTEND_DISTRIBUTION)
        grouped_root = distribution.locate_file("cudnn/grouped_gemm")
        utils_path = grouped_root / "utils.py"
        swiglu_root = grouped_root / "grouped_gemm_swiglu"
        kernel_path = swiglu_root / "grouped_gemm_swiglu_quant.py"
        if not utils_path.is_file() or not kernel_path.is_file():
            raise ImportError(
                f"{_CUDNN_FRONTEND_DISTRIBUTION} {distribution.version} does not contain "
                "the grouped_gemm_swiglu CuTeDSL kernel sources"
            )

        _namespace_package(_PRIVATE_PACKAGE, grouped_root)
        swiglu_package = f"{_PRIVATE_PACKAGE}.grouped_gemm_swiglu"
        _namespace_package(swiglu_package, swiglu_root)

        # Do not shadow a real Torch installation. If it is installed, the
        # unmodified utility source can import it normally; the sentinel is
        # only for JAX-only environments where no Torch module exists.
        try:
            importlib.metadata.distribution("torch")
            torch_installed = True
        except importlib.metadata.PackageNotFoundError:
            torch_installed = False
        torch_stubbed = "torch" not in sys.modules and not torch_installed
        if torch_stubbed:
            torch_stub = types.ModuleType("torch")
            torch_stub.Tensor = object
            torch_stub.float4_e2m1fn_x2 = object()
            torch_stub.__transformer_engine_cutedsl_stub__ = True
            sys.modules["torch"] = torch_stub
        try:
            _load_source_module(f"{_PRIVATE_PACKAGE}.utils", utils_path)
            kernel_module = _load_source_module(
                f"{swiglu_package}.grouped_gemm_swiglu_quant", kernel_path
            )
        except Exception:
            if torch_stubbed:
                sys.modules.pop("torch", None)
            raise

        try:
            return kernel_module.BlockScaledContiguousGroupedGemmKernel
        except AttributeError as exc:
            raise ImportError(
                f"{_CUDNN_FRONTEND_DISTRIBUTION} {distribution.version} has an "
                "incompatible grouped-GEMM SwiGLU kernel API"
            ) from exc


def pack_swiglu_pair(gate: jax.Array, up: jax.Array) -> jax.Array:
    """Interleave 32-column gate/up blocks as required by the kernel."""
    if gate.shape != up.shape:
        raise ValueError(f"gate shape {gate.shape} must match up shape {up.shape}")
    if gate.shape[-1] % 32:
        raise ValueError(f"SwiGLU intermediate dimension {gate.shape[-1]} must be divisible by 32")
    blocks = gate.shape[-1] // 32
    return jnp.stack(
        (
            gate.reshape(*gate.shape[:-1], blocks, 32),
            up.reshape(*up.shape[:-1], blocks, 32),
        ),
        axis=-2,
    ).reshape(*gate.shape[:-1], 2 * gate.shape[-1])


def unpack_swiglu_pair(interleaved: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Undo :func:`pack_swiglu_pair`."""
    if interleaved.shape[-1] % 64:
        raise ValueError(
            f"Interleaved SwiGLU dimension {interleaved.shape[-1]} must be divisible by 64"
        )
    intermediate = interleaved.shape[-1] // 2
    blocks = intermediate // 32
    paired = interleaved.reshape(*interleaved.shape[:-1], blocks, 2, 32)
    return (
        paired[..., 0, :].reshape(*interleaved.shape[:-1], intermediate),
        paired[..., 1, :].reshape(*interleaved.shape[:-1], intermediate),
    )


@lru_cache(maxsize=None)
def _make_launcher(
    expert_count: int,
    sf_vec_size: int,
    mma_tiler_m: int,
    mma_tiler_n: int,
):
    try:
        import cutlass
        from cutlass import cute
    except ImportError as exc:
        raise ImportError(
            "NVTE_JAX_MOE_USE_CUDNN_CUTEDSL_FUSION=1 requires nvidia-cutlass-dsl"
        ) from exc

    kernel_cls = load_grouped_gemm_swiglu_kernel()
    use_2cta_instrs = mma_tiler_m == 256
    cluster_shape = (2, 1) if use_2cta_instrs else (1, 1)
    kernel = kernel_cls(
        sf_vec_size=sf_vec_size,
        acc_dtype=cutlass.Float32,
        use_2cta_instrs=use_2cta_instrs,
        mma_tiler_mn=(mma_tiler_m, mma_tiler_n),
        cluster_shape_mn=cluster_shape,
        vector_f32=False,
        generate_sfd=False,
        discrete_col_sfd=False,
        expert_cnt=expert_count,
        use_mono_increase_expert_idx=True,
    )
    max_active_clusters = cutlass.utils.HardwareInfo().get_max_active_clusters(
        cluster_shape[0] * cluster_shape[1]
    )

    @cute.jit
    def launch(
        stream,
        a,
        b,
        sfa,
        sfb,
        padded_offsets,
        alpha,
        prob,
        c,
        d,
        d_col,
    ):
        kernel(
            a,
            b,
            c,
            d,
            d_col,
            sfa,
            sfb,
            None,
            None,
            None,
            None,
            padded_offsets,
            alpha,
            prob,
            max_active_clusters,
            stream,
        )

    return launch


def grouped_gemm_swiglu_mxfp8(
    a: jax.Array,
    b: jax.Array,
    sfa: jax.Array,
    sfb: jax.Array,
    padded_offsets: jax.Array,
    prob: jax.Array,
    *,
    compute_dtype: Any,
    output_dtype: Any,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Call cuDNN frontend's grouped MXFP8 GEMM+SwiGLU kernel.

    Args:
        a: Quantized activation payload with physical shape ``[M, K, 1]``.
        b: Quantized, block-interleaved colwise weights with physical shape
            ``[E, N, K]``.
        sfa/sfb: Pre-swizzled E8M0 inverse-scale buffers.
        padded_offsets: Exclusive, 256-aligned end offset for each expert.
        prob: Per-row multiplier, physical shape ``[M, 1, 1]``.
        compute_dtype: Unquantized combined-projection dtype.
        output_dtype: Unquantized SwiGLU output dtype. The caller applies
            TE's grouped output quantizer so its backward ABI is preserved.

    Returns:
        Raw combined projection, SwiGLU output, and the kernel's secondary
        output buffer (unused by the MoE integration).
    """
    try:
        from cutlass.jax import TensorSpec, cutlass_call
    except ImportError as exc:
        raise ImportError(
            "NVTE_JAX_MOE_USE_CUDNN_CUTEDSL_FUSION=1 requires CUTLASS JAX bindings"
        ) from exc

    if a.ndim != 3 or a.shape[-1] != 1:
        raise ValueError(f"Expected A[M,K,1], got {a.shape}")
    if b.ndim != 3:
        raise ValueError(f"Expected physical B[E,N,K], got {b.shape}")
    expert_count, n, k_b = b.shape
    m, k_a, _ = a.shape
    if k_a != k_b:
        raise ValueError(f"A K={k_a} does not match B K={k_b}")
    if n % 2:
        raise ValueError(f"Combined SwiGLU N={n} must be even")
    intermediate = n // 2

    # The public kernel requires 256-aligned expert ranges.  M is static at
    # trace time, while the individual offsets remain runtime values.
    if m % 256:
        raise ValueError(f"Padded activation rows M={m} must be divisible by 256")

    sf_vec_size = 32
    outputs = (
        jax.ShapeDtypeStruct((m, n, 1), compute_dtype),
        jax.ShapeDtypeStruct((m, intermediate, 1), output_dtype),
        jax.ShapeDtypeStruct((m, intermediate, 1), output_dtype),
    )
    launcher = _make_launcher(expert_count, sf_vec_size, 256, 256)
    call = cutlass_call(
        launcher,
        output_shape_dtype=outputs,
        input_spec=(
            # Singleton L must be outermost so K, rather than L, is the
            # leading (stride-1) dimension seen by the kernel.
            TensorSpec(layout=(1, 0, 2)),
            # physical colwise [E,N,K] -> logical K-major [N,K,E]
            TensorSpec(mode=(1, 2, 0)),
            TensorSpec(),
            TensorSpec(),
            TensorSpec(),
            TensorSpec(),
            TensorSpec(),
        ),
        output_spec=(
            TensorSpec(layout=(1, 0, 2)),
            TensorSpec(layout=(1, 0, 2)),
            TensorSpec(layout=(1, 0, 2)),
        ),
        allow_cuda_graph=True,
    )
    alpha = jnp.ones((expert_count,), dtype=jnp.float32)
    return call(
        a,
        b,
        sfa.reshape(-1),
        sfb.reshape(-1),
        padded_offsets.astype(jnp.int32),
        alpha,
        prob.astype(jnp.float32),
    )


__all__ = [
    "grouped_gemm_swiglu_mxfp8",
    "load_grouped_gemm_swiglu_kernel",
    "pack_swiglu_pair",
    "unpack_swiglu_pair",
]
