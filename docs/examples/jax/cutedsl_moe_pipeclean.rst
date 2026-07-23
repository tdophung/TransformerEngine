CuTeDSL MXFP8 MoE pipeclean
===========================

The JAX MoE path has an opt-in Blackwell fusion for FC1 grouped GEMM,
SwiGLU, and rowwise/colwise MXFP8 quantization. The forward leaf is compiled
from abstract tensor descriptors by cuDNN-FE and invoked as a native
``tvm_ffi.Function``. The surrounding EP path continues to use ``shard_map``
and sees only shard-local tensors. The fused dSwiGLU backward leaf continues
to use CUTLASS DSL's JAX integration.

Environment baseline
--------------------

Use the versions in ``tests/jax/requirements_cutedsl.txt`` on an SM100 CUDA
host and install the sibling cuDNN-FE checkout with its compile-only extra.
The end-to-end driver installs both editable trees, runs standalone forward
and backward parity, exercises custom partitioning and ``shard_map`` across
two processes, and runs the four-GPU MoEBlock integration test::

   CUDNN_FE_ROOT=/path/to/cudnn-frontend NUM_GPUS=4 \
       bash tests/jax/run_tvm_ffi_fused_moe_e2e.sh

If ``CUDNN_FE_ROOT`` does not exist, the driver clones the pinned NVIDIA
upstream revision and applies the bundled compile-only patch. An existing
checkout that already provides the API is left unchanged. Set
``INSTALL_DEPS=0`` to reuse an existing environment. The EP launcher currently
requires four or more ranks even though the EP mesh itself uses groups of two
ranks. Logs from every stage are retained in the timestamped artifact directory
printed by the driver.

Support and fallback contract
-----------------------------

The default value of ``NVTE_JAX_MOE_USE_CUDNN_CUTEDSL_FUSION`` is ``0`` and
does not import the optional TVM-FFI or CUTLASS DSL packages. Existing
unfused runs therefore remain supported when the optional packages or SM100
hardware are absent. Explicit opt-in validates the GPU architecture, SwiGLU
shapes and biases, MXFP8 quantizers, JAX FFI availability, and both compiler
paths before lowering. Unsupported configurations emit one warning with the
complete validation list and use the unfused implementation.

The cuDNN-FE compile-only API is exported from::

   cudnn/grouped_gemm/grouped_gemm_swiglu/compile.py

It accepts self-describing operand metadata, compiles without Torch or live
buffers, and returns a native function plus an exact ABI descriptor. The
native launch wrapper reorders the eight arguments, five results, and stream
into the volatile raw kernel ABI. This is necessary because ``jax-tvm-ffi``
currently describes only complete argument and result groups. The fused path
uses 256-token dispatch alignment; the unfused path remains at 128.

Custom partitioning migration design
------------------------------------

The standalone single-expert primitive now has a ``custom_partitioning``
wrapper and is tested against an explicit ``shard_map`` path. Integrated
multi-expert MoE remains inside the established ``shard_map`` boundary.
The five kernel outputs are:

* combined projection: ``[tokens, 2 * intermediate, 1]``
* rowwise MXFP8 payload: ``[tokens, intermediate, 1]``
* colwise MXFP8 payload: ``[tokens, intermediate, 1]``
* rowwise inverse scales: one flat buffer
* colwise inverse scales: one flat buffer

The draft Shardy factors are ``tokens`` (sharded by the caller's DP/EP token
axes), ``experts`` (sharded by EP), ``hidden``, ``intermediate``, distinct
singleton factors for each singleton dimension, and a constant factor
``swiglu_pair=2``. Inputs map as follows:

* A ``[M,K,1]``: ``(tokens, hidden, a_l)``
* B ``[E,2I,K]``: ``(experts, intermediate*swiglu_pair, hidden)``
* padded offsets ``[E]``: ``(experts,)``
* probability ``[M,1,1]``: ``(tokens, prob_n, prob_l)``
* pre-swizzled A/B scales: opaque until their JAX boundary is structured

The first three outputs inherit the ``tokens`` factor and otherwise remain
local. Scale buffers remain opaque flat native-ABI values at lowering time.
The standalone partitioner can shard them along the data axis because every
shard compiles a complete local scale buffer. Multi-expert production
partitioning stays below ``shard_map`` so expert-local padded offsets and
pre-swizzled scale layouts are preserved without exposing an invalid global
flat-buffer factor to Shardy.
