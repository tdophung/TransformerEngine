CuTeDSL MXFP8 MoE pipeclean
===========================

The JAX MoE path has an opt-in Blackwell fusion for FC1 grouped GEMM,
SwiGLU, and rowwise/colwise MXFP8 quantization. All CuTe-specific code lives
in ``transformer_engine/jax/cutedsl_extensions``; the surrounding EP path
continues to use ``shard_map`` and sees only shard-local tensors.

Environment baseline
--------------------

Use the versions in ``tests/jax/requirements_cutedsl.txt`` on an SM100 CUDA
host. Build Transformer Engine with JAX and NCCL EP support, then run::

   python3 tests/jax/cutedsl_smoke.py
   python3 -m pytest -c tests/jax/pytest.ini tests/jax/test_cutedsl_moe.py -v
   bash tests/jax/run_te_ep_moe.sh
   NVTE_JAX_MOE_USE_CUDNN_CUTEDSL_FUSION=1 \
   bash tests/jax/run_te_ep_moe.sh -k TestTeEpMoeCudnnCutedslFusion

The first command is intentionally independent of Transformer Engine. It
must report ``cutlass.jax.is_available()`` and execute its vector-add kernel
inside ``jax.jit`` before failures in the fused MoE tests are investigated.
The EP launcher currently requires four or more ranks even though the EP
mesh itself uses groups of two ranks. The CuTeDSL opt-in is strict, so run
only the CuTeDSL fusion class with that environment variable enabled.

Support and fallback contract
-----------------------------

The default value of ``NVTE_JAX_MOE_USE_CUDNN_CUTEDSL_FUSION`` is ``0`` and
never imports CUTLASS DSL. Existing unfused runs therefore remain supported
when the optional packages or SM100 hardware are absent. Explicit opt-in
validates the GPU architecture, SwiGLU shapes and biases, MXFP8 quantizers,
JAX/CUTLASS FFI availability, and the cuDNN frontend source layout before
kernel lowering. An unsupported explicit opt-in fails with the complete validation
list instead of failing later during kernel compilation.

The pinned cuDNN frontend layout is::

   cudnn/grouped_gemm/utils.py
   cudnn/grouped_gemm/grouped_gemm_swiglu/grouped_gemm_swiglu_quant.py

and must export ``BlockScaledContiguousGroupedGemmKernel``. The fused path
uses 256-token dispatch alignment; the unfused path remains at 128.

Custom partitioning migration design
------------------------------------

Do not add a ``custom_partitioning`` wrapper until nvbug 6432162 is fixed.
Today the five kernel outputs are:

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
local. The two scale outputs must eventually be exposed as structured block
layouts carrying ``tokens``/``intermediate`` (and the per-expert padding
factor where applicable), then flattened only inside ``cutlass_call``. Do
not assign a Shardy factor to the current giant flat dimension: that is the
failure mode tracked by nvbug 6432162. Until structured scales and the bug
fix are both available, the leaf stays under ``shard_map`` and needs no
partitioning rule.

``tests/jax/repro_cutedsl_moe_shardy.py`` lowers a shape-only custom
partitioning leaf with the production EP2/FSDP2 output shapes. Run it before
starting the migration; it should lower successfully after the Shardy fix.
After that gate passes, replace the shape-only leaf with
``grouped_gemm_swiglu_mxfp8``, structure both scale outputs, add the rule
above, and compare its shardings and numerics with the existing ``shard_map``
tests before removing ``shard_map``.
