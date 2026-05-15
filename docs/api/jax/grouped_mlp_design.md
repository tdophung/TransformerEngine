# `grouped_mlp` design notes

Developer-facing rationale behind
`transformer_engine/jax/grouped_mlp.py`. End users do not need to
read this; it documents the design decisions behind the single-VJP
shape and the `# TODO` markers in the source.

## Why a single `jax.custom_vjp` (the "gigantic VJP")

`jax.custom_vjp` requires the inputs and outputs of the wrapped
function to be ordinary JAX-traced types (`jnp.ndarray`s, possibly
inside a pytree).

What flows between a TE quantize op and the GEMM that consumes it is
not a `jnp.ndarray`. It is a `ScaledTensor1x` / `ScaledTensor2x`
(`GroupedScaledTensor1x` for the grouped path) wrapper class that
carries:

* the FP8 (or MXFP8) data buffer,
* the per-tensor or per-block `scale_inv` (in `e8m0fnu` for MXFP8),
* a `data_layout` flag (rowwise vs colwise),
* often *both* the rowwise (`LHS`) and colwise (`RHS_TRANS`) views of
  the same source tensor packed together (`ScaledTensor2x`), which
  the GEMM picks via `get_tensor(TensorUsage.LHS / RHS / LHS_TRANS /
  RHS_TRANS)`.

These wrappers can be pytree-registered, but the moment they cross a
`custom_vjp` boundary you have handed JAX a thing whose AD rules
(transpose, AD-cast etc.) the framework does not know about. The TE
convention is therefore: **do not let `ScaledTensor` wrappers cross a
`custom_vjp` boundary; keep both producer and consumer inside the same
VJP.**

That is why `_layernorm_mlp` packs LayerNorm + GEMM1 + activation +
GEMM2 into one VJP and why `grouped_mlp` does the analogous thing for
the two grouped GEMMs and the activation between them.

### Concretely, what the single VJP buys

1. **Save quantized residuals straight into `ctx`.** `_grouped_mlp_fwd_rule`
   stashes the colwise (`LHS_TRANS` / `RHS_TRANS`) views of every
   quantized operand into the bwd context. The bwd's dgrad/wgrad
   GEMMs consume the *exact* FP8 buffers — and the *exact* amax
   values — that the fwd used. Splitting the block across multiple
   VJPs would force each bwd rule to re-quantize from bf16 with a
   fresh amax, paying both runtime cost and a small numerical drift
   between the fwd kernel and the bwd kernel.

2. **Single LHS quantize + single grouped GEMM up to the activation
   ("fused MLP" layout).** GEMM1 takes a single fused `kernel_1` of
   shape `(G, K, activation_len, N)` and produces a single output of
   shape `(M, activation_len, N)`. The activation function then
   reduces along the `activation_len` axis (e.g. SwiGLU =
   `silu(out[..., 0, :]) * out[..., 1, :]`). This matches Maxtext's
   `fused_mlp` knob and the layout that `layernorm_mlp` already
   uses for `kernel_1`. Without the single VJP, you cannot stack
   `wi_0` / `wi_1` into one kernel and still have a clean
   per-call quantizer set, because each `grouped_dense` `custom_vjp`
   owns its own `quantizer_set`.

3. **Future fusion of activation with quantization.** `tex.act_lu` on
   the non-grouped path takes a bf16 GEMM output, applies the
   multi-component activation, and writes the FP8 quantized output in
   one FFI call — the bf16 intermediate never materialises in HBM.
   The grouped equivalent (`grouped_act_lu`) does not exist yet;
   today the activation runs in Python between two separate FFI
   calls. The TODO marker in `_grouped_mlp_fwd_rule` (and its
   counterpart in `_grouped_mlp_bwd_rule` for `quantize_dact_dbias`)
   points at the exact site where it lands. Crucially, this fusion
   is *only reachable* because both endpoints (the GEMM1 output and
   the GEMM2 input) live inside the same VJP — across a VJP
   boundary the intermediate would have to be a plain `jnp.ndarray`
   anyway, defeating the point of the fused FFI.

## Why `.checkpoint(quantizer)` on the `ctx` residuals

`tex.grouped_quantize` (and `tex.quantize`) in 2x mode produces a
`ScaledTensor2x` containing both rowwise (`LHS` / `RHS`) and colwise
(`LHS_TRANS` / `RHS_TRANS`) views of the same source tensor in one
FFI call. The fwd consumes the rowwise view in its GEMMs and saves
the colwise view into `ctx`.

When the surrounding code wraps the call in `jax.checkpoint` /
`jax.remat` (gradient checkpointing), JAX re-runs the fwd to
recompute saved-values it dropped. If the colwise residual in `ctx`
is left as a plain `ScaledTensor`, the remat tracer cannot tell that
it was already produced by the original quantize FFI and re-emits
the whole quantize FFI a second time. Calling
`tensor.checkpoint(quantizer)` registers the tensor's provenance with
JAX so the remat sees it as already-materialised and DCEs the
redundant call.

`_layernorm_mlp_fwd_rule` checkpoints only the `ctx`-bound colwise
residuals. `_grouped_mlp_fwd_rule` follows the same convention for
consistency. (`_grouped_dense_fwd_rule` additionally checkpoints the
rowwise inputs because its single-GEMM VJP must let the
`te_grouped_quantize_ffi` get DCE'd in the backward-scan remat block;
inside `_grouped_mlp_fwd_rule` the rowwise tensors are consumed by
GEMMs whose outputs are already traced through to `dot_2_output`, so
remat keeps them alive for free.)

The call is **unconditional** -- there is no `isinstance(tensor,
ScaledTensor)` guard around it. In the no-quantization path
`noop_quantizer_set.x` is `None`, so `tex.grouped_quantize` returns a
`GroupedNoScaleTensor` whose `.checkpoint(None)` is a safe no-op
(`assert quantizer is None; return self`). In the quantized path the
returned tensor is a `GroupedScaledTensor1x` whose `.checkpoint(real_quantizer)`
either applies the checkpoint name or returns `self` if no checkpoint
name is configured. Either way the call is safe; the `isinstance`
guard found in `_grouped_dense_fwd_rule` is leftover defensive style,
not load-bearing.

## Distributed / parallelism handling

`grouped_mlp` does **not** open a `shard_map`, perform a
`ragged_all_to_all`, or otherwise reach for any expert-parallel
collective. Those concerns live one level up (today: the EP wrapper
inside `MoEBlock`). Inside `grouped_mlp` everything is local to a
shard.

* **FSDP / DP / TP.** Sharding propagates through the
  `custom_partitioning` rules of `tex.grouped_quantize` /
  `tex.grouped_gemm`. As long as the caller hands sharded inputs and
  weights with the right `NamedSharding`, the grouped GEMMs gather /
  scatter their operands and emit the matching wgrad
  reduce-scatter on their own. No `lax.all_gather` or
  `lax.psum_scatter` lives inside `_grouped_mlp_fwd_rule` or
  `_grouped_mlp_bwd_rule`.

* **Quantize-before-FSDP-AG (QB4AG).** A separate optimisation that
  *replaces* the bf16 FSDP all-gather of each kernel with a per-shard
  quantize + FP8 all-gather. It needs explicit `lax.all_gather`
  inside the fwd rule and `lax.psum_scatter` inside the bwd rule, and
  is being landed on its own branch. The `grouped_mlp` API kept here
  is the natural integration target for that work — it can grow a
  single `kernel_fsdp_info_per_kernel` argument and centralise the
  AG / scatter calls in one place, instead of having to thread the
  knob through three separate `grouped_dense` invocations.

* **Distributed test.** `tests/jax/test_distributed_grouped_mlp.py`
  ships in this PR. It mirrors `tests/jax/test_distributed_layernorm_mlp.py`:
  a single-GPU `value_and_grad` reference is compared against the
  same function compiled on an FSDP / TPSP mesh with kernels sharded
  as `kernel_1: P(None, fsdp, None, tpsp)` /
  `kernel_2: P(None, tpsp, fsdp)`. This is what proves the claim
  above (no partitioning rule in the `custom_vjp` shell + correct
  wgrad reduce-scatter through the underlying primitives) for every
  recipe the platform supports.

## Roadmap (`# TODO` markers in the source)

The single-VJP shape unlocks three follow-up fusions, each tagged
with a `# TODO` comment at the site it touches:

1. **Fused grouped activation + quantize.** Replace `_apply_activation`
   + `tex.grouped_quantize(act_out, ...)` in fwd, and `_apply_dactivation`
   + `tex.grouped_quantize(dact_out, ...)` in bwd, with one fused
   FFI each. Expected to drop the bf16 activation intermediate from
   HBM entirely.

2. **`kernel_fsdp_info_per_kernel`.** Add the QB4AG entry point and
   centralise the per-kernel `lax.all_gather` / `lax.psum_scatter`
   inside this VJP. Mechanical once (1) lands and the QB4AG branch
   is rebased.

3. **Single LHS `grouped_quantize` for stacked downstream consumers.**
   Already half-realised by the fused-MLP layout: GEMM1 is now a
   single GEMM so its LHS is quantized once. The remaining
   opportunity is to share `ScaledTensor2x` views across GEMMs in
   future variants that re-introduce parallel branches (e.g. a
   bias-routing branch).
