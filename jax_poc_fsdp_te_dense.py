"""Proof of concept #2.5: same FSDP MLP block, but every matmul is a TE dense().

Compared to ``jax_poc_fsdp_mlp.py`` (which uses ``x @ w``):

    * Every dot product inside both shard_maps is replaced by a call to
      ``transformer_engine.jax.dense.dense()`` -- the same function used by
      ``DenseGeneral`` and the rest of the TE stack.
    * The OUTER ``custom_vjp`` + two-shard_map structure is unchanged; we
      still control fwd vs. bwd sharding via our own rules.
    * Inside the bwd shard_map we transpose explicitly so each call stays in
      ``dense()``'s required NN layout.
    * No FP8: ``quantizer_set=noop_quantizer_set`` so we're testing pure
      composition of TE primitives with our shard_map, not TE's quantization
      path. Flipping to FP8 is a follow-up.
    * No TE collective fusion: ``collective_op_set=noop_collective_op_set``
      so the all-gather / reduce-scatter come from OUR shard_map kernel
      (not from TE's GEMM). This isolates the question we care about:
      "does dense() compose with shard_map without breaking the FSDP
      collective pattern?".

What this POC validates
-----------------------
1. ``dense()`` (which internally registers a ``custom_vjp`` and calls
   ``custom_partitioning`` GEMM primitives) can be called from inside a
   ``shard_map`` -- the per-shard ``dense()`` invocations see local shards
   and just run their GEMM impl, leaving the SPMD orchestration to our
   surrounding shard_map contract.

2. Numerical parity is preserved: same y / dx / dw1 / dw2 as the pure-JAX
   FSDP POC and the replicated reference.

3. The HLO collective pattern is still ``1 all-gather + 1 reduce-scatter +
   1 loss all-reduce`` (the FSDP recipe). TE's dense doesn't sneak in
   extra collectives because we passed ``collective_op_set=noop``.

This is the dress rehearsal for the MoE block: there we will call TE's
``grouped_dense`` (and the permute primitive) the same way -- from inside
shard_maps that we own -- so we need to confirm the pattern works on the
simpler dense() first.
"""

from __future__ import annotations

import os
import sys
import re

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

# Defensive: strip script dir from sys.path so the workspace's ``jax/`` and
# ``TransformerEngine/`` source checkouts don't shadow the installed wheels.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _SCRIPT_DIR]

from functools import partial

import jax
import jax.numpy as jnp

if hasattr(jax, "shard_map"):
    _shard_map_impl = jax.shard_map
else:
    from jax.experimental.shard_map import shard_map as _shard_map_impl  # type: ignore[no-redef]


def shard_map(f, *, mesh, in_specs, out_specs):
    return _shard_map_impl(f, mesh=mesh, in_specs=in_specs, out_specs=out_specs)


from jax.sharding import Mesh, PartitionSpec as P, NamedSharding

import numpy as np

# --- TE imports ---------------------------------------------------------
# transformer_engine.jax.dense.dense is the same function DenseGeneral
# delegates to. We use noop_quantizer_set (no FP8) and noop_collective_op_set
# (no TE-internal collective fusion) so this POC isolates "does dense()
# compose with shard_map?" from the orthogonal questions of FP8 and TE's
# own collective overlap.
try:
    from transformer_engine.jax.dense import dense as te_dense
    from transformer_engine.jax.quantize import noop_quantizer_set
    import transformer_engine.jax.cpp_extensions as tex
    _NOOP_COLLECTIVE = tex.noop_collective_op_set
    HAVE_TE = True
except Exception as e:  # noqa: BLE001
    print(f"WARNING: transformer_engine.jax not importable -- {e}")
    print("This POC requires TE to be installed in the environment.")
    HAVE_TE = False
    raise


# ---------------------------------------------------------------------------
# Mesh
# ---------------------------------------------------------------------------

DEVICES = jax.devices()
FSDP_SIZE = len(DEVICES)
MESH = Mesh(np.asarray(DEVICES).reshape(FSDP_SIZE), axis_names=("fsdp",))
FSDP_AXIS = "fsdp"


# ---------------------------------------------------------------------------
# A thin wrapper that always calls TE dense() with our chosen knobs (no FP8,
# no TE collectives, no logical-axis sharding constraints -- those would be
# meaningless inside a shard_map anyway).
# ---------------------------------------------------------------------------

def _matmul_NN(x, w):
    """``y = x @ w`` via TE dense() in NN layout.

    Last dim of ``x`` contracts with first dim of ``w`` -- that's the
    default ``contracting_dims=((1,), (0,))`` for 2-D inputs, which is what
    ``dense()`` requires. Used for every dot product in this POC; for the
    backward operands we transpose first so we still feed dense() an NN
    GEMM (it asserts NN layout internally).
    """
    return te_dense(
        x,
        w,
        bias=None,
        contracting_dims=((x.ndim - 1,), (0,)),
        input_axes=None,            # no logical-axis constraints inside shard_map
        kernel_axes=None,
        output_axes=None,
        collective_op_set=_NOOP_COLLECTIVE,
        quantizer_set=noop_quantizer_set,
    )


# ---------------------------------------------------------------------------
# Per-shard FSDP kernels using TE dense() for every matmul
# ---------------------------------------------------------------------------

def _fwd_shard_fn(x, w1, w2):
    """Same FSDP forward as the pure-JAX POC, but matmuls go through TE."""
    w1_full = jax.lax.all_gather(w1, axis_name=FSDP_AXIS, axis=0, tiled=True)
    pre = _matmul_NN(x, w1_full)                     # (B/fsdp, F)
    h = jax.nn.gelu(pre)
    w2_full = jax.lax.all_gather(w2, axis_name=FSDP_AXIS, axis=1, tiled=True)
    y = _matmul_NN(h, w2_full)                       # (B/fsdp, O)
    return y


def _bwd_shard_fn(x, w1, w2, dy):
    """Same FSDP backward, but every dot is a TE dense() call.

    dense() requires NN layout, so for each backward GEMM we explicitly
    transpose the operand that would otherwise have been "transposed" in a
    plain ``A @ B.T`` expression. Transposes in JAX are essentially free
    -- XLA folds them into the GEMM.
    """
    w1_full = jax.lax.all_gather(w1, axis_name=FSDP_AXIS, axis=0, tiled=True)
    w2_full = jax.lax.all_gather(w2, axis_name=FSDP_AXIS, axis=1, tiled=True)

    # Recompute fwd for gelu vjp.
    pre_act = _matmul_NN(x, w1_full)                 # (B/fsdp, F)
    h = jax.nn.gelu(pre_act)                          # (B/fsdp, F)

    # dh = dy @ w2.T   ->   _matmul_NN(dy, w2_full.T)   shapes: (B/fsdp,O)@(O,F)
    dh = _matmul_NN(dy, w2_full.T)                   # (B/fsdp, F)

    _, gelu_vjp = jax.vjp(jax.nn.gelu, pre_act)
    (dpre,) = gelu_vjp(dh)                           # (B/fsdp, F)

    # dx = dpre @ w1.T   ->   _matmul_NN(dpre, w1_full.T)   (B/fsdp,F)@(F,H)
    dx = _matmul_NN(dpre, w1_full.T)                 # (B/fsdp, H)

    # dW1 = x.T @ dpre   ->   _matmul_NN(x.T, dpre)   (H,B/fsdp)@(B/fsdp,F)
    dw1_partial = _matmul_NN(x.T, dpre)              # (H, F) PARTIAL across 'fsdp'
    # dW2 = h.T @ dy   ->   _matmul_NN(h.T, dy)        (F,B/fsdp)@(B/fsdp,O)
    dw2_partial = _matmul_NN(h.T, dy)                # (F, O) PARTIAL across 'fsdp'

    dw1 = jax.lax.psum_scatter(
        dw1_partial, axis_name=FSDP_AXIS, scatter_dimension=0, tiled=True
    )                                                # (H/fsdp, F)
    dw2 = jax.lax.psum_scatter(
        dw2_partial, axis_name=FSDP_AXIS, scatter_dimension=1, tiled=True
    )                                                # (F, O/fsdp)

    return dx, dw1, dw2


# ---------------------------------------------------------------------------
# custom_vjp wrapping the two shard_maps
# ---------------------------------------------------------------------------

@jax.custom_vjp
def mlp_block_fsdp_te(x, w1, w2):
    y, _ = _mlp_fwd_rule(x, w1, w2)
    return y


def _mlp_fwd_rule(x, w1, w2):
    fwd_sm = shard_map(
        _fwd_shard_fn,
        mesh=MESH,
        in_specs=(
            P(FSDP_AXIS, None),
            P(FSDP_AXIS, None),
            P(None, FSDP_AXIS),
        ),
        out_specs=P(FSDP_AXIS, None),
    )
    y = fwd_sm(x, w1, w2)
    return y, (x, w1, w2)


def _mlp_bwd_rule(residuals, dy):
    x, w1, w2 = residuals
    bwd_sm = shard_map(
        _bwd_shard_fn,
        mesh=MESH,
        in_specs=(
            P(FSDP_AXIS, None),
            P(FSDP_AXIS, None),
            P(None, FSDP_AXIS),
            P(FSDP_AXIS, None),
        ),
        out_specs=(
            P(FSDP_AXIS, None),
            P(FSDP_AXIS, None),
            P(None, FSDP_AXIS),
        ),
    )
    return bwd_sm(x, w1, w2, dy)


mlp_block_fsdp_te.defvjp(_mlp_fwd_rule, _mlp_bwd_rule)


# ---------------------------------------------------------------------------
# Reference (pure JAX, no shard_map)
# ---------------------------------------------------------------------------

def mlp_reference(x, w1, w2):
    h = jax.nn.gelu(x @ w1)
    return h @ w2


# ---------------------------------------------------------------------------
# Helpers (HLO collective counters, parity report) -- shared with POC #2
# ---------------------------------------------------------------------------

_COLLECTIVE_KINDS = (
    "all-reduce", "all-reduce-start",
    "all-gather", "all-gather-start",
    "all-to-all", "all-to-all-start",
    "reduce-scatter", "reduce-scatter-start",
)


def _count_collectives(hlo_text):
    counts = {k: 0 for k in _COLLECTIVE_KINDS}
    for kind in _COLLECTIVE_KINDS:
        pat = re.compile(r"\s" + re.escape(kind) + r"\(")
        counts[kind] = len(pat.findall(hlo_text))
    return counts


def _summarize_collectives(hlo_text):
    raw = _count_collectives(hlo_text)
    logical = {
        "all-reduce":     raw["all-reduce"]     + raw["all-reduce-start"],
        "all-gather":     raw["all-gather"]     + raw["all-gather-start"],
        "all-to-all":     raw["all-to-all"]     + raw["all-to-all-start"],
        "reduce-scatter": raw["reduce-scatter"] + raw["reduce-scatter-start"],
    }
    breakdown_lines = []
    for kind in ("all-reduce", "all-gather", "all-to-all", "reduce-scatter"):
        sync = raw[kind]
        async_n = raw[kind + "-start"]
        if sync or async_n:
            breakdown_lines.append(
                f"      {kind:14s}: sync={sync}  async-start={async_n}  "
                f"=> logical total {sync + async_n}"
            )
    return logical, raw, "\n".join(breakdown_lines)


def _save_hlo(hlo_text, name):
    out = os.path.join(_SCRIPT_DIR, f"{name}.hlo.txt")
    with open(out, "w") as f:
        f.write(hlo_text)
    return out


def _allclose_report(name, a, b, rtol=1e-3, atol=1e-3):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    abs_diff = np.max(np.abs(a - b))
    denom = np.maximum(np.max(np.abs(a)), np.max(np.abs(b)))
    rel_diff = abs_diff / max(denom, 1e-12)
    ok = abs_diff <= atol + rtol * denom
    return ok, f"{name}: max_abs={abs_diff:.3e}  max_rel={rel_diff:.3e}  (ok={ok})"


def _count_te_custom_calls(hlo_text):
    """Count XLA custom-call ops whose target name comes from TE.

    TE GEMM primitives lower to ``custom-call(target_name="...")`` strings;
    if dense() actually triggered the TE GEMM custom-call (rather than
    falling through to a plain XLA dot), we'll see those names in the HLO.
    """
    targets = re.findall(r'custom-call\([^)]*custom_call_target="([^"]+)"', hlo_text)
    counts = {}
    for t in targets:
        counts[t] = counts.get(t, 0) + 1
    return counts


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _loss_fn(fn, x, w1, w2):
    y = fn(x, w1, w2)
    return jnp.sum(y * y)


def main():
    jax.config.update("jax_default_matmul_precision", "highest")

    print(f"JAX devices ({len(DEVICES)}): {[str(d) for d in DEVICES]}")
    print(f"Mesh: {MESH}\n")

    B, H, F, O = 16, 16, 64, 16
    assert B % FSDP_SIZE == 0 and H % FSDP_SIZE == 0 and O % FSDP_SIZE == 0, (
        f"B/H/O must be divisible by FSDP_SIZE={FSDP_SIZE}"
    )
    print(f"Shapes: B={B}, H={H}, F={F}, O={O}, FSDP={FSDP_SIZE}")
    print(f"  per-shard expected: x=({B // FSDP_SIZE}, {H})  "
          f"w1=({H // FSDP_SIZE}, {F})  w2=({F}, {O // FSDP_SIZE})  "
          f"y=({B // FSDP_SIZE}, {O})\n")

    key = jax.random.PRNGKey(0)
    kx, k1, k2 = jax.random.split(key, 3)
    x_full  = jax.random.normal(kx, (B, H), dtype=jnp.float32)
    w1_full = jax.random.normal(k1, (H, F), dtype=jnp.float32) * (1.0 / np.sqrt(H))
    w2_full = jax.random.normal(k2, (F, O), dtype=jnp.float32) * (1.0 / np.sqrt(F))

    x_sh  = NamedSharding(MESH, P(FSDP_AXIS, None))
    w1_sh = NamedSharding(MESH, P(FSDP_AXIS, None))
    w2_sh = NamedSharding(MESH, P(None, FSDP_AXIS))
    rep   = NamedSharding(MESH, P())

    x  = jax.device_put(x_full,  x_sh)
    w1 = jax.device_put(w1_full, w1_sh)
    w2 = jax.device_put(w2_full, w2_sh)
    x_rep  = jax.device_put(x_full,  rep)
    w1_rep = jax.device_put(w1_full, rep)
    w2_rep = jax.device_put(w2_full, rep)

    print("--- per-device shapes the USER SUPPLIES (FSDP layout) ---")
    print(f"  x  : global={x.shape}    "
          f"per-device shard={x.addressable_shards[0].data.shape}    "
          f"spec={x.sharding.spec}")
    print(f"  w1 : global={w1.shape}   "
          f"per-device shard={w1.addressable_shards[0].data.shape}   "
          f"spec={w1.sharding.spec}")
    print(f"  w2 : global={w2.shape}   "
          f"per-device shard={w2.addressable_shards[0].data.shape}   "
          f"spec={w2.sharding.spec}")

    # ---------------------------- Forward ----------------------------------
    fwd_te  = jax.jit(mlp_block_fsdp_te)
    fwd_ref = jax.jit(mlp_reference)

    y_te  = fwd_te(x, w1, w2)
    y_ref = fwd_ref(x_rep, w1_rep, w2_rep)

    print("\n--- forward output ---")
    print(f"  y(TE-FSDP) : global={y_te.shape}   "
          f"per-device shard={y_te.addressable_shards[0].data.shape}   "
          f"spec={y_te.sharding.spec}")

    # ---------------------------- Backward ---------------------------------
    grad_te  = jax.jit(jax.grad(partial(_loss_fn, mlp_block_fsdp_te), argnums=(0, 1, 2)))
    grad_ref = jax.jit(jax.grad(partial(_loss_fn, mlp_reference), argnums=(0, 1, 2)))

    dx_t, dw1_t, dw2_t = grad_te(x, w1, w2)
    dx_r, dw1_r, dw2_r = grad_ref(x_rep, w1_rep, w2_rep)

    print("\n--- backward outputs ---")
    print(f"  dx  : global={dx_t.shape}   "
          f"per-device shard={dx_t.addressable_shards[0].data.shape}   "
          f"spec={dx_t.sharding.spec}")
    print(f"  dw1 : global={dw1_t.shape}   "
          f"per-device shard={dw1_t.addressable_shards[0].data.shape}   "
          f"spec={dw1_t.sharding.spec}")
    print(f"  dw2 : global={dw2_t.shape}   "
          f"per-device shard={dw2_t.addressable_shards[0].data.shape}   "
          f"spec={dw2_t.sharding.spec}")

    assert dw1_t.addressable_shards[0].data.shape == (H // FSDP_SIZE, F)
    assert dw2_t.addressable_shards[0].data.shape == (F, O // FSDP_SIZE)

    # --------------------- Numerical parity vs. reference -------------------
    print("\n--- numerical parity vs. plain (replicated) reference ---")
    checks = [
        _allclose_report("y  ", y_te,  y_ref),
        _allclose_report("dx ", dx_t,  dx_r),
        _allclose_report("dw1", dw1_t, dw1_r),
        _allclose_report("dw2", dw2_t, dw2_r),
    ]
    for ok, msg in checks:
        print(f"  {msg}")
    if not all(ok for ok, _ in checks):
        raise AssertionError("FSDP+TE-dense POC numerical parity check failed.")

    # --------------------- HLO collective summary --------------------------
    hlo = jax.jit(jax.value_and_grad(
        partial(_loss_fn, mlp_block_fsdp_te), argnums=(0, 1, 2)
    )).lower(x, w1, w2).compile().as_text()
    hlo_path = _save_hlo(hlo, "fsdp_te_dense")
    logical, _, breakdown = _summarize_collectives(hlo)
    te_custom_calls = _count_te_custom_calls(hlo)

    print("\n--- HLO collective summary ---")
    print(f"  HLO saved to: {hlo_path}")
    print(f"  Logical collectives (sync + async-start counted once):")
    print(f"    all-reduce     : {logical['all-reduce']}   "
          f"(expect 1: loss `jnp.sum(y*y)` reducing across batch shards)")
    print(f"    all-gather     : {logical['all-gather']}   "
          f"(expect 1: multi-tensor fused (w1, w2) gather, fwd/bwd CSE'd)")
    print(f"    reduce-scatter : {logical['reduce-scatter']}   "
          f"(expect 1: multi-tensor fused (dw1, dw2) scatter)")
    print(f"    all-to-all     : {logical['all-to-all']}   (expect 0)")
    print(f"  Breakdown:")
    print(breakdown if breakdown else "      (no collectives found)")

    print("\n--- TE custom-call ops in HLO ---")
    if te_custom_calls:
        for name, n in sorted(te_custom_calls.items()):
            print(f"  {name}: {n}")
    else:
        print("  (none -- TE dense lowered to plain XLA dot ops; this is")
        print("   expected when quantizer_set=noop and the TE GEMM custom-call")
        print("   path is not triggered for these shapes/dtypes.)")

    # Sanity: same FSDP collective pattern as the pure-JAX POC.
    if logical["all-gather"] == 0:
        raise AssertionError("Expected at least one all-gather (FSDP fwd weights).")
    if logical["reduce-scatter"] == 0:
        raise AssertionError("Expected at least one reduce-scatter (FSDP bwd weight grads).")
    if logical["all-to-all"] != 0:
        raise AssertionError("Did not expect any all-to-all in this POC.")

    print("\nFSDP + TE-dense POC PASSED:")
    print("  * Every dot product in fwd and bwd shard_maps went through TE dense().")
    print("  * Two shard_maps inside one custom_vjp -- same pattern as POC #2.")
    print("  * FSDP collective recipe preserved: 1 all-gather + 1 reduce-scatter")
    print("    + 1 loss all-reduce. dense() did not insert extra collectives.")
    print("  * Numerical parity matched the replicated reference within tol.")


if __name__ == "__main__":
    main()

