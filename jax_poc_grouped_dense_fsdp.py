"""POC #4: TE grouped_dense inside an FSDP shard_map (mimicking the MoE FFN).

This is the smallest possible smoke test for the *expert MLP* phase of the MoE
block. The router / dispatch / combine machinery is replaced by a synthetic
uniform routing (every expert gets the same number of tokens), so we can focus
purely on the question:

    Does TE's ``grouped_dense`` (which has its own ``custom_vjp`` wrapping
    a CUDA grouped-GEMM custom_call) compose correctly with our outer
    ``shard_map`` orchestration when the EXPERT-WEIGHT axis is FSDP-sharded?

Two variants are tested in one script:

  Variant A  -- ONE shard_map, NO outer custom_vjp.
                This is exactly the pattern the current MoEBlock follows
                (``_forward_ring_ep`` wraps a single ``shard_map`` and lets
                ``grouped_dense``'s own custom_vjp + JAX's autodiff produce
                the backward shard_map automatically with mirror in/out specs).

  Variant B  -- ONE outer custom_vjp + TWO shard_maps (one for fwd, one for
                bwd, written by us).
                This is the pattern from POC #1 / #2, allowing the bwd to use
                a *different* sharding contract from the fwd if we ever want
                to. For the FFN we don't need different specs, so this variant
                is mostly here to show the option exists and that it composes.

Both variants must produce identical numerical results to a single-device,
fully-replicated reference that calls ``grouped_dense`` directly with no
shard_map at all.

Layout (FSDP across ``fsdp`` axis, no EP for this POC):

    Mesh: ('fsdp',) with N devices.
    x         : P('fsdp', None)       on (M, K)        # batch-sharded tokens
    kernel    : P('fsdp', None, None) on (E, K, N)     # experts FSDP-sharded
    grp_sizes : P('fsdp',)            on (E,)          # per-shard local counts
    y         : P('fsdp', None)       on (M, N)        # batch-sharded output

Inside the fwd shard_map, each device:
    1. ``all_gather`` the kernel along 'fsdp' (axis 0 = E) -> full (E, K, N).
    2. Call ``grouped_dense(x_local, kernel_full, group_sizes_local)``.
    3. Output is each device's own (M/fsdp, N) slice -- no combine needed
       because tokens never crossed devices in this synthetic setup.

The autodiff'd backward (Variant A) or hand-written backward (Variant B):
    1. Same ``all_gather`` of kernel.
    2. ``grouped_dense.bwd`` produces dx (batch-sharded) and dW (full (E, K, N)
       partial across 'fsdp' shards).
    3. ``psum_scatter`` of dW along 'fsdp' (axis 0) -> back to FSDP layout.

Run:
    python jax_poc_grouped_dense_fsdp.py
"""

from __future__ import annotations

import os
import sys
import re

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

# IMPORTANT: this POC MUST be launched as N processes (one per GPU), e.g.::
#
#     mpirun -np 4 python jax_poc_grouped_dense_fsdp.py
#     # -- or, if you don't have mpirun, see the os.environ-driven launch
#     # below, which works with any rendezvous mechanism.
#
# *Why not single-process multi-GPU?* TE's ``multi_stream_cublas_gemm`` (the
# kernel that backs every ``grouped_dense`` call) caches its CUDA streams in a
# process-static ``std::vector`` (see ``common/util/multi_stream.cpp:22-35``).
# CUDA streams are device-bound, so streams created when device 0 is current
# cannot be used while device N is current -- you get
# ``CUDA Error: invalid resource handle`` from cuBLAS the first time the
# kernel runs on any device other than the one TE happened to initialise on.
# The fix is to put each GPU in its own process (each process has its own
# device context, so its own per-process static cache is correct).
# TE's own multi-process test uses exactly this pattern; see
# ``tests/jax/test_multi_process_distributed_grouped_gemm.py``.
#
# IMPORTANT: TE's ``GroupedGemmPrimitive`` (cpp_extensions/gemm.py:1427) is a
# ``custom_partitioning`` primitive with 13+ FFI operands. It does NOT define a
# ``shardy_sharding_rule`` and (on JAX > 0.9.1) does NOT register GSPMD
# callbacks either (TE's ``base.py`` only registers GSPMD on JAX <= 0.9.1).
# The result is two failure modes for *standalone* calls (no shard_map wrapper):
#   - With Shardy on  -> "Sharding rule has 1 operands, but the operation has 13"
#   - With Shardy off -> "TypeError: 'NoneType' object is not callable"
#                        (XLA's GSPMD propagation pass calls a null callback)
#
# But the *only* way TE itself uses grouped_gemm in distributed mode is INSIDE
# a ``shard_map`` (see ``tests/jax/test_multi_process_distributed_grouped_gemm.py``):
# in a shard_map body the mesh axes are "manual" so the partitioner doesn't
# need to propagate sharding through the custom_call -- the rule check is
# short-circuited. Our Variant A and Variant B both follow that pattern, so
# they work.
#
# The reference call below (which we use only for numerical parity) used to
# call ``grouped_dense`` directly (no shard_map); that's the path that hits
# the partitioner and crashes. We replaced the reference with a pure-JAX
# einsum (mathematically equivalent for our uniform-routing setup) so the
# reference doesn't depend on TE primitives at all.

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _SCRIPT_DIR]

from functools import partial

import jax
import jax.numpy as jnp

# ---------------------------------------------------------------------------
# Multi-process launcher
# ---------------------------------------------------------------------------
# This script follows TE's canonical multi-process pattern -- the same one
# used by ``tests/jax/test_multi_process_distributed_grouped_gemm.py`` and its
# launcher ``tests/jax/multi_process_launch.sh``. Why exactly that pattern?
# Because it's the *only* multi-process JAX setup TE explicitly tests against
# its grouped GEMM, and grouped_gemm has a static-stream bug that only
# manifests in single-process multi-GPU JAX (see the long comment at the top
# of this file).
#
# Three launch styles are supported, in priority order:
#
#   (1) TE-style positional args (recommended):
#         python jax_poc_grouped_dense_fsdp.py <coord_addr> <rank> <num_procs>
#       This is what TE's ``multi_process_launch.sh`` invokes. To launch a 4-GPU
#       run with TE's own launcher::
#
#           cp jax_poc_grouped_dense_fsdp.py $TE_PATH/tests/jax/
#           SCRIPT_NAME=$TE_PATH/tests/jax/jax_poc_grouped_dense_fsdp.py \
#               bash $TE_PATH/tests/jax/multi_process_launch.sh
#
#       Or use the included ``jax_poc_grouped_dense_fsdp_launch.sh`` (a
#       lightly-modified copy of TE's launcher that does NOT swallow stderr).
#
#   (2) MPI / Slurm env vars (OMPI_COMM_WORLD_RANK / SLURM_PROCID):
#         mpirun -np 4 python jax_poc_grouped_dense_fsdp.py
#         srun  -n 4 --gpus-per-task=1 python jax_poc_grouped_dense_fsdp.py
#
#   (3) Single-process (only legal if the workload avoids ``grouped_dense``):
#         python jax_poc_grouped_dense_fsdp.py
#
# In all multi-process modes we set ``CUDA_VISIBLE_DEVICES=<rank>`` if the
# launcher hasn't already done so, ensuring each process sees exactly one GPU.

def _init_distributed_or_single():
    """Return (is_multiproc, rank, num_procs). Initialise jax.distributed on demand."""

    rank = num_procs = coord = None

    # (1) TE-style positional args: <script> <coord> <rank> <num_procs>
    #     This is the format ``test_multi_process_distributed_grouped_gemm.py``
    #     uses and is what TE's ``multi_process_launch.sh`` produces.
    if len(sys.argv) >= 4 and ":" in sys.argv[1]:
        coord = sys.argv[1]
        rank = int(sys.argv[2])
        num_procs = int(sys.argv[3])

    # (2) MPI / Slurm env vars.
    elif "OMPI_COMM_WORLD_RANK" in os.environ and "OMPI_COMM_WORLD_SIZE" in os.environ:
        rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
        num_procs = int(os.environ["OMPI_COMM_WORLD_SIZE"])
    elif "SLURM_PROCID" in os.environ and "SLURM_NTASKS" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        num_procs = int(os.environ["SLURM_NTASKS"])
    else:
        return False, 0, 1

    coord = coord or os.environ.get("JAX_POC_COORD") or "127.0.0.1:12345"

    # Pin each rank to one GPU. The launcher script normally already sets
    # ``CUDA_VISIBLE_DEVICES`` (and we respect that); if not, default to
    # CVD=<rank> assuming ranks 0..N-1 map to local GPUs 0..N-1.
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)

    print(
        f"[rank {rank}/{num_procs}] coord={coord} "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}",
        flush=True,
    )
    jax.distributed.initialize(
        coordinator_address=coord, num_processes=num_procs, process_id=rank
    )
    return True, rank, num_procs


_IS_MULTIPROC, _RANK, _NUM_PROCS = _init_distributed_or_single()


def _is_rank0() -> bool:
    return _RANK == 0


def _rprint(*a, **kw):
    """Print only from rank 0 (or always, in single-proc mode)."""
    if _is_rank0():
        print(*a, **kw)


if hasattr(jax, "shard_map"):
    _shard_map_impl = jax.shard_map
else:
    from jax.experimental.shard_map import shard_map as _shard_map_impl  # type: ignore[no-redef]


def shard_map(f, *, mesh, in_specs, out_specs, check_vma: bool = True):
    """Version-agnostic shard_map wrapper.

    The ``check_vma`` flag (renamed from the older ``check_rep`` in recent JAX)
    enforces Varying-Manual-Axis (VMA) annotation propagation through the body
    of the shard_map. When a TE custom_call (e.g. ``tex.grouped_gemm``) appears
    inside the body, it doesn't propagate the ``{V:axis}`` annotation through
    its FFI boundary, so the bwd of ``grouped_dense`` produces a tensor without
    the annotation that the primal had -- which JAX's ``custom_vjp`` reports as
    "manual axis types do not match".
    Set ``check_vma=False`` whenever the body wraps such a custom_call. This is
    exactly what TE's own multi-process grouped-GEMM test does (passing
    ``check_rep=False`` -- same semantics, older name).
    """
    kwargs = dict(mesh=mesh, in_specs=in_specs, out_specs=out_specs)
    # Try the modern (check_vma) name first, then the legacy (check_rep) name,
    # then no flag (very old JAX). This keeps the script portable across the
    # 0.8 -> 0.9 -> 0.10 API churn.
    for kw in ("check_vma", "check_rep"):
        try:
            return _shard_map_impl(f, **kwargs, **{kw: check_vma})
        except TypeError:
            continue
    return _shard_map_impl(f, **kwargs)


from jax.sharding import Mesh, PartitionSpec as P, NamedSharding

import numpy as np

# --- TE imports ---------------------------------------------------------
try:
    from transformer_engine.jax.dense import grouped_dense
    from transformer_engine.jax.quantize import noop_quantizer_set
    HAVE_TE = True
except Exception as e:  # noqa: BLE001
    print(f"FATAL: transformer_engine.jax not importable -- {e}")
    raise


# ---------------------------------------------------------------------------
# Mesh
# ---------------------------------------------------------------------------

DEVICES = jax.devices()
FSDP_SIZE = len(DEVICES)
MESH = Mesh(np.asarray(DEVICES).reshape(FSDP_SIZE), axis_names=("fsdp",))
FSDP_AXIS = "fsdp"


# ---------------------------------------------------------------------------
# Synthetic uniform routing helper
# ---------------------------------------------------------------------------
# In a real MoE block, the dispatch step produces:
#   - a permuted activation tensor (M, K) sorted by expert
#   - a group_sizes vector (E,) where group_sizes[i] = #tokens going to expert i
# For this POC we skip the dispatch and synthetically arrange tokens so each
# expert gets the same number of tokens, on each FSDP shard. That means:
#   - group_sizes_local = [tokens_per_expert_per_shard] * E  (same on each shard)
#   - x is batch-sharded along 'fsdp', sized so M/fsdp/E is an integer.

def _make_uniform_grouped_inputs(rng, M, K, N, E, dtype=jnp.float32):
    """Make (x, kernel, group_sizes) for uniform per-expert routing."""
    rx, rk = jax.random.split(rng)
    x = jax.random.normal(rx, (M, K), dtype=dtype)
    # Init with 1/sqrt(K) so outputs stay O(1) and we can use tight tolerances.
    kernel = jax.random.normal(rk, (E, K, N), dtype=dtype) * (1.0 / np.sqrt(K))
    tokens_per_expert = M // E
    group_sizes = jnp.full((E,), tokens_per_expert, dtype=jnp.int32)
    return x, kernel, group_sizes


# ---------------------------------------------------------------------------
# Reference: pure-JAX grouped GEMM (no TE primitive, no shard_map).
# ---------------------------------------------------------------------------
# Why not call ``grouped_dense`` here? See the long comment at the top of the
# file: a *standalone* (non-shard_map) call to TE's grouped GEMM crashes the
# partitioner because the primitive lacks a Shardy/GSPMD rule. For our POC the
# reference only needs to be mathematically correct; since we route the same
# number of tokens to every expert (synthetic uniform routing built by
# ``_make_uniform_grouped_inputs``), grouped_dense reduces to a single batched
# einsum that we can write in plain JAX.
#
# Mathematical equivalence: for ``group_sizes = [M/E] * E``, the M-axis of x
# is partitioned into E equal contiguous chunks of size M/E, with chunk e
# matmul'd against ``kernel[e] : (K, N)``. Reshaping x to (E, M/E, K) and
# kernel to (E, K, N), this is exactly ``einsum('esk,ekn->esn', x_g, kernel)``
# reshaped back to (M, N).

def grouped_dense_reference(x, kernel, group_sizes):
    """Pure-JAX equivalent of Variant A's grouped_dense + shard_map call.

    The ``group_sizes`` arg is accepted for API compatibility with the TE
    function but not consumed -- we instead reproduce the *exact* token-to-
    expert layout that Variant A's per-shard ``group_sizes_local = [M/(FSDP*E)]
    * E`` produces.

    Why this is more subtle than ``einsum('esk,ekn->esn', x.reshape(E,M/E,K), kernel)``:

      * The naive einsum reference assumes expert e processes a single
        contiguous M/E-token chunk: ``x_global[e*(M/E) : (e+1)*(M/E)]``.
      * Variant A instead has every shard run ``grouped_dense`` on its own
        local slice with ``group_sizes_local = [M/(FSDP*E)] * E`` -- so on
        shard r, expert e processes ``x_local[e*sz : (e+1)*sz]`` where
        ``sz = M/(FSDP*E)``. Globally that means expert e's input is the
        union of FSDP scattered slices, NOT one contiguous chunk.

    These produce different y values for the same x and kernel, even with
    the same total per-expert token count. To compare Variant A against a
    single-device reference we must mirror the FSDP-aware slicing, which
    is exactly what reshaping x to ``(FSDP, E, sz, K)`` achieves: index
    ``[r, e, j, :]`` is then ``x_local_on_shard_r[e*sz + j]``, matching
    Variant A's per-shard grouped_dense exactly.
    """
    del group_sizes  # uniform routing assumed
    M, K = x.shape
    E, K2, N = kernel.shape
    assert K == K2, f"K mismatch: x has {K}, kernel has {K2}"
    assert M % (FSDP_SIZE * E) == 0, (
        f"This reference assumes uniform routing with M divisible by "
        f"FSDP_SIZE*E; got M={M}, FSDP={FSDP_SIZE}, E={E}."
    )
    sz = M // (FSDP_SIZE * E)
    # x_4d[r, e, j, :] is the j-th token assigned to expert e on shard r.
    x_4d = x.reshape(FSDP_SIZE, E, sz, K)
    # einsum: expert weight kernel[e] applied to all (r, j) slots for that e.
    y_4d = jnp.einsum("rezk,ekn->rezn", x_4d, kernel)
    return y_4d.reshape(M, N)


# ===========================================================================
# Variant A: ONE shard_map, no outer custom_vjp.
#   This is exactly the pattern the current MoEBlock follows
#   (one shard_map wraps the body, JAX autodiff produces the bwd shard_map
#    with mirror specs, and grouped_dense's own custom_vjp handles the
#    primitive-level gradient).
# ===========================================================================

def _ffn_shard_fn_A(x, kernel, group_sizes_local):
    # FSDP all-gather of expert weights along the E (axis=0) dim.
    kernel_full = jax.lax.all_gather(
        kernel, axis_name=FSDP_AXIS, axis=0, tiled=True
    )
    # Each shard now sees all experts and only its own tokens.
    y = grouped_dense(
        x,
        kernel_full,
        group_sizes_local,
        contracting_dims=((1,), (1,)),
        quantizer_set=noop_quantizer_set,
        kernel_fsdp_info=(None, -1),
    )
    return y


def ffn_block_A(x, kernel, group_sizes_local):
    sm = shard_map(
        _ffn_shard_fn_A,
        mesh=MESH,
        in_specs=(
            P(FSDP_AXIS, None),         # x batch-sharded
            P(FSDP_AXIS, None, None),   # kernel sharded along expert dim
            P(FSDP_AXIS,),              # group_sizes -- per-shard local counts
        ),
        out_specs=P(FSDP_AXIS, None),   # y batch-sharded
        # check_vma=False: TE's grouped_gemm custom_call does not propagate
        # the {V:fsdp} (Varying-Manual-Axis) annotation through its FFI
        # boundary, so the autodiff'd bwd of grouped_dense produces a dx
        # tensor without the annotation that x had on entry, which the
        # default check rejects with "manual axis types do not match".
        # Disabling the check is what TE's own multi-process grouped-GEMM
        # test does (under the older name ``check_rep=False``).
        check_vma=False,
    )
    return sm(x, kernel, group_sizes_local)


# ===========================================================================
# Variant B: ONE outer custom_vjp + TWO shard_maps (fwd + bwd written by us).
#   Demonstrates the option to give the bwd a *different* sharding contract.
#   For this FFN we keep the bwd contract identical to the fwd (mirror specs),
#   so the only thing different from Variant A is who controls the bwd
#   shard_map -- us (Variant B) vs. JAX autodiff (Variant A).
# ===========================================================================

def _ffn_fwd_shard_fn_B(x, kernel, group_sizes_local):
    """Same kernel as Variant A's fwd. Save residuals for our bwd."""
    kernel_full = jax.lax.all_gather(
        kernel, axis_name=FSDP_AXIS, axis=0, tiled=True
    )
    y = grouped_dense(
        x,
        kernel_full,
        group_sizes_local,
        contracting_dims=((1,), (1,)),
        quantizer_set=noop_quantizer_set,
        kernel_fsdp_info=(None, -1),
    )
    return y


def _ffn_bwd_shard_fn_B(x, kernel, group_sizes_local, dy):
    """Backward kernel.

    We don't manually re-implement grouped_dense's bwd -- we just call
    ``jax.vjp(grouped_dense, ...)`` to get its bwd as a JAX function and
    then pipe ``dy`` through it. That lets us reuse all of TE's grouped-GEMM
    bwd logic while still controlling the shard_map's in/out specs.
    """
    kernel_full = jax.lax.all_gather(
        kernel, axis_name=FSDP_AXIS, axis=0, tiled=True
    )

    # vjp gives us the backward closure for grouped_dense.
    def _gd(x_arg, kernel_arg):
        return grouped_dense(
            x_arg,
            kernel_arg,
            group_sizes_local,
            contracting_dims=((1,), (1,)),
            quantizer_set=noop_quantizer_set,
            kernel_fsdp_info=(None, -1),
        )
    _, gd_vjp = jax.vjp(_gd, x, kernel_full)
    dx, dkernel_full = gd_vjp(dy)
    # dx        : (M/fsdp, K)               batch-sharded -- matches x's spec
    # dkernel_full: (E, K, N) PARTIAL across 'fsdp' (each shard's contribution
    #               from its own tokens). We psum_scatter to land it back in
    #               the FSDP layout (E/fsdp, K, N) per shard.
    dkernel = jax.lax.psum_scatter(
        dkernel_full, axis_name=FSDP_AXIS, scatter_dimension=0, tiled=True
    )

    # ``group_sizes_local`` is an int32 input, mathematically non-differentiable.
    # We can't ``return None`` because the surrounding shard_map declares an
    # ``out_specs`` entry for it (every shard_map output must be a real array
    # whose per-shard shape matches the spec). The cleanest way to express
    # "this is a zero/no-op tangent" is to return a same-shape zero tensor;
    # the outer ``_ffn_block_B_bwd_rule`` then forwards it as the tangent for
    # ``group_sizes_local`` at the ``custom_vjp`` boundary, where JAX accepts
    # a same-shape zero for an int input.
    dgs = jnp.zeros_like(group_sizes_local)
    return dx, dkernel, dgs


@jax.custom_vjp
def ffn_block_B(x, kernel, group_sizes_local):
    y, _ = _ffn_block_B_fwd_rule(x, kernel, group_sizes_local)
    return y


def _ffn_block_B_fwd_rule(x, kernel, group_sizes_local):
    fwd_sm = shard_map(
        _ffn_fwd_shard_fn_B,
        mesh=MESH,
        in_specs=(
            P(FSDP_AXIS, None),
            P(FSDP_AXIS, None, None),
            P(FSDP_AXIS,),
        ),
        out_specs=P(FSDP_AXIS, None),
        check_vma=False,  # see ffn_block_A for rationale
    )
    y = fwd_sm(x, kernel, group_sizes_local)
    return y, (x, kernel, group_sizes_local)


def _ffn_block_B_bwd_rule(residuals, dy):
    x, kernel, group_sizes_local = residuals
    bwd_sm = shard_map(
        _ffn_bwd_shard_fn_B,
        mesh=MESH,
        in_specs=(
            P(FSDP_AXIS, None),
            P(FSDP_AXIS, None, None),
            P(FSDP_AXIS,),
            P(FSDP_AXIS, None),         # dy batch-sharded (mirrors y)
        ),
        out_specs=(
            P(FSDP_AXIS, None),         # dx -- mirrors x
            P(FSDP_AXIS, None, None),   # dkernel -- mirrors kernel
            P(FSDP_AXIS,),              # group_sizes grad placeholder
        ),
        check_vma=False,  # see ffn_block_A for rationale
    )
    dx, dkernel, dgs = bwd_sm(x, kernel, group_sizes_local, dy)
    return dx, dkernel, dgs


ffn_block_B.defvjp(_ffn_block_B_fwd_rule, _ffn_block_B_bwd_rule)


# ---------------------------------------------------------------------------
# Helpers (HLO collective counter, parity report)
# ---------------------------------------------------------------------------

_COLLECTIVE_KINDS = (
    "all-reduce", "all-reduce-start",
    "all-gather", "all-gather-start",
    "all-to-all", "all-to-all-start",
    "reduce-scatter", "reduce-scatter-start",
)


def _summarize_collectives(hlo_text):
    counts = {k: 0 for k in _COLLECTIVE_KINDS}
    for kind in _COLLECTIVE_KINDS:
        pat = re.compile(r"\s" + re.escape(kind) + r"\(")
        counts[kind] = len(pat.findall(hlo_text))
    logical = {
        "all-reduce":     counts["all-reduce"]     + counts["all-reduce-start"],
        "all-gather":     counts["all-gather"]     + counts["all-gather-start"],
        "all-to-all":     counts["all-to-all"]     + counts["all-to-all-start"],
        "reduce-scatter": counts["reduce-scatter"] + counts["reduce-scatter-start"],
    }
    return logical


def _save_hlo(hlo_text, name):
    out = os.path.join(_SCRIPT_DIR, f"{name}.hlo.txt")
    with open(out, "w") as f:
        f.write(hlo_text)
    return out


def _to_global_numpy(arr):
    """Gather a possibly-multi-process jax.Array into a numpy array.

    In single-process mode, ``np.asarray(arr)`` works directly because every
    shard is addressable. In multi-process mode, each process only owns its
    local shards; touching the value of the global array raises
    ``RuntimeError: Fetching value for jax.Array that spans non-addressable
    (non process local) devices``. The standard fix -- and what TE's own
    ``test_multi_process_distributed_grouped_gemm.py`` uses (line 144) -- is
    ``jax.experimental.multihost_utils.process_allgather``, which performs
    a cross-process all-gather and returns a fully-replicated numpy array
    on every process.
    """
    if _IS_MULTIPROC:
        from jax.experimental.multihost_utils import process_allgather as _pag
        # ``tiled=True`` is required when the input has non-fully-addressable
        # shards (i.e. parts of the array live on devices owned by other
        # processes); without it JAX errors with "Gathering global
        # non-fully-addressable arrays only supports tiled=True". TE's test
        # gets away without it because ``jem.process_allgather((out, ref))``
        # there happens to feed it arrays whose layout is fully addressable
        # by every rank (their reference path uses w_no_sharding).
        return np.asarray(_pag(arr, tiled=True))
    return np.asarray(arr)


def _allclose_report(name, a, b, rtol=1e-3, atol=1e-3):
    a = _to_global_numpy(a).astype(np.float64)
    b = _to_global_numpy(b).astype(np.float64)
    abs_diff = np.max(np.abs(a - b))
    denom = np.maximum(np.max(np.abs(a)), np.max(np.abs(b)))
    rel_diff = abs_diff / max(denom, 1e-12)
    ok = abs_diff <= atol + rtol * denom
    return ok, f"{name}: max_abs={abs_diff:.3e}  max_rel={rel_diff:.3e}  (ok={ok})"


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _loss_fn(fn, x, kernel, group_sizes_local):
    y = fn(x, kernel, group_sizes_local)
    return jnp.sum(y * y)


import time as _time


def _stage(label):
    """Tiny progress helper so we can see where compile time goes.

    The very first jitted call inside a multi-device program triggers (a) full
    XLA compilation of the jaxpr (which for a custom_vjp + shard_map + TE
    custom_calls graph can easily be 30-60s) and (b) NCCL clique setup
    (another few seconds). JAX prints a "rendezvous waiting 10s" warning
    during NCCL setup; that is NOT a hang, just a heartbeat.
    """
    if _is_rank0():
        print(f"[{_time.strftime('%H:%M:%S')}] {label}", flush=True)


def main():
    jax.config.update("jax_default_matmul_precision", "highest")

    if not _IS_MULTIPROC and len(DEVICES) > 1:
        raise SystemExit(
            "ERROR: this script must be launched as one process per GPU.\n"
            "TE's grouped_gemm caches CUDA streams in a process-static array\n"
            "(see common/util/multi_stream.cpp:22-50); those streams are bound\n"
            "to whichever device was current at first use, so a single\n"
            "multi-GPU process gets 'CUDA: invalid resource handle' from\n"
            "cuBLAS as soon as the kernel runs on any other device. This is\n"
            "the same reason TE's own ``test_multi_process_distributed_grouped_gemm``\n"
            "uses ``tests/jax/multi_process_launch.sh`` instead of plain pytest.\n\n"
            "Launch with one of:\n"
            "\n"
            "  # (a) TE's official launcher (recommended; mirrors how TE tests run):\n"
            "  bash jax_poc_grouped_dense_fsdp_launch.sh\n"
            "\n"
            "  # (b) mpirun:\n"
            "  mpirun -np N python jax_poc_grouped_dense_fsdp.py\n"
            "\n"
            "  # (c) srun:\n"
            "  srun -n N --gpus-per-task=1 python jax_poc_grouped_dense_fsdp.py\n"
            "\n"
            "  # (d) Manual positional args (matches\n"
            "  # tests/jax/test_multi_process_distributed_grouped_gemm.py):\n"
            "  CUDA_VISIBLE_DEVICES=0 python jax_poc_grouped_dense_fsdp.py 127.0.0.1:12345 0 N\n"
            "  CUDA_VISIBLE_DEVICES=1 python jax_poc_grouped_dense_fsdp.py 127.0.0.1:12345 1 N\n"
            "  ..."
        )

    _rprint(f"JAX devices ({len(DEVICES)}): {[str(d) for d in DEVICES]}")
    _rprint(f"Mesh: {MESH}\n")

    # Sizes: every expert gets the same #tokens per shard.
    # Constraints:
    #   * E divisible by FSDP_SIZE (we shard experts across fsdp)
    #   * M divisible by E*FSDP_SIZE  (tokens_per_expert_per_shard is integer)
    E = 8 * FSDP_SIZE if FSDP_SIZE <= 4 else 8
    while E % FSDP_SIZE != 0:
        E += 1
    tokens_per_expert_per_shard = 4
    M = E * FSDP_SIZE * tokens_per_expert_per_shard
    K, N = 32, 64

    print(f"Shapes: M={M}, K={K}, N={N}, E={E}, FSDP={FSDP_SIZE}")
    print(f"  per-shard:")
    print(f"    x      : ({M // FSDP_SIZE}, {K})")
    print(f"    kernel : ({E // FSDP_SIZE}, {K}, {N})  (E/fsdp experts per shard)")
    print(f"    y      : ({M // FSDP_SIZE}, {N})")
    print(f"  group_sizes_local: [{tokens_per_expert_per_shard}] * {E}\n")

    rng = jax.random.PRNGKey(0)
    x_full, kernel_full, group_sizes_global = _make_uniform_grouped_inputs(
        rng, M, K, N, E
    )

    # Place inputs:
    x_sh    = NamedSharding(MESH, P(FSDP_AXIS, None))
    k_sh    = NamedSharding(MESH, P(FSDP_AXIS, None, None))
    rep     = NamedSharding(MESH, P())
    gs_sh   = NamedSharding(MESH, P(FSDP_AXIS,))

    x  = jax.device_put(x_full, x_sh)
    kernel = jax.device_put(kernel_full, k_sh)

    # group_sizes_local: per-shard, same on each shard. Construct by tiling
    # the per-shard vector across 'fsdp'. Each shard sees [tpe_per_shard]*E.
    group_sizes_local_per_shard = jnp.full(
        (E,), tokens_per_expert_per_shard, dtype=jnp.int32
    )
    group_sizes_tiled = jnp.tile(group_sizes_local_per_shard, FSDP_SIZE)
    group_sizes_local = jax.device_put(group_sizes_tiled, gs_sh)

    # For the reference we need the global group_sizes describing the FULL x.
    group_sizes_global = jnp.full((E,), M // E, dtype=jnp.int32)
    x_rep      = jax.device_put(x_full,            rep)
    kernel_rep = jax.device_put(kernel_full,       rep)
    gs_rep     = jax.device_put(group_sizes_global, rep)

    print("--- per-device shapes the USER SUPPLIES ---")
    print(f"  x       : global={x.shape}      "
          f"per-device={x.addressable_shards[0].data.shape}      "
          f"spec={x.sharding.spec}")
    print(f"  kernel  : global={kernel.shape} "
          f"per-device={kernel.addressable_shards[0].data.shape} "
          f"spec={kernel.sharding.spec}")
    print(f"  grp_loc : global={group_sizes_local.shape}     "
          f"per-device={group_sizes_local.addressable_shards[0].data.shape}      "
          f"spec={group_sizes_local.sharding.spec}")

    # ----------------------------------------------------------------------
    # Reference: single-device, fully replicated grouped_dense.
    # ----------------------------------------------------------------------
    _stage("ref fwd: jit + run")
    fwd_ref = jax.jit(grouped_dense_reference)
    y_ref = jax.block_until_ready(fwd_ref(x_rep, kernel_rep, gs_rep))
    _stage("ref bwd: jit + run")
    grad_ref = jax.jit(jax.grad(
        partial(_loss_fn, grouped_dense_reference), argnums=(0, 1)
    ))
    dx_r, dkernel_r = grad_ref(x_rep, kernel_rep, gs_rep)
    jax.block_until_ready(dx_r); jax.block_until_ready(dkernel_r)
    _stage("ref bwd: done")

    # ======================================================================
    # Variant A: shard_map only, no outer custom_vjp
    # ======================================================================
    print("\n" + "=" * 72)
    print("Variant A: ONE shard_map, no outer custom_vjp (mirrors current MoEBlock)")
    print("=" * 72)

    _stage("A fwd: jit + run (first call -> XLA compile + NCCL setup, may take 30-60s)")
    fwd_A = jax.jit(ffn_block_A)
    y_A = jax.block_until_ready(fwd_A(x, kernel, group_sizes_local))
    _stage("A fwd: done")
    print(f"  y_A : global={y_A.shape}  "
          f"per-device={y_A.addressable_shards[0].data.shape}  "
          f"spec={y_A.sharding.spec}")

    # Lower bwd separately so we can see HLO BEFORE running it. If the lower
    # step itself succeeds, the bwd graph is at least well-formed.
    _stage("A bwd: trace + lower (no NCCL yet)")
    grad_A_fn = jax.jit(jax.grad(
        partial(_loss_fn, ffn_block_A), argnums=(0, 1)
    ))
    grad_A_compiled = grad_A_fn.lower(x, kernel, group_sizes_local).compile()
    _stage("A bwd: compiled, dumping HLO before first run")
    pa_pre = _save_hlo(grad_A_compiled.as_text(), "grouped_dense_variant_A_bwd_precompile")
    cA_pre = _summarize_collectives(grad_A_compiled.as_text())
    print(f"  bwd HLO (pre-run) -> {pa_pre}")
    print(f"  bwd logical collectives: all-reduce={cA_pre['all-reduce']} "
          f"all-gather={cA_pre['all-gather']} "
          f"reduce-scatter={cA_pre['reduce-scatter']} "
          f"all-to-all={cA_pre['all-to-all']}")

    _stage("A bwd: run (this is where NCCL clique setup may take ~10-30s)")
    dx_A, dkernel_A = grad_A_fn(x, kernel, group_sizes_local)
    jax.block_until_ready(dx_A); jax.block_until_ready(dkernel_A)
    _stage("A bwd: done")
    print(f"  dx_A      : global={dx_A.shape}      "
          f"per-device={dx_A.addressable_shards[0].data.shape}      "
          f"spec={dx_A.sharding.spec}")
    print(f"  dkernel_A : global={dkernel_A.shape} "
          f"per-device={dkernel_A.addressable_shards[0].data.shape} "
          f"spec={dkernel_A.sharding.spec}")

    print("\n  --- numerical parity vs. replicated reference ---")
    for ok, msg in [
        _allclose_report("y_A      ", y_A,       y_ref),
        _allclose_report("dx_A     ", dx_A,      dx_r),
        _allclose_report("dkernel_A", dkernel_A, dkernel_r),
    ]:
        print(f"  {msg}")
        if not ok:
            raise AssertionError(f"Variant A parity failed for {msg}")

    hlo_A = jax.jit(jax.value_and_grad(
        partial(_loss_fn, ffn_block_A), argnums=(0, 1)
    )).lower(x, kernel, group_sizes_local).compile().as_text()
    pa = _save_hlo(hlo_A, "grouped_dense_variant_A")
    cA = _summarize_collectives(hlo_A)
    print(f"\n  HLO -> {pa}")
    print(f"  Logical collectives: all-reduce={cA['all-reduce']} "
          f"all-gather={cA['all-gather']} "
          f"reduce-scatter={cA['reduce-scatter']} "
          f"all-to-all={cA['all-to-all']}")

    # ======================================================================
    # Variant B: outer custom_vjp + two shard_maps (fwd + bwd)
    # ======================================================================
    print("\n" + "=" * 72)
    print("Variant B: outer custom_vjp + two shard_maps (POC #1/#2 pattern)")
    print("=" * 72)

    fwd_B = jax.jit(ffn_block_B)
    y_B = fwd_B(x, kernel, group_sizes_local)
    print(f"  y_B : global={y_B.shape}  "
          f"per-device={y_B.addressable_shards[0].data.shape}  "
          f"spec={y_B.sharding.spec}")

    grad_B = jax.jit(jax.grad(
        partial(_loss_fn, ffn_block_B), argnums=(0, 1)
    ))
    dx_B, dkernel_B = grad_B(x, kernel, group_sizes_local)
    print(f"  dx_B      : global={dx_B.shape}      "
          f"per-device={dx_B.addressable_shards[0].data.shape}      "
          f"spec={dx_B.sharding.spec}")
    print(f"  dkernel_B : global={dkernel_B.shape} "
          f"per-device={dkernel_B.addressable_shards[0].data.shape} "
          f"spec={dkernel_B.sharding.spec}")

    print("\n  --- numerical parity vs. replicated reference ---")
    for ok, msg in [
        _allclose_report("y_B      ", y_B,       y_ref),
        _allclose_report("dx_B     ", dx_B,      dx_r),
        _allclose_report("dkernel_B", dkernel_B, dkernel_r),
    ]:
        print(f"  {msg}")
        if not ok:
            raise AssertionError(f"Variant B parity failed for {msg}")

    hlo_B = jax.jit(jax.value_and_grad(
        partial(_loss_fn, ffn_block_B), argnums=(0, 1)
    )).lower(x, kernel, group_sizes_local).compile().as_text()
    pb = _save_hlo(hlo_B, "grouped_dense_variant_B")
    cB = _summarize_collectives(hlo_B)
    print(f"\n  HLO -> {pb}")
    print(f"  Logical collectives: all-reduce={cB['all-reduce']} "
          f"all-gather={cB['all-gather']} "
          f"reduce-scatter={cB['reduce-scatter']} "
          f"all-to-all={cB['all-to-all']}")

    # ======================================================================
    # A vs B comparison
    # ======================================================================
    print("\n" + "=" * 72)
    print("Variant A vs. Variant B comparison")
    print("=" * 72)
    # ``jnp.allclose`` here would try to materialise the global array on this
    # process, which is illegal in multi-process mode (each rank only owns
    # addressable shards). Use the same _to_global_numpy / process_allgather
    # path the parity report uses.
    def _eq(a, b):
        return bool(np.allclose(_to_global_numpy(a), _to_global_numpy(b),
                                atol=1e-5, rtol=1e-5))

    print(f"  Same y     : {_eq(y_A, y_B)}")
    print(f"  Same dx    : {_eq(dx_A, dx_B)}")
    print(f"  Same dW    : {_eq(dkernel_A, dkernel_B)}")
    print(f"  HLO collectives:")
    print(f"    A: all-reduce={cA['all-reduce']} all-gather={cA['all-gather']} "
          f"reduce-scatter={cA['reduce-scatter']} all-to-all={cA['all-to-all']}")
    print(f"    B: all-reduce={cB['all-reduce']} all-gather={cB['all-gather']} "
          f"reduce-scatter={cB['reduce-scatter']} all-to-all={cB['all-to-all']}")
    print()
    print("  Interpretation:")
    print("    * If the collective counts match, the explicit-bwd-shard_map (B)")
    print("      gives no perf advantage over the autodiff'd-bwd-shard_map (A)")
    print("      for THIS workload (because we picked mirror specs).")
    print("    * Variant B's value would only show up if the bwd benefited from")
    print("      a *different* sharding contract -- e.g. recompute differently,")
    print("      reorder collectives, or shard along a different mesh axis.")

    print("\nGrouped-dense FSDP POC PASSED (both variants).")


if __name__ == "__main__":
    main()

