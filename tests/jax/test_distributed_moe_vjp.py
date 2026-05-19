# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Multi-GPU (single-host) tests for the unified MoE custom_vjp.

Targets a 4-GPU box (e.g. a single GB200/B200 node) running a 2x2 mesh
``("ep", "fsdp")``. Two test classes:

* :class:`TestMoeVjpDistributedSmoke` -- "Level 2" structural / numerical
  correctness checks. Small shapes, fast. Verifies that
  :func:`transformer_engine.jax.moe.moe`'s internal ``shard_map`` opens
  cleanly under EP + FSDP-of-batch, that ``out_specs`` matches the
  returned ``ctx`` pytree, that gradients are finite, and that
  ``PURE_JAX`` and ``TRITON`` backends agree.

* :class:`TestMoeVjpDistributedPerf` -- "Level 3" Mixtral-ish-shape
  throughput + multi-step training-loop simulation. Reports
  ``tokens/sec``. Marked ``slow``; opt in with ``-m slow``.

Both share the same fixture-built mesh and ``MeshResource`` context. The
file is intentionally self-contained: no MaxText dependency, just JAX +
TE. To compare end-to-end against a MaxText / MoEBlock baseline, run the
existing ``test_perm.sh`` / ``test_router.sh`` scripts in the maxtext
repo (those drive a real Mixtral training step and report tokens/sec).

How to run
----------

CI invocation (canonical):

    bash qa/L0_jax_distributed_unittest/test.sh

Dev-loop invocation (thin shim around the same pytest command):

    bash tests/jax/run_distributed_moe_vjp.sh smoke

Both scripts apply ``-p no:typeguard`` -- see "CRITICAL" below.

Raw pytest invocation (do NOT use this in CI; only for one-off dev
work where you understand the typeguard gotcha):

    XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 \
        python -m pytest -c tests/jax/pytest.ini -v -s \
        -p no:typeguard \
        tests/jax/test_distributed_moe_vjp.py

The combination of ``XLA_PYTHON_CLIENT_PREALLOCATE=false`` (set at the
top of this file) and tests deliberately structured so each
parametrize variant only compiles the MoE custom_vjp once means a
single process runs the entire smoke suite in well under a minute.

CRITICAL: ``-p no:typeguard`` is REQUIRED
-----------------------------------------

If pytest's typeguard plugin is active (it is auto-loaded via
``jaxtyping``'s pytest entry point on most TE dev environments), the
runtime ``@typechecked`` shim that wraps every TE / jax / flax
callable will deadlock the first ``block.apply`` of the triton
backend: one GPU pins at 100%, three GPUs sit idle, no NCCL ops are
ever enqueued, and the Python MainThread parks in
``_pjit_call_impl_python``. The typeguard wrapper appears to either
materialise JAX tracers via ``isinstance`` checks during shard_map
tracing, or holds the GIL long enough to break the async-dispatch
pipeline that the MoE custom_vjp + Triton kernels +
``ragged_all_to_all`` rely on. The standalone equivalent of this test
(``tests/jax/standalone_smoke_triton.py``) runs in ~3s with no
pytest plugins active; under pytest with typeguard it hangs forever.

This is the first TE test that combines (a) Triton autotuned kernels
with input_output_aliases, (b) ``shard_map`` body, (c) NCCL
collectives (``ragged_all_to_all``, ``all_gather``), (d) ``custom_vjp``,
and (e) JAX async dispatch. None of the previous JAX tests exercised
this combination, which is why the typeguard interaction was not
observed before.

Both ``qa/L0_jax_distributed_unittest/test.sh`` and
``tests/jax/run_distributed_moe_vjp.sh`` pass ``-p no:typeguard``. We
do NOT disable typeguard in ``tests/jax/pytest.ini`` because other
jax tests rely on it for type-hint validation.

Heavier opt-in: pass ``--forked`` (requires ``pip install --user
pytest-forked``) to fork a fresh Python/JAX/XLA process per test
variant. This is rarely necessary now that preallocation is disabled,
but is still useful for diagnosing a flake suspected to come from
leftover state across tests in the same process. The
``run_distributed_moe_vjp.sh`` wrapper exposes both modes via
``FORKED=1``.

Why we previously needed ``--forked``: prior to the
``PREALLOCATE=false`` switch, JAX's default 90% HBM preallocation left
no headroom for NCCL to set up the EP communicator when a SECOND
custom_vjp executable was loaded in the same process (the typical
parametrize sweep ``[pure_jax, triton]`` did this). Now that
preallocation is off, JAX grows its pool on demand and NCCL always
finds room, so a single process handles the full sweep cleanly.
"""

import os

# IMPORTANT: configure JAX's HBM allocator BEFORE jax is imported.
#
# By default JAX preallocates ~90% of every visible GPU's HBM. That
# pool is fixed for the life of the process, so NCCL is starved for
# even the few KiB it needs to set up the EP communicator. On B200 /
# GB200 nodes (192 GiB HBM, possibly shared with another tenant) we
# saw NCCL `ncclCommInitRankConfig` fail with "Failed to CUDA calloc
# async 1216 bytes" -- 1 KiB! -- which then either crashes the test
# or deadlocks the all-to-all rendezvous because every rank waits
# forever for the leader stuck inside a failed NCCL init.
#
# Fix: disable preallocation entirely so JAX grows its pool on demand
# and always leaves room for NCCL allocations. The mem-fraction is
# still respected as a *cap*, so we also set it conservatively to
# leave headroom both for NCCL and for a co-resident tenant on
# shared clusters (e.g. prenyx batch partition).
#
# Users can override either knob in their env before invoking pytest
# if they know their node is dedicated and want max throughput.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")

# NOTE: ``NVTE_TRITON_PERMUTATION_BLOCK_SIZES`` (which shrinks the
# autotune sweep from 7 configs to 1 to skip 2-5 min of cold MLIR
# compiles) MUST be set BEFORE Python starts because
# ``tests/jax/conftest.py`` does ``import transformer_engine.jax`` at
# collection time -- by the time this test file's top level runs, the
# ``triton.autotune(configs=[...])`` wrappers have already frozen.
# The ``run_distributed_moe_vjp.sh`` wrapper exports it for us; if
# running pytest directly, export it manually:
#     export NVTE_TRITON_PERMUTATION_BLOCK_SIZES=128

import sys
import time
from typing import Optional

# ---------------------------------------------------------------------------
# Deadlock diagnostics. Install Python's built-in ``faulthandler`` BEFORE
# any heavy imports so a hang in jax/xla/triton/nccl can always be turned
# into a Python+C stack dump on demand. Two channels:
#
#   1. ``faulthandler.dump_traceback_later(N, repeat=True)`` -- prints all
#      thread stacks every ``MOE_VJP_WATCHDOG_SECS`` seconds. Off by
#      default; opt in with e.g. ``MOE_VJP_WATCHDOG_SECS=120``.
#   2. SIGUSR1 handler -- ``kill -USR1 <pid>`` dumps stacks once. Always
#      installed; zero overhead until the signal fires. Combined with the
#      ``_hb`` heartbeat (prints elapsed wallclock every few lines) this
#      lets you distinguish "slow compile" from "deadlock" in under a
#      minute without needing gdb / py-spy / ptrace permissions.
# ---------------------------------------------------------------------------
import faulthandler
import signal

faulthandler.enable()  # crash -> stack to stderr (no-op if already enabled)
try:
    faulthandler.register(signal.SIGUSR1, all_threads=True, chain=False)
except (AttributeError, ValueError):
    # Windows or signal already taken; not fatal.
    pass

_WATCHDOG_SECS = int(os.environ.get("MOE_VJP_WATCHDOG_SECS", "0") or "0")
if _WATCHDOG_SECS > 0:
    faulthandler.dump_traceback_later(_WATCHDOG_SECS, repeat=True)

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from flax.linen import partitioning as nn_partitioning


# Optional XLA persistent compilation cache. ~3-5x speed-up on the
# second and subsequent runs of this file in the same environment
# (e.g. across CI invocations sharing a /lustre mount). Off by default
# so a fresh checkout doesn't silently pick up artifacts from a
# stale TE build. Set ``MOE_VJP_COMPILE_CACHE_DIR=/some/path`` to opt
# in; we recommend a path on shared persistent storage in CI (e.g.
# ``/lustre/.../jax_compile_cache``) and a per-user path on workstations.
_compile_cache_dir = os.environ.get("MOE_VJP_COMPILE_CACHE_DIR")
if _compile_cache_dir:
    # Bump min size to 0 so even small jit'd helpers are cached, and
    # min entry size so single-device jit's qualify too. Default in
    # newer JAX is a multi-megabyte threshold that excludes most of
    # the test scaffolding (init, small reductions, etc.).
    jax.config.update("jax_compilation_cache_dir", _compile_cache_dir)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
    # Triton has its own per-kernel ptx/cubin cache (separate from
    # XLA's). Steer it to a sibling subdir so first-run Triton
    # autotune + compile artifacts also survive across runs. Massive
    # win on cold-start because every triton.autotune kernel evaluates
    # ~7 BLOCK_SIZE configs by compiling and timing each on the GPU.
    os.environ.setdefault("TRITON_CACHE_DIR", os.path.join(_compile_cache_dir, "triton"))


NUM_DEVICES_REQUIRED = 4
EP_AXIS = "ep"
FSDP_AXIS = "fsdp"
EP_SIZE = 2
FSDP_SIZE = 2

# Logical -> mesh axis rules. Match the TE / MaxText convention used by
# ``_MoEBlock``'s default ``wi_kernel_axes=("exp", "embed", "mlp")`` /
# ``wo_kernel_axes=("exp", "mlp", "embed")``.
LOGICAL_AXIS_RULES = (
    ("exp", EP_AXIS),
    ("embed", FSDP_AXIS),
    ("mlp", None),
    ("batch", (EP_AXIS, FSDP_AXIS)),
)


@pytest.fixture(scope="module")
def mesh():
    if jax.device_count() < NUM_DEVICES_REQUIRED:
        pytest.skip(
            f"Need >={NUM_DEVICES_REQUIRED} devices for ep={EP_SIZE} x fsdp={FSDP_SIZE};"
            f" have {jax.device_count()}"
        )
    devices = mesh_utils.create_device_mesh((EP_SIZE, FSDP_SIZE))
    return Mesh(devices, axis_names=(EP_AXIS, FSDP_AXIS))


@pytest.fixture(autouse=True, scope="function")
def _inject_moe(request):
    """Lazy-load TE MoE symbols only for ``triton``-marked tests so this
    file imports cleanly in environments without the fused-router CUDA
    kernel built.

    Aggressive cache cleanup between tests is OPT-IN via the
    ``MOE_VJP_AGGRESSIVE_CLEANUP=1`` env var. Off by default because:

    * with ``XLA_PYTHON_CLIENT_PREALLOCATE=false`` (set at the top of
      this file), JAX no longer hoards HBM so back-to-back compiled
      executables in the same process do not OOM NCCL;
    * keeping JAX's compilation cache alive across tests lets ``jit``
      re-hit on identical closures (e.g. the same ``block.apply``
      bound method), saving the full custom_vjp recompile.

    Set ``MOE_VJP_AGGRESSIVE_CLEANUP=1`` if you suspect a test is
    leaking state into the next one and want to bisect.
    """
    if os.environ.get("MOE_VJP_AGGRESSIVE_CLEANUP") == "1":
        import gc

        jax.clear_caches()
        gc.collect()

    if not request.node.get_closest_marker("triton"):
        yield
        return
    from transformer_engine.jax.flax import _MoEBlock as MoEBlock
    from transformer_engine.jax.moe import PermutationBackend
    from transformer_engine.jax.sharding import MeshResource, global_shard_guard

    mod = sys.modules[__name__]
    mod.MoEBlock = MoEBlock
    mod.PermutationBackend = PermutationBackend
    mod.MeshResource = MeshResource
    mod.global_shard_guard = global_shard_guard
    yield
    if os.environ.get("MOE_VJP_AGGRESSIVE_CLEANUP") == "1":
        import gc

        gc.collect()


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _make_block(
    *,
    num_experts: int,
    num_experts_per_tok: int,
    intermediate_size: int,
    permutation_backend,
    aux_loss_coeff: float = 0.0,
    dtype=jnp.bfloat16,
    align_size: int = 0,
):
    return MoEBlock(  # noqa: F821 -- injected by fixture
        num_experts=num_experts,
        num_experts_per_tok=num_experts_per_tok,
        intermediate_size=intermediate_size,
        permutation_backend=permutation_backend,
        data_parallelism_axes=(FSDP_AXIS,),
        aux_loss_coeff=aux_loss_coeff,
        dtype=dtype,
        _align_size=align_size,
    )


def _shard_inputs(x, mesh):
    """Constrain ``x`` to ``P(("ep", "fsdp"), None, None)``."""
    return jax.lax.with_sharding_constraint(
        x, NamedSharding(mesh, P((EP_AXIS, FSDP_AXIS), None, None))
    )


def _hb(msg: str):
    """Heartbeat print so a slow JAX/Triton compile shows progress in
    the pytest log instead of looking like a hang. Each line gets
    a wall-clock timestamp so the user can tell which step is the
    expensive one."""
    print(f"  [{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _init_apply(block, mesh, x, key):
    """Init + apply with logical_axis_rules + MeshResource context."""
    with mesh, global_shard_guard(  # noqa: F821
        MeshResource(ep_resource=EP_AXIS, fsdp_resource=FSDP_AXIS)  # noqa: F821
    ), nn_partitioning.axis_rules(LOGICAL_AXIS_RULES):
        x = _shard_inputs(x, mesh)
        _hb("  -> jit(block.init)")
        variables = jax.jit(block.init)(key, x)
        jax.block_until_ready(jax.tree_util.tree_leaves(variables)[0])
        _hb("  -> jit(block.apply) (fwd compile + Triton autotune if first call)")
        output, aux = jax.jit(block.apply)(variables, x)
        jax.block_until_ready(output)
    return variables, output, aux


def _grad_step(block, variables, mesh, x):
    with mesh, global_shard_guard(  # noqa: F821
        MeshResource(ep_resource=EP_AXIS, fsdp_resource=FSDP_AXIS)  # noqa: F821
    ), nn_partitioning.axis_rules(LOGICAL_AXIS_RULES):
        x = _shard_inputs(x, mesh)

        def loss_fn(variables, x):
            output, aux = block.apply(variables, x)
            main = jnp.mean(output.astype(jnp.float32) ** 2)
            return main + (aux.astype(jnp.float32) if aux is not None else 0.0)

        _hb("  -> jit(grad(loss_fn)) (fwd+bwd compile + Triton autotune if first call)")
        grads = jax.jit(jax.grad(loss_fn))(variables, x)
        jax.block_until_ready(jax.tree_util.tree_leaves(grads)[0])
        return grads


def _unwrap(x):
    return x.value if hasattr(x, "value") else x


# -----------------------------------------------------------------------------
# Level 2: smoke / correctness
# -----------------------------------------------------------------------------


# Smoke shapes are deliberately the smallest that still exercises every
# code path (FSDP-of-batch, EP-of-experts, top-k>1 routing, alignment
# padding -- though that one is off by default since align_size=0).
# Keeping these small directly cuts cold-compile time:
#   * Triton autotune times each BLOCK_SIZE config on the actual data;
#     16-token-per-shard inputs finish autotune in seconds instead of
#     minutes.
#   * XLA's HLO optimization passes process less data per pass.
# Constraints:
#   * SMOKE_BATCH must be a multiple of EP_SIZE * FSDP_SIZE = 4
#     (one micro-batch per device is the floor).
#   * SMOKE_NUM_EXPERTS must be a multiple of EP_SIZE = 2.
#   * SMOKE_HIDDEN and SMOKE_INTER should be multiples of 16 for bf16
#     GEMM alignment (32 is the practical floor).
SMOKE_BATCH = EP_SIZE * FSDP_SIZE  # 4 -- one micro-batch per device
SMOKE_SEQ = 16
SMOKE_HIDDEN = 32
SMOKE_INTER = 64
SMOKE_NUM_EXPERTS = 4  # divisible by EP_SIZE=2; 2 experts per shard locally
SMOKE_TOPK = 2


@pytest.mark.triton
class TestMoeVjpDistributedSmoke:
    """Level 2: structural + numerical correctness on 2x2 (ep, fsdp) mesh.

    Test design notes (CI-friendly):

    * The MoE custom_vjp compiles into a single large XLA executable.
      Each ``jax.jit``-wrapped invocation in a test triggers a fresh
      compile, so the number of compiles -- not the kernel shapes --
      dominates wall-clock time. We therefore compile **once per
      backend** and check fwd, bwd, and aux_loss within the same test.
    * The aux_loss path adds a second collective and a second
      ``fused_topk`` compile, so it would also be one extra compile per
      backend. We accept that cost as one extra parametrized test
      rather than rolling it into the main smoke (so a future
      aux-specific regression is reported cleanly).
    * The parity test deliberately compiles both backends a third
      time; that is unavoidable because by definition it needs both
      implementations side-by-side. Marked with a separate name so it
      can be skipped (``-k 'not parity'``) when bandwidth is tight.
    """

    @pytest.mark.parametrize("backend_name", ["pure_jax", "triton"])
    def test_fwd_and_bwd_smoke(self, mesh, backend_name):
        """One combined smoke check per backend: fwd shape / dtype /
        finiteness AND bwd finiteness + non-zero on every learnable
        parameter. Two compiles per backend (init+apply and grad)."""
        backend = PermutationBackend(backend_name)  # noqa: F821
        block = _make_block(
            num_experts=SMOKE_NUM_EXPERTS,
            num_experts_per_tok=SMOKE_TOPK,
            intermediate_size=SMOKE_INTER,
            permutation_backend=backend,
        )
        x = jax.random.normal(
            jax.random.PRNGKey(0),
            (SMOKE_BATCH, SMOKE_SEQ, SMOKE_HIDDEN),
            dtype=jnp.bfloat16,
        )
        variables, output, aux = _init_apply(block, mesh, x, jax.random.PRNGKey(1))
        # ---- Fwd checks ----
        assert output.shape == x.shape, f"got {output.shape}, want {x.shape}"
        assert output.dtype == x.dtype
        assert jnp.all(jnp.isfinite(output)).item(), "output has NaN/Inf"
        assert aux is None, "no aux_loss expected when aux_loss_coeff=0"
        # ---- Bwd checks ----
        grads = _grad_step(block, variables, mesh, x)
        for name in ("gate_kernel", "wi_0", "wi_1", "wo"):
            g = _unwrap(grads["params"][name])
            assert jnp.all(jnp.isfinite(g)).item(), f"{name} grad has NaN/Inf"
            assert jnp.any(g != 0.0).item(), f"{name} grad is identically zero"

    @pytest.mark.parametrize("backend_name", ["pure_jax", "triton"])
    def test_aux_loss_smoke(self, mesh, backend_name):
        """Aux-loss path: scalar returned + finite + gate receives a
        gradient through the aux branch."""
        backend = PermutationBackend(backend_name)  # noqa: F821
        block = _make_block(
            num_experts=SMOKE_NUM_EXPERTS,
            num_experts_per_tok=SMOKE_TOPK,
            intermediate_size=SMOKE_INTER,
            permutation_backend=backend,
            aux_loss_coeff=1e-2,
        )
        x = jax.random.normal(
            jax.random.PRNGKey(4),
            (SMOKE_BATCH, SMOKE_SEQ, SMOKE_HIDDEN),
            dtype=jnp.bfloat16,
        )
        variables, output, aux = _init_apply(block, mesh, x, jax.random.PRNGKey(5))
        assert output.shape == x.shape
        assert aux is not None
        assert aux.shape == ()
        assert jnp.isfinite(aux).item()
        grads = _grad_step(block, variables, mesh, x)
        g_gate = _unwrap(grads["params"]["gate_kernel"])
        assert jnp.all(jnp.isfinite(g_gate)).item(), "gate grad NaN/Inf under aux"

    def test_pure_jax_triton_parity(self, mesh):
        """Same params, swap backend, output + grads must match within
        bf16 tolerance."""
        block_pj = _make_block(
            num_experts=SMOKE_NUM_EXPERTS,
            num_experts_per_tok=SMOKE_TOPK,
            intermediate_size=SMOKE_INTER,
            permutation_backend=PermutationBackend.PURE_JAX,  # noqa: F821
        )
        block_tr = _make_block(
            num_experts=SMOKE_NUM_EXPERTS,
            num_experts_per_tok=SMOKE_TOPK,
            intermediate_size=SMOKE_INTER,
            permutation_backend=PermutationBackend.TRITON,  # noqa: F821
        )
        x = jax.random.normal(
            jax.random.PRNGKey(6),
            (SMOKE_BATCH, SMOKE_SEQ, SMOKE_HIDDEN),
            dtype=jnp.bfloat16,
        )
        # Share parameter init across the two blocks so routing + FFN
        # are identical and only the dispatch/combine implementation
        # differs.
        variables, out_pj, _ = _init_apply(block_pj, mesh, x, jax.random.PRNGKey(7))
        with mesh, global_shard_guard(  # noqa: F821
            MeshResource(ep_resource=EP_AXIS, fsdp_resource=FSDP_AXIS)  # noqa: F821
        ), nn_partitioning.axis_rules(LOGICAL_AXIS_RULES):
            x_sh = _shard_inputs(x, mesh)
            out_tr, _ = jax.jit(block_tr.apply)(variables, x_sh)

        diff = float(jnp.max(jnp.abs(out_pj - out_tr)))
        # bf16 compounded error budget on these shapes -- matches the
        # threshold the old single-device test used.
        assert diff < 5e-2, f"forward parity breach: max_abs_diff={diff}"

        grads_pj = _grad_step(block_pj, variables, mesh, x)
        grads_tr = _grad_step(block_tr, variables, mesh, x)
        for name in ("gate_kernel", "wi_0", "wi_1", "wo"):
            g_pj = _unwrap(grads_pj["params"][name])
            g_tr = _unwrap(grads_tr["params"][name])
            d = float(jnp.max(jnp.abs(g_pj - g_tr)))
            assert d < 5e-2, f"grad parity breach on {name}: max_abs_diff={d}"


# -----------------------------------------------------------------------------
# Level 3: Mixtral-ish-shape throughput
# -----------------------------------------------------------------------------
#
# Shapes intentionally smaller than full Mixtral-8x7B (which would need
# ~16 GPUs for a single replica) so the test fits on a single 4-GPU box
# at bf16. Keeps E=8, top_k=2, intermediate proportional to hidden,
# batch large enough to saturate. Adjust ``PERF_*`` constants below if
# you have more memory headroom.

PERF_BATCH = EP_SIZE * FSDP_SIZE * 4  # 16
PERF_SEQ = 2048
PERF_HIDDEN = 1024
PERF_INTER = 4096
PERF_NUM_EXPERTS = 8
PERF_TOPK = 2
PERF_WARMUP_STEPS = 5
PERF_TIMED_STEPS = 30


@pytest.mark.triton
@pytest.mark.slow
class TestMoeVjpDistributedPerf:
    """Level 3: tokens/sec on Mixtral-ish shapes."""

    @pytest.mark.parametrize("backend_name", ["pure_jax", "triton"])
    def test_throughput(self, mesh, backend_name, capsys):
        backend = PermutationBackend(backend_name)  # noqa: F821
        block = _make_block(
            num_experts=PERF_NUM_EXPERTS,
            num_experts_per_tok=PERF_TOPK,
            intermediate_size=PERF_INTER,
            permutation_backend=backend,
        )
        x = jax.random.normal(
            jax.random.PRNGKey(0),
            (PERF_BATCH, PERF_SEQ, PERF_HIDDEN),
            dtype=jnp.bfloat16,
        )

        with mesh, global_shard_guard(  # noqa: F821
            MeshResource(ep_resource=EP_AXIS, fsdp_resource=FSDP_AXIS)  # noqa: F821
        ), nn_partitioning.axis_rules(LOGICAL_AXIS_RULES):
            x = _shard_inputs(x, mesh)
            variables = jax.jit(block.init)(jax.random.PRNGKey(1), x)

            def loss_fn(variables, x):
                output, _ = block.apply(variables, x)
                return jnp.mean(output.astype(jnp.float32) ** 2)

            # value_and_grad mirrors a real training step.
            step = jax.jit(jax.value_and_grad(loss_fn))

            # Warmup (compile + a few iters).
            loss = None
            for _ in range(PERF_WARMUP_STEPS):
                loss, _ = step(variables, x)
            loss.block_until_ready()

            # Timed.
            t0 = time.perf_counter()
            for _ in range(PERF_TIMED_STEPS):
                loss, _ = step(variables, x)
            loss.block_until_ready()
            elapsed = time.perf_counter() - t0

        tokens_per_step = PERF_BATCH * PERF_SEQ
        tokens_per_sec = (tokens_per_step * PERF_TIMED_STEPS) / elapsed
        steps_per_sec = PERF_TIMED_STEPS / elapsed

        # Use capsys.disabled() so the perf line shows up even when
        # pytest captures stdout.
        with capsys.disabled():
            print(
                f"\n[perf] backend={backend_name}"
                f" batch={PERF_BATCH} seq={PERF_SEQ} hidden={PERF_HIDDEN}"
                f" inter={PERF_INTER} E={PERF_NUM_EXPERTS} k={PERF_TOPK}"
                f" mesh=ep{EP_SIZE}xfsdp{FSDP_SIZE}"
                f"\n[perf]   loss(final)     = {float(loss):.6f}"
                f"\n[perf]   elapsed         = {elapsed:.3f} s over"
                f" {PERF_TIMED_STEPS} steps"
                f"\n[perf]   steps/sec       = {steps_per_sec:.2f}"
                f"\n[perf]   tokens/sec      = {tokens_per_sec:.0f}",
                flush=True,
            )

        assert jnp.isfinite(loss).item(), "loss diverged during perf run"
