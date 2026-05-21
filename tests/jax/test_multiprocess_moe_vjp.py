# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Multi-process (one-GPU-per-process) tests for the unified MoE custom_vjp.

This is the **multiprocess companion** to
``test_distributed_moe_vjp.py``. The two files exercise the same code
paths in :func:`transformer_engine.jax.moe.moe`, but they bootstrap JAX
very differently:

* ``test_distributed_moe_vjp.py`` is the **single-process, multi-GPU**
  variant: one Python process sees all 4 GPUs as local devices, and
  ``shard_map`` dispatches work to each. This is the simplest setup
  but suffers from a CUDA-driver-level deadlock when **lazy Triton
  module loading** interleaves with **active NCCL collectives** -- see
  ``past_JAX_XLA_deadlock.txt`` for Olli Lupton's writeup
  (nvbug/5564750). Workaround used there: ``CUDA_LAUNCH_BLOCKING=1``
  to serialize every CUDA launch across all device threads.

* THIS file is the **multi-process, one-GPU-per-process** variant.
  Each pytest process binds to exactly one device via
  ``jax.distributed.initialize(..., local_device_ids=process_id)`` and
  the four processes form a global mesh through JAX's distributed
  runtime. Because every process has its **own** CUDA driver context,
  there is no global module-load lock shared between the threads
  driving different GPUs, and the lazy-load deadlock window does not
  exist. No ``CUDA_LAUNCH_BLOCKING=1`` is needed.

Why we keep BOTH files in tree
------------------------------

* The single-process file remains the simpler thing to read and is
  what most developers will reach for during dev-loop iteration; the
  ``CUDA_LAUNCH_BLOCKING=1`` workaround keeps it green.
* The multi-process file is what we want CI to run for guaranteed
  green-without-workarounds correctness, and it is what
  ``run_multiprocess_moe_vjp.sh`` (sister script to
  ``examples/jax/encoder/run_test_multiprocessing_encoder.sh``)
  invokes.
* If/when the upstream JAX FFI bug is fixed (or XLA gains an
  ``initialize`` stage for Triton custom_calls), we can collapse the
  two by dropping ``CUDA_LAUNCH_BLOCKING=1`` from the single-process
  file and deleting this one.

How to run
----------

You typically do NOT invoke pytest on this file directly -- the
``run_multiprocess_moe_vjp.sh`` launcher forks N pytest processes (one
per visible GPU), passing ``--num-process=N --process-id=i`` to each.
Driving it directly with only one process will skip every test
because :func:`jax.distributed.initialize` will report
``jax.device_count() == 1``.

    bash tests/jax/run_multiprocess_moe_vjp.sh

CI invocation lives in ``qa/L0_jax_distributed_unittest/test.sh``
alongside the single-process file's invocation.
"""

import os

# Same allocator config as the single-process variant: NCCL needs HBM
# headroom that JAX's default 90% preallocation does not leave. Set
# before any jax import below.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")

import sys
import time

import faulthandler
import signal

faulthandler.enable()
try:
    faulthandler.register(signal.SIGUSR1, all_threads=True, chain=False)
except (AttributeError, ValueError):
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


# Per-process distributed bootstrap. Each pytest invocation initializes
# JAX with exactly one local device (its assigned GPU). Once
# initialized, the four processes form one global mesh of 4 devices.
def _init_distributed(num_process: int, process_id: int) -> bool:
    """Initialize jax.distributed for this pytest process.

    Returns True if initialization succeeded (i.e. this is a real
    multi-process launch), False if num_process == 0 / 1 meaning the
    file is being collected without a launcher and tests should be
    skipped at module level.
    """
    if num_process <= 1:
        return False
    coord = os.environ.get("MOE_VJP_COORDINATOR_ADDRESS", "127.0.0.1:1234")
    jax.distributed.initialize(
        coordinator_address=coord,
        num_processes=num_process,
        process_id=process_id,
        local_device_ids=process_id,
    )
    assert jax.local_device_count() == 1, "one GPU per process is the whole point"
    assert (
        jax.device_count() == num_process
    ), f"global device_count {jax.device_count()} != num_process {num_process}"
    return True


# Read --num-process / --process-id BEFORE pytest collects any tests so
# we can fast-skip the whole module when not in a multiprocess launch.
def _read_mp_options():
    # Use pytest's option lookup via the request fixture isn't available
    # at module top-level; parse argv ourselves the same way encoder
    # test does. CLI form is e.g. "pytest ... --num-process=4 --process-id=0".
    num = int(os.environ.get("MP_NUM_PROCESS", "0") or "0")
    pid = int(os.environ.get("MP_PROCESS_ID", "0") or "0")
    for i, a in enumerate(sys.argv):
        if a.startswith("--num-process="):
            num = int(a.split("=", 1)[1])
        elif a == "--num-process" and i + 1 < len(sys.argv):
            num = int(sys.argv[i + 1])
        elif a.startswith("--process-id="):
            pid = int(a.split("=", 1)[1])
        elif a == "--process-id" and i + 1 < len(sys.argv):
            pid = int(sys.argv[i + 1])
    return num, pid


_MP_NUM_PROCESS, _MP_PROCESS_ID = _read_mp_options()
_MP_ACTIVE = _init_distributed(_MP_NUM_PROCESS, _MP_PROCESS_ID)

if not _MP_ACTIVE:
    # Skip the entire module if not launched via the multiprocess
    # runner. Lets `pytest tests/jax/` collect this file harmlessly.
    pytest.skip(
        "test_multiprocess_moe_vjp.py requires the multiprocess launcher "
        "(run_multiprocess_moe_vjp.sh). Skipping.",
        allow_module_level=True,
    )


NUM_DEVICES_REQUIRED = 4
EP_AXIS = "ep"
FSDP_AXIS = "fsdp"
EP_SIZE = 2
FSDP_SIZE = 2

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


# -----------------------------------------------------------------------------
# Helpers (identical to the single-process file's helpers; copied rather
# than imported because importing the single-process file would trigger
# its own faulthandler/setup and confuse pytest collection).
# -----------------------------------------------------------------------------


def _make_block(
    *,
    num_experts,
    num_experts_per_tok,
    intermediate_size,
    permutation_backend,
    aux_loss_coeff=0.0,
    dtype=jnp.bfloat16,
    align_size=0,
):
    return MoEBlock(  # noqa: F821
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
    return jax.lax.with_sharding_constraint(
        x, NamedSharding(mesh, P((EP_AXIS, FSDP_AXIS), None, None))
    )


def _hb(msg):
    if _MP_PROCESS_ID == 0:
        print(f"  [{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _init_apply(block, mesh, x, key):
    with mesh, global_shard_guard(  # noqa: F821
        MeshResource(ep_resource=EP_AXIS, fsdp_resource=FSDP_AXIS)  # noqa: F821
    ), nn_partitioning.axis_rules(LOGICAL_AXIS_RULES):
        x = _shard_inputs(x, mesh)
        _hb("  -> jit(block.init)")
        variables = jax.jit(block.init)(key, x)
        jax.block_until_ready(jax.tree_util.tree_leaves(variables)[0])
        _hb("  -> jit(block.apply)")
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

        _hb("  -> jit(grad(loss_fn))")
        grads = jax.jit(jax.grad(loss_fn))(variables, x)
        jax.block_until_ready(jax.tree_util.tree_leaves(grads)[0])
        return grads


def _unwrap(x):
    return x.value if hasattr(x, "value") else x


def _local_shard(x):
    """Return the local (this-process) shard of a global JAX Array as numpy.

    Every assertion in this file is structural ("is this finite", "is this
    non-zero", "is parity within 5e-2"). For all of these, checking the
    *local* shard on each process is just as valid as gathering everything
    to the host -- if any rank has NaN, that rank's assertion fires; if
    any rank's parity diverges, that rank's assertion fires. We avoid
    triggering a cross-process collective, which under JAX multi-host can
    deadlock if procs disagree on the order in which they emit it (we hit
    this on a first attempt with ``multihost_utils.process_allgather``).

    ``arr.addressable_data(0)`` returns the local-device view of the
    sharded array. With one GPU per process (which is the whole point of
    this multiprocess launcher), there is exactly one addressable shard.
    """
    return np.asarray(jax.device_get(x.addressable_data(0)))


def _describe_arr(name: str, x):
    """Print a one-line summary of a JAX Array on this process: global
    shape, sharding spec, local (addressable) shape, and whether the
    local shard contains any NaN/Inf. Used to determine, on first
    multiprocess run, whether ``addressable_data(0)`` is returning the
    correct slice or the global buffer with garbage outside the local
    window.
    """
    full_shape = tuple(x.shape) if hasattr(x, "shape") else None
    sharding = getattr(x, "sharding", None)
    local = x.addressable_data(0)
    local_np = np.asarray(jax.device_get(local))
    n_nan = int(np.isnan(local_np).sum())
    n_inf = int(np.isinf(local_np).sum())
    print(
        f"  [DESC pid={_MP_PROCESS_ID}] {name}: global={full_shape} "
        f"sharding={sharding} local_shape={local_np.shape} "
        f"local_nan={n_nan} local_inf={n_inf} local_size={local_np.size}",
        flush=True,
    )
    return local_np


# -----------------------------------------------------------------------------
# Smoke shapes (identical to the single-process file's SMOKE_* constants).
# -----------------------------------------------------------------------------

SMOKE_BATCH = EP_SIZE * FSDP_SIZE  # 4 -- one micro-batch per device
SMOKE_SEQ = 16
SMOKE_HIDDEN = 32
SMOKE_INTER = 64
SMOKE_NUM_EXPERTS = 4
SMOKE_TOPK = 2


@pytest.mark.triton
class TestMoeVjpMultiprocessSmoke:
    """Level 2 smoke under the multiprocess launcher (one GPU/process).

    Mirrors :class:`TestMoeVjpDistributedSmoke` from the
    single-process file. Same assertions, same shapes -- only the
    JAX bootstrap differs.
    """

    @pytest.mark.parametrize("backend_name", ["pure_jax", "triton"])
    def test_fwd_and_bwd_smoke(self, mesh, backend_name):
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
        # First-time diagnostic dump: print sharding + local shape +
        # local NaN/Inf count for the forward output and every named
        # gradient. Critical for telling apart "real multiprocess bwd
        # bug" from "addressable_data returning the wrong shape".
        _describe_arr("output", output)
        assert output.dtype == x.dtype
        assert aux is None
        grads = _grad_step(block, variables, mesh, x)
        # Print diagnostics on EVERY proc for EVERY gradient before we
        # raise any assertion. This keeps all procs synchronized at
        # this code point (no early-exit divergence) and gives us the
        # full picture across ranks. We then collect failures and
        # raise at the end if any.
        gradient_problems = []
        for name in ("gate_kernel", "wi_0", "wi_1", "wo"):
            g = _unwrap(grads["params"][name])
            g_local = _describe_arr(f"grad/{name}", g)
            if not np.all(np.isfinite(g_local)):
                gradient_problems.append(f"{name}: local shard has NaN/Inf")
            elif not np.any(g_local != 0.0):
                gradient_problems.append(f"{name}: local shard is all zero")
        if gradient_problems:
            raise AssertionError(
                "Per-rank gradient checks failed (one or more local shards):\n  "
                + "\n  ".join(gradient_problems)
            )

    @pytest.mark.parametrize("backend_name", ["pure_jax", "triton"])
    def test_aux_loss_smoke(self, mesh, backend_name):
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
        out_local = _local_shard(output)
        assert np.all(np.isfinite(out_local)), "output has NaN/Inf under aux"
        assert aux is not None
        assert aux.shape == ()
        aux_local = _local_shard(aux)
        assert np.isfinite(aux_local), "aux is NaN/Inf"
        grads = _grad_step(block, variables, mesh, x)
        g_gate_local = _local_shard(_unwrap(grads["params"]["gate_kernel"]))
        assert np.all(np.isfinite(g_gate_local)), "gate grad NaN/Inf under aux"

    def test_pure_jax_triton_parity(self, mesh):
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
        variables, out_pj, _ = _init_apply(block_pj, mesh, x, jax.random.PRNGKey(7))
        with mesh, global_shard_guard(  # noqa: F821
            MeshResource(ep_resource=EP_AXIS, fsdp_resource=FSDP_AXIS)  # noqa: F821
        ), nn_partitioning.axis_rules(LOGICAL_AXIS_RULES):
            x_sh = _shard_inputs(x, mesh)
            out_tr, _ = jax.jit(block_tr.apply)(variables, x_sh)

        out_pj_local = _local_shard(out_pj)
        out_tr_local = _local_shard(out_tr)
        diff = float(np.max(np.abs(out_pj_local - out_tr_local)))
        assert diff < 5e-2, f"forward parity breach: max_abs_diff={diff}"

        grads_pj = _grad_step(block_pj, variables, mesh, x)
        grads_tr = _grad_step(block_tr, variables, mesh, x)
        for name in ("gate_kernel", "wi_0", "wi_1", "wo"):
            g_pj = _local_shard(_unwrap(grads_pj["params"][name]))
            g_tr = _local_shard(_unwrap(grads_tr["params"][name]))
            d = float(np.max(np.abs(g_pj - g_tr)))
            assert d < 5e-2, f"grad parity breach on {name}: max_abs_diff={d}"
