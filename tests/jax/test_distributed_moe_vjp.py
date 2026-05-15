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
"""

import sys
import time
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from flax.linen import partitioning as nn_partitioning


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
    kernel built."""
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


def _init_apply(block, mesh, x, key):
    """Init + apply with logical_axis_rules + MeshResource context."""
    with mesh, global_shard_guard(  # noqa: F821
        MeshResource(ep_resource=EP_AXIS, fsdp_resource=FSDP_AXIS)  # noqa: F821
    ), nn_partitioning.axis_rules(LOGICAL_AXIS_RULES):
        x = _shard_inputs(x, mesh)
        variables = jax.jit(block.init)(key, x)
        output, aux = jax.jit(block.apply)(variables, x)
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

        return jax.jit(jax.grad(loss_fn))(variables, x)


def _unwrap(x):
    return x.value if hasattr(x, "value") else x


# -----------------------------------------------------------------------------
# Level 2: smoke / correctness
# -----------------------------------------------------------------------------


SMOKE_BATCH = EP_SIZE * FSDP_SIZE * 2  # 8 -- two micro-batches per device
SMOKE_SEQ = 32
SMOKE_HIDDEN = 64
SMOKE_INTER = 128
SMOKE_NUM_EXPERTS = 8  # divisible by EP_SIZE=2
SMOKE_TOPK = 2


@pytest.mark.triton
class TestMoeVjpDistributedSmoke:
    """Level 2: structural + numerical correctness on 2x2 (ep, fsdp) mesh."""

    @pytest.mark.parametrize("backend_name", ["pure_jax", "triton"])
    def test_forward_shape_and_finite(self, mesh, backend_name):
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
        _, output, aux = _init_apply(block, mesh, x, jax.random.PRNGKey(1))
        assert output.shape == x.shape, f"got {output.shape}, want {x.shape}"
        assert output.dtype == x.dtype
        assert jnp.all(jnp.isfinite(output)).item(), "output has NaN/Inf"
        assert aux is None

    @pytest.mark.parametrize("backend_name", ["pure_jax", "triton"])
    def test_backward_finite_and_nonzero(self, mesh, backend_name):
        backend = PermutationBackend(backend_name)  # noqa: F821
        block = _make_block(
            num_experts=SMOKE_NUM_EXPERTS,
            num_experts_per_tok=SMOKE_TOPK,
            intermediate_size=SMOKE_INTER,
            permutation_backend=backend,
        )
        x = jax.random.normal(
            jax.random.PRNGKey(2),
            (SMOKE_BATCH, SMOKE_SEQ, SMOKE_HIDDEN),
            dtype=jnp.bfloat16,
        )
        variables, _, _ = _init_apply(block, mesh, x, jax.random.PRNGKey(3))
        grads = _grad_step(block, variables, mesh, x)
        for name in ("gate_kernel", "wi_0", "wi_1", "wo"):
            g = _unwrap(grads["params"][name])
            assert jnp.all(jnp.isfinite(g)).item(), f"{name} grad has NaN/Inf"
            assert jnp.any(g != 0.0).item(), f"{name} grad is identically zero"

    @pytest.mark.parametrize("backend_name", ["pure_jax", "triton"])
    def test_aux_loss_returned_and_finite(self, mesh, backend_name):
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
        _, output, aux = _init_apply(block, mesh, x, jax.random.PRNGKey(5))
        assert output.shape == x.shape
        assert aux is not None
        assert aux.shape == ()
        assert jnp.isfinite(aux).item()
        # Gate should also receive grads via the aux path.
        variables, _, _ = _init_apply(block, mesh, x, jax.random.PRNGKey(5))
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
