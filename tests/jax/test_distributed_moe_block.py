# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Distributed tests for ``transformer_engine.jax.flax.MoEBlock``.

Parametrized over the supported quantization recipes (BF16 +
``Float8CurrentScaling``) so any one of them regressing on this
hardware fails the suite. FP8 mode uses looser tolerances because
the legacy code path (knob ``quantize_before_fsdp_ag=False``)
quantizes activations per-shard without an amax reduction across
FSDP peers, which introduces small per-peer scale drift relative to
the single-device run. The forthcoming Exp 1 Stage 3 plumbing will
tighten this back to BF16-level agreement.

Two recipes are intentionally excluded:

* ``DelayedScaling`` -- ``GroupedQuantizer`` doesn't currently accept
  the DelayedScaling-specific kwargs (``margin``, ``amax_history``,
  ``amax_compute_algo``) that ``QuantizerFactory.create`` forwards;
  pending TE-side fix.
* ``MXFP8BlockScaling`` -- the MXFP8 FFI requires ``M % 32 == 0`` on
  every quantized buffer; the toy distributed shapes produce
  random-routed per-group buffers that don't satisfy this without
  ``align_size = 32`` (and V2 dispatch wants 128-alignment + H/I %
  128 == 0). MXFP8 distributed coverage is provided by the realistic
  demo at ``moe_block_demo/run_realistic.sh
  DEMO_QUANT_RECIPE=mxfp8``.
"""

import sys

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, PartitionSpec

from transformer_engine.common import recipe as _recipe
from transformer_engine.jax.quantize import is_fp8_available

from utils import assert_allclose, is_devices_enough


_is_fp8_supported, _ = is_fp8_available()

SUPPORTED_RECIPES = [pytest.param(None, id="bf16")]
if _is_fp8_supported:
    SUPPORTED_RECIPES.append(pytest.param(_recipe.Float8CurrentScaling(), id="CurrentScaling"))


@pytest.fixture(autouse=True, scope="function")
def _inject_moe(request):
    """Lazy-load ``MoEBlock`` only for tests marked ``triton``."""
    if not request.node.get_closest_marker("triton"):
        yield
        return

    from transformer_engine.jax import MeshResource, autocast
    from transformer_engine.jax.flax import MoEBlock

    mod = sys.modules[__name__]
    mod.MeshResource = MeshResource
    mod.autocast = autocast
    mod.MoEBlock = MoEBlock
    yield


DTYPE = jnp.bfloat16
# Must be divisible by ep*fsdp = 4 so the batch dim can be sharded over
# the full ('ep','fsdp') axis tuple under Experiment 3.
BATCH_SIZE = 4
SEQUENCE_LENGTH = 16
HIDDEN_SIZE = 64
INTERMEDIATE_SIZE = 128
NUM_EXPERTS = 8
NUM_EXPERTS_PER_TOK = 2

# Larger dims for ``quantize_before_fsdp_ag=True`` -- the MoEBlock-
# level alignment check requires the per-shard size on the FSDP-sharded
# kernel dim to be a multiple of 128 (block_size * alignment_y for
# MXFP8 K-side AG). With FSDP=2 and ``embed -> fsdp`` (which shards H
# on wi axis 1 and on wo axis 2), we need H/2 % 128 == 0, so H>=256.
HIDDEN_SIZE_QB4AG = 256
INTERMEDIATE_SIZE_QB4AG = 256


def _make_inputs(key: jax.Array) -> jax.Array:
    return jax.random.normal(key, (BATCH_SIZE, SEQUENCE_LENGTH, HIDDEN_SIZE), dtype=DTYPE)


def _make_inputs_qb4ag(key: jax.Array) -> jax.Array:
    return jax.random.normal(
        key, (BATCH_SIZE, SEQUENCE_LENGTH, HIDDEN_SIZE_QB4AG), dtype=DTYPE
    )


def _unwrap_partitioned(x):
    return x.value if hasattr(x, "value") else x


def _tolerances_for(recipe_or_none):
    """Return ``(atol_out, rtol_out, atol_grad, rtol_grad)`` for the recipe.

    BF16 keeps the historical ``5e-2 / 5e-2`` for outputs and
    ``1e-1 / 1e-1`` for gradients. FP8 / MXFP8 are looser because the
    legacy code path (no ``quantize_before_fsdp_ag``) quantizes
    activations per-shard without an FSDP-wide amax reduction, so the
    sharded vs single-device runs see different scales for the same
    activation values. Stage 3 of Exp 1 will eliminate this drift.
    """
    if recipe_or_none is None:
        return 5e-2, 5e-2, 1e-1, 1e-1
    return 2e-1, 2e-1, 3e-1, 3e-1


@pytest.mark.triton
class TestDistributedMoEBlock:
    @pytest.mark.parametrize("permutation_backend", ["pure_jax", "triton"])
    @pytest.mark.parametrize("quantization_recipe", SUPPORTED_RECIPES)
    def test_ep2_fsdp2_matches_single_device(self, permutation_backend, quantization_recipe):
        if not is_devices_enough(4):
            pytest.skip("MoE distributed test requires 4 devices for EP=2 x FSDP=2.")

        atol_out, rtol_out, atol_grad, rtol_grad = _tolerances_for(quantization_recipe)

        key = jax.random.PRNGKey(11)
        init_key, data_key = jax.random.split(key)
        inputs = _make_inputs(data_key)

        base_kwargs = dict(
            num_experts=NUM_EXPERTS,
            num_experts_per_tok=NUM_EXPERTS_PER_TOK,
            intermediate_size=INTERMEDIATE_SIZE,
            permutation_backend=permutation_backend,
            aux_loss_coeff=1e-2,
            dtype=DTYPE,
        )

        single_block = MoEBlock(**base_kwargs)

        def _make_loss_and_grad(block):
            """Build a jitted ``value_and_grad`` over ``(variables, x)``.

            Capturing ``block`` in a closure (so it isn't a jit input)
            sidesteps having to mark it as static -- Flax modules are
            registered pytrees but they carry Python-level config that
            jit treats as part of the trace.
            """

            def loss_fn(variables, x):
                output, aux_loss = block.apply(variables, x)
                loss = jnp.mean(output.astype(jnp.float32) ** 2)
                if aux_loss is not None:
                    loss = loss + aux_loss.astype(jnp.float32)
                return loss, (output, aux_loss)

            return jax.jit(jax.value_and_grad(loss_fn, has_aux=True))

        # Single-device reference. Use the same recipe (or no autocast for
        # BF16) so the comparison apples-to-apples isolates the
        # sharding-induced drift, not the BF16-vs-FP8 dtype gap.
        single_autocast_kwargs = dict(mesh_resource=MeshResource())
        if quantization_recipe is None:
            single_ctx = autocast(enabled=False, **single_autocast_kwargs)
        else:
            single_ctx = autocast(
                enabled=True, recipe=quantization_recipe, **single_autocast_kwargs
            )
        with single_ctx:
            single_variables = single_block.init(init_key, inputs)
            (single_loss, (single_output, single_aux)), single_grads = _make_loss_and_grad(
                single_block
            )(single_variables, inputs)

        devices = np.asarray(jax.devices()[:4]).reshape(2, 2)
        mesh = Mesh(devices, ("ep", "fsdp"))
        # FSDP-style sharding: weights are sharded on a *non-contracting*
        # weight axis (gathered before the GEMM); activations stay sharded on
        # the *batch* axis throughout - the same fsdp mesh axis is reused for
        # both. The TE primitives' custom_partitioning rules expect activations
        # FSDP-sharded on batch, so we declare ("batch", "fsdp") AND pass
        # ``input_axes=("batch", None, None)`` to enforce it on the inputs to
        # the block. ("embed", "fsdp") shards the weight's hidden dim, which
        # is gathered inside grouped_dense's custom_partitioning before GEMM
        # (no reshard of activations needed because their layout is unchanged).
        logical_axis_rules = (
            ("exp", "ep"),
            ("batch", "fsdp"),
            ("embed", "fsdp"),
        )
        # ``data_parallelism_axes=("fsdp",)`` opts in to the true-FSDP
        # behavior: the ``shard_map``'s in_specs/out_specs become
        # ``P(("ep","fsdp"), None, None)`` for the batch dim, so each
        # device owns ``B/(ep*fsdp)`` unique tokens (no redundant compute
        # across fsdp peers within an ep group).
        sharded_block = MoEBlock(
            expert_parallelism_axis="ep",
            data_parallelism_axes=("fsdp",),
            mesh=mesh,
            input_axes=("batch", None, None),
            **base_kwargs,
        )

        sharded_autocast_kwargs = dict(mesh_resource=MeshResource(fsdp_resource="fsdp"))
        if quantization_recipe is None:
            sharded_ctx = autocast(enabled=False, **sharded_autocast_kwargs)
        else:
            sharded_ctx = autocast(
                enabled=True, recipe=quantization_recipe, **sharded_autocast_kwargs
            )
        with mesh, sharded_ctx:
            with nn.logical_axis_rules(logical_axis_rules):
                # ``MoEBlock`` registers params via ``with_logical_partitioning``
                # which only attaches LogicallyPartitioned metadata; the
                # underlying jax.Array stays single-device unless ``init``
                # is run inside ``jax.jit`` with ``out_shardings``. Use the
                # canonical Flax-Linen pattern (mirrors
                # ``examples/jax/encoder/test_model_parallel_encoder.py``):
                #   1. ``jax.eval_shape`` to trace abstract variables (keeps
                #      the LogicallyPartitioned wrappers; only the inner
                #      arrays become ShapeDtypeStruct);
                #   2. ``nn.get_partition_spec`` to extract a tree of logical
                #      PartitionSpecs from those wrappers (treats
                #      LogicallyPartitioned as a leaf);
                #   3. ``nn.logical_to_mesh_sharding`` to resolve those
                #      logical specs to NamedShardings via the active rules;
                #   4. ``jax.jit(init, out_shardings=...)`` to actually
                #      place the params on-device with those shardings.
                abstract_variables = jax.eval_shape(sharded_block.init, init_key, inputs)
                logical_partition_spec = nn.get_partition_spec(abstract_variables)
                out_shardings = nn.logical_to_mesh_sharding(
                    logical_partition_spec, mesh, logical_axis_rules
                )
                sharded_variables = jax.jit(sharded_block.init, out_shardings=out_shardings)(
                    init_key, inputs
                )
                (sharded_loss, (sharded_output, sharded_aux)), sharded_grads = (
                    _make_loss_and_grad(sharded_block)(sharded_variables, inputs)
                )

        wi_0 = _unwrap_partitioned(sharded_variables["params"]["wi_0"])
        wi_1 = _unwrap_partitioned(sharded_variables["params"]["wi_1"])
        wo = _unwrap_partitioned(sharded_variables["params"]["wo"])
        assert wi_0.sharding.spec == PartitionSpec("ep", "fsdp", None)
        assert wi_1.sharding.spec == PartitionSpec("ep", "fsdp", None)
        assert wo.sharding.spec == PartitionSpec("ep", None, "fsdp")

        assert_allclose(sharded_output, single_output, dtype=DTYPE, atol=atol_out, rtol=rtol_out)
        assert_allclose(
            sharded_loss, single_loss, dtype=jnp.float32, atol=atol_out, rtol=rtol_out
        )
        assert_allclose(
            sharded_aux, single_aux, dtype=jnp.float32, atol=atol_out, rtol=rtol_out
        )

        for name in ("gate_kernel", "wi_0", "wi_1", "wo"):
            grad_single = _unwrap_partitioned(single_grads["params"][name])
            grad_sharded = _unwrap_partitioned(sharded_grads["params"][name])
            assert_allclose(
                grad_sharded,
                grad_single,
                dtype=DTYPE,
                atol=atol_grad,
                rtol=rtol_grad,
                err_msg=f"Distributed gradient mismatch for {name}",
            )

    @pytest.mark.parametrize("permutation_backend", ["pure_jax", "triton"])
    @pytest.mark.parametrize("quantization_recipe", SUPPORTED_RECIPES)
    def test_quantize_before_fsdp_ag_matches_single_device(
        self, permutation_backend, quantization_recipe
    ):
        """Stage-1-through-4 of Experiment 1: ``quantize_before_fsdp_ag``
        keeps the FSDP dim of ``wi_*`` / ``wo`` sharded across the EP
        ``shard_map`` boundary, runs the per-shard quantize INSIDE the
        body, all-gathers the FP8 data (and per-block ``scale_inv`` for
        MXFP8 K-side) to materialize the full kernel for the GEMM, and
        in the bwd ``psum_scatter``s the wgrad along the same FSDP axis
        so it matches the per-shard kernel input.

        Functionally this should produce numerically equivalent loss /
        output / gradient to the legacy ``knob=False`` path (and to the
        single-device reference), within recipe-appropriate tolerance.
        For tensor scaling there is a small per-expert FP8 rounding gap
        because the fwd amax is now FSDP-pmax'd (tighter scale) instead
        of per-shard (looser scale); the gap is well within the FP8
        tolerance band already in :func:`_tolerances_for`.

        Larger ``H = I = 256`` is required so the per-shard FSDP size
        (``H / 2 = 128``) clears the MoEBlock-level mult-of-128
        alignment check enforced when the knob is on.
        """
        if not is_devices_enough(4):
            pytest.skip("MoE distributed test requires 4 devices for EP=2 x FSDP=2.")

        atol_out, rtol_out, atol_grad, rtol_grad = _tolerances_for(quantization_recipe)

        key = jax.random.PRNGKey(11)
        init_key, data_key = jax.random.split(key)
        inputs = _make_inputs_qb4ag(data_key)

        base_kwargs = dict(
            num_experts=NUM_EXPERTS,
            num_experts_per_tok=NUM_EXPERTS_PER_TOK,
            intermediate_size=INTERMEDIATE_SIZE_QB4AG,
            permutation_backend=permutation_backend,
            aux_loss_coeff=1e-2,
            dtype=DTYPE,
        )

        single_block = MoEBlock(**base_kwargs)

        def _make_loss_and_grad(block):
            def loss_fn(variables, x):
                output, aux_loss = block.apply(variables, x)
                loss = jnp.mean(output.astype(jnp.float32) ** 2)
                if aux_loss is not None:
                    loss = loss + aux_loss.astype(jnp.float32)
                return loss, (output, aux_loss)

            return jax.jit(jax.value_and_grad(loss_fn, has_aux=True))

        single_autocast_kwargs = dict(mesh_resource=MeshResource())
        if quantization_recipe is None:
            single_ctx = autocast(enabled=False, **single_autocast_kwargs)
        else:
            single_ctx = autocast(
                enabled=True, recipe=quantization_recipe, **single_autocast_kwargs
            )
        with single_ctx:
            single_variables = single_block.init(init_key, inputs)
            (single_loss, (single_output, single_aux)), single_grads = _make_loss_and_grad(
                single_block
            )(single_variables, inputs)

        devices = np.asarray(jax.devices()[:4]).reshape(2, 2)
        mesh = Mesh(devices, ("ep", "fsdp"))
        logical_axis_rules = (
            ("exp", "ep"),
            ("batch", "fsdp"),
            ("embed", "fsdp"),
        )
        sharded_block = MoEBlock(
            expert_parallelism_axis="ep",
            data_parallelism_axes=("fsdp",),
            mesh=mesh,
            input_axes=("batch", None, None),
            quantize_before_fsdp_ag=True,
            **base_kwargs,
        )

        sharded_autocast_kwargs = dict(mesh_resource=MeshResource(fsdp_resource="fsdp"))
        if quantization_recipe is None:
            sharded_ctx = autocast(enabled=False, **sharded_autocast_kwargs)
        else:
            sharded_ctx = autocast(
                enabled=True, recipe=quantization_recipe, **sharded_autocast_kwargs
            )
        with mesh, sharded_ctx:
            with nn.logical_axis_rules(logical_axis_rules):
                abstract_variables = jax.eval_shape(sharded_block.init, init_key, inputs)
                logical_partition_spec = nn.get_partition_spec(abstract_variables)
                out_shardings = nn.logical_to_mesh_sharding(
                    logical_partition_spec, mesh, logical_axis_rules
                )
                sharded_variables = jax.jit(sharded_block.init, out_shardings=out_shardings)(
                    init_key, inputs
                )
                (sharded_loss, (sharded_output, sharded_aux)), sharded_grads = (
                    _make_loss_and_grad(sharded_block)(sharded_variables, inputs)
                )

        assert_allclose(sharded_output, single_output, dtype=DTYPE, atol=atol_out, rtol=rtol_out)
        assert_allclose(
            sharded_loss, single_loss, dtype=jnp.float32, atol=atol_out, rtol=rtol_out
        )
        assert_allclose(
            sharded_aux, single_aux, dtype=jnp.float32, atol=atol_out, rtol=rtol_out
        )

        for name in ("gate_kernel", "wi_0", "wi_1", "wo"):
            grad_single = _unwrap_partitioned(single_grads["params"][name])
            grad_sharded = _unwrap_partitioned(sharded_grads["params"][name])
            assert_allclose(
                grad_sharded,
                grad_single,
                dtype=DTYPE,
                atol=atol_grad,
                rtol=rtol_grad,
                err_msg=f"Distributed gradient mismatch for {name}",
            )
