# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Multi-process (one-GPU-per-process) tests for the TE-EP MoE custom_vjp.

The launcher ``tests/jax/run_te_ep_moe.sh`` forks one pytest process per
visible GPU. Each process binds to exactly one device via
``jax.distributed.initialize(..., local_device_ids=process_id)``; the
participating processes form a global ``(ep, fsdp)`` mesh through JAX's
distributed runtime.

How to run
----------

You typically do NOT invoke pytest on this file directly -- use the
launcher, which passes ``--num-process=N --process-id=i`` to each
forked process. Driving it directly with only one process will skip
every test because :func:`jax.distributed.initialize` requires
multiple participants, and the TE EP NCCL primitives require at
least four ranks.

    bash tests/jax/run_te_ep_moe.sh

What this suite covers
----------------------

Each test exercises one MoE-block run and bundles every check that
single run supports — shape, dtype,
finiteness AND numerical parity vs a pure-JAX reference. Variations
on the block are pytest parametrize values rather than separate test
classes:

* ``test_forward`` covers the forward across a curated set of
  configurations (softmax/sigmoid scoring, optional non-zero
  expert_bias). Each config asserts shape, dtype, finiteness and
  numerical parity vs the reference in one run.
* ``test_backward`` mirrors that for gradients.
* ``TestTeEpMoeAuxLoss`` covers the second return value end-to-end
  (returned + parity + aux-only grad propagates to gate + combined
  main+aux grads stay finite) in two consolidated tests.
"""

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")

import sys
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from flax.linen import partitioning as nn_partitioning


def _init_distributed(num_process: int, process_id: int) -> bool:
    """Initialize jax.distributed for this pytest process.

    Returns True on a real multi-process launch, False otherwise so
    the module can fast-skip when pytest collects it without the
    launcher.
    """
    if num_process <= 1:
        return False
    coord = os.environ.get("TE_EP_MOE_COORDINATOR_ADDRESS", "127.0.0.1:13457")
    jax.distributed.initialize(
        coordinator_address=coord,
        num_processes=num_process,
        process_id=process_id,
        local_device_ids=process_id,
    )
    assert jax.local_device_count() == 1, "one GPU per process is required for TE EP"
    assert (
        jax.device_count() == num_process
    ), f"global device_count {jax.device_count()} != num_process {num_process}"
    return True


def _read_mp_options():
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
    pytest.skip(
        "test_te_ep_moe.py requires the multiprocess launcher (run_te_ep_moe.sh). Skipping.",
        allow_module_level=True,
    )

from transformer_engine_jax import get_device_compute_capability

# Grouped GEMM in the MoE custom_vjp requires Blackwell (sm_100+). The
# TE EP NCCL primitives themselves need SM>=90, but the FFN body uses
# grouped_gemm, so the file as a whole gates on sm_100+.
if get_device_compute_capability(0) < 100:
    pytest.skip(
        "MoE TE EP tests require Blackwell (sm_100+) for grouped GEMM",
        allow_module_level=True,
    )

from transformer_engine.jax.flax import _MoEBlock as MoEBlock
from transformer_engine.common.recipe import MXFP8BlockScaling
from transformer_engine.jax import autocast
from transformer_engine.jax.moe import (
    _ALIGN_SIZE,
    _CUDNN_CUTEDSL_ALIGN_SIZE,
    _use_cudnn_cutedsl_fusion_from_env,
    moe,
    record_ep_bootstrap_signature_for_moe,
)
from transformer_engine.jax.ep import ep_bootstrap
from transformer_engine.jax.sharding import MeshResource, global_shard_guard


# -----------------------------------------------------------------------------
# Mesh / shape config
# -----------------------------------------------------------------------------

EP_AXIS = "ep"
FSDP_AXIS = "fsdp"
EP_SIZE = 2
assert (
    jax.device_count() % EP_SIZE == 0
), f"device_count {jax.device_count()} must be divisible by EP_SIZE={EP_SIZE}"
FSDP_SIZE = jax.device_count() // EP_SIZE
NUM_DEVICES_REQUIRED = EP_SIZE * FSDP_SIZE

LOGICAL_AXIS_RULES = (
    ("exp", EP_AXIS),
    ("embed", FSDP_AXIS),
    ("mlp", None),
    ("batch", (EP_AXIS, FSDP_AXIS)),
)


@pytest.fixture(autouse=True)
def _use_cutedsl_only_for_cutedsl_tests(request, monkeypatch):
    # run_te_ep_moe.sh may be launched with the CuTeDSL env var enabled so
    # the dedicated MXFP8 tests run. The non-MXFP8 tests should still exercise
    # the ordinary CUDA C++ path instead of strict opt-in rejection.
    cls = getattr(request.node, "cls", None)
    if cls is None or cls.__name__ != "TestTeEpMoeCudnnCutedslFusion":
        monkeypatch.setenv("NVTE_JAX_MOE_USE_CUDNN_CUTEDSL_FUSION", "0")

# Small shapes so the parity tests stay tight on bf16. The block still
# has all four ranks participating in dispatch/combine.  The explicit
# MaxText-shape regression mode is selected by its dedicated test invocation;
# keeping it opt-in avoids making every distributed parity test allocate the
# production-slice buffers.
_MAXTEXT_CUTEDSL_REGRESSION = os.getenv("TE_EP_MOE_MAXTEXT_CUTEDSL_REGRESSION", "0") == "1"
DTYPE = jnp.bfloat16
if _MAXTEXT_CUTEDSL_REGRESSION:
    # Matches run-dsv3-prod-slice-ep2-fsdp2.sh: global batch 16, sequence
    # length 4096, 32 experts, top-k 8, H=1792, expert MLP=2048.
    BATCH = 16
    SEQ = 4096
    HIDDEN = 1792
    INTER = 2048
    NUM_EXPERTS = 32
    TOPK = 8
else:
    BATCH = EP_SIZE * FSDP_SIZE * 2  # 8 on 4-GPU, 16 on 8-GPU
    SEQ = 32
    HIDDEN = 64
    INTER = 128
    NUM_EXPERTS = 8
    TOPK = 2

# bf16 grouped_gemm + softmax-topk + ep all-to-all stack drifts ~1e-1 vs a
# fp32 numpy reference. Keep these tight enough to catch real bugs but
# loose enough to absorb expected bf16 rounding.
FWD_ATOL = 5e-2
FWD_RTOL = 5e-2
GRAD_FFN_ATOL = 1e-1
GRAD_FFN_RTOL = 1e-1
GRAD_GATE_ATOL = 5e-1
GRAD_GATE_RTOL = 5e-1

# Two TE EP runs that should be bitwise-equal modulo XLA fusion order
# (slot alignment rounding, etc.).
TE_TO_TE_ATOL = 5e-3
TE_TO_TE_RTOL = 5e-3

# Aux loss is computed in float32 from the SAME logits as the routing
# path. Numerical drift between TE-EP and the reference is dominated by
# the bf16-rounded softmax inside the topk kernel.
AUX_ATOL = 1e-3
AUX_RTOL = 1e-3


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


def _compute_worst_case_recv_pr(alignment=_ALIGN_SIZE):
    """Per-rank recv buffer the bootstrap must reserve.

    NCCL EP HT expert-major uses one flat recv buffer with variable
    per-expert zones. Each non-empty expert zone is padded to
    ``_ALIGN_SIZE`` slots, so the reserve must cover the worst-case
    total assignments plus independent per-zone padding.
    """
    num_procs = jax.device_count()
    num_local_experts = NUM_EXPERTS // EP_SIZE
    max_tokens_per_rank = (BATCH // num_procs) * SEQ
    tokens_per_ep_group = EP_SIZE * max_tokens_per_rank
    max_local_assignments = tokens_per_ep_group * min(TOPK, num_local_experts)
    max_nonempty_experts = min(num_local_experts, max_local_assignments)
    padded_total_bound = max_local_assignments + (alignment - 1) * max_nonempty_experts
    aligned_total_bound = ((padded_total_bound + alignment - 1) // alignment) * alignment
    per_expert_bound = (
        num_local_experts * ((tokens_per_ep_group + alignment - 1) // alignment) * alignment
    )
    return min(per_expert_bound, aligned_total_bound)


@pytest.fixture(scope="module")
def mesh():
    if jax.device_count() < NUM_DEVICES_REQUIRED:
        pytest.skip(
            f"Need >={NUM_DEVICES_REQUIRED} devices for ep={EP_SIZE} x fsdp={FSDP_SIZE};"
            f" have {jax.device_count()}"
        )
    # ``ep`` must be the inner axis: ``ep_bootstrap`` forms NCCL EP groups
    # from consecutive global ranks via ``dp_color = rank // ep_size``, so
    # only an (outer_fsdp, inner_ep) device layout groups ranks correctly.
    devices = mesh_utils.create_device_mesh((FSDP_SIZE, EP_SIZE))
    mesh_obj = Mesh(devices, axis_names=(FSDP_AXIS, EP_AXIS))

    num_procs = jax.process_count()
    max_tokens_per_rank = (BATCH // num_procs) * SEQ
    fusion_enabled = _use_cudnn_cutedsl_fusion_from_env()
    alignment = _CUDNN_CUTEDSL_ALIGN_SIZE if fusion_enabled else _ALIGN_SIZE
    recv_capacity_per_rank = _compute_worst_case_recv_pr(alignment)

    # Eager bootstrap: ep_bootstrap does a host-side NCCL UID allgather
    # and cannot run from inside jax.jit. Sized to the worst-case recv_pr
    # across _CONFIGS so every parametrized config is bootstrap-compatible.
    with mesh_obj, global_shard_guard(MeshResource(ep_resource=EP_AXIS, fsdp_resource=FSDP_AXIS)):
        ep_bootstrap(
            world_size=num_procs,
            rank=jax.process_index(),
            num_experts=NUM_EXPERTS,
            max_tokens_per_rank=max_tokens_per_rank,
            recv_capacity_per_rank=recv_capacity_per_rank,
            hidden_dim=HIDDEN,
            max_token_dtype=DTYPE,
        )
    record_ep_bootstrap_signature_for_moe(
        num_experts=NUM_EXPERTS,
        max_tokens_per_rank=max_tokens_per_rank,
        recv_capacity_per_rank=recv_capacity_per_rank,
        hidden_dim=HIDDEN,
        ep_size=EP_SIZE,
    )
    return mesh_obj


# -----------------------------------------------------------------------------
# Pure-JAX reference MoE (no EP). Mirrors the exact math of TE's fused
# router primitive (see tests/jax/test_fused_router.py for the same
# reference applied to the standalone router kernel):
#
# softmax + post-softmax (use_pre_softmax=False, the default):
#   1. top_k by raw logits
#   2. softmax over just the K selected logits (so weights sum to 1)
#
# sigmoid + optional expert_bias:
#   1. scores = sigmoid(logits)
#   2. top_k by (scores + expert_bias)  [bias only steers selection]
#   3. weights = scores at top_k positions, normalized when K > 1
#
# Then for both:
#   * weights *= scaling_factor (we leave scaling_factor=1.0 in this
#     suite, matching _make_block's default).
#   * per-expert FFN: silu(layer_w0) * layer_w1 → wo.
# -----------------------------------------------------------------------------


@partial(
    jax.jit,
    static_argnames=(
        "num_experts",
        "num_experts_per_tok",
        "aux_loss_coeff",
        "score_function",
    ),
)
def _pure_jax_moe_reference(
    x,
    gate_kernel,
    wi_0,
    wi_1,
    wo,
    expert_bias=None,
    *,
    num_experts,
    num_experts_per_tok,
    aux_loss_coeff: float = 0.0,
    score_function: str = "softmax",
):
    B, S, H = x.shape
    T = B * S
    K = num_experts_per_tok
    x_2d = x.reshape(T, H)

    gate_kernel_cast = gate_kernel.astype(x.dtype)
    logits = (x_2d @ gate_kernel_cast).astype(jnp.float32)  # [T, E]

    if score_function == "softmax":
        # use_pre_softmax=False: topk on raw logits, then softmax over K.
        top_logits, top_indices = jax.lax.top_k(logits, k=K)
        weights = jax.nn.softmax(top_logits, axis=-1)  # [T, K], sums to 1
    elif score_function == "sigmoid":
        scores = jax.nn.sigmoid(logits)  # [T, E]
        if expert_bias is not None and expert_bias.shape != (0,):
            scores_for_routing = scores + expert_bias.astype(jnp.float32)[None, :]
            _, top_indices = jax.lax.top_k(scores_for_routing, k=K)
            weights = jnp.take_along_axis(scores, top_indices, axis=-1)
        else:
            weights, top_indices = jax.lax.top_k(scores, k=K)
        # Sigmoid weights are normalized when K > 1 (matches the kernel).
        if K > 1:
            weights = weights / (weights.sum(axis=-1, keepdims=True) + 1e-20)
    else:
        raise ValueError(f"Unsupported score_function={score_function!r}")

    routing_weights_full = jnp.zeros((T, num_experts), dtype=jnp.float32)
    routing_weights_full = routing_weights_full.at[jnp.arange(T)[:, None], top_indices].set(weights)

    # FFN. ``apply_topk_weights_early`` is a fusion knob that doesn't
    # change the math (wo is linear), so the reference is identical for
    # both placements.
    layer_w0 = jnp.einsum("th,ehm->tem", x_2d, wi_0)
    layer_w1 = jnp.einsum("th,ehm->tem", x_2d, wi_1)
    # Activation runs in x.dtype (typically bf16) to mirror the impl --
    # the impl keeps silu+multiply in the wi GEMM output dtype because
    # storing higher precision than the consumer (wo) GEMM buys nothing.
    intermediate = jax.nn.silu(layer_w0) * layer_w1
    expert_out = jnp.einsum("tem,emh->teh", intermediate, wo)  # [T, E, H]
    output_2d = jnp.einsum("te,teh->th", routing_weights_full.astype(x.dtype), expert_out)
    output = output_2d.reshape(B, S, H).astype(x.dtype)

    if aux_loss_coeff > 0.0:
        # tex.fused_moe_aux_loss formula (matches the same
        # reference_aux_loss helper from test_fused_router.py). The
        # "aux scores" use the same score_function but always with
        # K-normalised sigmoid (when sigmoid) / plain softmax (when
        # softmax) — see tex.fused_topk_with_score_function_fwd with
        # compute_aux_scores=True.
        if score_function == "softmax":
            aux_scores = jax.nn.softmax(logits, axis=-1)
        else:  # sigmoid
            aux_scores = jax.nn.sigmoid(logits)
            if K > 1:
                aux_scores = aux_scores / (aux_scores.sum(axis=-1, keepdims=True) + 1e-20)
        routing_map = (routing_weights_full > 0).astype(jnp.int32)
        tokens_per_expert = jnp.sum(routing_map, axis=0)  # [E]
        sum_probs_per_expert = jnp.sum(aux_scores, axis=0)  # [E]
        aux_loss = (num_experts * aux_loss_coeff / (K * (T**2))) * jnp.sum(
            sum_probs_per_expert * tokens_per_expert.astype(jnp.float32)
        )
        aux_loss = aux_loss.astype(x.dtype)
    else:
        aux_loss = jnp.zeros((), dtype=x.dtype)
    return output, aux_loss


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _make_block(
    *,
    apply_topk_weights_early=False,
    aux_loss_coeff=0.0,
    use_expert_routing_bias=False,
    score_function="softmax",
    scaling_factor=1.0,
    expert_bias_init=None,
):
    kwargs = dict(
        num_experts=NUM_EXPERTS,
        num_experts_per_tok=TOPK,
        intermediate_size=INTER,
        data_parallelism_axes=(FSDP_AXIS,),
        apply_topk_weights_early=apply_topk_weights_early,
        aux_loss_coeff=aux_loss_coeff,
        use_expert_routing_bias=use_expert_routing_bias,
        score_function=score_function,
        scaling_factor=scaling_factor,
        dtype=DTYPE,
    )
    # Custom expert_bias_init lets tests inject a non-zero expert_bias without
    # poking variables['params'] post-init.
    if expert_bias_init is not None:
        kwargs["expert_bias_init"] = expert_bias_init
    return MoEBlock(**kwargs)


def _strong_expert_bias_init(key, shape, dtype):
    """Half +5, half -5 — large enough to force topk onto the +ve half."""
    del key
    n = shape[0]
    return jnp.concatenate(
        [
            jnp.full((n // 2,), 5.0, dtype=dtype),
            jnp.full((n - n // 2,), -5.0, dtype=dtype),
        ]
    )


def _shard_inputs(x, mesh):
    # Match the layout moe.py re-pins to: outer dp axes, then ep innermost.
    return jax.lax.with_sharding_constraint(
        x, NamedSharding(mesh, P((FSDP_AXIS, EP_AXIS), None, None))
    )


def _ctx(mesh):
    """Combined mesh + global_shard_guard + axis_rules context."""

    class _Combo:
        def __enter__(self_inner):
            self_inner._m = mesh.__enter__()
            self_inner._gs = global_shard_guard(
                MeshResource(ep_resource=EP_AXIS, fsdp_resource=FSDP_AXIS)
            )
            self_inner._gs.__enter__()
            self_inner._ar = nn_partitioning.axis_rules(LOGICAL_AXIS_RULES)
            self_inner._ar.__enter__()
            return self_inner._m

        def __exit__(self_inner, *args):
            self_inner._ar.__exit__(*args)
            self_inner._gs.__exit__(*args)
            mesh.__exit__(*args)

    return _Combo()


def _init_apply(block, mesh, x, key):
    with _ctx(mesh):
        x_sh = _shard_inputs(x, mesh)
        variables = jax.jit(block.init)(key, x_sh)
        jax.block_until_ready(jax.tree_util.tree_leaves(variables)[0])
        output, aux = jax.jit(block.apply)(variables, x_sh)
        jax.block_until_ready(output)
    return variables, output, aux


def _grad_step(block, variables, mesh, x, *, include_aux=False):
    """Run jax.grad of mean(out^2) [+ aux if include_aux] vs (params, x).

    Returns ``(grads_variables, grad_x)`` so callers can check both the
    weight gradients and the input-activation gradient that propagates
    back to the previous layer.
    """
    with _ctx(mesh):
        x_sh = _shard_inputs(x, mesh)

        def loss_fn(variables, x):
            output, aux = block.apply(variables, x)
            loss = jnp.mean(output.astype(jnp.float32) ** 2)
            if include_aux and aux is not None:
                loss = loss + aux.astype(jnp.float32)
            return loss

        grads_v, grad_x = jax.jit(jax.grad(loss_fn, argnums=(0, 1)))(variables, x_sh)
        jax.block_until_ready(jax.tree_util.tree_leaves(grads_v)[0])
        jax.block_until_ready(grad_x)
        return grads_v, grad_x


def _grad_aux_only(block, variables, mesh, x):
    """Jit'd grad of just the aux loss scalar — proves it reaches the
    gate even when no main-output contribution is present."""
    with _ctx(mesh):
        x_sh = _shard_inputs(x, mesh)

        def aux_only(variables, x):
            _, aux = block.apply(variables, x)
            return aux.astype(jnp.float32)

        grads = jax.jit(jax.grad(aux_only))(variables, x_sh)
        jax.block_until_ready(jax.tree_util.tree_leaves(grads)[0])
        return grads


def _unwrap(x):
    return x.value if hasattr(x, "value") else x


def _to_global_numpy(arr, mesh):
    """Replicate a sharded JAX array onto every rank and return as numpy.

    Triggers an all-gather inside JIT. The resulting addressable_data(0)
    contains the full global array on every process, so we can run the
    pure-JAX reference and compare against it from any process.
    """
    rep = NamedSharding(mesh, P())
    with mesh:
        full = jax.jit(lambda a: jax.lax.with_sharding_constraint(a, rep))(arr)
        full.block_until_ready()
    return np.asarray(jax.device_get(full.addressable_data(0)))


def _params_global_numpy(variables, mesh):
    """Pull every entry of variables['params'] to a replicated numpy array."""
    params = variables["params"]
    return {name: _to_global_numpy(_unwrap(p), mesh) for name, p in params.items()}


def _make_inputs(key):
    """Generate a globally-identical input tensor on every process."""
    return jax.random.normal(key, (BATCH, SEQ, HIDDEN), dtype=DTYPE)


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# Parametrize variants exercised by both the forward and the backward
# parity tests. Each config is one MoE-block configuration the suite
# wants covered; the test body checks shape, dtype, finiteness AND
# numerical parity vs the same pure-JAX reference (which understands
# the same set of knobs).
# -----------------------------------------------------------------------------

_CONFIGS = [
    pytest.param(
        dict(score_function="softmax"),
        id="softmax",
    ),
    pytest.param(
        dict(score_function="softmax", apply_topk_weights_early=True),
        id="softmax-early-weighting",
    ),
    pytest.param(
        dict(score_function="sigmoid"),
        id="sigmoid",
    ),
    # NOTE: a ``sigmoid-bias-zero`` config (use_expert_routing_bias=True
    # with a zero-initialised bias buffer) was previously exercised
    # here. It was dropped because the routing math collapses to the
    # no-bias case when the buffer is zero -- ``sigmoid`` already
    # covers that numerical path. The bias-aware codepath is still
    # exercised by ``sigmoid-bias-strong`` below, which uses a
    # non-zero bias.
    pytest.param(
        dict(
            score_function="sigmoid",
            use_expert_routing_bias=True,
            expert_bias_init=_strong_expert_bias_init,
        ),
        id="sigmoid-bias-strong",
    ),
]


def _reference_kwargs_from_config(config, params_np):
    """Pick out the reference-relevant pieces of a parametrize config."""
    return dict(
        score_function=config.get("score_function", "softmax"),
        expert_bias=(
            jnp.asarray(params_np["expert_bias"])
            if config.get("use_expert_routing_bias", False)
            else None
        ),
    )


class TestTeEpMoeForward:
    """Per-config forward correctness in a single run: shape, dtype,
    finiteness AND numerical parity vs the pure-JAX reference."""

    @pytest.mark.parametrize("config", _CONFIGS)
    def test_forward(self, mesh, config):
        block = _make_block(**config)
        x = _make_inputs(jax.random.PRNGKey(0))
        variables, output, aux = _init_apply(block, mesh, x, jax.random.PRNGKey(1))

        # Shape / dtype / finiteness (cheap; on the local shard).
        assert output.shape == x.shape
        assert output.dtype == x.dtype
        out_local = np.asarray(jax.device_get(output.addressable_data(0)))
        assert np.all(np.isfinite(out_local)), "output has NaN/Inf"
        assert aux is None, "aux_loss should be None when aux_loss_coeff == 0"

        # Numerical parity (replicated global view -> single rank's numpy).
        params_np = _params_global_numpy(variables, mesh)
        x_np = np.asarray(jax.device_get(x))
        out_te_np = _to_global_numpy(output, mesh)

        out_ref, _ = _pure_jax_moe_reference(
            jnp.asarray(x_np),
            jnp.asarray(params_np["gate_kernel"]),
            jnp.asarray(params_np["wi_0"]),
            jnp.asarray(params_np["wi_1"]),
            jnp.asarray(params_np["wo"]),
            num_experts=NUM_EXPERTS,
            num_experts_per_tok=TOPK,
            **_reference_kwargs_from_config(config, params_np),
        )
        np.testing.assert_allclose(
            out_te_np.astype(np.float32),
            np.asarray(jax.device_get(out_ref)).astype(np.float32),
            atol=FWD_ATOL,
            rtol=FWD_RTOL,
            err_msg=f"forward parity breach for config={config}",
        )


class TestTeEpMoeBackward:
    """Per-config backward correctness in a single run: per-tensor
    grads finite, non-zero AND parity vs the pure-JAX reference."""

    @pytest.mark.parametrize("config", _CONFIGS)
    def test_backward(self, mesh, config):
        block = _make_block(**config)
        x = _make_inputs(jax.random.PRNGKey(2))
        variables, _, _ = _init_apply(block, mesh, x, jax.random.PRNGKey(3))
        grads_te, grad_x_te = _grad_step(block, variables, mesh, x)

        # Reference grads via jax.grad over the pure-JAX MoE with the
        # same config. argnums=(0, 1) so the reference also produces a
        # d_x for the propagated-gradient parity check below.
        params_np = _params_global_numpy(variables, mesh)
        x_np = np.asarray(jax.device_get(x))
        ref_kwargs = _reference_kwargs_from_config(config, params_np)
        ref_expert_bias = ref_kwargs.pop("expert_bias")

        def loss_fn(params, x):
            out, _ = _pure_jax_moe_reference(
                x,
                params["gate_kernel"],
                params["wi_0"],
                params["wi_1"],
                params["wo"],
                ref_expert_bias,
                num_experts=NUM_EXPERTS,
                num_experts_per_tok=TOPK,
                **ref_kwargs,
            )
            return jnp.mean(out.astype(jnp.float32) ** 2)

        grads_ref, grad_x_ref = jax.jit(jax.grad(loss_fn, argnums=(0, 1)))(
            {k: jnp.asarray(v) for k, v in params_np.items() if k != "expert_bias"},
            jnp.asarray(x_np),
        )
        grads_ref_np = {k: np.asarray(jax.device_get(v)) for k, v in grads_ref.items()}
        grad_x_ref_np = np.asarray(jax.device_get(grad_x_ref))

        for name in ("gate_kernel", "wi_0", "wi_1", "wo"):
            # Per-tensor: finite + non-zero + parity in one pass.
            g_te = _to_global_numpy(_unwrap(grads_te["params"][name]), mesh)
            assert np.all(np.isfinite(g_te)), f"{name} grad has NaN/Inf [config={config}]"
            assert np.any(g_te != 0.0), f"{name} grad identically zero [config={config}]"
            atol, rtol = (
                (GRAD_GATE_ATOL, GRAD_GATE_RTOL)
                if name == "gate_kernel"
                else (GRAD_FFN_ATOL, GRAD_FFN_RTOL)
            )
            np.testing.assert_allclose(
                g_te.astype(np.float32),
                grads_ref_np[name].astype(np.float32),
                atol=atol,
                rtol=rtol,
                err_msg=f"grad parity breach on {name} [config={config}]",
            )

        # d_x: the gradient propagated back to the previous layer. Checks
        # shape, dtype (must match x.dtype — protects the
        # _with_sharding_constraint_cast_bwd wrapper that casts the
        # fp32-promoted gate path back to bf16), finiteness, non-zero
        # AND numerical parity vs the pure-JAX reference d_x.
        grad_x_te_np = _to_global_numpy(grad_x_te, mesh)
        assert (
            grad_x_te.shape == x.shape
        ), f"d_x shape {grad_x_te.shape} != x.shape {x.shape} [config={config}]"
        assert (
            grad_x_te.dtype == x.dtype
        ), f"d_x dtype {grad_x_te.dtype} != x.dtype {x.dtype} [config={config}]"
        assert np.all(np.isfinite(grad_x_te_np)), f"d_x has NaN/Inf [config={config}]"
        assert np.any(grad_x_te_np != 0.0), f"d_x identically zero [config={config}]"
        np.testing.assert_allclose(
            grad_x_te_np.astype(np.float32),
            grad_x_ref_np.astype(np.float32),
            atol=GRAD_FFN_ATOL,
            rtol=GRAD_FFN_RTOL,
            err_msg=f"d_x parity breach [config={config}]",
        )


class TestTeEpMoeCudnnCutedslFusion:
    """End-to-end MXFP8 coverage for the opt-in FC1+SwiGLU+quant fusion."""

    @pytest.mark.parametrize("apply_topk_weights_early", [False, True])
    def test_mxfp8_forward_and_backward(self, mesh, apply_topk_weights_early, monkeypatch):
        if not _use_cudnn_cutedsl_fusion_from_env():
            pytest.skip("run separately with NVTE_JAX_MOE_USE_CUDNN_CUTEDSL_FUSION=1")
        block = _make_block(apply_topk_weights_early=apply_topk_weights_early)
        x = _make_inputs(jax.random.PRNGKey(30))
        mesh_resource = MeshResource(ep_resource=EP_AXIS, fsdp_resource=FSDP_AXIS)

        with _ctx(mesh), autocast(
            enabled=True,
            recipe=MXFP8BlockScaling(),
            mesh_resource=mesh_resource,
        ):
            x_sh = _shard_inputs(x, mesh)
            variables = jax.jit(block.init)(jax.random.PRNGKey(31), x_sh)
            fused, _ = jax.jit(block.apply)(variables, x_sh)

            def loss_fn(vars_arg, x_arg):
                output, _ = block.apply(vars_arg, x_arg)
                return jnp.mean(output.astype(jnp.float32) ** 2)

            grads, grad_x = jax.jit(jax.grad(loss_fn, argnums=(0, 1)))(variables, x_sh)
            jax.block_until_ready((fused, grads, grad_x))

        # Compile the same block and parameters through the ordinary
        # grouped_gemm + SwiGLU + grouped_quantize path. The 256-aligned
        # bootstrap allocation used by the fused run is also wide enough for
        # the unfused 128-aligned dispatch.
        monkeypatch.setenv("NVTE_JAX_MOE_USE_CUDNN_CUTEDSL_FUSION", "0")
        with _ctx(mesh), autocast(
            enabled=True,
            recipe=MXFP8BlockScaling(),
            mesh_resource=mesh_resource,
        ):
            unfused, _ = jax.jit(block.apply)(variables, _shard_inputs(x, mesh))

            def unfused_loss_fn(vars_arg, x_arg):
                output, _ = block.apply(vars_arg, x_arg)
                return jnp.mean(output.astype(jnp.float32) ** 2)

            unfused_grads, unfused_grad_x = jax.jit(jax.grad(unfused_loss_fn, argnums=(0, 1)))(
                variables, _shard_inputs(x, mesh)
            )
            jax.block_until_ready((unfused, unfused_grads, unfused_grad_x))
        monkeypatch.setenv("NVTE_JAX_MOE_USE_CUDNN_CUTEDSL_FUSION", "1")

        fused_np = _to_global_numpy(fused, mesh).astype(np.float32)
        unfused_np = _to_global_numpy(unfused, mesh).astype(np.float32)
        assert np.all(np.isfinite(fused_np))
        relative_error = np.linalg.norm(fused_np - unfused_np) / max(
            np.linalg.norm(unfused_np), 1e-12
        )
        assert relative_error < 0.2, f"fused/unfused forward relative error {relative_error:.4f}"

        params_np = _params_global_numpy(variables, mesh)
        reference, _ = _pure_jax_moe_reference(
            jnp.asarray(jax.device_get(x)),
            jnp.asarray(params_np["gate_kernel"]),
            jnp.asarray(params_np["wi_0"]),
            jnp.asarray(params_np["wi_1"]),
            jnp.asarray(params_np["wo"]),
            num_experts=NUM_EXPERTS,
            num_experts_per_tok=TOPK,
        )
        reference_np = np.asarray(jax.device_get(reference), dtype=np.float32)
        relative_error = np.linalg.norm(fused_np - reference_np) / np.linalg.norm(reference_np)
        assert relative_error < 0.2

        for name in ("gate_kernel", "wi_0", "wi_1", "wo"):
            grad = _to_global_numpy(_unwrap(grads["params"][name]), mesh).astype(np.float32)
            assert np.all(np.isfinite(grad)), f"{name} fused MXFP8 grad has NaN/Inf"
            assert np.any(grad != 0), f"{name} fused MXFP8 grad is identically zero"
        grad_x_np = _to_global_numpy(grad_x, mesh).astype(np.float32)
        unfused_grad_x_np = _to_global_numpy(unfused_grad_x, mesh).astype(np.float32)
        assert np.all(np.isfinite(grad_x_np))
        assert np.any(grad_x_np != 0)
        relative_error = np.linalg.norm(grad_x_np - unfused_grad_x_np) / max(
            np.linalg.norm(unfused_grad_x_np), 1e-12
        )
        assert relative_error < 0.35, f"d_x fused/unfused relative error {relative_error:.4f}"

        for name in ("gate_kernel", "wi_0", "wi_1", "wo"):
            fused_grad = _to_global_numpy(_unwrap(grads["params"][name]), mesh).astype(np.float32)
            unfused_grad = _to_global_numpy(_unwrap(unfused_grads["params"][name]), mesh).astype(
                np.float32
            )
            relative_error = np.linalg.norm(fused_grad - unfused_grad) / max(
                np.linalg.norm(unfused_grad), 1e-12
            )
            assert (
                relative_error < 0.35
            ), f"{name} fused/unfused VJP relative error {relative_error:.4f} exceeds 0.35"

        # A finite gradient is not sufficient: a layout error can produce a
        # finite but numerically invalid VJP that corrupts parameters on the
        # first optimizer update. Compare all learnable gradients to the same
        # pure-JAX reference used by the non-quantized VJP tests.
        def reference_loss(params, x_arg):
            output, _ = _pure_jax_moe_reference(
                x_arg,
                params["gate_kernel"],
                params["wi_0"],
                params["wi_1"],
                params["wo"],
                num_experts=NUM_EXPERTS,
                num_experts_per_tok=TOPK,
            )
            return jnp.mean(output.astype(jnp.float32) ** 2)

        reference_params = {
            name: jnp.asarray(params_np[name]) for name in ("gate_kernel", "wi_0", "wi_1", "wo")
        }
        reference_grads = jax.jit(jax.grad(reference_loss))(
            reference_params, jnp.asarray(jax.device_get(x))
        )
        for name, reference_grad in reference_grads.items():
            fused_grad = _to_global_numpy(_unwrap(grads["params"][name]), mesh).astype(np.float32)
            reference_grad = np.asarray(jax.device_get(reference_grad), dtype=np.float32)
            relative_error = np.linalg.norm(fused_grad - reference_grad) / max(
                np.linalg.norm(reference_grad), 1e-12
            )
            assert (
                relative_error < 0.35
            ), f"{name} fused MXFP8 VJP relative error {relative_error:.4f} exceeds 0.35"

        # Exercise the failure mode seen in MaxText: apply one optimizer-like
        # update and require the next forward pass to remain finite.
        updated_variables = jax.tree_util.tree_map(
            lambda param, grad: param - jnp.asarray(1e-3, param.dtype) * grad.astype(param.dtype),
            variables,
            grads,
        )
        with _ctx(mesh), autocast(
            enabled=True,
            recipe=MXFP8BlockScaling(),
            mesh_resource=mesh_resource,
        ):
            updated_output, _ = jax.jit(block.apply)(updated_variables, _shard_inputs(x, mesh))
            updated_output.block_until_ready()
        updated_output_np = _to_global_numpy(updated_output, mesh).astype(np.float32)
        assert np.all(np.isfinite(updated_output_np)), "post-update output has NaN/Inf"

    def test_maxtext_shape_vjp_update_stays_finite(self, mesh):
        """Regression for the NaN observed after MaxText's first update."""
        if not _use_cudnn_cutedsl_fusion_from_env():
            pytest.skip("run separately with NVTE_JAX_MOE_USE_CUDNN_CUTEDSL_FUSION=1")
        if not _MAXTEXT_CUTEDSL_REGRESSION:
            pytest.skip("set TE_EP_MOE_MAXTEXT_CUTEDSL_REGRESSION=1 for the large regression")

        block = _make_block(
            score_function="sigmoid",
            use_expert_routing_bias=True,
            scaling_factor=2.5,
        )
        x = _make_inputs(jax.random.PRNGKey(40))
        mesh_resource = MeshResource(ep_resource=EP_AXIS, fsdp_resource=FSDP_AXIS)
        with _ctx(mesh), autocast(
            enabled=True,
            recipe=MXFP8BlockScaling(),
            mesh_resource=mesh_resource,
        ):
            x_sh = _shard_inputs(x, mesh)
            variables = jax.jit(block.init)(jax.random.PRNGKey(41), x_sh)

            compiled_forward = jax.jit(block.apply)
            forward_0, _ = compiled_forward(variables, x_sh)
            forward_1, _ = compiled_forward(variables, x_sh)
            jax.block_until_ready((forward_0, forward_1))
            forward_0_local = np.asarray(jax.device_get(forward_0.addressable_data(0)))
            forward_1_local = np.asarray(jax.device_get(forward_1.addressable_data(0)))
            assert np.all(np.isfinite(forward_0_local)), "first repeated forward has NaN/Inf"
            assert np.all(np.isfinite(forward_1_local)), "second repeated forward has NaN/Inf"

            def loss_fn(vars_arg, x_arg):
                output, _ = block.apply(vars_arg, x_arg)
                return jnp.mean(output.astype(jnp.float32) ** 2)

            def train_step(vars_arg, x_arg, learning_rate):
                loss, grads = jax.value_and_grad(loss_fn)(vars_arg, x_arg)
                updated_vars = jax.tree_util.tree_map(
                    lambda param, grad: param
                    - learning_rate.astype(param.dtype) * grad.astype(param.dtype),
                    vars_arg,
                    grads,
                )
                return updated_vars, loss, grads

            compiled_train_step = jax.jit(train_step)
            # MaxText's two-step warmup uses LR=0 at step 0. The second
            # invocation therefore checks that replaying the same compiled
            # kernel/VJP is safe even before any parameter value changes.
            variables, loss_0, grads = compiled_train_step(
                variables, x_sh, jnp.asarray(0.0, jnp.float32)
            )
            jax.block_until_ready((loss_0, grads))
            assert np.isfinite(float(loss_0.addressable_data(0))), "step-0 loss has NaN/Inf"
            for path, grad in jax.tree_util.tree_leaves_with_path(grads):
                grad_local = np.asarray(jax.device_get(_unwrap(grad).addressable_data(0)))
                assert np.all(np.isfinite(grad_local)), f"gradient {path} has NaN/Inf"
            variables, loss_1, _ = compiled_train_step(
                variables, x_sh, jnp.asarray(1.5e-5, jnp.float32)
            )
            jax.block_until_ready(loss_1)
            assert np.isfinite(float(loss_1.addressable_data(0))), "step-1 loss has NaN/Inf"

            updated_output, _ = jax.jit(block.apply)(variables, x_sh)
            updated_output.block_until_ready()

        updated_local = np.asarray(jax.device_get(updated_output.addressable_data(0)))
        assert np.all(np.isfinite(updated_local)), "MaxText-shape post-update output has NaN/Inf"


class TestTeEpMoeAuxLoss:
    """Aux-loss path. Consolidated into:
    * ``test_aux_loss``: one run that checks the returned scalar's
      shape / dtype / finiteness / magnitude AND numerical parity vs the
      reference AND that the aux-only bwd propagates to gate_kernel.
    * ``test_combined_loss_grads``: one run for joint main+aux bwd
      finite + non-zero per tensor.
    """

    def test_aux_loss(self, mesh):
        coeff = 1e-2
        block = _make_block(aux_loss_coeff=coeff)
        x = _make_inputs(jax.random.PRNGKey(20))
        variables, _, aux = _init_apply(block, mesh, x, jax.random.PRNGKey(21))

        # Shape / dtype / finiteness / magnitude.
        assert aux is not None, "aux_loss should be returned when coeff > 0"
        assert aux.shape == (), f"aux_loss must be 0-d scalar, got {aux.shape}"
        assert aux.dtype == DTYPE, f"aux_loss dtype {aux.dtype} != {DTYPE}"
        aux_np = _to_global_numpy(aux, mesh)
        assert np.isfinite(aux_np), "aux_loss is NaN/Inf"
        assert abs(float(aux_np)) < 1e2, f"aux_loss looks unreasonable: {aux_np}"

        # Numerical parity vs the reference.
        params_np = _params_global_numpy(variables, mesh)
        x_np = np.asarray(jax.device_get(x))
        _, aux_ref = _pure_jax_moe_reference(
            jnp.asarray(x_np),
            jnp.asarray(params_np["gate_kernel"]),
            jnp.asarray(params_np["wi_0"]),
            jnp.asarray(params_np["wi_1"]),
            jnp.asarray(params_np["wo"]),
            num_experts=NUM_EXPERTS,
            num_experts_per_tok=TOPK,
            aux_loss_coeff=coeff,
        )
        np.testing.assert_allclose(
            float(aux_np),
            float(jax.device_get(aux_ref)),
            atol=AUX_ATOL,
            rtol=AUX_RTOL,
        )

        # Aux-only bwd must propagate to gate_kernel — proves the
        # fused_moe_aux_loss_bwd → topk(compute_aux_scores)_bwd chain is
        # wired.
        aux_grads = _grad_aux_only(block, variables, mesh, x)
        g_gate = np.asarray(
            jax.device_get(_unwrap(aux_grads["params"]["gate_kernel"]).addressable_data(0))
        )
        assert np.all(np.isfinite(g_gate)), "gate grad NaN/Inf under aux-only loss"
        assert np.any(g_gate != 0.0), "aux bwd should propagate to gate_kernel"

    def test_combined_loss_grads(self, mesh):
        """Joint main + aux loss bwd: per-tensor finite + non-zero in
        one pass."""
        block = _make_block(aux_loss_coeff=1e-2)
        x = _make_inputs(jax.random.PRNGKey(22))
        variables, _, _ = _init_apply(block, mesh, x, jax.random.PRNGKey(23))
        grads, _ = _grad_step(block, variables, mesh, x, include_aux=True)
        for name in ("gate_kernel", "wi_0", "wi_1", "wo"):
            g_local = np.asarray(jax.device_get(_unwrap(grads["params"][name]).addressable_data(0)))
            assert np.all(np.isfinite(g_local)), f"{name} grad NaN/Inf under main+aux"
            assert np.any(g_local != 0.0), f"{name} grad zero under main+aux"
