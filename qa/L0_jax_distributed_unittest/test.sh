# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas

function error_exit() {
    echo "Error: $1"
    exit 1
}

function test_fail() {
    RET=1
    FAILED_CASES="$FAILED_CASES $1"
    echo "Error: sub-test failed: $1"
}

RET=0
FAILED_CASES=""

export NVTE_JAX_TEST_TIMING=1

: ${TE_PATH:=/opt/transformerengine}
: ${XML_LOG_DIR:=/logs}
mkdir -p "$XML_LOG_DIR"

pip3 install -r $TE_PATH/examples/jax/encoder/requirements.txt || error_exit "Failed to install requirements"

# Make encoder tests to have run-to-run deterministic to have the stable CI results
export XLA_FLAGS="${XLA_FLAGS} --xla_gpu_deterministic_ops"
python3 -m pytest -c $TE_PATH/tests/jax/pytest.ini -v --junitxml=$XML_LOG_DIR/pytest_test_multigpu_encoder.xml $TE_PATH/examples/jax/encoder/test_multigpu_encoder.py || test_fail "test_multigpu_encoder.py"
wait
python3 -m pytest -c $TE_PATH/tests/jax/pytest.ini -v --junitxml=$XML_LOG_DIR/pytest_test_model_parallel_encoder.xml $TE_PATH/examples/jax/encoder/test_model_parallel_encoder.py || test_fail "test_model_parallel_encoder.py"
wait
TE_PATH=$TE_PATH bash $TE_PATH/examples/jax/encoder/run_test_multiprocessing_encoder.sh || test_fail "run_test_multiprocessing_encoder.sh"
wait

TE_PATH=$TE_PATH bash $TE_PATH/examples/jax/collective_gemm/run_test_cgemm.sh || test_fail "run_test_cgemm.sh"
wait

# MoE custom_vjp distributed (Level 2 smoke + Level 3 perf). Single-host
# multi-GPU; requires >=4 visible GPUs.
#
# Flags required for this file (mirrored in tests/jax/run_distributed_moe_vjp.sh):
#
# * ``-p no:typeguard`` — jaxtyping's pytest plugin auto-loads typeguard,
#   whose @typechecked import hook materialises JAX tracers via isinstance()
#   checks during shard_map tracing. We disable it only here (other jax tests
#   need it for type-hint validation).
# * ``XLA_PYTHON_CLIENT_PREALLOCATE=false`` + ``MEM_FRACTION=0.5`` —
#   prevents NCCL OOM during EP all-to-all communicator setup (default 90%
#   preallocation leaves no room).
# * ``CUDA_LAUNCH_BLOCKING=1`` — workaround for an async-dispatch hang
#   between Triton custom_calls with ``input_output_aliases`` and the
#   downstream NCCL ragged_all_to_all in this test's bwd path. Without it,
#   MainThread parks in _pjit_call_impl_python and one GPU pins at 100%
#   forever. With it, the smoke suite passes in <1 min. See
#   ``tests/jax/test_distributed_moe_vjp.py`` module docstring for the
#   bisection record and TODO for the proper fix.
XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 \
    CUDA_LAUNCH_BLOCKING=1 \
    python3 -m pytest -c $TE_PATH/tests/jax/pytest.ini -v -s \
    -p no:typeguard \
    --junitxml=$XML_LOG_DIR/pytest_test_distributed_moe_vjp.xml \
    $TE_PATH/tests/jax/test_distributed_moe_vjp.py || test_fail "test_distributed_moe_vjp.py"
wait

if [ $RET -ne 0 ]; then
    echo "Error: some sub-tests failed: $FAILED_CASES"
    exit 1
fi
echo "All tests passed"
exit 0
