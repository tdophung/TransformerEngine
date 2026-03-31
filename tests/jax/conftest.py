# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""conftest for tests/jax"""
import os
import pathlib
import subprocess
import sys
import time
import jax
import pytest
from collections import defaultdict


import transformer_engine.jax
from transformer_engine_jax import get_device_compute_capability
from transformer_engine.jax.version_utils import (
    TRITON_EXTENSION_MIN_JAX_VERSION,
    is_triton_extension_supported,
)


@pytest.fixture(autouse=True, scope="function")
def clear_live_arrays():
    """
    Clear all live arrays to keep the resource clean
    """
    yield
    for arr in jax.live_arrays():
        arr.delete()


@pytest.fixture(autouse=True, scope="module")
def enable_fused_attn_after_hopper():
    """
    Enable fused attn for hopper+ arch.
    Fused attn kernels on pre-hopper arch are not deterministic.
    """
    if get_device_compute_capability(0) >= 90:
        os.environ["NVTE_FUSED_ATTN"] = "1"
    yield
    if "NVTE_FUSED_ATTN" in os.environ:
        del os.environ["NVTE_FUSED_ATTN"]


class TestTimingPlugin:
    """
    Plugin to measure test execution time. Enable test timing by setting NVTE_JAX_TEST_TIMING=1
    in the environment.
    """

    def __init__(self):
        self.test_timings = defaultdict(list)

    @pytest.hookimpl(tryfirst=True)
    def pytest_runtest_setup(self, item):
        item._timing_start = time.time()

    @pytest.hookimpl(trylast=True)
    def pytest_runtest_teardown(self, item, nextitem):
        if hasattr(item, "_timing_start"):
            duration = time.time() - item._timing_start

            # Extract base function name without parameters
            test_name = item.name
            if "[" in test_name:
                base_name = test_name.split("[")[0]
            else:
                base_name = test_name

            self.test_timings[base_name].append(duration)

    def pytest_sessionfinish(self, session, exitstatus):
        print("\n" + "=" * 80)
        print("TEST RUNTIME SUMMARY (grouped by function)")
        print("=" * 80)

        total_overall = 0
        for test_name, durations in sorted(self.test_timings.items()):
            total_time = sum(durations)
            count = len(durations)
            avg_time = total_time / count if count > 0 else 0
            total_overall += total_time

            print(f"{test_name:<60} | {count:3}x | {total_time:7.2f}s | avg: {avg_time:6.2f}s")

        print("=" * 80)
        print(f"{'TOTAL RUNTIME':<60} | {'':>3}  | {total_overall:7.2f}s |")
        print("=" * 80)


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "triton: mark test (or test class) as requiring JAX Triton kernel support"
        f" (JAX >= {TRITON_EXTENSION_MIN_JAX_VERSION})."
        " Apply per test/class with @pytest.mark.triton so non-Triton tests in the same file run on"
        " old JAX.",
    )
    if os.getenv("NVTE_JAX_TEST_TIMING", "0") == "1":
        config.pluginmanager.register(TestTimingPlugin(), "test_timing")


@pytest.hookimpl(trylast=True)
def pytest_runtest_makereport(item, call):
    """On test_sort_chunks_by_index failure, spawn a VLOG+HLO-dump re-run of the minimal trigger.

    Activated only when NVTE_SORT_CHUNKS_VLOG_ON_FAIL=1 is set. Useful when running the full
    test suite and you want VLOG output from XLA's buffer-assignment pass specifically for the
    failing sort_chunks compilation, without flooding the main log with VLOG from all other tests.

    Environment variables used in the spawned subprocess:
        TF_CPP_MIN_LOG_LEVEL=0          — show all C++ log messages
        TF_CPP_VMODULE=buffer_assignment=3  — VLOG level 3 for buffer_assignment.cc only
        XLA_FLAGS+=--xla_dump_to=<dir>  — write HLO files to disk
        XLA_FLAGS+=--xla_dump_hlo_pass_re=buffer-assignment  — only the buffer-assignment pass
        XLA_FLAGS+=--xla_dump_hlo_module_re=sort_chunks      — only sort_chunks modules

    The subprocess runs test_fused_attn.py (to recreate XLA memory pressure) followed by
    test_permutation.py filtered to test_sort_chunks_by_index. Whether the bug reproduces in
    the subprocess is probabilistic, but the HLO dump is written regardless and VLOG output
    is captured to <dump_dir>/vlog_repro.log.
    """
    if os.getenv("NVTE_SORT_CHUNKS_VLOG_ON_FAIL", "0") != "1":
        return
    if call.when != "call" or call.excinfo is None:
        return
    if "test_sort_chunks_by_index" not in item.nodeid:
        return

    te_root = os.environ.get(
        "TE_PATH",
        str(pathlib.Path(str(item.fspath)).parent.parent.parent),
    )
    dump_dir = f"/tmp/xla_sort_chunks_{int(time.time())}"
    os.makedirs(dump_dir, exist_ok=True)
    vlog_log = f"{dump_dir}/vlog_repro.log"

    env = os.environ.copy()
    env["TF_CPP_MIN_LOG_LEVEL"] = "0"
    env["TF_CPP_VMODULE"] = "buffer_assignment=3"
    existing_xla = env.get("XLA_FLAGS", "")
    env["XLA_FLAGS"] = " ".join(
        filter(
            None,
            [
                existing_xla,
                f"--xla_dump_to={dump_dir}",
                "--xla_dump_hlo_pass_re=buffer-assignment",
                "--xla_dump_hlo_module_re=sort_chunks",
            ],
        )
    )
    # Do not re-trigger this hook in the subprocess
    env["NVTE_SORT_CHUNKS_VLOG_ON_FAIL"] = "0"

    print(f"\n[conftest] test_sort_chunks_by_index FAILED — spawning VLOG re-run")
    print(f"[conftest] Failing test: {item.nodeid}")
    print(f"[conftest] XLA dump dir: {dump_dir}")
    print(f"[conftest] VLOG log:     {vlog_log}")

    cmd = [
        sys.executable,
        "-m",
        "pytest",
        f"{te_root}/tests/jax/test_fused_attn.py",
        f"{te_root}/tests/jax/test_permutation.py",
        "-k",
        "test_sort_chunks_by_index",
        "-v",
        "--no-header",
        "-c",
        f"{te_root}/tests/jax/pytest.ini",
    ]

    with open(vlog_log, "w") as f:
        subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)

    print(f"[conftest] VLOG re-run complete. Check {dump_dir}/")


def pytest_collection_modifyitems(config, items):
    """Skip tests marked 'triton' when JAX is too old for Triton kernel dispatch."""
    if is_triton_extension_supported():
        return
    skip_triton = pytest.mark.skip(
        reason=(
            f"JAX >= {TRITON_EXTENSION_MIN_JAX_VERSION} required for Triton kernel support. "
            "Triton kernel dispatch segfaults with older jaxlib. "
            "Upgrade with: pip install --upgrade jax jaxlib"
        )
    )
    for item in items:
        if item.get_closest_marker("triton"):
            item.add_marker(skip_triton)
