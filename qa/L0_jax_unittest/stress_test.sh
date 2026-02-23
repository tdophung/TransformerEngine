!/bin/bash
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
#
# Stress test script for JAX L0 unittest suite.
# Runs the test suite multiple times and reports pass/fail statistics.
#
# Usage:
#   bash stress_test.sh [OPTIONS]
#
# Options:
#   -n NUM_RUNS       Number of times to run the suite (default: 30)
#   -k TEST_FILTER    pytest -k filter expression (default: run all non-distributed)
#   -c                Clear Triton cache between runs
#   -f                Fail fast: stop on first failure (for investigation)
#   -t TEST_FILE      Specific test file (default: all jax tests)
#   -o OUTPUT_DIR     Directory for logs (default: ./stress_test_logs)
#   -h                Show this help
#
# Examples:
#   # Run full L0 suite 30 times:
#   bash stress_test.sh -n 30
#
#   # Run only sort_chunks tests 50 times, clearing cache:
#   bash stress_test.sh -n 50 -k "test_sort_chunks" -c
#
#   # Run test_permutation.py 100 times, stop on first failure:
#   bash stress_test.sh -n 100 -t tests/jax/test_permutation.py -f
#
#   # Reproduce CI exactly (L0 suite):
#   bash stress_test.sh -n 30

set -o pipefail

export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
export NVTE_JAX_TEST_TIMING=1

# --- Defaults ---
NUM_RUNS=30
TEST_FILTER=""
CLEAR_CACHE=0
FAIL_FAST=0
TEST_FILE=""
OUTPUT_DIR="./stress_test_logs"

# --- Parse args ---
usage() {
    head -35 "$0" | tail -30
    exit 0
}

while getopts "n:k:cft:o:h" opt; do
    case $opt in
        n) NUM_RUNS=$OPTARG ;;
        k) TEST_FILTER=$OPTARG ;;
        c) CLEAR_CACHE=1 ;;
        f) FAIL_FAST=1 ;;
        t) TEST_FILE=$OPTARG ;;
        o) OUTPUT_DIR=$OPTARG ;;
        h) usage ;;
        *) usage ;;
    esac
done

# --- Resolve paths ---
#: ${TE_PATH:=$(cd "$(dirname "$0")/../.." && pwd)}
TE_PATH="/mnt/tdophung/ptyche-lustre-home/TransformerEngine"
PYTEST_INI="$TE_PATH/tests/jax/pytest.ini"

if [ -n "$TEST_FILE" ]; then
    # User specified a specific file (relative to TE_PATH or absolute)
    if [[ "$TEST_FILE" = /* ]]; then
        TEST_TARGET="$TEST_FILE"
    else
        TEST_TARGET="$TE_PATH/$TEST_FILE"
    fi
else
    TEST_TARGET="$TE_PATH/tests/jax"
fi

PYTEST_ARGS="-c $PYTEST_INI -v"
if [ -n "$TEST_FILTER" ]; then
    # Always exclude distributed; add user filter
    PYTEST_ARGS="$PYTEST_ARGS -k 'not distributed and ($TEST_FILTER)'"
else
    PYTEST_ARGS="$PYTEST_ARGS -k 'not distributed'"
fi

mkdir -p "$OUTPUT_DIR"

# --- Environment info ---
echo "================================================================================"
echo "  JAX L0 Unittest Stress Test"
echo "================================================================================"
echo "TE_PATH:         $TE_PATH"
echo "Test target:     $TEST_TARGET"
echo "Filter (-k):     ${TEST_FILTER:-<none>}"
echo "Num runs:        $NUM_RUNS"
echo "Clear cache:     $CLEAR_CACHE"
echo "Fail fast:       $FAIL_FAST"
echo "Output dir:      $OUTPUT_DIR"
echo ""

# Print environment details
python3 -c "
import jax; import triton
print(f'JAX version:     {jax.__version__}')
print(f'Triton version:  {triton.__version__}')
print(f'Devices:         {jax.devices()}')
" 2>/dev/null || echo "(Could not query JAX/Triton versions)"

nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | head -1 | \
    awk -F',' '{printf "GPU:             %s\nDriver:          %s\n", $1, $2}'
echo "================================================================================"
echo ""

# --- Run loop ---
PASS_COUNT=0
FAIL_COUNT=0
FAIL_RUNS=""
declare -A FAIL_TESTS  # track which tests failed and how many times

START_ALL=$(date +%s)

for i in $(seq 1 $NUM_RUNS); do
    RUN_LOG="$OUTPUT_DIR/run_${i}.log"
    RUN_XML="$OUTPUT_DIR/run_${i}.xml"

    echo -n "Run $i/$NUM_RUNS ... "

    # Optionally clear Triton cache
    if [ $CLEAR_CACHE -eq 1 ]; then
        rm -rf ~/.triton/cache
    fi

    # Run pytest (-s to show print output for diagnostics, --tb=long for details)
    RUN_START=$(date +%s)
    eval python3 -m pytest $PYTEST_ARGS \
        --junitxml="$RUN_XML" \
        --tb=long \
        -s \
        "$TEST_TARGET" \
        > "$RUN_LOG" 2>&1
    EXIT_CODE=$?
    RUN_END=$(date +%s)
    RUN_DURATION=$((RUN_END - RUN_START))

    if [ $EXIT_CODE -eq 0 ]; then
        PASS_COUNT=$((PASS_COUNT + 1))
        # Extract summary line
        SUMMARY=$(grep -E "passed" "$RUN_LOG" | tail -1)
        echo "PASSED (${RUN_DURATION}s) - $SUMMARY"
    else
        FAIL_COUNT=$((FAIL_COUNT + 1))
        FAIL_RUNS="$FAIL_RUNS $i"

        # Extract which tests failed
        FAILED_NAMES=$(grep -E "^FAILED " "$RUN_LOG" | sed 's/^FAILED //' | sed 's/ -.*$//')
        if [ -z "$FAILED_NAMES" ]; then
            # Try alternate format
            FAILED_NAMES=$(grep -E "FAILED" "$RUN_LOG" | head -5)
        fi

        SUMMARY=$(grep -E "passed|failed|error" "$RUN_LOG" | tail -1)
        echo "FAILED (${RUN_DURATION}s) - $SUMMARY"
        echo "         Failed tests:"

        # Track per-test failure counts
        while IFS= read -r test_name; do
            if [ -n "$test_name" ]; then
                echo "           - $test_name"
                FAIL_TESTS["$test_name"]=$(( ${FAIL_TESTS["$test_name"]:-0} + 1 ))
            fi
        done <<< "$FAILED_NAMES"

        if [ $FAIL_FAST -eq 1 ]; then
            echo ""
            echo ">>> Fail-fast enabled. Stopping after first failure."
            echo ">>> Full log: $RUN_LOG"
            break
        fi
    fi
done

END_ALL=$(date +%s)
TOTAL_DURATION=$((END_ALL - START_ALL))

# --- Summary ---
echo ""
echo "================================================================================"
echo "  STRESS TEST SUMMARY"
echo "================================================================================"
TOTAL_RUNS=$((PASS_COUNT + FAIL_COUNT))
if [ $TOTAL_RUNS -gt 0 ]; then
    PASS_PCT=$(awk "BEGIN {printf \"%.1f\", ($PASS_COUNT/$TOTAL_RUNS)*100}")
    FAIL_PCT=$(awk "BEGIN {printf \"%.1f\", ($FAIL_COUNT/$TOTAL_RUNS)*100}")
else
    PASS_PCT="N/A"
    FAIL_PCT="N/A"
fi

echo "Total runs:      $TOTAL_RUNS"
echo "Passed:          $PASS_COUNT ($PASS_PCT%)"
echo "Failed:          $FAIL_COUNT ($FAIL_PCT%)"
echo "Total duration:  ${TOTAL_DURATION}s (avg: $(awk "BEGIN {printf \"%.1f\", $TOTAL_DURATION/$TOTAL_RUNS}")s/run)"
echo ""

if [ $FAIL_COUNT -gt 0 ]; then
    echo "Failed runs:     $FAIL_RUNS"
    echo ""
    echo "--- Per-test failure counts ---"
    for test_name in "${!FAIL_TESTS[@]}"; do
        count=${FAIL_TESTS[$test_name]}
        pct=$(awk "BEGIN {printf \"%.1f\", ($count/$TOTAL_RUNS)*100}")
        echo "  $test_name: $count/$TOTAL_RUNS ($pct%)"
    done | sort -t: -k2 -rn
    echo ""
    echo "--- Logs for failed runs ---"
    for run_num in $FAIL_RUNS; do
        echo "  $OUTPUT_DIR/run_${run_num}.log"
    done
fi
echo "================================================================================"

# Exit with failure if any run failed
if [ $FAIL_COUNT -gt 0 ]; then
    exit 1
fi
exit 0

