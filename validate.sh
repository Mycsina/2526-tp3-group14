#!/usr/bin/env bash
# Validation script for BunnyMIP
# Compares GPU output against CPU "gold standard" across multiple parameter sets.
#
# Usage: ./validate.sh [--verbose]

set -euo pipefail

THRESHOLD_PCT=1.0
PASS=0
FAIL=0
RESULTS=()

run_test() {
    local name="$1"
    shift
    echo "==> Test: ${name}"
    echo "    args: $@"

    local output
    output=$("./bunnyMIP" "$@" 2>&1) || true

    local line
    line=$(echo "${output}" | grep "VALIDATION:" || true)

    if [[ -z "${line}" ]]; then
        echo "    RESULT: ERROR (no VALIDATION line found)"
        FAIL=$((FAIL + 1))
        RESULTS+=("FAIL  ${name}  (crashed / no output)")
        return
    fi

    local pct
    pct=$(echo "${line}" | sed -n 's/.*diff_pct=\([0-9.]*\).*/\1/p')

    if [[ -z "${pct}" ]]; then
        echo "    RESULT: ERROR (could not parse diff_pct)"
        FAIL=$((FAIL + 1))
        RESULTS+=("FAIL  ${name}  (parse error)")
        return
    fi

    local exceeded
    exceeded=$(echo "${line}" | sed -n 's/.*pixels_exceeding=\([0-9]*\).*/\1/p')

    if (( $(echo "${pct} < ${THRESHOLD_PCT}" | bc -l) )); then
        echo "    RESULT: PASS  (diff=${pct}%, ${exceeded} pixels > 2)"
        PASS=$((PASS + 1))
        RESULTS+=("PASS  ${name}  diff=${pct}%  exceeded=${exceeded}")
    else
        echo "    RESULT: FAIL  (diff=${pct}% exceeds ${THRESHOLD_PCT}%, ${exceeded} pixels > 2)"
        FAIL=$((FAIL + 1))
        RESULTS+=("FAIL  ${name}  diff=${pct}%  exceeded=${exceeded}")
    fi
}

echo "============================================"
echo " BunnyMIP Validation Suite"
echo " Threshold: ${THRESHOLD_PCT}% max difference"
echo "============================================"
echo

# Build
echo "==> Compiling..."
make clean > /dev/null 2>&1 || true
make
echo

# --- Default parameters ---
run_test "defaults" ""

# --- Varying threshold ---
run_test "threshold=8192"  --threshold 8192
run_test "threshold=32768" --threshold 32768
run_test "threshold=49152" --threshold 49152

# --- Varying sigma ---
run_test "sigma=0.5"  --sigma 0.5
run_test "sigma=2.0"  --sigma 2.0
run_test "sigma=3.0"  --sigma 3.0

# --- Varying rotations ---
run_test "pitch=30"  --pitch 30
run_test "yaw=45"    --yaw 45
run_test "roll=60"   --roll 60
run_test "pitch=15,yaw=25,roll=5" --pitch 15 --yaw 25 --roll 5

# --- Combined parameters ---
run_test "threshold=16384,sigma=1.5,yaw=30" --threshold 16384 --sigma 1.5 --yaw 30
run_test "threshold=24576,sigma=0.8,pitch=45,roll=20" --threshold 24576 --sigma 0.8 --pitch 45 --roll 20

echo
echo "============================================"
echo " Results"
echo "============================================"
for r in "${RESULTS[@]}"; do
    echo "  ${r}"
done
echo
echo "  PASS: ${PASS}  FAIL: ${FAIL}  TOTAL: $((PASS + FAIL))"
echo "============================================"

if [[ ${FAIL} -gt 0 ]]; then
    echo "VALIDATION FAILED"
    exit 1
else
    echo "VALIDATION PASSED"
    exit 0
fi