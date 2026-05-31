set -euo pipefail

echo "==> Compiling for Benchmark..."
make clean > /dev/null 2>&1 || true
make > /dev/null 2>&1

echo
echo "### Performance Benchmark Results"
echo
echo "| Threshold | Sigma | Pitch | Yaw | Roll | CPU Time (ms) | GPU Time (ms) | Speedup |"
echo "|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|"

run_bench() {
    local threshold=$1
    local sigma=$2
    local pitch=$3
    local yaw=$4
    local roll=$5

    local output
    output=$("./bunnyMIP" --threshold "$threshold" --sigma "$sigma" --pitch "$pitch" --yaw "$yaw" --roll "$roll" 2>&1) || true

    # Extract timings using grep and awk
    local cpu_time
    cpu_time=$(echo "$output" | grep "CPU Time:" | awk '{print $3}')
    
    local gpu_time
    gpu_time=$(echo "$output" | grep "GPU Time:" | awk '{print $3}')
    
    local speedup
    speedup=$(echo "$output" | grep "Speedup:" | awk '{print $2}' | tr -d 'x')

    if [[ -z "$cpu_time" || -z "$gpu_time" || -z "$speedup" ]]; then
        echo "| $threshold | $sigma | $pitch | $yaw | $roll | ERROR | ERROR | ERROR |"
    else
        printf "| %9s | %5s | %5s | %3s | %4s | %13s | %13s | %7sx |\n" \
            "$threshold" "$sigma" "$pitch" "$yaw" "$roll" "$cpu_time" "$gpu_time" "$speedup"
    fi
}

# Base Case
run_bench 32768 1.0 0 0 0

# --- Varying Threshold ---
run_bench 8192 1.0 0 0 0
run_bench 16384 1.0 0 0 0
run_bench 49152 1.0 0 0 0

# --- Varying Sigma (Blur intensity) ---
run_bench 32768 0.5 0 0 0
run_bench 32768 2.0 0 0 0
run_bench 32768 3.0 0 0 0

# --- Varying Rotations (Ray casting complexity) ---
run_bench 32768 1.0 45 0 0
run_bench 32768 1.0 0 45 0
run_bench 32768 1.0 45 45 45

# --- Combined Heavy Load ---
run_bench 8192 2.5 30 60 90

echo
echo "==> Done."