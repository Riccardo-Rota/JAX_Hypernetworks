#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status

# Dynamically find the directory where this script lives
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

IMAGE_NAME="leonardobocchieri/jax-hypernetworks:latest"

# Number of epochs per run. Both the jitted and non-jitted runs use the same
# value, so the per-epoch comparison is fair. The jitted per-epoch time includes
# a one-time XLA compilation amortized over the epochs: raise EPOCHS to make that
# warm-up negligible (the non-jitted run has no compilation, it just costs more
# wall-clock).
EPOCHS="${EPOCHS:-100}"

# Determine if a GPU is available
if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
    echo "[Hardware Profiler] NVIDIA GPU detected and drivers active. Will run on CPU and GPU."
    GPU_AVAILABLE=true
else
    echo "[Hardware Profiler] No active NVIDIA GPU found. Will run on CPU only."
    GPU_AVAILABLE=false
fi

# Clean up previous results to ensure a fresh run
echo "Cleaning up old results..."
rm -rf "$PROJECT_ROOT/results/jit_comparison"

# -----------------------------------------------------------------------------
# run_case <device: cpu|gpu> <mode: jitted|non_jitted>
# Runs one training inside the container with the right device/JIT settings and
# writes its results under results/jit_comparison/<device>/<mode>/.
# -----------------------------------------------------------------------------
run_case() {
    local device="$1"
    local mode="$2"
    local out_container="results/jit_comparison/${device}/${mode}"

    local gpu_flag=""
    local env_flags=()

    if [ "$device" = "gpu" ]; then
        gpu_flag="--gpus all"            # expose the GPU; JAX auto-selects CUDA
    else
        env_flags+=(-e JAX_PLATFORMS=cpu) # force CPU even if a GPU is present
    fi

    if [ "$mode" = "non_jitted" ]; then
        env_flags+=(-e JAX_DISABLE_JIT=1) # turn jax.jit (and nnx.jit) into no-ops
    fi

    echo ""
    echo ">>> Running ${device} / ${mode} (${EPOCHS} epochs)..."
    docker run --rm $gpu_flag "${env_flags[@]}" \
      -v "$PROJECT_ROOT:/app" \
      -w /app \
      "$IMAGE_NAME" \
      bash -c "python main.py problem=toy training.epochs=$EPOCHS use_wandb=False hydra.run.dir=$out_container"
}

# -----------------------------------------------------------------------------
# Execute the cases
# -----------------------------------------------------------------------------
DEVICES=(cpu)
if [ "$GPU_AVAILABLE" = true ]; then
    DEVICES+=(gpu)
fi

for device in "${DEVICES[@]}"; do
    run_case "$device" "jitted"
    run_case "$device" "non_jitted"
done

echo ""
echo "Executions finished. Extracting results..."

# jq is needed to parse the JSON results
if ! command -v jq &> /dev/null; then
    echo "Error: 'jq' is not installed. Please install it to parse the results automatically."
    echo "Raw results are in: $PROJECT_ROOT/results/jit_comparison/<device>/<mode>/run_data.json"
    exit 1
fi

# read_time <device> <mode> -> prints time_per_epoch_seconds (or "NA")
read_time() {
    local f="$PROJECT_ROOT/results/jit_comparison/$1/$2/run_data.json"
    if [ -f "$f" ]; then
        jq -r '.time_per_epoch_seconds' "$f"
    else
        echo "NA"
    fi
}

# speedup <non_jitted_time> <jitted_time> -> prints "<x>x" or "NA"
speedup() {
    awk -v n="$1" -v j="$2" 'BEGIN {
        if (n == "NA" || j == "NA" || j == 0) { print "NA" }
        else { printf "%.1fx", n / j }
    }'
}

echo ""
echo "=== JIT vs. No-JIT comparison (${EPOCHS} epochs) ==="
echo "Note: the jitted figure includes a one-time XLA compilation amortized over the epochs."
echo ""
printf "%-8s | %-22s | %-22s | %-10s\n" "Device" "Jitted (s/epoch)" "Non-jitted (s/epoch)" "Speedup"
printf -- "---------+------------------------+------------------------+-----------\n"
for device in "${DEVICES[@]}"; do
    JT="$(read_time "$device" jitted)"
    NJT="$(read_time "$device" non_jitted)"
    SP="$(speedup "$NJT" "$JT")"
    printf "%-8s | %-22s | %-22s | %-10s\n" "$device" "$JT" "$NJT" "$SP"
done
