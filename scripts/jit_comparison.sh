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

# This benchmark runs on CPU only: the toy model is too small to benefit from a
# GPU (kernel-launch and host/device transfer overheads dominate the tiny
# matmuls), so CPU timings are the meaningful ones for the JIT comparison.

# Clean up previous results to ensure a fresh run
echo "Cleaning up old results..."
rm -rf "$PROJECT_ROOT/results/jit_comparison"

# -----------------------------------------------------------------------------
# run_case <mode: jitted|non_jitted>
# Runs one training inside the container (forced on CPU) with the right JIT
# setting and writes its results under results/jit_comparison/<mode>/.
# -----------------------------------------------------------------------------
run_case() {
    local mode="$1"
    local out_container="results/jit_comparison/${mode}"

    local env_flags=(-e JAX_PLATFORMS=cpu) # force CPU even if a GPU is present
    if [ "$mode" = "non_jitted" ]; then
        env_flags+=(-e JAX_DISABLE_JIT=1)  # turn jax.jit (and nnx.jit) into no-ops
    fi

    echo ""
    echo ">>> Running ${mode} (${EPOCHS} epochs, CPU)..."
    docker run --rm "${env_flags[@]}" \
      -v "$PROJECT_ROOT:/app" \
      -w /app \
      "$IMAGE_NAME" \
      bash -c "python main.py problem=toy training.epochs=$EPOCHS use_wandb=False hydra.run.dir=$out_container"
}

# -----------------------------------------------------------------------------
# Execute the cases
# -----------------------------------------------------------------------------
run_case "jitted"
run_case "non_jitted"

echo ""
echo "Executions finished. Extracting results..."

# jq is needed to parse the JSON results
if ! command -v jq &> /dev/null; then
    echo "Error: 'jq' is not installed. Please install it to parse the results automatically."
    echo "Raw results are in: $PROJECT_ROOT/results/jit_comparison/<mode>/run_data.json"
    exit 1
fi

# read_time <mode> -> prints time_per_epoch_seconds (or "NA")
read_time() {
    local f="$PROJECT_ROOT/results/jit_comparison/$1/run_data.json"
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

JITTED_TIME="$(read_time jitted)"
NON_JITTED_TIME="$(read_time non_jitted)"
SPEEDUP="$(speedup "$NON_JITTED_TIME" "$JITTED_TIME")"

echo ""
echo "=== JIT vs. No-JIT comparison (CPU, ${EPOCHS} epochs) ==="
echo "Note: the jitted figure includes a one-time XLA compilation amortized over the epochs."
echo ""
echo "Time per epoch (jitted):     $JITTED_TIME s"
echo "Time per epoch (non-jitted): $NON_JITTED_TIME s"
echo "Speedup (non-jitted / jitted): $SPEEDUP"
