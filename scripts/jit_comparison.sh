#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status

# Dynamically find the directory where this script lives
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

IMAGE_NAME="leonardobocchieri/jax-hypernetworks:latest"

# Total number of epochs for the "full" run. The first epoch is timed separately
# (via a dedicated 1-epoch run) so we can isolate the one-time XLA compilation of
# the jitted version from the steady-state per-epoch cost. Must be >= 2.
EPOCHS="${EPOCHS:-100}"

# This benchmark runs on CPU only: the toy model is too small to benefit from a
# GPU (kernel-launch and host/device transfer overheads dominate the tiny
# matmuls), so CPU timings are the meaningful ones for the JIT comparison.

if [ "$EPOCHS" -lt 2 ]; then
    echo "Error: EPOCHS must be >= 2 (need more than one epoch to separate first from steady-state)."
    exit 1
fi

# Clean up previous results to ensure a fresh run
echo "Cleaning up old results..."
rm -rf "$PROJECT_ROOT/results/jit_comparison"

# -----------------------------------------------------------------------------
# run_case <mode: jitted|non_jitted> <epochs> <subdir>
# Runs one training inside the container (forced on CPU, no plots) with the right
# JIT setting and writes its results under results/jit_comparison/<mode>/<subdir>.
# -----------------------------------------------------------------------------
run_case() {
    local mode="$1"
    local epochs="$2"
    local subdir="$3"
    local out_container="results/jit_comparison/${mode}/${subdir}"

    local env_flags=(-e JAX_PLATFORMS=cpu) # force CPU even if a GPU is present
    if [ "$mode" = "non_jitted" ]; then
        env_flags+=(-e JAX_DISABLE_JIT=1)  # turn jax.jit (and nnx.jit) into no-ops
    fi

    echo ""
    echo ">>> Running ${mode} (${epochs} epoch(s), CPU)..."
    docker run --rm "${env_flags[@]}" \
      -v "$PROJECT_ROOT:/app" \
      -w /app \
      "$IMAGE_NAME" \
      bash -c "python main.py problem=toy postprocessing=none use_wandb=False training.epochs=$epochs hydra.run.dir=$out_container"
}

# -----------------------------------------------------------------------------
# Execute the cases: for each mode, a 1-epoch run (to time the first epoch) and a
# full EPOCHS run (to derive the steady-state per-epoch time).
# -----------------------------------------------------------------------------
for mode in jitted non_jitted; do
    run_case "$mode" 1 "first_epoch"
    run_case "$mode" "$EPOCHS" "full"
done

echo ""
echo "Executions finished. Extracting results..."

# jq is needed to parse the JSON results
if ! command -v jq &> /dev/null; then
    echo "Error: 'jq' is not installed. Please install it to parse the results automatically."
    echo "Raw results are in: $PROJECT_ROOT/results/jit_comparison/<mode>/<subdir>/run_data.json"
    exit 1
fi

# jq_field <mode> <subdir> <field> -> prints the JSON field (or "NA")
jq_field() {
    local f="$PROJECT_ROOT/results/jit_comparison/$1/$2/run_data.json"
    if [ -f "$f" ]; then
        jq -r ".$3" "$f"
    else
        echo "NA"
    fi
}

# First-epoch time: the 1-epoch run's total training time.
FIRST_JIT="$(jq_field jitted first_epoch training_time_seconds)"
FIRST_NOJIT="$(jq_field non_jitted first_epoch training_time_seconds)"

# Steady-state per-epoch time: (full total time - first epoch time) / (EPOCHS - 1).
steady() {
    awk -v total="$1" -v first="$2" -v e="$EPOCHS" 'BEGIN {
        if (total == "NA" || first == "NA" || e <= 1) { print "NA" }
        else { printf "%.6f", (total - first) / (e - 1) }
    }'
}
TOTAL_JIT="$(jq_field jitted full training_time_seconds)"
TOTAL_NOJIT="$(jq_field non_jitted full training_time_seconds)"
STEADY_JIT="$(steady "$TOTAL_JIT" "$FIRST_JIT")"
STEADY_NOJIT="$(steady "$TOTAL_NOJIT" "$FIRST_NOJIT")"

# speedup <non_jitted> <jitted> -> "<x>x" or "NA"
speedup() {
    awk -v n="$1" -v j="$2" 'BEGIN {
        if (n == "NA" || j == "NA" || j == 0) { print "NA" }
        else { printf "%.1fx", n / j }
    }'
}
SP_FIRST="$(speedup "$FIRST_NOJIT" "$FIRST_JIT")"
SP_STEADY="$(speedup "$STEADY_NOJIT" "$STEADY_JIT")"

echo ""
echo "=== JIT vs. No-JIT comparison (CPU, ${EPOCHS} epochs) ==="
printf "%-28s | %-14s | %-16s | %-10s\n" "Metric" "Jitted (s)" "Non-jitted (s)" "Speedup"
printf -- "-----------------------------+----------------+------------------+-----------\n"
printf "%-28s | %-14s | %-16s | %-10s\n" "Per epoch (steady-state)"    "$STEADY_JIT" "$STEADY_NOJIT" "$SP_STEADY"
