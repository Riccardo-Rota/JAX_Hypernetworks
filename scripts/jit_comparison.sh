#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status

# Dynamically find the directory where this script lives
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

IMAGE_NAME="leonardobocchieri/jax-hypernetworks:latest"

# Determine if GPU is available
if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
    echo "[Hardware Profiler] NVIDIA GPU detected and drivers active. Enabling GPU acceleration."
    GPU_FLAGS="--gpus all"
else
    echo "[Hardware Profiler] No active NVIDIA GPU found. Defaulting to CPU execution."
    GPU_FLAGS=""
fi

# Define output directories for host and container
JITTED_DIR_HOST="$PROJECT_ROOT/results/jit_comparison/jitted"
NON_JITTED_DIR_HOST="$PROJECT_ROOT/results/jit_comparison/non_jitted"
JITTED_DIR_CONTAINER="results/jit_comparison/jitted"
NON_JITTED_DIR_CONTAINER="results/jit_comparison/non_jitted"

# Clean up previous results to ensure a fresh run
echo "Cleaning up old results..."
rm -rf "$PROJECT_ROOT/results/jit_comparison"

# JITTED
echo "Starting container to run the JITTED version (100 epochs)..."
docker run --rm $GPU_FLAGS \
  -v "$PROJECT_ROOT:/app" \
  -w /app \
  "$IMAGE_NAME" \
  bash -c "python main.py problem=toy training.epochs=100 use_wandb=False hydra.run.dir=$JITTED_DIR_CONTAINER"

# NON JITTED
echo "Starting container to run the NON-JITTED version (100 epochs)..."
docker run --rm $GPU_FLAGS \
  -e JAX_DISABLE_JIT=1 \
  -v "$PROJECT_ROOT:/app" \
  -w /app \
  "$IMAGE_NAME" \
  bash -c "python main.py problem=toy training.epochs=100 use_wandb=False hydra.run.dir=$NON_JITTED_DIR_CONTAINER"

echo "Executions finished. Extracting results..."

# Check if jq is installed, as it's needed to parse the JSON results
if ! command -v jq &> /dev/null
then
    echo "Error: 'jq' is not installed. Please install it to parse the results automatically."
    echo "You can find the results in the following files:"
    echo "Jitted results: $JITTED_DIR_HOST/run_data.json"
    echo "Non-jitted results: $NON_JITTED_DIR_HOST/run_data.json"
    exit 1
fi

# Extract time_per_epoch_seconds from the run_data.json of each run
JITTED_TIME=$(jq '.time_per_epoch_seconds' "$JITTED_DIR_HOST/run_data.json")
NON_JITTED_TIME=$(jq '.time_per_epoch_seconds' "$NON_JITTED_DIR_HOST/run_data.json")

echo ""
echo "--- JIT vs. No-JIT Comparison ---"
echo "Time per epoch (jitted):     $JITTED_TIME seconds"
echo "Time per epoch (non-jitted): $NON_JITTED_TIME seconds"
