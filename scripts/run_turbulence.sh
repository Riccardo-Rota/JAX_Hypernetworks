#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.

# Dynamically find the directory where this script lives, then step up one level to the repo root.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Define the image name. You MUST replace this placeholder before submitting.
IMAGE_NAME="leonardobocchieri/jax-hypernetworks:latest"

# Determine if GPU is available
if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
    echo "[Hardware Profiler] NVIDIA GPU detected and drivers active. Enabling GPU acceleration."
    GPU_FLAGS="--gpus all"
else
    echo "[Hardware Profiler] No active NVIDIA GPU found. Defaulting to CPU execution."
    GPU_FLAGS=""
fi

echo "Starting container to run the Turbulence Problem..."

docker run -it --rm $GPU_FLAGS \
  -v "$PROJECT_ROOT/config:/app/config" \
  -v "$PROJECT_ROOT/main.py:/app/main.py" \
  -v "$PROJECT_ROOT/results:/app/results" \
  "$IMAGE_NAME" bash -c "python main.py problem=turbulence"

echo "Execution finished. Container removed."