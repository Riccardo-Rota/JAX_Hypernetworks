#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.

# Dynamically find the directory where this script lives, then step up one level to the repo root.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Define the image name. You MUST replace this placeholder before submitting.
IMAGE_NAME="leonardobocchieri/jax-hypernetworks:latest"

echo "Starting container to run the Turbulence Problem..."

docker run -it --rm --gpus all \
  -v "$PROJECT_ROOT/config:/app/config" \
  -v "$PROJECT_ROOT/main.py:/app/main.py" \
  -v "$PROJECT_ROOT/results:/app/results" \
  "$IMAGE_NAME" bash -c "python main.py problem=turbulence"

echo "Execution finished. Container removed."