#!/bin/bash
set -e

# Read infrastructure variables
ENGINE=${ENGINE:-docker}
USE_GPU=${USE_GPU:-false}

# HARDWARE PROFILER (Fault Tolerance)
if [ "$USE_GPU" = "true" ]; then
    if ! command -v nvidia-smi &> /dev/null || ! nvidia-smi &> /dev/null; then
        echo "⚠️  [Hardware Warning] NVIDIA drivers/hardware not detected."
        echo "⚠️  [Hardware Warning] Safely falling back to CPU mode..."
        USE_GPU="false"
    fi
fi

# WANDB API KEY PROFILER
WANDB_KEY=""
if [ -f "$HOME/.netrc" ]; then
    # Parse the .netrc file securely. Find the wandb block, then grab the password.
    WANDB_KEY=$(awk '/machine api\.wandb\.ai/ {f=1} f && /password/ {print $2; exit}' "$HOME/.netrc" 2>/dev/null || true)
    
    if [ -n "$WANDB_KEY" ]; then
        echo "[Security] WandB credentials securely loaded from ~/.netrc"
    else
        echo "[Security] ~/.netrc found, but no WandB credentials detected."
    fi
else
    echo "[Security] No ~/.netrc file found. WandB will run offline or fail."
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "[Infrastructure] Engine: $ENGINE | GPU: $USE_GPU"
echo "[Hydra Overrides] Passing to Python: $@"

if [ "$ENGINE" = "docker" ]; then
    GPU_FLAG=$([ "$USE_GPU" = "true" ] && echo "--gpus all" || echo "")
    
    docker run -it --rm $GPU_FLAG \
      -e WANDB_API_KEY="$WANDB_KEY" \
      -v "$PROJECT_ROOT/config:/app/config" \
      -v "$PROJECT_ROOT/main.py:/app/main.py" \
      -v "$PROJECT_ROOT/results:/app/results" \
      "leonardobocchieri/jax-hypernetworks:latest" bash -c "python main.py $*"

elif [ "$ENGINE" = "apptainer" ]; then
    GPU_FLAG=$([ "$USE_GPU" = "true" ] && echo "--nv" || echo "")
    
    apptainer exec $GPU_FLAG \
      --env WANDB_API_KEY="$WANDB_KEY" \
      --bind "$PROJECT_ROOT/config:/app/config" \
      --bind "$PROJECT_ROOT/main.py:/app/main.py" \
      --bind "$PROJECT_ROOT/results:/app/results" \
      "$PROJECT_ROOT/jax-hypernetworks.sif" bash -c "cd /app && python main.py $*"
fi