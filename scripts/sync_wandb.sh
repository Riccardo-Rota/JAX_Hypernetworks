#!/bin/bash
# Upload all OFFLINE W&B runs to the cloud, using the wandb library that lives
# INSIDE the apptainer image (no host wandb install required).
# Usage:
#   bash scripts/sync_wandb.sh            # sync results/ under the project root
#   RESULTS_DIR=<path_to_specific_dir> bash scripts/sync_wandb.sh   # narrow it
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SIF="${SIF:-$PROJECT_ROOT/jax-hypernetworks.sif}"
RESULTS_DIR="${RESULTS_DIR:-results}"

if [ ! -f "$SIF" ]; then
    echo "ERROR: apptainer image not found at $SIF"
    echo "       Pull it with: apptainer pull jax-hypernetworks.sif docker://leonardobocchieri/jax-hypernetworks:latest"
    exit 1
fi

# Load the W&B API key from ~/.netrc (same parse as run_experiment.sh).
WANDB_KEY=$(awk '/machine api\.wandb\.ai/ {f=1} f && /password/ {print $2; exit}' "$HOME/.netrc" 2>/dev/null || true)
if [ -z "$WANDB_KEY" ]; then
    echo "ERROR: no W&B API key found in ~/.netrc. Create it as described in README.md."
    exit 1
fi

if [ ! -d "$PROJECT_ROOT/$RESULTS_DIR" ]; then
    echo "ERROR: results directory not found: $PROJECT_ROOT/$RESULTS_DIR"
    exit 1
fi

echo "[sync] Uploading offline runs under $RESULTS_DIR via $SIF ..."
apptainer exec \
  --env WANDB_API_KEY="$WANDB_KEY" \
  --bind "$PROJECT_ROOT:/app" \
  "$SIF" \
  bash -c "cd /app && find '$RESULTS_DIR' -type d -name 'offline-run-*' -print0 \
             | xargs -0 --no-run-if-empty -n1 wandb sync"

echo "[sync] Done."
