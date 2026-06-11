#!/bin/bash
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SIF="${SIF:-$PROJECT_ROOT/jax-hypernetworks.sif}"
RESULTS_DIR="${RESULTS_DIR:-results}"
SYNC_TIMEOUT="${SYNC_TIMEOUT:-300}"
PARALLEL="${PARALLEL:-4}"
SERVICE_WAIT="${SERVICE_WAIT:-60}"

if [ ! -f "$SIF" ]; then
    echo "ERROR: apptainer image not found at $SIF"
    echo "       Pull it with: apptainer pull jax-hypernetworks.sif docker://leonardobocchieri/jax-hypernetworks:latest"
    exit 1
fi

WANDB_KEY=$(awk '/machine api\.wandb\.ai/ {f=1} f && /password/ {print $2; exit}' "$HOME/.netrc" 2>/dev/null || true)
if [ -z "$WANDB_KEY" ]; then
    echo "ERROR: no W&B API key found in ~/.netrc. Create it as described in README.md."
    exit 1
fi

if [ ! -d "$PROJECT_ROOT/$RESULTS_DIR" ]; then
    echo "ERROR: results directory not found: $PROJECT_ROOT/$RESULTS_DIR"
    exit 1
fi

echo "[sync] Uploading offline runs under $RESULTS_DIR (timeout=${SYNC_TIMEOUT}s, parallel=${PARALLEL}) via $SIF ..."
apptainer exec \
  --env WANDB_API_KEY="$WANDB_KEY" \
  --env WANDB_START_METHOD=thread \
  --env WANDB__SERVICE_WAIT="$SERVICE_WAIT" \
  --env RESULTS_DIR="$RESULTS_DIR" \
  --env SYNC_TIMEOUT="$SYNC_TIMEOUT" \
  --env PARALLEL="$PARALLEL" \
  --bind "$PROJECT_ROOT:/app" \
  "$SIF" bash -c '
    cd /app
    : "${RESULTS_DIR:=results}" "${SYNC_TIMEOUT:=300}" "${PARALLEL:=4}"
    while IFS= read -r -d "" d; do
      (
        if timeout "$SYNC_TIMEOUT" wandb sync "$d"; then
          echo "[sync] OK       $d"
        elif [ "$?" -eq 124 ]; then
          echo "[sync] TIMEOUT  (skipped after ${SYNC_TIMEOUT}s) $d"
        else
          echo "[sync] FAILED   $d"
        fi
      ) &
      while [ "$(jobs -r | wc -l)" -ge "$PARALLEL" ]; do wait -n; done
    done < <(find "$RESULTS_DIR" -type d -name "offline-run-*" -print0)
    wait
  '

echo "[sync] Done."
