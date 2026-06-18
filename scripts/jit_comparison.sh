#!/bin/bash
set -e

# Compare training speed with and without JIT compilation.
# Runs on CPU (the toy model is too small to benefit from a GPU).

ENGINE="${ENGINE:-docker}"
EPOCHS="${EPOCHS:-100}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_NAME="leonardobocchieri/jax-hypernetworks:latest"

run() {
    # $1: extra env (e.g. JAX_DISABLE_JIT=1), $2: output subdir
    local overrides="problem=toy use_wandb=False test_model=False plot_inference=False training.epochs=$EPOCHS hydra.run.dir=results/jit_comparison/$2"

    if [ "$ENGINE" = "docker" ]; then
        local jit_flag=""
        [ -n "$1" ] && jit_flag="-e $1"
        docker run --rm -e JAX_PLATFORMS=cpu $jit_flag \
          -v "$PROJECT_ROOT:/app" -w /app \
          "$IMAGE_NAME" bash -c "python main.py $overrides"
    elif [ "$ENGINE" = "venv" ]; then
        env JAX_PLATFORMS=cpu $1 python "$PROJECT_ROOT/main.py" $overrides
    else
        echo "Error: unsupported ENGINE='$ENGINE' (use docker or venv)."
        exit 1
    fi
}

echo ">>> Training with JIT ($EPOCHS epochs, engine=$ENGINE)..."
start=$SECONDS
run "" jitted
jit_time=$((SECONDS - start))

echo ">>> Training without JIT ($EPOCHS epochs, engine=$ENGINE)..."
start=$SECONDS
run "JAX_DISABLE_JIT=1" non_jitted
nojit_time=$((SECONDS - start))

echo ""
echo "=== JIT vs No-JIT (CPU, $EPOCHS epochs) ==="
echo "Jitted:     ${jit_time}s"
echo "Non-jitted: ${nojit_time}s"
