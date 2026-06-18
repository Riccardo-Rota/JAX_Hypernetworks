"""
Lightweight JIT vs no-JIT micro-benchmark.

Loops a single training step (forward + grad + optimizer update) over a fixed batch
of mock data and measures wall-clock after the 1st step (which includes XLA compilation
for the jitted version) and after STEPS steps (to derive the steady-state per-step time).

No Hydra run, no data pipeline, no logging/plots — just the compute we care about.

Env vars:
    STEPS  number of steps for the steady-state measurement (default 100, must be >= 2)
    BATCH  batch size of the mock data (default 4096)
"""

import os
# CPU-only benchmark: the toy model is too small to benefit from a GPU.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import jax
import jax.numpy as jnp
from flax import nnx
import hydra
from hydra import compose, initialize_config_dir
from utils import register_resolvers

register_resolvers()

STEPS = int(os.environ.get("STEPS", "100"))
BATCH = int(os.environ.get("BATCH", "4096"))
if STEPS < 2:
    raise SystemExit("STEPS must be >= 2 (need more than one step to separate first from steady-state).")

# Build the toy model / optimizer / criterion straight from the Hydra configs.
with initialize_config_dir(config_dir=str(PROJECT_ROOT / "config"), version_base="1.3"):
    cfg = compose(
        config_name="config",
        overrides=["problem=toy", "use_wandb=false", f"runtime.N={BATCH}"],
    )

num_hyper = int(cfg.model.num_hypervariables)
num_vars = int(cfg.model.num_variables)
criterion = hydra.utils.instantiate(cfg.loss)


def make_model_and_optimizer():
    model = hydra.utils.instantiate(cfg.model.manager)
    optimizer = hydra.utils.instantiate(cfg.optimizer, model=model)
    return model, optimizer


# Fixed batch of mock data with the shapes the model expects.
key = jax.random.PRNGKey(0)
k1, k2, k3 = jax.random.split(key, 3)
DATA = {
    "hypervars": jax.random.uniform(k1, (BATCH, num_hyper)),
    "vars": jax.random.uniform(k2, (BATCH, num_vars)),
}
LABELS = jax.random.uniform(k3, (BATCH, 1))


def train_step(model, optimizer, data, labels):
    def forward(model):
        pred = model(data)
        return jnp.mean(criterion(pred, labels))

    loss, grads = nnx.value_and_grad(forward)(model)
    optimizer.update(grads, value=loss)
    return loss


def benchmark(use_jit):
    """Return (first_step_seconds, steady_state_seconds_per_step)."""
    model, optimizer = make_model_and_optimizer()
    step = nnx.jit(train_step) if use_jit else train_step

    # First step: for the jitted version this includes the one-time XLA compilation.
    t0 = time.perf_counter()
    loss = step(model, optimizer, DATA, LABELS)
    jax.block_until_ready(loss)
    t1 = time.perf_counter()

    # Remaining steps: steady-state cost.
    for _ in range(STEPS - 1):
        loss = step(model, optimizer, DATA, LABELS)
    jax.block_until_ready(loss)
    t2 = time.perf_counter()

    return (t1 - t0), (t2 - t1) / (STEPS - 1)


def speedup(no_jit, jit):
    return f"{no_jit / jit:.1f}x" if jit > 0 else "NA"


jit_first, jit_steady = benchmark(use_jit=True)
nojit_first, nojit_steady = benchmark(use_jit=False)

print()
print(f"=== JIT vs no-JIT (CPU, toy model, batch={BATCH}, {STEPS} steps) ===")
print(f"{'Metric':<22} | {'jitted (s)':>12} | {'no-jit (s)':>12} | {'speedup':>8}")
print("-" * 22 + "-+-" + "-" * 12 + "-+-" + "-" * 12 + "-+-" + "-" * 8)
print(f"{'First step':<22} | {jit_first:>12.6f} | {nojit_first:>12.6f} | {speedup(nojit_first, jit_first):>8}")
print(f"{'Per step (steady)':<22} | {jit_steady:>12.6f} | {nojit_steady:>12.6f} | {speedup(nojit_steady, jit_steady):>8}")
