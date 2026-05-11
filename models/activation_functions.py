import flax.nnx as nnx
import jax
import jax.numpy as jnp

def get_tanh():
    return nnx.tanh

def get_relu():
    return nnx.relu

def get_sigmoid():
    return nnx.sigmoid

def get_gelu():
    return nnx.gelu

def uniform_init(limit: float):
    """Custom initializer to strictly enforce [-limit, limit] bounds."""
    def init_fn(key, shape, dtype=jnp.float32):
        return jax.random.uniform(key, shape, dtype, minval=-limit, maxval=limit)
    return init_fn
