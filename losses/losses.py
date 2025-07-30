import jax.numpy as jnp

def compute_rrmse(predictions, targets, epsilon=1e-8):
    """
    Computes RRMSE for the predictions and targets. Epsilon to avoid division by zero
    """
    relative_errors = (predictions - targets) / (jnp.abs(targets) + epsilon)
    rrmse = jnp.sqrt(jnp.mean(relative_errors ** 2))
    return rrmse

def compute_rrmse_alt1(predictions, targets, epsilon=1e-8):
    """Alternative that uses target magnitude for normalization."""
    target_magnitude = jnp.sqrt(jnp.mean(targets ** 2)) + epsilon
    rmse = jnp.sqrt(jnp.mean((predictions - targets) ** 2))
    rrmse = rmse / target_magnitude
    return rrmse