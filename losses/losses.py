import jax.numpy as jnp
import optax
from typing import Callable, Union, Optional, List
from utils import to_list

#Guidelines for defining losses or metrics:
# - The arguments passed during the training cycle should be (predictions, targets, weights) or (predictions, targets), all other
#   arguments should be set as default values
# - If using a different signature, please modify the compute_metrics helper function in training/train.py accordingly


def RRMSE(predictions, targets, epsilon=1e-8):
    """
    Compute the Relative Root Mean Square Error (RRMSE) between predictions and targets.
    RRMSE is defined as the RMSE divided by the magnitude of the target values.
    Args:
        predictions (jax.Array): The predicted values.
        targets (jax.Array): The true target values.
        epsilon (float): A small value to avoid division by zero. Default is 1e-8.
    Returns:
        jax.Array: The computed RRMSE value.
    """
    target_magnitude = jnp.sqrt(jnp.mean(targets ** 2)) + epsilon
    rmse = jnp.sqrt(jnp.mean((predictions - targets) ** 2))
    rrmse = rmse / target_magnitude
    return rrmse

def l2_loss(predictions, targets):
    """
    Compute the Mean Squared Error (MSE) between predictions and targets.
    Args:
        predictions (jax.Array): The predicted values.
        targets (jax.Array): The true target values.
    Returns:
        jax.Array: The computed MSE value.
    """
    return jnp.mean(optax.l2_loss(predictions, targets))

def MAE(predictions, targets):
    """
    Compute the Mean Absolute Error (MAE) between predictions and targets.
    Args:
        predictions (jax.Array): The predicted values.
        targets (jax.Array): The true target values.
    Returns:
        jax.Array: The computed MAE value.
    """
    return jnp.mean(jnp.abs(predictions - targets))

class CustomLoss:
    def __init__(self, 
                 l2_loss_weight: float=1.0, 
                 l1_penalty_weight: float=0.0, 
                 l2_penalty_weight: float=0.0, 
                 extra_loss_functions: Optional[Union[Callable, List[Callable]]]=None,
                 extra_loss_weights: Optional[Union[float, List[float]]]=None,
                 extra_penalty_functions: Optional[Union[Callable, List[Callable]]]=None,
                 extra_penalty_weights: Optional[Union[float, List[float]]]=None):

        assert len(to_list(extra_loss_functions)) == len(to_list(extra_loss_weights)), "Length of extra_loss_functions must be equal to length of extra_loss_weights."
        assert len(to_list(extra_penalty_functions)) == len(to_list(extra_penalty_weights)), "Length of extra_penalty_functions must be equal to length of extra_penalty_weights."

        self.loss_weights = [l2_loss_weight] + to_list(extra_loss_weights)
        self.loss_functions = [optax.l2_loss] + to_list(extra_loss_functions)
        self.penalty_weights = [l1_penalty_weight, l2_penalty_weight] + to_list(extra_penalty_weights)
        self.penalty_functions = [lambda w: jnp.mean(jnp.abs(w)), lambda w: jnp.mean(w**2)] + to_list(extra_penalty_functions)
    
    def __call__(self, predictions, targets, weights):
        total_loss = 0.0

        # Compute losses
        for weight, loss_fn in zip(self.loss_weights, self.loss_functions):
            if weight != 0.0:
                total_loss += weight * jnp.mean(loss_fn(predictions, targets))
        
        # Compute penalties
        for weight, penalty_fn in zip(self.penalty_weights, self.penalty_functions):
            if weight != 0.0:
                total_loss += weight * penalty_fn(weights)
        
        return total_loss    