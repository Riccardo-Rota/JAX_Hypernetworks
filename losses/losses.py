import jax.numpy as jnp
import optax
from typing import Callable, Union, Optional, List
from utils import to_list
from flax import nnx

# Guidelines for defining losses:
# - Each loss should be an instance of nnx.Module
# - The forward method should return the SUM of the loss over the batch, the averaging should be done outside the loss function
# - The arguments passed to the forward during the training cycle should be (predictions, targets, weights) or (predictions, targets), all other
#   arguments should be set as default values
# - If using a different signature for the forward method, please modify the compute_metrics helper function in training/train.py accordingly

    
class L2Loss(nnx.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, predictions, targets):
        """Compute the Mean Squared Error (MSE) between predictions and targets."""
        return optax.l2_loss(predictions, targets)


class CustomLoss(nnx.Module):
    """
    Customizable loss function that combines L2 loss with optional additional losses and penalties.
    Args:
    l2_loss_weight (float): Weight for the L2 loss component. Default is 1.0.
    l1_penalty_weight (float): Weight for the L1 penalty on model weights. Default is 0.0.
    l2_penalty_weight (float): Weight for the L2 penalty on model weights. Default is 0.0.
    extra_loss_functions (Callable or List[Callable], optional): Additional loss functions to include.
    extra_loss_weights (float or List[float], optional): Weights for the additional loss functions.
    extra_penalty_functions (Callable or List[Callable], optional): Additional penalty functions to include.
    extra_penalty_weights (float or List[float], optional): Weights for the additional penalty functions.
    """
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
        self.penalty_functions = [lambda w: jnp.mean(jnp.abs(w), axis=-1), lambda w: jnp.mean(w**2, axis=-1)] + to_list(extra_penalty_functions)

    def __call__(self, predictions, targets, weights):
        """
        Compute the total loss as a weighted sum of individual losses and penalties.
        Args:
            predictions (jax.Array): The predicted values.
            targets (jax.Array): The true target values.
            weights (jax.Array): The model weights for penalty computation.
        Returns:
            jax.Array: The computed total loss.
        """
        total_loss = 0.0

        # Compute losses
        for weight, loss_fn in zip(self.loss_weights, self.loss_functions):
            if weight != 0.0:
                total_loss += weight * jnp.sum(loss_fn(predictions, targets))
        
        # Compute penalties
        for weight, penalty_fn in zip(self.penalty_weights, self.penalty_functions):
            if weight != 0.0:
                total_loss += weight * jnp.sum(penalty_fn(weights))
        
        return total_loss    