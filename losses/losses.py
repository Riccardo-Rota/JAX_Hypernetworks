import jax.numpy as jnp
import optax
from typing import Callable, Union, Optional, List
from utils import to_list
from flax import nnx

# Guidelines for defining losses:
# - Each loss should be an instance of nnx.Module
# - The forward method should return the SUM of the loss over the batch, the averaging should be done outside the loss function
# - The arguments passed to the forward during the training cycle should be (predictions, targets), all other
#   arguments should be set as default values or initialized by the constructor.
# - If using a different signature for the forward method, please modify the compute_metrics helper function in training/train.py accordingly

    
class L2Loss(nnx.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, predictions, targets):
        """Compute the Mean Squared Error (MSE) between predictions and targets."""
        return optax.l2_loss(predictions, targets)

class LpLoss(nnx.Module):
    def __init__(self, p: float = 2.0):
        super().__init__()
        self.p = p

    def __call__(self, predictions, targets):
        """Compute the Lp loss between predictions and targets."""
        return jnp.sum(jnp.abs(predictions - targets) ** self.p)
    
class CombinedLoss(nnx.Module):
    def __init__(self, losses: List[nnx.Module], weights: Optional[List[float]] = None):
        super().__init__()
        self.losses = to_list(losses)
        if weights is None:
            self.weights = [1.0] * len(self.losses)
        else:
            self.weights = weights

    def __call__(self, predictions, targets):
        """Compute the weighted sum of multiple loss functions."""
        total_loss = 0.0
        for loss, weight in zip(self.losses, self.weights):
            total_loss += weight * loss(predictions, targets)
        return total_loss