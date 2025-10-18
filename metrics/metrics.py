from flax.nnx.training.metrics import Metric, Average, MetricState
import jax.numpy as jnp

# Guidelines for defining metrics:
# - Each metric should be an instance of nnx.training.metrics.Metric
# - Each subclass of Metric should implement a ``compute``, ``reset`` and ``update`` method.
# - The arguments passed to update should be (predictions, targets)
# - It is useful to inherit from nnx.training.metrics.Average to exploit built-in updating and resetting logic

class MSE(Average):
    """
    Compute the Mean Squared Error (MSE) between predictions and targets.
    """
    def update(self, predictions, targets):
        """
        Update the metric state with new predictions and targets.
        Args:
            predictions (jax.Array): The predicted values.
            targets (jax.Array): The true target values.
        """
        mse_values = (predictions - targets) ** 2
        super().update(values=mse_values)

class RMSE(MSE):
    """
    Compute the Root Mean Squared Error (RMSE) between predictions and targets.
    """        
    def compute(self):
        """
        Compute the RMSE based on the accumulated state.
        Returns:
            jax.Array: The computed RMSE value.
        """
        mse = super().compute()
        rmse = jnp.sqrt(mse)
        return rmse

class RRMSE(Metric):
    """ 
    Compute the Relative Root Mean Square Error (RRMSE) between predictions and targets.
    RRMSE is defined as the RMSE divided by the average magnitude of the target values.
    """
    def __init__(self, epsilon=1e-8):
        self.total_mse = MetricState(jnp.array(0, dtype=jnp.float32))
        self.total_target_magnitude = MetricState(jnp.array(0, dtype=jnp.float32))
        self.count = MetricState(jnp.array(0, dtype=jnp.int32))
        self.epsilon = epsilon
    
    def reset(self):
        self.total_mse.value = jnp.array(0, dtype=jnp.float32)
        self.total_target_magnitude.value = jnp.array(0, dtype=jnp.float32)
        self.count.value = jnp.array(0, dtype=jnp.int32)

    def update(self, predictions, targets):
        """
        Update the metric state with new predictions and targets.
        Args:
        predictions (jax.Array): The predicted values.
        targets (jax.Array): The true target values.
        """
        target_magnitude = jnp.abs(targets)
        mse = (predictions - targets) ** 2
        self.total_mse.value += (mse if isinstance(mse, (int, float)) else mse.sum())
        self.total_target_magnitude.value += (target_magnitude if isinstance(target_magnitude, (int, float)) else target_magnitude.sum())   
        self.count.value += 1 if isinstance(mse, (int, float)) else mse.size
        
    def compute(self):
        """
        Compute the RRMSE based on the accumulated state.
        Returns:
            jax.Array: The computed RRMSE value.
        """
        mean_mse = self.total_mse.value / self.count.value
        mean_target_magnitude = self.total_target_magnitude.value / self.count.value
        rmse = jnp.sqrt(mean_mse)
        rrmse = rmse / (mean_target_magnitude + self.epsilon)
        return rrmse

class MAE(Average):
    """
    Compute the Mean Absolute Error (MAE) between predictions and targets.
    """
    def update(self, predictions, targets):
        """
        Update the metric state with new predictions and targets.
        Args:
            predictions (jax.Array): The predicted values.
            targets (jax.Array): The true target values.
        """
        mae_values = jnp.abs(predictions - targets)
        super().update(values=mae_values)

def old_RRMSE(predictions, targets, epsilon=1e-8): # check: were we cheating?
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