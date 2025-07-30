import jax
import jax.numpy as jnp
from flax import nnx
import optax
from typing import Callable
from .hypernet_utils import build_state_from_parameters

def train_step(
        hypernetwork: nnx.Module,
        targetnetwork: nnx.Module, 
        hypervariables: jax.Array, 
        x: jax.Array, 
        y: jax.Array, 
        optimizer: optax.GradientTransformationExtraArgs, 
        criterion: Callable = optax.l2_loss) -> jax.Array:
    """
    Performs a single training step, updating the hypernetwork state.
    Args:
        hypernetwork (nnx.Module): The hypernetwork that generates the parameters for the target network.
        targetnetwork (nnx.Module): The target network that will be modified by the hypernetwork.
        hypervariables (jax.Array): Variables that the hypernetwork uses to generate parameters.
        x (jax.Array): Input data for the target network.
        y (jax.Array): Target labels for the input data.
        optimizer (optax.GradientTransformationExtraArgs): Optimizer used to update the hypernetwork parameters.
        criterion (Callable): Function used to compute the loss. It must take as inputs the predictions and targets. Default: optax.l2_loss.
    Returns:
        loss (jax.Array): The computed loss for the current batch.
    """
    def compute_loss(hypernetwork, hypervariables, x, y):
        w = hypernetwork(hypervariables)
        graphdef, template_state = nnx.split(targetnetwork)
        state = nnx.vmap(build_state_from_parameters, in_axes=(None, 0), out_axes=0)(template_state, w)
        modified_targetnetwork = nnx.merge(graphdef, state)
        
        pred = nnx.vmap(type(modified_targetnetwork).__call__)(modified_targetnetwork, x)
        # Or, maybe more readable:
        # @nnx.vmap
        # def compute_predictions(modified_targetnetwork, x):
        #     return modified_targetnetwork(x)
        # pred = compute_predictions(modified_targetnetwork, x)
        loss = jnp.mean(criterion(pred, y))
        return loss
    loss, grads = nnx.value_and_grad(compute_loss)(hypernetwork, hypervariables, x, y)
    optimizer.update(grads)
    return loss

train_step = nnx.jit(train_step, static_argnames=('criterion'))

def evaluation_step(
    hypernetwork: nnx.Module,
    targetnetwork: nnx.Module,
    hypervariables: jax.Array,
    x: jax.Array,
    y: jax.Array,
    criterion: Callable = optax.l2_loss
) -> jax.Array:
    """
    Evaluates the network on a batch of data.
    Args:
        hypernetwork (nnx.Module): The hypernetwork that generates the parameters for the target network.
        targetnetwork (nnx.Module): The target network that will be modified by the hypernetwork.
        hypervariables (jax.Array): Variables that the hypernetwork uses to generate parameters.
        x (jax.Array): Input data for the target network.
        y (jax.Array): Target labels for the input data.
        criterion (Callable): Function used for evaluation. It must take as inputs the predictions and targets. Default: optax.l2_loss.
    Returns:
        loss (jax.Array): The computed loss for the current batch.
    """
    def compute_loss(hypernetwork, hypervariables, x, y):
        w = hypernetwork(hypervariables)
        graphdef, template_state = nnx.split(targetnetwork)
        state = nnx.vmap(build_state_from_parameters, in_axes=(None, 0), out_axes=0)(template_state, w)
        modified_targetnetwork = nnx.merge(graphdef, state) 
        pred = nnx.vmap(type(modified_targetnetwork).__call__)(modified_targetnetwork, x)
        loss = jnp.mean(criterion(pred, y))
        return loss
    loss = compute_loss(hypernetwork, hypervariables, x, y)
    return loss

evaluation_step = nnx.jit(evaluation_step, static_argnames=('criterion'))
