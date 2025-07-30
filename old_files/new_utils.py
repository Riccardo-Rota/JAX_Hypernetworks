# File: utils.py
# This file contains utility functions and classes for JAX-based machine learning tasks.

import jax
import flax
import jax.numpy as jnp
from flax import nnx
from flax.nnx.training.metrics import Metric, Average
import optax
import jax.random as random
from typing import Union, Sequence, Callable, Optional

class DataLoader:
    """
    DataLoader for JAX that supports batching and shuffling.
    The data and labels are stored as squeezed JAX arrays.
    When iterating, batches of data and labels are returned as arrays with an additional dimension.
    """

    def __init__(self, data, labels, batch_size=32, shuffle=True, seed=0):
        """
        Initialize the DataLoader.
        Parameters:
            data (array-like): Input data.
            labels (array-like): Corresponding labels.
            batch_size (int): Size of each batch.
            shuffle (bool): Whether to shuffle the data.
            seed (int): Random seed for shuffling.
        """
        self.data = jnp.array(data).squeeze()
        self.labels = jnp.array(labels).squeeze()
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.key = jax.random.key(seed)
        self.n_samples = self.data.shape[0]
        self._reset_indices()
    
    def _reset_indices(self):
        self.indices = jnp.arange(self.n_samples)
        if self.shuffle:
            self.key, subkey = jax.random.split(self.key) 
            self.indices = jax.random.permutation(subkey, self.indices)
    
    def __iter__(self):
        self._current_idx = 0
        return self
    
    def __next__(self):
        """Return the next batch of data and labels."""
        if self._current_idx >= self.n_samples:
            raise StopIteration
        
        start = self._current_idx
        end = start + self.batch_size
        batch_indices = self.indices[start:end]
        
        batch_data = self.data[batch_indices]
        batch_labels = self.labels[batch_indices]

        if batch_data.ndim == 1:
            batch_data = batch_data[:, None]
        if batch_labels.ndim == 1:
            batch_labels = batch_labels[:, None]
        
        self._current_idx += self.batch_size
        return batch_data, batch_labels
    
    def __len__(self):
        """Return the number of batches."""
        return (self.n_samples + self.batch_size - 1) // self.batch_size  

class MLP(nnx.Module):
    """
    A flexible MLP class that allows for a variable number of hidden layers and dimensions.
    Attributes:
        input_dim (int): Dimension of the input layer.
        output_dim (int): Dimension of the output layer.
        num_hidden_layers (int): Number of hidden layers.
        hidden_dims (list): Dimensions of the hidden layers.
        activation_functions (list): Activation functions for the hidden layers.
        num_parameters (int): Total number of parameters in the network.
    Methods:
        __call__(x): Forward pass through the network.
    """
   
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_hidden_layers: Optional[int] = None,
        hidden_dims: Union[int, Sequence[int]] = 8,
        activation_functions: Union[Callable, Sequence[Callable]] = nnx.relu,
        rngs: nnx.Rngs = nnx.Rngs(0)
    ):
        """
        Initializes the MLP with the specified parameters.
        Args:
            input_dim (int): Dimension of the input layer.
            output_dim (int): Dimension of the output layer.
            num_hidden_layers (int, optional): Number of hidden layers. If None, it is inferred
                from the length of hidden_dims. Default: None.
            hidden_dims (int or Sequence[int], optional): Dimension(s) of the hidden layers.
                If an int is provided, it is used for all hidden layers. If a sequence is 
                provided, its length must match `num_hidden_layers` if specified. Default: 8.
            activation_functions (Callable or Sequence[Callable], optional): Activation function(s)
                for the hidden layers. If a single callable is provided, it is used for all
                hidden layers. If a sequence is provided, its length must match the number of
                hidden layers. Default: nnx.relu.
            rngs (nnx.Rngs): Random number generators used to initialize the network. Default: nnx.Rngs(0).
        """

        if isinstance(hidden_dims,Sequence):
            if not all(isinstance(dim, int) for dim in hidden_dims):
                raise TypeError("hidden_dims must be an int or a sequence of ints")
        elif not isinstance(hidden_dims, int):
            raise TypeError("hidden_dims must be an int or a sequence of ints")
        if isinstance(activation_functions, Sequence):
            if not all(callable(func) for func in activation_functions):
                raise TypeError("activation_functions must be a callable or a sequence of callables")
        elif not callable(activation_functions):
            raise TypeError("activation_functions must be a callable or a sequence of callables")
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_parameters = 0

        if num_hidden_layers:
            if isinstance(hidden_dims, int):
                self.hidden_dims = [hidden_dims] * num_hidden_layers
                self.num_hidden_layers = num_hidden_layers
            elif len(hidden_dims) == num_hidden_layers:
                self.hidden_dims = list(hidden_dims)
                self.num_hidden_layers = len(hidden_dims)
            else:
                raise ValueError("Length of hidden_dims must match num_hidden_layers")
            
            if isinstance(activation_functions, Callable):
                self.activation_functions = [activation_functions] * self.num_hidden_layers
            elif len(activation_functions) == self.num_hidden_layers:
                self.activation_functions = list(activation_functions)
            else:
                raise ValueError("Length of activation_functions must match num_hidden_layers")
            
        else:
            self.hidden_dims = [hidden_dims] if isinstance(hidden_dims, int) else list(hidden_dims)
            self.num_hidden_layers = len(self.hidden_dims)
            self.activation_functions = [activation_functions] * self.num_hidden_layers if isinstance(activation_functions, Callable) else list(activation_functions)
        
        layers = []
        
        if self.num_hidden_layers > 0:
            layers.append(nnx.Linear(input_dim, self.hidden_dims[0], rngs=rngs))  # First hidden layer
            layers.append(self.activation_functions[0])
            self.num_parameters = self.hidden_dims[0] * (input_dim + 1)  # +1 for bias

            for i in range(self.num_hidden_layers - 1): # Additional hidden layers
                layers.append(nnx.Linear(self.hidden_dims[i], self.hidden_dims[i+1], rngs=rngs))
                layers.append(self.activation_functions[i+1])
                self.num_parameters += self.hidden_dims[i+1] * (self.hidden_dims[i] + 1)

            layers.append(nnx.Linear(self.hidden_dims[-1], output_dim, rngs=rngs)) # Output layer
            self.num_parameters += output_dim * (self.hidden_dims[-1] + 1)

        else: # Direct input to output
            layers.append(nnx.Linear(input_dim, output_dim, rngs=rngs))
            self.num_parameters = output_dim * (input_dim + 1)

        self.layers = nnx.Sequential(*layers)

    def __call__(self, x: jax.Array):
        """
        Forward pass through the network.
        Args:
            x (jax.Array): Input data.
        Returns:
            jax.Array: Output of the network.
        """
        return self.layers(x)

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

@nnx.jit
#@nnx.vmap(in_axes=(None, 0), out_axes=0) mi sembra più comodo tenere come base la funzione senza vmap, e poi usare nnx.vmap quando serve
def build_state_from_parameters(template_state: nnx.statelib.State, parameters: jax.Array) -> nnx.statelib.State:
    """
    Builds a state from the parameters, reshaping them according to the template state.
    Args:
        template_state (nnx.statelib.State): The template state that defines the structure of the parameters.
        parameters (jax.Array): The parameters to be reshaped and assigned to the template state.
    Returns:
        nnx.statelib.State: The state with the parameters reshaped according to the template.
    """
    treedef = jax.tree.structure(template_state)
    reshaped_parameters = []
    shapes = []
    sizes = []
    for _, param in nnx.to_flat_state(template_state):
        shapes.append(param.value.shape)
        sizes.append(param.value.size)
    i = 0
    for shape, size in zip(shapes, sizes):
        reshaped_parameters.append(parameters[i:i+size].reshape(shape))
        i += size
    state = jax.tree.unflatten(treedef, reshaped_parameters)
    return state

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


@nnx.jit
def assign_parameters(model: nnx.Module, parameters: jax.Array) -> nnx.Module:
    """
    Assigns the parameters from an array to the state of the model.
    Args:
        model (nnx.Module): The model whose state will be updated with the new parameters.
        parameters (jax.Array): The parameters to assign to the model's state.
    Returns:
        nnx.Module: The model with the updated state.
    """
    
    graphdef, template_state = nnx.split(model)
    state = build_state_from_parameters(template_state = template_state, parameters = parameters)
    return nnx.merge(graphdef, state)


@nnx.jit
def apply(model, parameters, x):
    """
    Applies the model to the input data using the specified parameters.
    Args:
        model (nnx.Module): The model to apply.
        parameters (jax.Array): The parameters to use for the model.
        x (jax.Array): Input data to the model.
    Returns:
        jax.Array: The output of the model after applying it to the input data.
    """
    return assign_parameters(model, parameters)(x)


def variables_generator(N: int, domains: list, key: random.PRNGKey = random.key(0)) -> list:
    """
    Generates N random variables from the specified domains.
    Each domain is a tuple (min, max) representing the range of the variable.
    """
    variables = []
    for domain in domains:
        key, subkey = random.split(key)
        var = random.uniform(subkey, (N,), minval=domain[0], maxval=domain[1])
        variables.append(var)
    return variables