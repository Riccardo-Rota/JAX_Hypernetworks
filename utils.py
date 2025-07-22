# File: utils.py
# This file contains utility functions and classes for JAX-based machine learning tasks.

import jax
import jax.numpy as jnp
from flax import nnx
from flax.nnx.training.metrics import Metric, Average
import optax

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
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, num_hidden_layers: int, *, rngs: nnx.Rngs):
        
        self.num_hidden_layers = num_hidden_layers
        
        layers = []
        
        if num_hidden_layers > 0:
            # First hidden layer
            layers.append(nnx.Linear(input_dim, hidden_dim, rngs=rngs))
            layers.append(nnx.relu)
            
            # Additional hidden layers
            for _ in range(num_hidden_layers - 1):
                layers.append(nnx.Linear(hidden_dim, hidden_dim, rngs=rngs))
                layers.append(nnx.relu)
                
            # Output layer
            layers.append(nnx.Linear(hidden_dim, output_dim, rngs=rngs))

        else:
            # Direct input to output
            layers.append(nnx.Linear(input_dim, output_dim, rngs=rngs))
            
        # Store in sequential container
        self.layers = nnx.Sequential(*layers)
        
    def __call__(self, x: jax.Array):
        return self.layers(x)
    

# New trial: hidden layers with different lengths DA SISTEMARE
class MLP_trial(nnx.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim1: int, hidden_dim2: int, num_hidden_layers1: int, num_hidden_layers2: int, *, rngs: nnx.Rngs):
        
        self.num_hidden_layers = num_hidden_layers1
        self.num_hidden_layers2 = num_hidden_layers2
        
        layers = []
        
        if num_hidden_layers1 > 0:
            # First hidden layer
            layers.append(nnx.Linear(input_dim, hidden_dim1, rngs=rngs))
            layers.append(nnx.relu)
            
            # Additional hidden layers
            for _ in range(num_hidden_layers1 - 1):
                layers.append(nnx.Linear(hidden_dim1, hidden_dim1, rngs=rngs))
                layers.append(nnx.relu)
        
        if num_hidden_layers2 > 0:
            # First hidden layer
            layers.append(nnx.Linear(hidden_dim1, hidden_dim2, rngs=rngs))
            layers.append(nnx.relu)
            
            # Additional hidden layers
            for _ in range(num_hidden_layers2 - 1):
                layers.append(nnx.Linear(hidden_dim2, hidden_dim2, rngs=rngs))
                layers.append(nnx.relu)
        
            # Output layer
            layers.append(nnx.Linear(hidden_dim2, output_dim, rngs=rngs))
            
        # Store in sequential container
        self.layers = nnx.Sequential(*layers)
        
    def __call__(self, x: jax.Array):
        return self.layers(x)
    

@nnx.jit
@nnx.vmap
def apply(network, parameters, x):
    """
    Assigns the parameters from an array to the state. state and parameters must be batched with the same first dimension.
    """
    parameters = parameters.squeeze()
    graphdef, state = nnx.split(network)
    flat_state = nnx.to_flat_state(state)

    # TODO: Check if the number of parameters matches
    
    i = 0
    for key, param in flat_state:
        param_size = param.value.size
        # Extract parameters for this specific parameter
        param_values = parameters[i:i + param_size]
        # Reshape to match the original parameter shape
        param.value = param_values.reshape(param.value.shape)
        i += param_size

    modified_network = nnx.merge(graphdef, state)

    return modified_network(x)


def train_step(hypernetwork, targetnetwork_fun, hyperparams, x, y, optimizer, batch_size):
    """
    Performs a single training step."""
    def loss_fn(hypernetwork, hyperparams, x, y, batch_size):
        w = hypernetwork(hyperparams)

        @nnx.split_rngs(splits=x.shape[0])
        @nnx.vmap(in_axes=(0, None), out_axes=0)
        def make_model(rngs, targetnetwork_fun):
            return targetnetwork_fun(1, 1, 8, 2, rngs=rngs)
        
        targetnetwork = make_model(nnx.Rngs(0), targetnetwork_fun)

        pred = apply(targetnetwork, w, x)
        loss = jnp.mean(optax.l2_loss(pred, y))
        #eps = 1e-8  # Small epsilon to avoid division by zero
        #relative_errors = (pred - y) / (y + eps)
        #loss = jnp.sqrt(jnp.mean(relative_errors ** 2))
        return loss
    loss, grads = nnx.value_and_grad(loss_fn)(hypernetwork, hyperparams, x, y, batch_size)
    optimizer.update(grads)
    return loss

train_step = nnx.jit(train_step, static_argnames=('targetnetwork_fun','batch_size'))

def evaluation_step(hypernetwork, targetnetwork_fun, hyperparams, x, y, batch_size):
    """
    Performs a single training step."""
    def loss_fn(hypernetwork, hyperparams, x, y, batch_size):
        w = hypernetwork(hyperparams)

        @nnx.split_rngs(splits=batch_size)
        @nnx.vmap(in_axes=(0, None), out_axes=0)
        def make_model(rngs, targetnetwork_fun):
            return targetnetwork_fun(1, 1, 8, 2, rngs=rngs)
        
        targetnetwork = make_model(nnx.Rngs(0), targetnetwork_fun)
        
        pred = apply(targetnetwork, w, x)
        loss = jnp.mean(optax.l2_loss(pred, y))
        #eps = 1e-8  # Small epsilon to avoid division by zero
        #relative_errors = (pred - y) / (y + eps)
        #loss = jnp.sqrt(jnp.mean(relative_errors ** 2))
        return loss
    loss = loss_fn(hypernetwork, hyperparams, x, y, batch_size)
    return loss

evaluation_step = nnx.jit(evaluation_step, static_argnames=('targetnetwork_fun','batch_size'))