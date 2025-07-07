# File: utils.py
# This file contains utility functions and classes for JAX-based machine learning tasks.

import jax
import jax.numpy as jnp

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