import numpy as np
import jax
import jax.numpy as jnp
from typing import Any, Dict, Optional, Sequence, Union

ArrayLike = Union[np.ndarray, jax.Array]


class Dataset:
    """Dataset container for variables, hypervariables, and labels. Data are stored in a dictionary, divided in
       "hypervariables", "variables" and "labels". Data are kept as numpy arrays, to avoid unnecessary copies. Data are then
       converted to jax.Arrays by the DataLoader."""

    def __init__(self,
                 vars: ArrayLike,
                 hypervars: ArrayLike,
                 labels: ArrayLike,
                 ):
        
        # Convert inputs to dicts of jax arrays (asarray do not force-copy if the input is already a jax array)
        self.vars = np.asarray(vars)
        self.hypervars = np.asarray(hypervars)
        self.labels =  np.asarray(labels)
        self.length = self.vars.shape[0]

        # Check dataset dimensionality consistency
        assert self.hypervars.shape[0] == self.length, "Mismatched leading dimension for hypervars"
        assert self.labels.shape[0] == self.length, "Mismatched leading dimension for labels"        
    
    def __len__(self) -> int:
        return self.length
    
    def __getitem__(self, idx: Union[int, ArrayLike]) -> Dict[str, jax.Array]:
        return {
            "hypervars": self.hypervars[idx],
            "vars": self.vars[idx],
            "labels": self.labels[idx],
        }

class JaxDataLoader:
    """
    
    """

    def __init__(self,
                 dataset: Dataset,
                 batch_size: int = 256,
                 shuffle: bool = True,
                 drop_last: bool = False,
                 seed: int = 0):
        
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self._key = jax.random.key(seed)
        self.n_samples = len(self.dataset)
        self._indices = None
        self._current_idx = 0
        self._reset_indices()
    
    def _reset_indices(self):
        self._indices = jnp.arange(self.n_samples)
        if self.shuffle:
            self._key, subkey = jax.random.split(self._key) 
            self._indices = jax.random.permutation(subkey, self._indices)

    def __iter__(self):
        self._current_idx = 0
        return self
    
    def __next__(self):
        """Return the next batch of data and labels."""
        if self._current_idx >= self.n_samples:
            raise StopIteration
        
        start = self._current_idx
        end = start + self.batch_size

        if end > self.n_samples and self.drop_last:
            raise StopIteration
        
        batch_indices = self._indices[start:end]
        
        batch_data = {k: jnp.asarray(v) for k,v in self.dataset[batch_indices].items()}
        
        self._current_idx += self.batch_size
        return batch_data
    
    
    def __len__(self) -> int:
        """Return the number of batches."""
        if self.drop_last:
            return self.n_samples // self.batch_size
        else:
            return (self.n_samples + self.batch_size - 1) // self.batch_size