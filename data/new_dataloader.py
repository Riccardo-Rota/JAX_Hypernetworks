import numpy as np
import jax
import jax.numpy as jnp
from typing import Any, Dict, Optional, Sequence, Union

ArrayLike = Union[np.ndarray, jax.Array]


class JaxDataset:
    """Dataset container for variables, hypervariables, and labels. Data are stored as dictionaries of JAX arrays,
       where each value of the dictionary should be a full dataset array (not a single sample)."""

    def __init__(self,
                 vars: Union[ArrayLike, Dict[str, ArrayLike]],
                 hypervars: Union[ArrayLike, Dict[str, ArrayLike]],
                 labels: Union[ArrayLike, Dict[str, ArrayLike]],
                 ):
        
        # Convert inputs to dicts of jax arrays
        self.vars = _convert_to_jax_dict(vars, "vars")
        self.hypervars = _convert_to_jax_dict(hypervars, "hypervars")
        self.labels = _convert_to_jax_dict(labels, "labels")

        self.length = self._compute_length()
        self._check_length_consistency()


    def _convert_to_jax_dict(field: Union[ArrayLike, Dict[str, ArrayLike]], name: str) -> Dict[str, jax.Array]:
        """Convert input field to a dict of jax arrays."""
        if isinstance(field, dict):
            return {k: jnp.asarray(v) for k, v in field.items()}
        else:
            return {f"{name}_0": jnp.asarray(field)}
        
    def _compute_length(self) -> int:
        """Compute the length of the dataset."""
        return self.vars.value.shape[0]
    
    # TODO: check that slow the computation, decide what to do
    def _check_length_consistency(self) -> None:
        for group_name, group in (("vars", self.vars), ("hypervars", self.hypervars), ("labels", self.labels)):
            for k, v in group.items():
                if v.shape[0] != self.length:
                    raise ValueError(f"Mismatched leading dimension for {group_name}.{k}")
    
    def __len__(self) -> int:
        return self.length
    
    def __getitem__(self, idx: int) -> Dict[str, Dict[str, jax.Array]]:
        out = {}
        for name, group in (("vars", self.vars), ("hypervars", self.hypervars), ("labels", self.labels)):
            out[name] = {k: v[idx] for k, v in group.items()}
        return out

class JaxDataLoader:
    """
    
    """

    def __init__(self,
                 dataset: JaxDataset,
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
        
        # TODO: discuss if dictionary of dictionaries is okay
        batch_data = {
            "vars": {k: v[batch_indices] for k, v in self.dataset.vars.items()},
            "hypervars": {k: v[batch_indices] for k, v in self.dataset.hypervars.items()},
            "labels": {k: v[batch_indices] for k, v in self.dataset.labels.items()},
        }
        
        self._current_idx += self.batch_size
        return batch_data
    
    
    def __len__(self) -> int:
        """Return the number of batches."""
        if self.drop_last:
            return self.n_samples // self.batch_size
        else:
            return (self.n_samples + self.batch_size - 1) // self.batch_size