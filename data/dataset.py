import numpy as np
from pathlib import Path
import jax
import jax.numpy as jnp
from typing import Dict, Union, Sequence, Optional


ArrayLike = Union[np.ndarray, jax.Array]


class Dataset:
    """Dataset container for variables, hypervariables, and labels. Data are stored in a dictionary, divided in
       "hypervariables", "variables" and "labels"."""
    # TODO: supergeneralized needed?

    def __init__(self,
                vars: Optional[ArrayLike] = None,
                hypervars: Optional[ArrayLike] = None,
                labels: Optional[ArrayLike] = None,
                *,
                data: Optional[ArrayLike] = None,
                column_map: Optional[Dict[str, Sequence[int]]] = None
                ):
        if data is None:
            assert vars is not None and hypervars is not None and labels is not None, "If data is not provided, vars, hypervars and labels must be provided"
            
            self._mode = "separated"
            self.vars = vars
            self.hypervars = hypervars
            self.labels =  labels
            self.length = self.vars.shape[0]

            # Check dataset dimensionality consistency
            assert self.hypervars.shape[0] == self.length, "Mismatched leading dimension for hypervars"
            assert self.labels.shape[0] == self.length, "Mismatched leading dimension for labels"
        else:
            assert column_map is not None, "If data is provided, column_map must be provided"
            
            self._mode = "combined"
            self.data = data
            self.length = self.data.shape[0]

            self._vars_idx = np.asarray(column_map["vars"])
            self._hypervars_idx = np.asarray(column_map["hypervars"])
            self._labels_idx = np.asarray(column_map["labels"])

            
    @classmethod
    def from_npy(cls,
                 dataset_path: Union[str, Path],
                 column_map: Dict[str, Sequence[int]]
                 ):
        data = np.load(dataset_path, mmap_mode='r')
        return cls(data=data, column_map=column_map)
    
    def __len__(self) -> int:
        return self.length
    
    def _normalize_index(self, idx: Union[int, ArrayLike]):
        # Cast index to numpy array if it's a JAX array (otherwise no compatibility with numpy data)
        if isinstance(idx, jax.Array):
            return np.asarray(idx)  # small batch index
        return idx
    
    def __getitem__(self, idx: Union[int, ArrayLike]) -> Dict[str, ArrayLike]:
        idx = self._normalize_index(idx)

        if self._mode == "separated":
            hypervars = self.hypervars[idx]
            vars = self.vars[idx]
            labels = self.labels[idx]

        else:  # combined mode
            rows = self.data[idx]
            hypervars = rows[..., self._hypervars_idx]
            vars = rows[..., self._vars_idx]
            labels = rows[..., self._labels_idx]

        if hypervars.ndim == 1:
            hypervars = hypervars[:, None]
        if vars.ndim == 1:
            vars = vars[:, None]
        if labels.ndim == 1:
            labels = labels[:, None]

        return {
            "hypervars": hypervars,
            "vars": vars,
            "labels": labels,
        }

class JaxDataLoader:
    """
    TODO: document
    TODO: NamedTuple/dataclass to avoid static argument class?
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
        
        self._rng = np.random.default_rng(seed)
        self.n_samples = len(self.dataset)
        self._indices: np.ndarray = None
        self._current_idx = 0

        self._reset_indices()
    
    def _reset_indices(self) -> None:
        """Create (and optionally shuffle) the index array for a new epoch."""
        self._indices = np.arange(self.n_samples)
        if self.shuffle:
            self._rng.shuffle(self._indices)

    def __iter__(self):
        self._reset_indices()
        self._current_idx = 0
        return self
    
    def __next__(self):
        """Return the next batch of data and labels."""
        if self._current_idx >= self.n_samples:
            raise StopIteration
        
        start = self._current_idx
        end = start + self.batch_size

        if end > self.n_samples:
            if self.drop_last:
                # no partial batch
                raise StopIteration
            else:
                end = self.n_samples
        
        batch_indices = self._indices[start:end]
        self._current_idx = end

        batch_np = self.dataset[batch_indices]
        
        batch_jax = {k: jnp.asarray(v) for k,v in batch_np.items()}
        
        return batch_jax
    
    
    def __len__(self) -> int:
        """Return the number of batches."""
        if self.drop_last:
            return self.n_samples // self.batch_size
        else:
            return (self.n_samples + self.batch_size - 1) // self.batch_size