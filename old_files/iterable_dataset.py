from typing import Dict, Sequence, Iterator, Optional
import numpy as np
import jax.numpy as jnp

class IterableDataset:
    # TODO: add documentation

    # NOTE: since it is an iterable dataset, no __getitem__ but __iter__ method,
    # which creates an iterator (in particular, a python generator)
    # np.load with mmap_mode="r" creates a memory map to the .npy file,
    # so that data are read from disk only when accessed (lazy loading)

    def __init__(
        self,
        npy_path: str,
        field_specs: Dict[str, Sequence[int]],
        *,
        shuffle: bool = False,
        seed: int = 0,
        start: Optional[int] = None,
        stop: Optional[int] = None,
    ):
        self.npy_path = npy_path
        self.field_specs = {k: list(v) for k, v in field_specs.items()}
        self.shuffle = shuffle
        self.base_seed = int(seed)
        tmp = np.load(self.npy_path, mmap_mode="r")
        N = tmp.shape[0]
        del tmp  # Release the memory map
        # Theory: this creates a memory map (mmap), with which data elements are read from disk only when accessed
        # (lazy loading, without reading it fully into memory)
        self.start = 0 if start is None else start
        self.stop = N if stop is None else stop
        assert 0 <= self.start <= self.stop <= N

    def __len__(self) -> int:
        return self.stop - self.start

    def __iter__(self) -> Iterator[Dict[str, np.ndarray]]:
        """
        Create an iterator over the dataset.
        
        Note: This does NOT use self.base_seed directly. The epoch offset
        should be provided by the DataLoader via iter_with_epoch().
        """
        return self.iter_with_epoch(epoch=0)
    
    def iter_with_epoch(self, epoch: int = 0) -> Iterator[Dict[str, np.ndarray]]:
        """
        Create an iterator with a specific epoch for reproducible shuffling.
        # NOTE: I need this to have different shuffling each epoch in the dataloader.
        # Another option would be to change self.seed inside the dataloader before calling dataset.__iter__
        # (but this would not be safe for multiple dataloaders sharing the same dataset).
        
        Args:
            epoch: Epoch number to use for seed offset
        """
        # Load data with memory mapping
        data = np.load(self.npy_path, mmap_mode="r")
        n = len(self)
        indices = np.arange(self.start, self.stop)
        if self.shuffle:
            # Use base_seed + epoch for reproducible shuffling per epoch
            rng = np.random.default_rng(self.base_seed + epoch)
            indices = rng.permutation(indices)
            
        for i in indices:
            sample = data[i]
            out = {}
            for k, field_indices in self.field_specs.items():
                v = sample[field_indices]
                # Ensure 1D even if a single index was provided
                if v.ndim == 0:
                    v = v[None]
                out[k] = v
            yield out


class IterableJaxDataLoader:
    """
    TODO: add documentation
    NOTE: no shuffle since already in dataset class. this can be a problem if we want same dataloader for both dataset and iterable dataset
    """
    def __init__(self, dataset, batch_size: int, drop_last: bool = False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self._epoch = 0
        self.n_samples = len(self.dataset)

    def __len__(self) -> int:
        """Return the number of batches"""
        if self.drop_last:
            return self.n_samples // self.batch_size
        else:
            return (self.n_samples + self.batch_size - 1) // self.batch_size
        
    def __iter__(self) -> Iterator[Dict[str, jnp.ndarray]]:
        # Update the dataset seed for this epoch to get different shuffling
        # Get iterator from dataset with current epoch for shuffling
        dataset_iter = self.dataset.iter_with_epoch(self._epoch)
                
        batch = None
        batch_count = 0
        
        for sample in dataset_iter:
            # Initialize batch on first sample
            if batch is None:
                batch = {k: [] for k in sample.keys()}
            
            # Add sample to batch
            for key, value in sample.items():
                batch[key].append(value)
            
            batch_count += 1
            
            # Yield batch when full
            if batch_count == self.batch_size:
                # Convert lists to JAX arrays
                yield {k: jnp.stack(v, axis=0) for k, v in batch.items()}
                batch = None
                batch_count = 0
        
        # Handle remaining samples
        if batch is not None and not self.drop_last:
            # Convert lists to JAX arrays
            yield {k: jnp.stack(v, axis=0) for k, v in batch.items()}
        
        # Increment epoch counter for next iteration
        self._epoch += 1
    
    def reset_epoch(self) -> None:
        """Reset the epoch counter. Useful for reproducible training."""
        self._epoch = 0