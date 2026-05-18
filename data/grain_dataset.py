import os
import numpy as np
from typing import Callable, List, Tuple
import h5py
import pickle
import jax
import jax.numpy as jnp
import grain.python as grain

class InMemoryHDF5Source(grain.RandomAccessDataSource):
    """Dataset Source to be used when loading the HDF5 file entirely into RAM for fast access."""

    def __init__(self, hdf5_path: str):
        with h5py.File(hdf5_path, 'r') as f:
            self._data = f['data'][:] # Load entirely into memory

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx: int):
        single_sample = self._data[idx]

        return {
            "hypervars": single_sample[0:1],       # time
            "vars": single_sample[1:3],            # x,y
        }, single_sample[3:] # density, pressure, velocity_x, velocity_y
    
    def dim_hypervars(self):
        return self.__getitem__(0)[0]["hypervars"].shape[0]
    
    def dim_vars(self):
        return self.__getitem__(0)[0]["vars"].shape[0]
    
    def dim_labels(self):
        return self.__getitem__(0)[1].shape[0]


class ToyDataSource(grain.RandomAccessDataSource):

    def __init__(
        self,
        f: Callable,
        hyper_domains: List[Tuple[float, float]],
        var_domains: List[Tuple[float, float]],
        N: int,
        n_realizations: int,
        seed: int = 42):

        # NOTE: if N % n_realization != 0, last group will have N % n_realization samples,
        # and the rest will have n_realization samples (same as drop_remainder=False in batching operation)

        # Calculate the exact grouping and remainder
        n_full_groups = N // n_realizations
        remainder = N % n_realizations
        
        # Total unique set of hypervariable
        n_hyper_samples = n_full_groups + (1 if remainder > 0 else 0)
        
        # Create the repeat pattern
        repeats = [n_realizations] * n_full_groups + ([remainder] if remainder > 0 else [])

        # jax generation of the dataset must be done on CPU to avoid GPU memory usage
        cpu_device = jax.devices("cpu")[0]

        # Force all JAX operations inside this block to execute in system RAM (CPU)
        with jax.default_device(cpu_device):
            repeats_jax = jnp.array(repeats)

            # Initialize JAX PRNG Key
            key = jax.random.PRNGKey(seed)

            # Generate Hypervariables
            hyper_cols = []
            for low, high in hyper_domains:
                key, subkey = jax.random.split(key)
                col = jax.random.uniform(subkey, shape=(n_hyper_samples,), minval=low, maxval=high)
                hyper_cols.append(col)
            
            unique_hypervars = jnp.stack(hyper_cols, axis=-1)
            
            # Duplicate rows according to the repeat pattern to get the full dataset of hypervariables
            hypervars_jax = jnp.repeat(unique_hypervars, repeats_jax, axis=0)

            # Generate Variables
            # Variables are completely independent, so we always just generate exactly N.
            var_cols = []
            for low, high in var_domains:
                key, subkey = jax.random.split(key)
                col = jax.random.uniform(subkey, shape=(N,), minval=low, maxval=high)
                var_cols.append(col)
                
            vars_jax = jnp.stack(var_cols, axis=-1)

            # Evaluate the parametric function
            # NOTE: variables must be (num_vars, N), since in Python single indexing inside f definition accesses rows
            labels_jax = f(hypervars_jax.T, vars_jax.T)
            if labels_jax.ndim == 1:
                labels_jax = labels_jax[:, jnp.newaxis] # to ensure (N, 1) and not (N,)
            else:
                labels_jax = labels_jax.T # to ensure (N, num_labels) and not (num_labels, N)

        # NOTE: if needed, we have to convert to np.ndarrays (not clear from documentation)
        self._hypervars = np.asarray(hypervars_jax)
        self._vars      = np.asarray(vars_jax)
        self._labels    = np.asarray(labels_jax)
        
        self._num_records = N

        # Attributes for plotting at the end of training
        self.hyper_domains = hyper_domains
        self.var_domains = var_domains
        self.f = f

    def __len__(self):
        return self._num_records

    def __getitem__(self, idx: int):    
        return {
                "hypervars": self._hypervars[idx],
                "vars": self._vars[idx],
            }, self._labels[idx]
    
    def dim_hypervars(self):
        return self._hypervars.shape[1]
    
    def dim_vars(self):
        return self._vars.shape[1]
    
    def dim_labels(self):
        return self._labels.shape[1]    


def build_dataset(
    source: grain.RandomAccessDataSource,
    is_training: bool,
    batch_size: int = 32,
    drop_remainder: bool = False,
    seed: int = 42,
    num_threads: int = None,
    prefetch_size: int = None
):
    """
    Builds and returns an iterator over batched samples.
 
    Args:
        source:       Any RandomAccessDataSource
        is_training:  If True, shuffles and repeats indefinitely;
                      if False (val/test), iterates once without shuffling
        batch_size:   Number of samples per batch
        drop_remainder: If True, drops the last batch if it's smaller than batch_size
        seed:         Random seed used for shuffling
        num_threads:  Override the number of reader threads (None = auto).
        prefetch_size: Override the prefetch buffer size (None = auto).
 
    Returns:
        A Python iterator yielding batches as dicts of arrays.
    """

    # Create MapDataset from the source
    dataset = grain.MapDataset.source(source)

    if is_training:
        dataset = dataset.shuffle(seed=seed)

    dataset = dataset.batch(batch_size=batch_size, drop_remainder=drop_remainder)

    # From documentation: "If the data are already loaded in memory,
    # we recommend setting num_threads to 0 to avoid Python GIL contention by multiple threads."
    # "https://google-grain.readthedocs.io/en/stable/grain.dataset.html#grain.ReadOptions"
    # TODO: study about GIL and optimal number of threads
    if num_threads is None:
        num_threads = 0
    
    if prefetch_size is None:
        prefetch_size = 0

    # Convert to IterDataset
    iter_dataset = dataset.to_iter_dataset(
        # TODO: check if num_threads and prefetch_size are ok
        grain.ReadOptions(
            num_threads=num_threads, 
            prefetch_buffer_size=prefetch_size
        )
    )
    
    # Return DatasetIterator
    return iter(iter_dataset)