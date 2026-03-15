import os
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
            "labels": single_sample[3:]            # density, pressure, velocity_x, velocity_y
        }


class ToyDataSource(grain.RandomAccessDataSource):

    def __init__(
        self, f: Callable,
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
        self._hypervars = hypervars_jax
        self._vars = vars_jax
        self._labels = labels_jax
        
        self._num_records = N

    def __len__(self):
        return self._num_records

    def __getitem__(self, idx: int):
        return {
            "hypervars": self._hypervars[idx],
            "vars": self._vars[idx],
            "labels": self._labels[idx]
        }
    

def get_pipeline(
    file_path: str,
    is_training: bool,
    use_array_record: bool = False,
    batch_size: int = 32,
    seed: int = 42,
    num_threads: int = None,
    prefetch_size: int = None
):
    """
    Builds the Grain MapDataset object depending on use_array_record.
    """
    
    if use_array_record:
        # TODO: check if it works
        raw_source = grain.ArrayRecordDataSource(file_path)
        dataset = grain.MapDataset.source(raw_source).map(pickle.loads)
    else:
        raw_source = InMemoryHDF5Source(file_path)
        dataset = grain.MapDataset.source(raw_source)

    if is_training:
        dataset = dataset.shuffle(seed=seed).repeat()
        drop_remainder = True
    else:
        # No shuffle, no repeat for validation and testing
        drop_remainder = False

    dataset = dataset.batch(batch_size=batch_size, drop_remainder=drop_remainder)

    # From documentation: "If the data are already loaded in memory,
    # we recommend setting num_threads to 0 to avoid Python GIL contention by multiple threads."
    # "https://google-grain.readthedocs.io/en/stable/grain.dataset.html#grain.ReadOptions"
    # TODO: study about GIL and optimal number of threads
    if num_threads is None:
        if not use_array_record:
            # Data is IN MEMORY. Follow the docs: set to 0.
            num_threads = 0 
        else:
            # Data is ON DISK. Use threads to hide I/O latency.
            total_cores = os.cpu_count() or 4
            num_threads = max(1, total_cores // 4)
    
    if prefetch_size is None:
        if not use_array_record:
            # Data is IN MEMORY.
            prefetch_size = 0 
        else:
            # Data is ON DISK. Prefetch to avoid I/O bottlenecks.
            prefetch_size = 2

    # Convert to IterDataset
    iter_dataset = dataset.to_iter_dataset(
        # TODO: check if num_threads and prefetch_size are ok
        grain.ReadOptions(
            num_threads=num_threads, 
            prefetch_buffer_size=prefetch_size
        )
    )
    
    return iter(iter_dataset)