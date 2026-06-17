import numpy as np
from typing import Callable, List, Sequence, Tuple, Dict
import h5py
import jax
import jax.numpy as jnp
import grain.python as grain

class InMemoryHDF5Source(grain.RandomAccessDataSource):
    """
    Dataset Source to be used when loading the HDF5 file entirely into RAM for fast access.
    """

    def __init__(
        self, 
        hdf5_path: str,
        schema: Dict[str, int],
        hypervar_keys: Sequence[str],
        var_keys: Sequence[str],
        target_keys: Sequence[str],
        dataset_key: str = 'data',
        var_bounds: List[Tuple[float, float]] = None
    ):
        """
        Initializes the in-memory HDF5 data source.

        Args:
            hdf5_path (str): The file path to the HDF5 dataset.
            schema (Dict[str, int]): A mapping from variable names to their column indices in the dataset.
            hypervar_keys (Sequence[str]): A list of keys corresponding to hypervariables.
            var_keys (Sequence[str]): A list of keys corresponding to standard variables.
            target_keys (Sequence[str]): A list of keys corresponding to the target labels.
            dataset_key (str, optional): The internal HDF5 key where the data matrix is stored. Defaults to 'data'.
            var_bounds (List[Tuple[float, float]], optional): A list of (min, max) bounds to filter rows. 
                Must match the length of var_keys. Defaults to None.
        """
        with h5py.File(hdf5_path, 'r') as f:
            data = f[dataset_key][:] # Load entirely into memory

        try:
            self.hypervar_indices = np.array([schema[key] for key in hypervar_keys], dtype=int)
            self.var_indices = np.array([schema[key] for key in var_keys], dtype=int)
            self.target_indices = np.array([schema[key] for key in target_keys], dtype=int)
        except KeyError as e:
            raise ValueError(f"Key {e} not found in schema. Available keys: {list(schema.keys())}")
    
        # If required, rescrict variable domains
        if var_bounds is not None:
            vars_data = data[:, self.var_indices]
            mask = np.ones(len(data), dtype=bool)
            
            for i, (low, high) in enumerate(var_bounds):
                mask &= (vars_data[:, i] >= low) & (vars_data[:, i] <= high)
                
            data = data[mask]

        self._data = data

    def __len__(self) -> int:
        """
        Returns the total number of samples in the filtered dataset.

        Returns:
            int: Number of rows in the dataset.
        """
        return len(self._data)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """
        Retrieves a single sample from memory at the specified index.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            Tuple[Dict[str, np.ndarray], np.ndarray]: A tuple containing a dictionary of inputs 
            ('hypervars', 'vars') and an array of target labels.
        """
        single_sample = self._data[idx]

        return {
            "hypervars": single_sample[self.hypervar_indices],
            "vars": single_sample[self.var_indices],
        }, single_sample[self.target_indices]

    def dim_hypervars(self) -> int:
        """
        Returns the number of hypervariable features.

        Returns:
            int: Dimension of hypervariables.
        """
        return len(self.hypervar_indices)
    
    def dim_vars(self) -> int:
        """
        Returns the number of standard variable features.

        Returns:
            int: Dimension of standard variables.
        """
        return len(self.var_indices)
    
    def dim_labels(self) -> int:
        """
        Returns the number of target label features.

        Returns:
            int: Dimension of labels.
        """
        return len(self.target_indices)


class ToyDataSource(grain.RandomAccessDataSource):
    """
    In-memory data source that generates toy data using JAX, evaluating 
    a provided parametric function over generated hypervariables and variables.
    """

    def __init__(
        self,
        f: Callable,
        hyper_domains: List[Tuple[float, float]],
        var_domains: List[Tuple[float, float]],
        N: int,
        n_realizations: int,
        seed: int = 42):
        """
        Initializes the generated toy dataset.

        Args:
            f (Callable): The parametric function to evaluate. Must accept transposed arrays 
                of shape (num_hypervars, N) and (num_vars, N).
            hyper_domains (List[Tuple[float, float]]): List of (min, max) bounds for each hypervariable.
            var_domains (List[Tuple[float, float]]): List of (min, max) bounds for each standard variable.
            N (int): The total number of records to generate.
            n_realizations (int): The number of times each unique hypervariable configuration is repeated.
            seed (int, optional): Random seed for JAX PRNG.
        """

        # Calculate the exact grouping and remainder
        n_full_groups = N // n_realizations
        remainder = N % n_realizations
        
        # Total unique set of hypervariable
        n_hyper_samples = n_full_groups + (1 if remainder > 0 else 0)
        
        # Create the repeat pattern
        repeats = [n_realizations] * n_full_groups + ([remainder] if remainder > 0 else [])

        # jax generation of the dataset must be done on CPU to avoid GPU VRAM exhaustion
        cpu_device = jax.devices("cpu")[0]

        with jax.default_device(cpu_device):
            repeats_jax = jnp.array(repeats)
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
            labels_jax = f(hypervars_jax.T, vars_jax.T)

            # Ensure correct dimensional output structures
            if labels_jax.ndim == 1:
                labels_jax = labels_jax[:, jnp.newaxis] # to ensure (N, 1) and not (N,)
            else:
                labels_jax = labels_jax.T # to ensure (N, num_labels) and not (num_labels, N)

        # Convert back to standard NumPy arrays for Grain compatibility
        self._hypervars = np.asarray(hypervars_jax)
        self._vars      = np.asarray(vars_jax)
        self._labels    = np.asarray(labels_jax)
        
        self._num_records = N

        # Attributes for plotting at the end of training
        self.hyper_domains = hyper_domains
        self.var_domains = var_domains
        self.f = f

    def __len__(self) -> int:
        """
        Returns the total number of procedurally generated records.

        Returns:
            int: Number of records (N).
        """
        return self._num_records

    def __getitem__(self, idx: int) -> Tuple[Dict[str, np.ndarray], np.ndarray]:   
        """
        Retrieves a single generated sample.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            Tuple[Dict[str, np.ndarray], np.ndarray]: A tuple containing a dictionary of inputs 
            ('hypervars', 'vars') and an array of target labels.
        """ 
        return {
                "hypervars": self._hypervars[idx],
                "vars": self._vars[idx],
            }, self._labels[idx]
    
    def dim_hypervars(self) -> int:
        """
        Returns the number of hypervariable features.

        Returns:
            int: Dimension of hypervariables.
        """
        return self._hypervars.shape[1]
    
    def dim_vars(self) -> int:
        """
        Returns the number of standard variable features.

        Returns:
            int: Dimension of standard variables.
        """
        return self._vars.shape[1]
    
    def dim_labels(self) -> int:
        """
        Returns the number of target label features.

        Returns:
            int: Dimension of labels.
        """
        return self._labels.shape[1]    


def build_dataset(
    source: grain.RandomAccessDataSource,
    is_training: bool,
    batch_size: int = 32,
    drop_remainder: bool = False,
    seed: int = 42
):
    """
    Builds and returns an iterator over batched samples using Google Grain.
 
    Args:
        source (grain.RandomAccessDataSource): Any valid Grain RandomAccessDataSource.
        is_training (bool): If True, shuffles and repeats the dataset indefinitely. 
            If False (validation/testing), iterates exactly once without shuffling.
        batch_size (int, optional): Number of samples per batch.
        drop_remainder (bool, optional): If True, drops the final batch if it is smaller 
            than batch_size.
        seed (int, optional): Random seed used for the shuffling operation
 
    Returns:
        Iterator: A Python iterator yielding batches of data.
    """

    # Create MapDataset from the source
    dataset = grain.MapDataset.source(source)

    if is_training:
        dataset = dataset.shuffle(seed=seed)

    dataset = dataset.batch(batch_size=batch_size, drop_remainder=drop_remainder)

    # Convert to IterDataset
    iter_dataset = dataset.to_iter_dataset(
        grain.ReadOptions(
            num_threads=0, 
            prefetch_buffer_size=0
        )
    )
    
    # Return DatasetIterator
    return iter(iter_dataset)