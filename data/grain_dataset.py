import h5py
import pickle
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

def get_train_pipeline(
    file_path: str,
    is_training: bool,
    use_array_record: bool = False,
    batch_size: int = 32,
    num_threads: int = 1,  
    prefetch_size: int = 2
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
        dataset = dataset.shuffle(seed=42).repeat()
        drop_remainder = True
    else:
        # No shuffle, no repeat for validation and testing
        drop_remainder = False

    dataset = dataset.batch(batch_size=batch_size, drop_remainder=drop_remainder)

    # Convert to IterDataset
    iter_dataset = dataset.to_iter_dataset(
        # TODO: check if num_threads and prefetch_size are ok
        grain.ReadOptions(
            num_threads=num_threads, 
            prefetch_buffer_size=prefetch_size
        )
    )
    
    return iter(iter_dataset)