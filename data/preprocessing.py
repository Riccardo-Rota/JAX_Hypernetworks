import h5py
import numpy as np
import pickle
from array_record.python import array_record_module

def prepare_datasets(
    master_hdf5_path: str = 'turbulent_radiative_layer_tcool_0.10.hdf5', 
    base_name: str = 'turbulence_dataset',
    trajectory_index: int = 0, 
    train_ratio: float = 0.8, 
    val_ratio: float = 0.1, 
    test_ratio: float = 0.1,
    seed: int = 18
):
    """
    Helper function that reads .hdf5 file, extracts a trajectory, flattens it,
    shuffles globally, splits into train/val/test,
    and saves to .hdf5 files.
    """

    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError(f"Split ratios must sum to 1. Got {train_ratio + val_ratio + test_ratio}")

    # Extract all data for the specified trajectory
    with h5py.File(master_hdf5_path, 'r') as f:
        # Check if trajectory_index is within bounds
        n_trajectories = f["t0_fields"]["density"].shape[0]
        if trajectory_index >= n_trajectories:
            raise IndexError(f"trajectory_index {trajectory_index} out of range (dataset has {n_trajectories} trajectories)")

        time_points = f["dimensions"]['time'][:]
        x_points = f["dimensions"]['x'][:]
        y_points = f["dimensions"]['y'][:]
        
        density = f["t0_fields"]["density"][trajectory_index] 
        pressure = f["t0_fields"]["pressure"][trajectory_index]
        velocity = f["t1_fields"]["velocity"][trajectory_index]

    # Combine into (time, x, y, density, pressure, vel_x, vel_y)
    labels = np.concatenate([density[..., np.newaxis],
                             pressure[..., np.newaxis],
                             velocity],
                             axis=-1)
    
    # Reshape labels to (n_timesteps*x*y, 4)
    labels_reshaped = labels.reshape(-1, 4)

    # Create meshgrid for t,x,y
    T, X, Y = np.meshgrid(time_points, x_points, y_points, indexing='ij')

    # Create final dataset: (time, x, y, density, pressure, vel_x, vel_y)
    dataset_array = np.concatenate([
        T.ravel()[..., np.newaxis], 
        X.ravel()[..., np.newaxis], 
        Y.ravel()[..., np.newaxis], 
        labels_reshaped
    ], axis=-1)

    # Shuffle the entire dataset once before saving
    rng = np.random.default_rng(seed)
    shuffled_indices = rng.permutation(len(dataset_array))
    dataset_array = dataset_array[shuffled_indices]

    # Calculate splits
    N = len(dataset_array)
    n_train = int(N * train_ratio)
    n_val = int(N * val_ratio)

    splits = {
        "train": dataset_array[:n_train],
        "val": dataset_array[n_train:n_train+n_val],
        "test": dataset_array[n_train+n_val:]
    }

    # Save to physically separate files
    for split_name, data in splits.items():
        hdf5_path = f"{base_name}_{split_name}.hdf5"
        
        # Save to HDF5 (For fast In-Memory testing)
        with h5py.File(hdf5_path, 'w') as f:
            f.create_dataset('data', data=data.astype(np.float32))

# TODO: for now, we create datasets by running this .py file. In the future, we can decide to move this process somewhere else,
# e.g. a separate data processing script or a notebook.
if __name__ == "__main__":
    prepare_datasets()