import h5py
import numpy as np

def create_dataset_turbulence(path: str = 'turbulent_radiative_layer_tcool_0.10.hdf5', npy_path: str = 'turbulent_radiative_layer_tcool_0.10.npy', num_labels: int = 4, trajectory: int = 0, seed: int = 42) -> None:
    #TODO: Add docstring
    # From .hdf5 file, create a .npy dataset

    if num_labels > 4:
        raise ValueError("num_labels cannot be greater than 4 (density, pressure, vel_x, vel_y)")

    # Extract datasets for density, pressure and velocity
    with h5py.File(path, 'r') as f:
        time_points = f["dimensions"]['time'][:]
        x_points = f["dimensions"]['x'][:]
        y_points = f["dimensions"]['y'][:]
        density = f["t0_fields"]["density"][trajectory]  # shape (n_timesteps, x, y)
        pressure = f["t0_fields"]["pressure"][trajectory]  # shape (n_timesteps, x, y)
        velocity = f["t1_fields"]["velocity"][trajectory]  # shape (n_timesteps, x, y, 2)

    labels = np.concatenate([density[..., np.newaxis], pressure[..., np.newaxis], velocity], axis=-1)
    # Reshape labels to (n_timesteps*x*y, 4)
    labels_reshaped = labels.reshape(-1, 4)

    # Create meshgrid for t,x,y
    T, X, Y = np.meshgrid(time_points, x_points, y_points, indexing='ij')

    # Flatten the coordinates and field
    T_flat = T.ravel()
    X_flat = X.ravel()
    Y_flat = Y.ravel()

    # Create final dataset: (time, x, y, density, pressure, vel_x, vel_y)
    dataset_array = np.concatenate([T_flat[..., np.newaxis], X_flat[..., np.newaxis], Y_flat[..., np.newaxis], labels_reshaped], axis=-1)

    # Shuffle the entire dataset once before saving
    rng = np.random.default_rng(seed)
    shuffled_indices = rng.permutation(len(dataset_array))
    dataset_array = dataset_array[shuffled_indices]
    
    np.save(npy_path, dataset_array)

def split_dataset_turbulence(npy_path: str = 'turbulent_radiative_layer_tcool_0.10.npy', train_path: str = 'turbulent_radiative_layer_tcool_0.10_train.npy', val_path: str = 'turbulent_radiative_layer_tcool_0.10_val.npy', test_path: str = 'turbulent_radiative_layer_tcool_0.10_test.npy', train_frac: float = 0.7, val_frac: float = 0.15, test_frac: float = 0.15, seed: int = 42) -> None:
    #TODO: Add docstring
    # Split .npy dataset into train, val and test

    # Normalize fractions if they don't sum to 1
    if not np.isclose(train_frac + val_frac + test_frac, 1.0):
        train_frac = train_frac / (train_frac + val_frac + test_frac)
        val_frac = val_frac / (train_frac + val_frac + test_frac)
        test_frac = test_frac / (train_frac + val_frac + test_frac)
    
    # Load and shuffle dataset
    data = np.load(npy_path)
    n_samples = data.shape[0]
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_samples)
    data = data[indices]

    # Get split indices
    train_end = int(train_frac * n_samples)
    val_end = train_end + int(val_frac * n_samples)
    test_end = val_end + int(test_frac * n_samples)

    # Split the data
    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:test_end]

    # Save the splits
    np.save(train_path, train_data)
    np.save(val_path, val_data)
    np.save(test_path, test_data)