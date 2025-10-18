from data import Dataset
import h5py
import numpy as np

def create_dataset_turbulence(path: str = 'turbulent_radiative_layer_tcool_0.10.hdf5', num_labels: int = 4, num_data: int = 1000, trajectory: int = 0) -> Dataset:

    if num_labels >4:
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
    dataset = Dataset(vars = dataset_array[:num_data,1:3], hypervars = dataset_array[:num_data,0], labels = dataset_array[:num_data,3:3+num_labels])
    return dataset