import numpy as np
import jax
import jax.numpy as jnp
import flax.nnx as nnx
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional, Callable, Union
import h5py
from pathlib import Path
import os

def plot_loss_curves(train_history: List[dict], val_history: List[dict], save_path: str, logx: bool=False, logy: bool=False, loss_key: Optional[str]=None):
    """
    Plots the training and validation loss (or metrics) curves.
    Args:        
        train_history (dict): List of dicts with train loss/metric values for each epoch.
        val_history (dict): List of dicts with validation loss/metric values for each epoch.
        save_path (str): Path to save the generated plot.
        logx (bool): Whether to use logarithmic scale for the x-axis. Default: False.
        logy (bool): Whether to use logarithmic scale for the y-axis. Default: False.
        loss_key (Optional[str]): Key in the train_results and val_results dicts to use for plotting. If None, it will use the first key found in the dicts. Default: None.
    """

    # Setup directories
    directory = Path("figures")
    directory.mkdir(parents=True, exist_ok=True)
    base_name = "loss_curve"
    extension = ".png"

    if not logx:
        plotter = plt.plot if not logy else plt.semilogy
    else:
        plotter = plt.semilogx if not logy else plt.loglog
    if loss_key is None:
        loss_key = next(iter(val_history[0].keys()))
    train_values = [epoch[loss_key] for epoch in train_history]
    val_values = [epoch[loss_key] for epoch in val_history]
    plt.figure()
    plotter(range(len(train_history)), train_values, label=f'Training {loss_key}')
    plotter(range(len(val_history)), val_values, label=f'Validation {loss_key}')
    plt.xlabel('Epochs')
    plt.ylabel(loss_key)
    plt.legend()

    unique_save_path = directory / f"{base_name}_{extension}"
    plt.savefig(unique_save_path, bbox_inches='tight')
    plt.savefig(save_path)

    plt.close()


def plot_1d_predictions(
    model: nnx.Module,
    hypervars_set: Tuple[jnp.ndarray, ...],
    var_domain: Tuple[float, float],
    output_idx: Optional[int] = None,
    exact_function: Optional[Callable] = None,
    n_points: int = 400
):
    """
    Plots 1D predictions for a given set of hypervariables.

    Args:
        model (nnx.Module): The final trained network
        hypervars_set (Tuple[jnp.ndarray, ...]): A tuple containing individual 1D arrays 
            of hypervariables. Each array represents a distinct configuration ($\theta$) to evaluate.
            The function create a plot for each configuration of hypervariables provided in this tuple.
        var_domain (Tuple[float, float]): The (min, max) boundary for the 1D input variable ($x$).
        output_idx (Optional[int], optional): The index to extract from the model's output 
            if the model returns a tuple. Must be explicitly provided if the model output is 
            a tuple. Defaults to None.
        exact_function (Optional[Callable], optional): A reference ground-truth function to 
            superimpose on the plot. It must accept two arguments: an unbatched hypervariable 
            array and a batched variable array (e.g., `f(theta, x)`). Defaults to None.
        n_points (int, optional): The resolution of the plot, representing the number of 
            linearly spaced points generated across the `var_domain`. Defaults to 400.
    """
    var_values = jnp.linspace(var_domain[0], var_domain[1], n_points).reshape(-1, 1)  # Shape (n_points, 1)

    # Vectorize the exact function (for unbarched hypervars)
    if exact_function is not None:
        batched_exact_fn = jax.vmap(exact_function, in_axes=(None, 0))
    
    # Setup directories
    directory = Path("figures")
    directory.mkdir(parents=True, exist_ok=True)
    base_name = "prediction_set"
    extension = ".png"

    for i, hypervars in enumerate(hypervars_set):
        # Initialize figure
        plt.figure(figsize=(10, 6))

        # Format the hyipervars into a clean string for plot legends
        hv_formatted = ", ".join([f"{v:.3f}" for v in hypervars.tolist()])
        label_suffix = fr"$\theta$=[{hv_formatted}]"

        data = {"hypervars": hypervars, "vars": var_values}
        
        # Forward pass
        output = model(data, unbatched_keys=["hypervars"])

        # Handle tuple output
        if isinstance(output, tuple):
            if output_idx is None:
                plt.close()
                raise ValueError(f"Model returned a tuple of {len(output)} outputs, but output_idx is None.")
            try:
                pred = output[output_idx]
            except IndexError:
                plt.close()
                raise IndexError(f"Error: output_idx {output_idx} is out of bounds for output tuple of length {len(output)}.")
        else:
            pred = output

        # Check dimensions
        if pred.ndim < 2 or pred.shape[1] != 1:
            plt.close()
            raise ValueError(f"Expected second dimension to be 1, got shape {pred.shape}")

        # Plot prediction
        plt.plot(var_values, pred, label=f"Prediction ({label_suffix})")

        # Superimpose exact function
        if exact_function is not None:
            # Compute exact values using the vmapped function
            exact_y = batched_exact_fn(hypervars, var_values)
            plt.plot(var_values, exact_y, label=f"Exact ({label_suffix})", linestyle="--")

        plt.xlabel("x")
        plt.ylabel("Output")
        plt.title("Model Predictions vs Exact Function")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)

        unique_save_path = directory / f"{base_name}_{i}{extension}"
        plt.savefig(unique_save_path, bbox_inches='tight')
        
        plt.close()


def plot_2d_predictions(
    model: nnx.Module,
    hypervars_set: Tuple[jnp.ndarray, ...],
    var_domains: Tuple[Tuple[float, float], Tuple[float, float]],
    output_idx: Optional[int] = None,
    exact_function: Optional[Callable] = None,
    n_points: int = 100
):
    """
    Evaluates a hypernetwork model across a 2D domain and plots the predictions as filled contours, 
    saving a distinct figure for each hypervariable set automatically inside a 'figures' folder.

    Args:
        model (nnx.Module): The final trained network.
        hypervars_set (Tuple[jnp.ndarray, ...]):  A tuple containing individual 1D arrays 
            of hypervariables. Each array represents a distinct configuration ($\theta$) to evaluate.
            The function create a plot for each configuration of hypervariables provided in this tuple.
        var_domains (Tuple[Tuple[float, float], Tuple[float, float]]): The (min, max) boundaries 
            for the two 1D input variables (var1, var2).
        output_idx (Optional[int], optional): The index to extract from the model output 
            if the model returns a tuple. Defaults to None.
        exact_function (Optional[Callable], optional): A reference ground-truth function. 
            Accepts an unbatched hypervariable array and a batched variable array of shape (N, 2).
        n_points (int, optional): The resolution of the grid per axis. Defaults to 100 
            (which means 100x100 = 10000  points).
    """
    # Construct the 2D grid
    x1 = jnp.linspace(var_domains[0][0], var_domains[0][1], n_points)
    x2 = jnp.linspace(var_domains[1][0], var_domains[1][1], n_points)
    
    # Create meshgrid and flatten for the network
    X1, X2 = jnp.meshgrid(x1, x2)
    var_values = jnp.stack([X1.ravel(), X2.ravel()], axis=-1)

    if exact_function is not None:
        batched_exact_fn = jax.vmap(exact_function, in_axes=(None, 0))

    # Setup directories
    directory = Path("figures")
    directory.mkdir(parents=True, exist_ok=True)
    base_name = "prediction_2d_set"
    extension = ".png"

    for i, hypervars in enumerate(hypervars_set):
        hv_formatted = ", ".join([f"{v:.3f}" for v in hypervars.tolist()])
        label_suffix = fr"$\theta$=[{hv_formatted}]"

        data = {"hypervars": hypervars, "vars": var_values}
        output = model(data, unbatched_keys=["hypervars"])

        if isinstance(output, tuple):
            if output_idx is None:
                raise ValueError(f"Model returned a tuple of {len(output)} outputs, but output_idx is None.")
            try:
                pred = output[output_idx]
            except IndexError:
                raise IndexError(f"Error: output_idx {output_idx} is out of bounds for output tuple of length {len(output)}.")
        else:
            pred = output

        if pred.ndim < 2 or pred.shape[1] != 1:
            raise ValueError(f"Expected second dimension to be 1, got shape {pred.shape}")

        # Reshape the flattened 1D prediction back into the 2D grid shape
        pred_grid = pred.reshape(n_points, n_points)

        # Plotting
        if exact_function is None:
            # Single plot if no exact function is provided
            fig, ax = plt.subplots(figsize=(8, 6))
            c = ax.contourf(X1, X2, pred_grid, levels=50, cmap='viridis')
            fig.colorbar(c, ax=ax, label="Model Output")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_title(f"Model Prediction\n{label_suffix}")
        
        else:
            # 1x3 Subplots for rigorous comparison
            exact_y = batched_exact_fn(hypervars, var_values)
            exact_grid = exact_y.reshape(n_points, n_points)
            error_grid = jnp.abs(pred_grid - exact_grid)

            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            
            # Subplot 1: Prediction
            c0 = axes[0].contourf(X1, X2, pred_grid, levels=50, cmap='viridis')
            fig.colorbar(c0, ax=axes[0])
            axes[0].set_title("Model Prediction")
            axes[0].set_xlabel("x")
            axes[0].set_ylabel("y")

            # Subplot 2: Exact
            c1 = axes[1].contourf(X1, X2, exact_grid, levels=50, cmap='viridis')
            fig.colorbar(c1, ax=axes[1])
            axes[1].set_title("Exact Function")
            axes[1].set_xlabel("x")
            axes[1].set_ylabel("y")

            # Subplot 3: Absolute Error (Using a different colormap to emphasize error magnitude)
            c2 = axes[2].contourf(X1, X2, error_grid, levels=50, cmap='Reds')
            fig.colorbar(c2, ax=axes[2])
            axes[2].set_title("Absolute Error")
            axes[2].set_xlabel("x")
            axes[2].set_ylabel("y")

            fig.suptitle(f"Evaluation for {label_suffix}", fontsize=14, y=1.05)

        # Ensure layout isn't squashed
        plt.tight_layout()
        
        # Save and close
        unique_save_path = directory / f"{base_name}_{i}{extension}"
        plt.savefig(unique_save_path, bbox_inches='tight')
        
        plt.close(fig)



def plot_2d_hdf5_comparison(
    model: nnx.Module,
    file_path: str,
    hypervars_set: Tuple[jnp.ndarray, ...],
    output_label: str,
    output_idx: Optional[int] = None,
    dataset_key: Optional[str] = None,
    time_tolerance: float = 1e-5
):
    """
    Evaluates a hypernetwork against ground-truth data from an HDF5 file.
    Generates a 1x3 grid (Prediction, Exact, Error) for each timestep.

    Args:
        model (nnx.Module): The final trained network.
        file_path (str): Path to the .hdf5 dataset file.
        hypervars_set (Tuple[jnp.ndarray, ...]): A tuple of 1D arrays, where each array 
            contains the time value (t) for evaluation.
        output_label (str): The target variable ("density", "pressure", "velocity_x", "velocity_y").
        output_idx (Optional[int], optional): Index to extract if the model returns a tuple. Defaults to None.
        dataset_key (Optional[str], optional): The HDF5 internal key. Defaults to the first available key.
        time_tolerance (float, optional): Tolerance for floating-point time matching. Defaults to 1e-5.
    """
    # Map column indices of .hdf5 file to variable names
    column_map = {
        "time": 0, "x": 1, "y": 2, 
        "density": 3, "pressure": 4, "velocity_x": 5, "velocity_y": 6
    }

    if output_label not in column_map or output_label in ["time", "x", "y"]:
        raise ValueError(f"Invalid output_label: '{output_label}'.")

    # Setup output directory
    directory = Path("figures")
    directory.mkdir(parents=True, exist_ok=True)
    base_name = f"comparison_2d_{output_label}"
    extension = ".png"

    # Load HDF5 Data into memory once
    with h5py.File(file_path, 'r') as f:
        if dataset_key is None:
            dataset_key = list(f.keys())[0]
        data = f[dataset_key][:]

    time_column = data[:, column_map["time"]]

    for i, hypervars in enumerate(hypervars_set):
        # Extract the scalar time value from the hypervariable array
        target_time = float(hypervars[0])

        # Select only data with the target time
        mask = np.isclose(time_column, target_time, atol=time_tolerance)
        timestep_data = data[mask]

        if len(timestep_data) == 0:
            print(f"Skipping set {i}: No HDF5 data found for t={target_time} within tolerance.")
            continue

        x_np = timestep_data[:, column_map["x"]]
        y_np = timestep_data[:, column_map["y"]]
        exact_z = timestep_data[:, column_map[output_label]]

        # Prepare model inputs: stack x and y into shape (N, 2)
        var_values = jnp.stack([jnp.array(x_np), jnp.array(y_np)], axis=-1)
        model_data = {"hypervars": hypervars, "vars": var_values}

        # Execute forward pass
        output = model(model_data, unbatched_keys=["hypervars"])

        # Check output dimensionality and extract prediction
        if isinstance(output, tuple):
            if output_idx is None:
                raise ValueError("Model returned a tuple, but output_idx is None.")
            try:
                pred_z = output[output_idx]
            except IndexError:
                raise IndexError(f"output_idx {output_idx} is out of bounds for tuple of length {len(output)}.")
            
            if pred_z.ndim > 1 and pred_z.shape[1] > 1:
                raise ValueError(f"Extracted array from tuple has second dimension > 1 (shape: {pred_z.shape}). Expected a 1D or single-column array.")
        else:
            if output.ndim > 1 and output.shape[1] > 1:
                if output_idx is None:
                    raise ValueError(f"Model returned an array with multiple columns (shape {output.shape}), but output_idx is None.")
                try:
                    # Extract specific column
                    pred_z = output[:, output_idx]
                except IndexError:
                    raise IndexError(f"output_idx {output_idx} is out of bounds for array with {output.shape[1]} columns.")
            else:
                pred_z = output

        # tricontourf expects a flat 1D array for z. Safely flatten before plotting.
        pred_z = pred_z.flatten()

        # Calculate absolute error
        error_z = jnp.abs(pred_z - exact_z)

        # Plotting
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Lock color scales between Prediction and Exact for accurate visual comparison
        vmin = min(np.nanmin(pred_z), np.nanmin(exact_z))
        vmax = max(np.nanmax(pred_z), np.nanmax(exact_z))

        # 1. Prediction
        c0 = axes[0].tricontourf(x_np, y_np, pred_z, levels=50, cmap='viridis', vmin=vmin, vmax=vmax)
        fig.colorbar(c0, ax=axes[0])
        axes[0].set_title("Model Prediction")
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("y")

        # 2. Exact
        c1 = axes[1].tricontourf(x_np, y_np, exact_z, levels=50, cmap='viridis', vmin=vmin, vmax=vmax)
        fig.colorbar(c1, ax=axes[1])
        axes[1].set_title(f"Exact ({output_label.replace('_', ' ').title()})")
        axes[1].set_xlabel("x")
        axes[1].set_ylabel("y")

        # 3. Absolute Error
        c2 = axes[2].tricontourf(x_np, y_np, error_z, levels=50, cmap='Reds')
        fig.colorbar(c2, ax=axes[2])
        axes[2].set_title("Absolute Error")
        axes[2].set_xlabel("x")
        axes[2].set_ylabel("y")

        fig.suptitle(f"Evaluation at t = {target_time:.3f}", fontsize=14, y=1.05)
        plt.tight_layout()

        # Save and close
        unique_save_path = directory / f"{base_name}_set_{i}{extension}"
        plt.savefig(unique_save_path, bbox_inches='tight')
        print(f"Plot saved to {unique_save_path.absolute()}")
        plt.close(fig)


def generate_prediction_plots(
    model: nnx.Module,
    hypervars_sets: jnp.ndarray,
    var_domains: Union[Tuple[float, float], List[Tuple[float, float]]],
    var_labels: Union[str, List[str]],
    output_labels: List[str],
    save_dir: str,
    n_points: int = 100
):
    """
    Generates and saves prediction plots for different sets of hypervariables.
    Dispatches to 1D or 2D plotting functions based on number of variables.
    """
    os.makedirs(save_dir, exist_ok=True)
    num_vars = 1 if isinstance(var_labels, str) else len(var_labels)

    for i, hypervars in enumerate(hypervars_sets):
        hypervar_str = "_".join([f"{v:.2f}" for v in hypervars])
        if num_vars == 1:
            save_path = os.path.join(save_dir, f"plot_1d_{hypervar_str}.png")
            plot_1d_predictions(model=model, hypervars_set=hypervars, var_domain=var_domains, var_label=var_labels, output_labels=output_labels, save_path=save_path, n_points=n_points)
        elif num_vars == 2:
            save_path = os.path.join(save_dir, f"plot_2d_{hypervar_str}.png")
            plot_2d_predictions(model=model, hypervars_set=hypervars, var_domains=var_domains, var_labels=var_labels, output_labels=output_labels, save_path=save_path, n_points_per_dim=n_points)
        else:
            print(f"Plotting for {num_vars} variables not supported. Skipping.")
            continue
