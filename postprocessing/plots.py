import numpy as np
import jax
import jax.numpy as jnp
import flax.nnx as nnx
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List, Optional, Callable, Union, Sequence
from collections.abc import Sequence
import h5py
from pathlib import Path

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
        plt.plot(var_values, pred, label=f"Prediction")

        # Superimpose exact function
        if exact_function is not None:
            # Compute exact values using the vmapped function
            exact_y = batched_exact_fn(hypervars, var_values)
            plt.plot(var_values, exact_y, label=f"Exact", linestyle="--")

        plt.xlabel("x")
        plt.ylabel("Output")
        plt.title(f"Model Predictions vs Exact Function for ({label_suffix})")
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
    schema: Dict[str, int],
    target_keys: Sequence[str],
    num_plots: Optional[int] = None,
    dataset_key: Optional[str] = None,
    time_tolerance: float = 1e-5
):
    """
    Evaluates a hypernetwork against ground-truth data from an HDF5 file.
    Plots every output from the model. Matches the first outputs to the provided target_keys.

    Args:
        model (nnx.Module): The final trained network.
        file_path (str): Path to the .hdf5 dataset file.
        schema (Dict[str, int]): Mapping from variable names to HDF5 column indices.
        target_keys (Sequence[str]): Ordered list of target variables expected from the model 
            (e.g., ["density", "pressure"]).
        num_plots (Optional[int], optional): Number of distinct time points to plot.
        dataset_key (Optional[str], optional): The HDF5 internal key. Defaults to first available.
        time_tolerance (float, optional): Tolerance for floating-point time matching.
    """
    required_keys = ["time", "x", "y"]
    for req in required_keys:
        if req not in schema:
            raise KeyError(f"Required key '{req}' is missing from the provided schema.")

    for key in target_keys:
        if key not in schema:
            raise KeyError(f"Target key '{key}' is missing from the provided schema.")

    directory = Path("figures")
    directory.mkdir(parents=True, exist_ok=True)
    extension = ".png"

    with h5py.File(file_path, 'r') as f:
        if dataset_key is None:
            dataset_key = list(f.keys())[0]
        data = f[dataset_key][:]

    time_column = data[:, schema["time"]]

    unique_times = np.unique(time_column)
    unique_times.sort()

    if len(unique_times) == 0:
        raise ValueError("No time data found in the dataset.")

    if num_plots is not None and num_plots < len(unique_times):
        indices = np.linspace(0, len(unique_times) - 1, num_plots, dtype=int)
        selected_times = unique_times[indices]
    else:
        selected_times = unique_times

    for i, target_time in enumerate(selected_times):
        hypervars = jnp.array([target_time])

        mask = np.isclose(time_column, target_time, atol=time_tolerance)
        timestep_data = data[mask]

        if len(timestep_data) == 0:
            print(f"Skipping set {i}: No HDF5 data found for t={target_time} within tolerance.")
            continue

        x_np = timestep_data[:, schema["x"]]
        y_np = timestep_data[:, schema["y"]]

        var_values = jnp.stack([jnp.array(x_np), jnp.array(y_np)], axis=-1)
        model_data = {"hypervars": hypervars, "vars": var_values}

        output = model(model_data, unbatched_keys=["hypervars"])

        # Flatten the model output into a sequential list of 1D arrays
        predictions = []
        if isinstance(output, tuple):
            for out_tensor in output:
                if out_tensor.ndim == 1 or (out_tensor.ndim > 1 and out_tensor.shape[1] == 1):
                    predictions.append(out_tensor.flatten())
                else:
                    for col in range(out_tensor.shape[1]):
                        predictions.append(out_tensor[:, col].flatten())
        else:
            if output.ndim == 1 or (output.ndim > 1 and output.shape[1] == 1):
                predictions.append(output.flatten())
            else:
                for col in range(output.shape[1]):
                    predictions.append(output[:, col].flatten())

        # Iterate through every extracted prediction
        for out_idx, pred_z in enumerate(predictions):
            
            # Case A: Prediction has a corresponding exact target in the HDF5 file
            if out_idx < len(target_keys):
                target_name = target_keys[out_idx]
                exact_z = timestep_data[:, schema[target_name]]
                error_z = jnp.abs(pred_z - exact_z)

                fig, axes = plt.subplots(1, 3, figsize=(18, 5))
                vmin = min(np.nanmin(pred_z), np.nanmin(exact_z))
                vmax = max(np.nanmax(pred_z), np.nanmax(exact_z))

                c0 = axes[0].tricontourf(x_np, y_np, pred_z, levels=50, cmap='viridis', vmin=vmin, vmax=vmax)
                fig.colorbar(c0, ax=axes[0])
                axes[0].set_title(f"Prediction ({target_name})")
                axes[0].set_xlabel("x")
                axes[0].set_ylabel("y")

                c1 = axes[1].tricontourf(x_np, y_np, exact_z, levels=50, cmap='viridis', vmin=vmin, vmax=vmax)
                fig.colorbar(c1, ax=axes[1])
                axes[1].set_title(f"Exact ({target_name.replace('_', ' ').title()})")
                axes[1].set_xlabel("x")
                axes[1].set_ylabel("y")

                c2 = axes[2].tricontourf(x_np, y_np, error_z, levels=50, cmap='Reds')
                fig.colorbar(c2, ax=axes[2])
                axes[2].set_title("Absolute Error")
                axes[2].set_xlabel("x")
                axes[2].set_ylabel("y")

                fig.suptitle(f"Evaluation at t = {target_time:.3f}", fontsize=14, y=1.05)
                save_name = f"comparison_2d_t{i}_{target_name}{extension}"

            # Case B: Extra model output with no corresponding exact data
            else:
                target_name = f"extra_output_{out_idx}"
                
                fig, ax = plt.subplots(1, 1, figsize=(7, 5))
                c0 = ax.tricontourf(x_np, y_np, pred_z, levels=50, cmap='viridis')
                fig.colorbar(c0, ax=ax)
                ax.set_title(f"Prediction ({target_name})\nNo Exact Data Available")
                ax.set_xlabel("x")
                ax.set_ylabel("y")
                
                fig.suptitle(f"Evaluation at t = {target_time:.3f}", fontsize=12, y=1.05)
                save_name = f"comparison_2d_t{i}_{target_name}{extension}"

            plt.tight_layout()
            unique_save_path = directory / save_name
            plt.savefig(unique_save_path, bbox_inches='tight')
            plt.close(fig)
            

def generate_toy_plots(
    model: nnx.Module,
    hypervars_set: Sequence[Union[jnp.ndarray, Sequence[float]]],
    var_domains: Union[Sequence[float], Sequence[Sequence[float]]],
    output_idx: Optional[int] = None,
    exact_function: Optional[Callable] = None,
    n_points: Optional[int] = 400
):
    """
    General router for generating toy problem plots. 
    Redirects to 1D or 2D plotting functions based on the number of input variables.

    Args:
        model (nnx.Module): The final trained network.
        hypervars_set (Sequence[Union[jnp.ndarray, Sequence[float]]]): A sequence containing individual 1D arrays of hypervariables.
        var_domains (Union[Sequence[float], Sequence[Sequence[float]]]): 
            For 1D: A single sequence (min, max).
            For 2D: A sequence of two sequences ((min1, max1), (min2, max2)).
        output_idx (Optional[int], optional): The index to extract from the model output if it returns a tuple.
        exact_function (Optional[Callable], optional): A reference ground-truth function.
        n_points (Optional[int], optional): Resolution of the grid.
    """

    if len(hypervars_set) > 0 and isinstance(hypervars_set[0], jnp.ndarray):
        # Ensure it is a tuple for downstream consistency
        clean_hypervars = tuple(hypervars_set)
    else:
        # Cast Hydra ListConfigs or standard lists to JAX arrays
        clean_hypervars = tuple(jnp.array(hv) for hv in hypervars_set)
    
    # Determine dimensionality from var_domains structure
    if isinstance(var_domains[0], (int, float)):
        num_vars = 1
        domain_1d = tuple(var_domains)
    elif isinstance(var_domains[0], Sequence):
        # Nested sequence provided
        num_vars = len(var_domains)
        if num_vars == 1:
            domain_1d = tuple(var_domains[0])
    else:
        raise ValueError("Invalid var_domains format. Must be a flat tuple for 1D or a sequence of tuples for ND.")

    # Check limits
    if num_vars > 2:
        raise ValueError(f"Function only supports up to 2 variables. Received var_domains for {num_vars} variables.")

    # Route to specific plotting functions
    if num_vars == 1:
        plot_1d_predictions(
            model=model,
            hypervars_set=clean_hypervars,
            var_domain=domain_1d,
            output_idx=output_idx,
            exact_function=exact_function,
            n_points=n_points
        )
    elif num_vars == 2:
        clean_var_domains = tuple(tuple(d) for d in var_domains)
        plot_2d_predictions(
            model=model,
            hypervars_set=clean_hypervars,
            var_domains=clean_var_domains,
            output_idx=output_idx,
            exact_function=exact_function,
            n_points=n_points
        )
