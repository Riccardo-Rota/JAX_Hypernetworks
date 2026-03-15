import matplotlib.pyplot as plt
from typing import List, Optional, Union
import jax.numpy as jnp
from flax import nnx
import os
from utils import variables_generator
from training import assign_parameters
import jax.random as random

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
    plt.savefig(save_path)
    plt.close()

def plot_1D_predictions(variables: jnp.ndarray, hypervariables_set: jnp.ndarray, hypernetwork: nnx.Module, targetnetwork: nnx.Module, save_path: str, var_label: Optional[str], hypervar_labels: Optional[List[str]], output_label: Optional[str]):
    """
    Plots the predicted vs true function for 1D input data.
    Args:
        variables (jnp.ndarray): 1D array of input variable values (shape: [N]).
        hypervariables_set (jnp.ndarray): fixed hypervariable set (shape: [num_hypervars]).
        hypernetwork (nnx.Module): The hypernetwork that generates the parameters for the target network.
        targetnetwork (nnx.Module): The target network that will be modified by the hypernetwork.
        save_path (str): Path to save the generated plot.
        var_label (Optional[str]): Label for the input variable.
        hypervar_labels (Optional[List[str]]): List of labels for the hypervariables.
        output_label (Optional[str]): Label for the output variable.
    """
    # TODO: implement this function to plot predictions for 1D input data when the data loading is well defined
    pass

def plot_2D_predictions(variables: jnp.ndarray, hypervariables_set: jnp.ndarray, hypernetwork: nnx.Module, targetnetwork: nnx.Module, save_path: str, var_labels: Optional[List[str]], hypervar_labels: Optional[List[str]], output_label: Optional[str]):
    """
    Plots the predicted vs true function for 2D input data.
    Args:
        variables (jnp.ndarray): 2D array of input variable values (shape: [N, M]).
        hypervariables_set (jnp.ndarray): fixed hypervariable set (shape: [num_hypervars]).
        hypernetwork (nnx.Module): The hypernetwork that generates the parameters for the target network.
        targetnetwork (nnx.Module): The target network that will be modified by the hypernetwork.
        save_path (str): Path to save the generated plot.
        var_labels (Optional[List[str]]): List of labels for the input variables.
        hypervar_labels (Optional[List[str]]): List of labels for the hypervariables.
        output_label (Optional[str]): Label for the output variable.
    """
    # TODO: implement this function to plot predictions for 2D input data when the data loading is well defined
    pass

def plot_predictions(variables: jnp.ndarray, hypervariables: jnp.ndarray, hypernetwork: nnx.Module, targetnetwork: nnx.Module, save_path: str, var_labels: Optional[Union[str, List[str]]]=None, hypervar_labels: Optional[List[str]]=None, output_label: Optional[str]=None):
    """
    Plots the predicted vs true function for input data for different values of the hypervariables set.
    Args:
        variables (jnp.ndarray): Array of input variable values (shape: [N] or [N, M]).
        hypervariables (jnp.ndarray): multiple hypervariable sets (shape: [num_sets, num_hypervars]).
        hypernetwork (nnx.Module): The hypernetwork that generates the parameters for the target network.
        targetnetwork (nnx.Module): The target network that will be modified by the hypernetwork.
        save_path (str): Path to folder where to save the generated plots.
        var_labels (Optional[Union[str, List[str]]): List of labels for the input variable(s).
        hypervar_labels (Optional[List[str]]): List of labels for the hypervariables.
        output_label (Optional[str]): Label for the output variable.
    """
    os.makedirs(save_path, exist_ok=True)
    for hypervars in hypervariables:
        file_name = "hypervars" + "_".join([f"{value:.2f}" for value in hypervars]) + ".png"
        file_path = os.path.join(save_path, file_name)
        
        if variables.ndim == 1:
            plot_1D_predictions(variables, hypervariables, hypernetwork, targetnetwork, file_path, var_labels, hypervar_labels, output_label)
        else:
            plot_2D_predictions(variables, hypervariables, hypernetwork, targetnetwork, file_path, var_labels, hypervar_labels, output_label)


def plot_predictions_legacy(N_examples, f_to_learn, hypernetwork, targetnetwork, mu_domain, l_domain, k_domain, save_path):
    x_vector = jnp.linspace(-1, 1, 101)[:, None]
    mu_example, l_example, k_example = variables_generator(
        N=N_examples,
        n_realizations=1,
        var_names=['mu', 'l', 'k'],
        var_domains=[mu_domain, l_domain, k_domain],
        key=random.key(1)
    ).values()
    example_hypervars = jnp.stack([mu_example, l_example, k_example], axis=1)
    f_to_learn = eval(f_to_learn, {"__builtins__": None, "jnp": jnp})

    for i, hypervars in enumerate(example_hypervars):
        w = hypernetwork(hypervars)
        modified_targetnetwork = assign_parameters(targetnetwork, w)
        mu, l, k = hypervars
        y_pred = modified_targetnetwork(x_vector)
        v_f_to_learn = nnx.vmap(lambda x: f_to_learn(mu, l, k, x))
        y_vector = v_f_to_learn(x_vector)

        plt.figure()
        plt.plot(x_vector, y_pred, '--b')
        plt.plot(x_vector, y_vector, '-r')
        plt.legend(['Predicted', 'True'], loc='upper left')
        plt.title(f'l = {l:.2f}, k = {k:.2f}, mu = {mu:.2f}')
        plot_path = os.path.join(save_path, f'prediction_{i}.png')
        plt.savefig(plot_path)
        plt.close()
