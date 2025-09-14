import jax
import jax.numpy as jnp
from flax import nnx
from flax.nnx.training.metrics import Average, MultiMetric
import optax
from typing import Callable, Union, Optional, List, Tuple
from .hypernet_utils import build_state_from_parameters
from data import DataLoader
from tqdm import tqdm
import inspect
from utils import to_tuple
from .early_stopping import EarlyStopping
from datetime import datetime

#TODO: controllare tutti i tipi e i docstring

def train_step(
    hypernetwork: nnx.Module,
    targetnetwork: nnx.Module, 
    hypervariables: jax.Array, 
    x: jax.Array, 
    y: jax.Array, 
    optimizer: nnx.Optimizer, 
    criterion: Callable = optax.l2_loss,
    metrics: Tuple[Callable, ...] = (),
    evaluation: bool = False,
):
    """
    TODO: commentare
    """
    def compute_loss_and_metrics(hypernetwork, hypervariables, x, y):
        w = hypernetwork(hypervariables)
        graphdef, template_state = nnx.split(targetnetwork)
        state = nnx.vmap(build_state_from_parameters, in_axes=(None, 0), out_axes=0)(template_state, w)
        modified_targetnetwork = nnx.merge(graphdef, state)

        pred = nnx.vmap(type(modified_targetnetwork).__call__)(modified_targetnetwork, x)
        loss = jnp.mean(compute_metrics(criterion, pred, y, w))

        metrics_vals = [jnp.mean(compute_metrics(m, pred, y, w)) for m in metrics] if metrics else []
        return loss, metrics_vals
    
    if evaluation:
        loss, metrics_vals = compute_loss_and_metrics(hypernetwork, hypervariables, x, y)
    else:
        (loss, metrics_vals), grads = nnx.value_and_grad(compute_loss_and_metrics, has_aux=True)(hypernetwork, hypervariables, x, y)
        optimizer.update(grads)

    return loss, metrics_vals

train_step = nnx.jit(train_step, static_argnames=("criterion", "metrics", "evaluation")) #QUESTION: Each time we change evaluation to false, does the compliler overwrite the function or not?

#TODO: cambiare in base alla nuova implementazione del dataloader
def train_epoch(
        hypernetwork: nnx.Module,
        targetnetwork: nnx.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: nnx.Optimizer,
        criterion: Callable = optax.l2_loss,
        metrics: Tuple[Callable, ...] = (),
        metrics_names: Optional[List[str]] = None,
        ) -> tuple:
    """
    Trains and evaluates the model for one epoch.
    Args:
        hypernetwork (nnx.Module): The hypernetwork that generates the parameters for the target network.
        targetnetwork (nnx.Module): The target network that will be modified by the hypernetwork.
        hypervariables (jax.Array): Variables that the hypernetwork uses to generate parameters.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        optimizer (nnx.Optimizer): Optimizer used to update the hypernetwork parameters.
        criterion (Callable): Function used to compute the loss. It must take as inputs the predictions and targets. Default: optax.l2_loss.
        metrics (Union[Callable, dict, None]): Metrics to compute during training and evaluation. Default: None.
    Returns:
        train_loss (jax.Array): The computed training loss for the epoch.
        val_loss (jax.Array): The computed validation loss for the epoch.
    """
    if metrics_names:
        assert len(metrics_names) == len(metrics) or len(metrics_names) == len(metrics) + 1, "Length of metrics_names must be equal to length of metrics or length of metrics + 1 (for loss)."
        if len(metrics_names) == len(metrics):
            loss_name = 'loss'
        else:
            loss_name = metrics_names[0]
            metrics_names = metrics_names[1:]
    else:
        loss_name = 'loss'
        metrics_names = [m.__name__ for m in metrics]

    training_metrics = MultiMetric(**{k: Average(argname=k) for k in [loss_name] + metrics_names})
    validation_metrics = MultiMetric(**{k: Average(argname=k) for k in [loss_name] + metrics_names})

    for data, label in train_loader:
        hypervariables = data[:, :-1] # mu, l, k TODO: SOSTITUIRE CON UN DIZIONARIO UNA VOLTA IMPLEMENTATO UN DATALOADER
        x = data[:, -1:] # x
        train_loss, train_metrics = train_step(hypernetwork, targetnetwork, hypervariables, x, label, optimizer, criterion, metrics)
        training_results = {loss_name: train_loss}
        for m in range(len(train_metrics)):
            training_results[metrics_names[m]] = train_metrics[m]
        training_metrics.update(**training_results)


    for data, label in val_loader:
        hypervariables = data[:, :-1] # mu, l, k TODO: SOSTITUIRE CON UN DIZIONARIO UNA VOLTA IMPLEMENTATO UN DATALOADER
        x = data[:, -1:] # x
        val_loss, val_metrics = train_step(hypernetwork, targetnetwork, hypervariables, x, label, optimizer, criterion, metrics, evaluation=True)
        validation_results = {loss_name: val_loss}
        for m in range(len(val_metrics)):
            validation_results[metrics_names[m]] = val_metrics[m]
        validation_metrics.update(**validation_results)
    return training_metrics, validation_metrics

train_epoch = nnx.jit(train_epoch, static_argnames=("criterion", "metrics", "train_loader", "val_loader"))

def train_and_evaluate(
        hypernetwork: nnx.Module,
        targetnetwork: nnx.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: nnx.Optimizer,
        num_epochs: int,
        criterion: Callable = optax.l2_loss,
        metrics: Union[List[Callable], Tuple[Callable, ...], None] = None,
        early_stopping: Optional[EarlyStopping] = None,
        early_stopping_metric: Optional[Union[str, int]] = None,
        log_file_path: Optional[str] = None,
        ) -> tuple:
    """
    Trains and evaluates the model for a specified number of epochs.
    Args:
        hypernetwork (nnx.Module): The hypernetwork that generates the parameters for the target network.
        targetnetwork (nnx.Module): The target network that will be modified by the hypernetwork.
        hypervariables (jax.Array): Variables that the hypernetwork uses to generate parameters.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        optimizer (nnx.Optimizer): Optimizer used to update the hypernetwork parameters.
        num_epochs (int): Number of epochs to train and evaluate.
        criterion (Callable): Function used to compute the loss. It must take as inputs the predictions and targets. Default: optax.l2_loss.
        metrics (Union[List[Callable], Tuple[Callable, ...], None]): Metrics to compute during training and evaluation. Default: None.
        early_stopping (Optional[EarlyStopping]): Early stopping callback to stop training when a metric stops improving. Default: None.
        early_stopping_metric (Optional[Union[str, int]]): Metric to monitor for early stopping. Can be a string (metric name) or an integer (metric index). Default: None (first metric).
        log_to_file (bool): If True, logs training and validation metrics to a file. Default: False.
    Returns:
        history (dict): A dictionary containing training and validation losses for each epoch.
    """

    history = {'train_metrics': [], 'val_metrics': []}
    metrics = to_tuple(metrics)
    if log_file_path:
        with open(log_file_path, "w") as f:
            f.write(f"Training Log - Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    pbar = tqdm(range(num_epochs))
    for epoch in pbar:
        train_metrics, val_metrics = train_epoch(hypernetwork, targetnetwork, train_loader, val_loader, optimizer, criterion, metrics)
        history['train_metrics'].append(train_metrics.compute())
        history['val_metrics'].append(val_metrics.compute())
        log_compact = f"Epoch {epoch+1}/{num_epochs} - " + \
              f"Train: {', '.join(f'{k}: {v.item():.4f}' for k,v in train_metrics.compute().items())} - " + \
              f"Val: {', '.join(f'{k}: {v.item():.4f}' for k,v in val_metrics.compute().items())}"
        log_detail = f"Epoch {epoch+1}/{num_epochs} - " + \
              f"Train: {', '.join(f'{k}: {v.item():.8f}' for k,v in train_metrics.compute().items())} - " + \
              f"Val: {', '.join(f'{k}: {v.item():.8f}' for k,v in val_metrics.compute().items())}"
        pbar.set_description(log_compact)

        if log_file_path:
            with open(log_file_path, "a") as f:
                f.write(log_detail + "\n")

        if early_stopping:
            if isinstance(early_stopping_metric, str):
                if early_stopping_metric not in val_metrics.compute():
                    raise ValueError(f"Metric for early stopping '{early_stopping_metric}' not found in validation metrics.")
                metric_index = list(val_metrics.compute().keys()).index(early_stopping_metric)
            elif isinstance(early_stopping_metric, int):
                if early_stopping_metric < 0 or early_stopping_metric >= len(val_metrics.compute()):
                    raise ValueError(f"Metric index for early stopping '{early_stopping_metric}' is out of range.")
                metric_index = early_stopping_metric
            else:
                metric_index = 0

            early_stopping(current_loss = list(val_metrics.compute().values())[metric_index], 
                           current_model = hypernetwork, 
                           current_epoch = epoch)
            if early_stopping.should_stop:
                print(f"Early stopping at epoch {epoch+1}. Best epoch was {early_stopping.best_epoch+1} with loss {early_stopping.best_loss:.8f}.")
                if log_file_path:
                    with open(log_file_path, "a") as f:
                        f.write(f"Early stopping at epoch {epoch+1}. Best epoch was {early_stopping.best_epoch+1} with loss {early_stopping.best_loss:.8f}.\n")
                if early_stopping.best_model:
                    hypernetwork = early_stopping.best_model
                break

    if log_file_path:
        with open(log_file_path, "a") as f:
            f.write(f"Training Log - End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # TODO: sistemare il plateau scheduler
    return history


def compute_metrics(metric: Callable, preds: jax.Array, targets: jax.Array, weights: jax.Array) -> jax.Array:
    """
    Computes the specified metric between predictions and targets.
    Args:
        metric (Callable): The metric function to compute. It must take as inputs the predictions and targets.
        preds (jax.Array): The predicted values.
        targets (jax.Array): The ground truth values.
        weights (jax.Array): The weights generated by the hypernetwork.
    Returns:
        jax.Array: The computed metric value.
    """
    if 'weights' in inspect.getfullargspec(metric).args:
        return metric(preds, targets, weights)
    else:
        return metric(preds, targets)
    
compute_metrics = nnx.jit(compute_metrics, static_argnames=("metric",))