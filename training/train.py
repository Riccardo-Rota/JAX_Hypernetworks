import jax
import jax.numpy as jnp
from flax import nnx
from flax.nnx.training.metrics import Average, MultiMetric
import optax
from typing import Callable, Union, Optional, List, Tuple, Dict
from .hypernet_utils import build_state_from_parameters
from data import JaxDataLoader
from tqdm import tqdm
import inspect
from utils import to_tuple
from .early_stopping import EarlyStopping
from datetime import datetime

#TODO: controllare tutti i tipi e i docstring

def perform_step(
    hypernetwork: nnx.Module,
    targetnetwork: nnx.Module, 
    hypervariables: jax.Array, 
    x: jax.Array, 
    y: jax.Array, 
    criterion: Callable = optax.l2_loss,
    evaluation: bool = False,
    optimizer: Optional[nnx.Optimizer] = None,
    loss_eval_prev: Optional[float] = None
):
    """
    TODO: commentare
    """
    def forward_pass(hypernetwork, hypervariables, x, y):
        w = hypernetwork(hypervariables)
        graphdef, template_state = nnx.split(targetnetwork)
        state = nnx.vmap(build_state_from_parameters, in_axes=(None, 0), out_axes=0)(template_state, w)
        modified_targetnetwork = nnx.merge(graphdef, state)

        pred = nnx.vmap(type(modified_targetnetwork).__call__)(modified_targetnetwork, x)
        loss = jnp.mean(compute(criterion, pred, y, w))

        return loss, (pred, w)

    if evaluation:
        loss, (pred, w) = forward_pass(hypernetwork, hypervariables, x, y)
    else:
        (loss, (pred, w)), grads = nnx.value_and_grad(forward_pass, has_aux=True)(hypernetwork, hypervariables, x, y)
        optimizer.update(grads, value=loss_eval_prev)


    # def optimize(hypernetwork, hypervariables, x, y):
    #     (loss, (pred, w)), grads = nnx.value_and_grad(forward_pass, has_aux=True)(hypernetwork, hypervariables, x, y)
    #     optimizer.update(grads, value=loss_eval_prev)
    #     return loss, pred, w
    
    # loss, pred, w = nnx.cond(optimizer is None,
    #                              forward_pass,
    #                              optimize,
    #                              hypernetwork, hypervariables, x, y)

    return loss, pred, w

perform_step = nnx.jit(perform_step, static_argnames=("criterion", "evaluation"))


def perform_epoch(
        hypernetwork: nnx.Module,
        targetnetwork: nnx.Module,
        train_loader: JaxDataLoader,
        val_loader: JaxDataLoader,
        optimizer: Optional[nnx.Optimizer] = None,
        criterion: Callable = optax.l2_loss,
        metrics: Tuple[Callable, ...] = (),
        loss_eval_prev: Optional[float] = None
        ) -> tuple:
    """
    TODO: commentare
    """
    train_loss_epoch = 0.0
    val_loss_epoch = 0.0
    n_samples_train = 0
    train_metrics_epoch = [0.0 for _ in range(len(metrics))]
    val_metrics_epoch = [0.0 for _ in range(len(metrics))]
    n_samples_val = 0

    for data in train_loader:
        hypervariables = data['hypervars'] # mu, l, k
        x = data['vars'] # x
        y = data['labels'] # y
        train_loss, pred, w = perform_step(hypernetwork = hypernetwork, 
                                        targetnetwork = targetnetwork,
                                        hypervariables = hypervariables, 
                                        x = x, 
                                        y = y, 
                                        optimizer = optimizer, 
                                        criterion = criterion, 
                                        loss_eval_prev = loss_eval_prev)
        train_loss_epoch += train_loss * x.shape[0]
        n_samples_train += x.shape[0]
        for m in range(len(metrics)):
            train_metrics_epoch[m] += jnp.mean(compute(metrics[m], pred, y, w)) * x.shape[0]
    train_loss_epoch /= n_samples_train
    train_metrics_epoch = [m / n_samples_train for m in train_metrics_epoch]

    for data in val_loader:
        hypervariables = data['hypervars'] # mu, l, k
        x = data['vars'] # x
        y = data['labels'] # y
        val_loss, pred, w = perform_step(hypernetwork = hypernetwork, 
                                        targetnetwork = targetnetwork,
                                        hypervariables = hypervariables, 
                                        x = x, 
                                        y = y, 
                                        criterion = criterion,
                                        evaluation = True)
        val_loss_epoch += val_loss * x.shape[0]
        n_samples_val += x.shape[0]
        for m in range(len(metrics)):
            val_metrics_epoch[m] += jnp.mean(compute(metrics[m], pred, y, w)) * x.shape[0]
    val_loss_epoch /= n_samples_val
    val_metrics_epoch = [m / n_samples_val for m in val_metrics_epoch]

    return train_loss_epoch, train_metrics_epoch, val_loss_epoch, val_metrics_epoch

perform_epoch = nnx.jit(perform_epoch, static_argnames=("criterion", "metrics", "train_loader", "val_loader"))

def train_model(
        hypernetwork: nnx.Module,
        targetnetwork: nnx.Module,
        train_loader: JaxDataLoader,
        val_loader: JaxDataLoader,
        optimizer: nnx.Optimizer,
        num_epochs: int,
        criterion: Callable = optax.l2_loss,
        metrics: Optional[Dict[str, Callable]] = None,
        early_stopping: Optional[EarlyStopping] = None,
        early_stopping_metric: Optional[Union[str, int]] = None,
        plateau_scheduler_metric: Optional[Union[str, int]] = None,
        log_file_path: Optional[str] = None,
        ) -> tuple:
    """
    TODO: commentare
    """

    history = {'train_results': [], 'val_results': []}
    if metrics is None:
        fn_metrics = ()
        name_metrics = ()
    else:
        fn_metrics = tuple(metrics.values())
        name_metrics = tuple(metrics.keys())

    early_stopping_metric_name = check_metric_name(early_stopping_metric, name_metrics)
    if early_stopping_metric_name == -1:
        raise ValueError("Invalid early_stopping_metric. It must be either None, a valid metric name or a valid metric index.")
    plateau_scheduler_metric_name = check_metric_name(plateau_scheduler_metric, name_metrics)
    if plateau_scheduler_metric_name == -1:
        raise ValueError("Invalid plateau_scheduler_metric. It must be either None, a valid metric name or a valid metric index.")

    if log_file_path:
        with open(log_file_path, "w") as f:
            f.write(f"Training Log - Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    loss_plateau_scheduler = jnp.inf

    pbar = tqdm(range(num_epochs))
    for epoch in pbar:
        train_loss, train_metrics, val_loss, val_metrics = perform_epoch(hypernetwork=hypernetwork,
                                                                       targetnetwork=targetnetwork,
                                                                       train_loader=train_loader, 
                                                                       val_loader=val_loader, 
                                                                       optimizer=optimizer, 
                                                                       criterion=criterion, 
                                                                       metrics=fn_metrics,
                                                                       loss_eval_prev=loss_plateau_scheduler)
        train_results_epoch = {'loss': train_loss, **dict(zip(name_metrics, train_metrics))}
        val_results_epoch   = {'loss': val_loss, **dict(zip(name_metrics, val_metrics))}
        history['train_results'].append(train_results_epoch)
        history['val_results'].append(val_results_epoch)
        log_compact = f"Epoch {epoch+1}/{num_epochs} - " + \
              f"Train: {', '.join(f'{k}: {v.item():.4f}' for k,v in train_results_epoch.items())} - " + \
              f"Val: {', '.join(f'{k}: {v.item():.4f}' for k,v in val_results_epoch.items())}"
        log_detail = f"Epoch {epoch+1}/{num_epochs} - " + \
              f"Train: {', '.join(f'{k}: {v.item():.8f}' for k,v in train_results_epoch.items())} - " + \
              f"Val: {', '.join(f'{k}: {v.item():.8f}' for k,v in val_results_epoch.items())}"
        pbar.set_description(log_compact)

        if log_file_path:
            with open(log_file_path, "a") as f:
                f.write(log_detail + "\n")

        loss_plateau_scheduler = val_results_epoch[plateau_scheduler_metric_name]
        if early_stopping:
            loss_early_stopping = val_results_epoch[early_stopping_metric_name]
            early_stopping(current_loss=loss_early_stopping,
                           current_model=hypernetwork,
                           current_epoch=epoch)
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


def compute(function: Callable, preds: jax.Array, targets: jax.Array, weights: jax.Array) -> jax.Array:
    """
    Auxiliary function to compute a loss, handling both cases where the loss requires weights and where it does not.
    Args:
        function (Callable): The loss function to compute. It must take as inputs the predictions and targets.
        preds (jax.Array): The predicted values.
        targets (jax.Array): The ground truth values.
        weights (jax.Array): The weights generated by the hypernetwork.
    Returns:
        jax.Array: The computed loss value.
    """
    if 'weights' in inspect.getfullargspec(function).args:
        return function(preds, targets, weights)
    else:
        return function(preds, targets)

compute = nnx.jit(compute, static_argnames=("function",))

def check_metric_name(metric: Union[str, int, None], name_metrics: Tuple[str]) -> int:
    if metric is None:
        return 'loss'
    if isinstance(metric, str):
        if metric not in name_metrics:
            return -1
        return metric
    elif isinstance(metric, int):
        if metric < 0 or metric >= len(name_metrics):
            return -1
        return name_metrics[metric]
    else:
        return -1