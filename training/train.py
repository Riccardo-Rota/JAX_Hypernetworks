import jax
import jax.numpy as jnp
from flax import nnx
from flax.nnx.training.metrics import Metric, Average, MultiMetric
import optax
from typing import Callable, Union, Optional, List, Tuple, Dict
from losses import L2Loss
from typing import Iterator
from tqdm import tqdm
import inspect
import copy
from datetime import datetime
import grain.python as grain
from optax.contrib._reduce_on_plateau import ReduceLROnPlateauState
from data.grain_dataset import build_dataset
from flax.training import early_stopping as flax_early_stopping
from utils import extract_lr_info


def perform_step(
    model: nnx.Module,
    data: jax.Array,
    labels: jax.Array,
    criterion: nnx.Module = L2Loss(),
    evaluation: bool = False,
    optimizer: Optional[nnx.Optimizer] = None,
    loss_eval_prev: Optional[float] = None
):
    """
    Perform a single training (or evaluation) step.
    Args:
        model (nnx.Module): The model to be trained or evaluated.
        data (jax.Array): The input data for the model.
        labels (jax.Array): The ground truth labels for the model.
        criterion (Callable, optional): The loss function to use. Default is optax.l2_loss.
        evaluation (bool, optional): If True, performs an evaluation step without updating the optimizer. Default is False.
        optimizer (nnx.Optimizer, optional): The optimizer to use for updating the hypernetwork parameters. Required if evaluation is False.
        loss_eval_prev (float, optional): Previous epoch evaluation loss, used for optimizers that require it. Default is None.
    Returns:
        loss (float): The computed loss for the step.
        pred (jax.Array): The predictions made by the model.
    """
    def forward_pass(model, data, labels):
        pred = model(data)
        loss = jnp.mean(criterion(pred, labels))
        return loss, pred

    if evaluation:
        loss, pred = forward_pass(model, data, labels)
    else:
        (loss, pred), grads = nnx.value_and_grad(forward_pass, has_aux=True)(model, data, labels)
        optimizer.update(grads, value=loss_eval_prev)

    return loss, pred

perform_step = nnx.jit(perform_step, static_argnames=("evaluation"))


def perform_epoch(
        model: nnx.Module,
        train_loader: Iterator,
        val_loader: Iterator,
        optimizer: Optional[nnx.Optimizer] = None,
        criterion: nnx.Module = L2Loss(),
        metrics: MultiMetric = MultiMetric(),
        loss_eval_prev: Optional[float] = None
        ) -> tuple:
    """
    Perform a single training epoch.
    Args:
        model (nnx.Module): The model to be trained or evaluated.
        train_loader (Iterator): Iterator over training batches (dicts of numpy arrays).
        val_loader (Iterator):   Iterator over validation batches (dicts of numpy arrays).
        optimizer (nnx.Optimizer, optional): The optimizer to use for updating the model parameters. Required for training.
        criterion (nnx.Module, optional): The loss function to use. Default is L2Loss().
        metrics (MultiMetric, optional): The metrics to use for evaluation. Default is MultiMetric().
        loss_eval_prev (float, optional): Previous epoch evaluation loss, used for optimizers that require it. Default is None.
    Returns:
        train_loss_epoch (float): The average training loss for the epoch.
        train_metrics (Dict[str, float]): The average training metrics for the epoch.
        val_loss_epoch (float): The average validation loss for the epoch.
        val_metrics (Dict[str, float]): The average validation metrics for the epoch.
    """
    train_loss_epoch = 0.0
    val_loss_epoch = 0.0
    n_samples_train = 0
    n_samples_val = 0
    train_metrics = metrics
    val_metrics = copy.deepcopy(train_metrics)

    #for batch in train_loader:
    for data, labels in train_loader:
        batch_size = labels.shape[0]
        train_loss, pred = perform_step(model = model,
                                       data = data,
                                       labels = labels,
                                       optimizer = optimizer,
                                       criterion = criterion,
                                       loss_eval_prev = loss_eval_prev)
        train_loss_epoch += train_loss * batch_size
        n_samples_train += batch_size
        train_metrics.update(predictions=pred, targets=labels)
    train_loss_epoch /= n_samples_train

    for data, labels in val_loader:
        batch_size = labels.shape[0]
        val_loss, pred = perform_step(model=model,
                                         data = data,
                                         labels = labels,
                                         criterion = criterion,
                                         evaluation = True)
        val_loss_epoch += val_loss * batch_size
        n_samples_val += batch_size
        val_metrics.update(predictions=pred, targets=labels)
    val_loss_epoch /= n_samples_val

    return train_loss_epoch, train_metrics.compute(), val_loss_epoch, val_metrics.compute()

def train_model(
        model: nnx.Module,
        train_source: grain.RandomAccessDataSource,
        val_source: grain.RandomAccessDataSource,
        optimizer: nnx.Optimizer,
        num_epochs: int,
        batch_size: int = 32,
        criterion: nnx.Module = L2Loss(),
        metrics: Optional[Union[MultiMetric, Dict[str, Metric]]] = None,
        early_stopping: Optional[flax_early_stopping.EarlyStopping] = None,
        early_stopping_metric: Optional[Union[str, int]] = None,
        plateau_scheduler_metric: Optional[Union[str, int]] = None,
        log_file_path: Optional[str] = None,
        ) -> tuple:
    """
    Train the model for a specified number of epochs, with optional early stopping and logging.
    Args:
        model (nnx.Module): The model to train.
        train_source (grain.RandomAccessDataSource): The training data source.
        val_source   (grain.RandomAccessDataSource): The validation data source.
        optimizer (nnx.Optimizer): The optimizer to use for updating the model parameters.
        num_epochs (int): The number of epochs to train the model.
        batch_size   (int): Batch size for both train and val pipelines.
        criterion (Callable, optional): The loss function to use. Default is optax.l2_loss.
        metrics (Dict[str, Callable], optional): A dictionary of metric functions to compute during the training. Default is None.
        early_stopping (flax.training.early_stopping.EarlyStopping, optional): An EarlyStopping object to monitor validation performance and stop training early if needed. Default is None.
        early_stopping_metric (str or int, optional): The metric to monitor for early stopping. Can be a metric name or index. Default is None (uses the validation loss).
        plateau_scheduler_metric (str or int, optional): The metric to monitor for learning rate plateau scheduling. Can be a metric name or index. Default is None (uses the validation loss).
        log_file_path (str, optional): Path to a log file where training progress will be logged. Default is None (no logging).
    Returns:
        A tuple containing:
            - history (Dict[str, List[float]]): A dictionary containing training and validation losses and metrics for each epoch.
            - early_stopping (Optional[flax.training.early_stopping.EarlyStopping]): The final state of the early stopping object.
            - best_epoch (Optional[int]): The epoch with the best validation metric, if early stopping was used.
    """

    if not isinstance(metrics, MultiMetric):
        if isinstance(metrics, dict):
            metrics = MultiMetric(**metrics)
        elif metrics is None:
            metrics = MultiMetric()
        else:
            raise ValueError("metrics must be either a MultiMetric instance, a dictionary of metrics, or None.")

    # Initialize history to store train and validation results for each epoch. Learning rate info added if found.
    history = {'train_results': [], 'val_results': []}

    early_stopping_metric_name = check_metric_name(early_stopping_metric, metrics._metric_names)
    if early_stopping_metric_name == -1:
        raise ValueError("Invalid early_stopping_metric. It must be either None, a valid metric name or a valid metric index.")
    plateau_scheduler_metric_name = check_metric_name(plateau_scheduler_metric, metrics._metric_names)
    if plateau_scheduler_metric_name == -1:
        raise ValueError("Invalid plateau_scheduler_metric. It must be either None, a valid metric name or a valid metric index.")

    if log_file_path:
        with open(log_file_path, "w") as f:
            f.write(f"Training Log - Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    loss_plateau_scheduler = jnp.inf

    best_epoch = None
    best_state = None

    pbar = tqdm(range(num_epochs))
    for epoch in pbar:
        metrics.reset()
        train_iter = build_dataset(train_source, is_training=True,  batch_size=batch_size, seed=epoch)
        val_iter   = build_dataset(val_source,   is_training=False, batch_size=batch_size)
        train_loss, train_metrics, val_loss, val_metrics = perform_epoch(model=model,
                                                                       train_loader=train_iter, 
                                                                       val_loader=val_iter, 
                                                                       optimizer=optimizer, 
                                                                       criterion=criterion, 
                                                                       metrics=metrics,
                                                                       loss_eval_prev=loss_plateau_scheduler)
        train_results_epoch = {'loss': train_loss, **train_metrics}
        val_results_epoch   = {'loss': val_loss, **val_metrics}

        history['train_results'].append(train_results_epoch)
        history['val_results'].append(val_results_epoch)

        # Extract learning rate information from the optimizer
        base_lr, lr_scale = extract_lr_info(optimizer.opt_state)

        # Nothing loged if inject_hyperparams was not used
        lr_log_str_compact = ""
        lr_log_str_detail = ""

        # Save learning rate info to history if found
        if base_lr is not None:
            # If reduce_on_plateau not used, scale defaults to 1.0
            current_scale = lr_scale if lr_scale is not None else 1.0
            effective_lr = base_lr * current_scale

            # Initialize history keys dynamically on the first pass
            if 'lr' not in history:
                history['lr'] = []
                history['lrs'] = []

            history['lr'].append(effective_lr)
            history['lrs'].append(current_scale)

            # Format strings for the terminal output
            lr_log_str_compact = f"LR: {effective_lr:.2e} (Scale: {current_scale:.4f})"
            lr_log_str_detail = f"LR: {effective_lr:.8f} - LR multiplier: {current_scale:.8f}"
    
        log_compact = (
            f"Epoch {epoch+1}/{num_epochs} - "
            f"Train: {compact_format(train_results_epoch)} - "
            f"Val: {compact_format(val_results_epoch)} - "
            f"{lr_log_str_compact}"
        )
        log_detail = (
            f"Epoch {epoch+1}/{num_epochs} - "
            f"Train: {', '.join(f'{k}: {v.item():.8f}' for k,v in train_results_epoch.items())} - "
            f"Val: {', '.join(f'{k}: {v.item():.8f}' for k,v in val_results_epoch.items())} - "
            f"{lr_log_str_detail}"
        )
        pbar.set_description(log_compact)

        if log_file_path:
            with open(log_file_path, "a") as f:
                f.write(log_detail + "\n")

        loss_plateau_scheduler = val_results_epoch[plateau_scheduler_metric_name]
        if early_stopping:
            metric_for_es = val_results_epoch[early_stopping_metric_name]
            new_early_stopping = early_stopping.update(metric_for_es)

            # If metric improved, save the best epoch and model state
            if new_early_stopping.has_improved:
                best_epoch = epoch
                # TODO: fix using orbax checkpoints (saving whole state in RAM can be bottleneck)
                #best_state = nnx.state(model)

            early_stopping = new_early_stopping

            if early_stopping.should_stop:
                if log_file_path:
                    with open(log_file_path, "a") as f:
                        f.write(f"Early stopping at epoch {epoch+1}. Best epoch was {best_epoch+1} with loss {early_stopping.best_metric:.8f}.\n")
                # TODO: fix this: nnx.update does not work with current model structure (Python lists inside nnx.Modules)
                # if best_state:
                    # nnx.update(model, best_state)
                break

    if log_file_path:
        with open(log_file_path, "a") as f:
            f.write(f"Training Log - End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    return history, early_stopping, best_epoch


def check_metric_name(metric: Union[str, int, None], name_metrics: Tuple[str]) -> int:
    """
    Auxiliary function to check if the given metric name or index is valid.
    Args:
        metric (str, int, or None): The metric to check. Can be a metric name, index, or None.
        name_metrics (Tuple[str]): A tuple of valid metric names.
    Returns:
        str: The valid metric name if found, or -1 if invalid.
    """
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
    
def compact_format(metrics_dict, k=2):
        items = list(metrics_dict.items())[:k] 
        return ', '.join(f'{key}: {val.item():.4f}' for key, val in items)
