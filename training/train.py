import jax
import jax.numpy as jnp
from flax import nnx
from flax.nnx.training.metrics import Average, MultiMetric
import optax
from typing import Callable, Union, Optional, List, Tuple
from .hypernet_utils import build_state_from_parameters
from data import DataLoader
from tqdm import tqdm

def train_step( #TODO: GESTIRE IL CASO IN CUI LA TUPLA METRICS CONTENGA UN SOLO ELEMENTO + COMMENTARE
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
        loss = jnp.mean(criterion(pred, y))

        metrics_vals = [jnp.mean(m(pred, y)) for m in metrics] if metrics else []
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
        metrics: Union[List[Callable], None] = None,
        ) -> tuple:
    """
    Trains and evaluates the model for one epoch.
    Args:
        hypernetwork (nnx.Module): The hypernetwork that generates the parameters for the target network.
        targetnetwork (nnx.Module): The target network that will be modified by the hypernetwork.
        hypervariables (jax.Array): Variables that the hypernetwork uses to generate parameters.
        train_ds (tuple): A tuple containing training data (x_train, y_train).
        val_ds (tuple): A tuple containing validation data (x_val, y_val).
        optimizer (nnx.Optimizer): Optimizer used to update the hypernetwork parameters.
        criterion (Callable): Function used to compute the loss. It must take as inputs the predictions and targets. Default: optax.l2_loss.
        metrics (Union[Callable, dict, None]): Metrics to compute during training and evaluation. Default: None.
    Returns:
        train_loss (jax.Array): The computed training loss for the epoch.
        val_loss (jax.Array): The computed validation loss for the epoch.
    """
    metrics_dict = {"loss": 'loss'}
    if metrics:
        for metric in metrics:
            metrics_dict[metric.__name__] = metric.__name__

    training_metrics = MultiMetric(**{k: Average(argname=k) for k in metrics_dict})
    validation_metrics = MultiMetric(**{k: Average(argname=k) for k in metrics_dict}) #TODO: controllare se funziona

    for data, label in train_loader:
        hypervariables = data[:, :-1] # mu, l, k TODO: SOSTITUIRE CON UN DIZIONARIO UNA VOLTA IMPLEMENTATO UN DATALOADER
        x = data[:, -1:] # x
        train_loss, train_metrics = train_step(hypernetwork, targetnetwork, hypervariables, x, label, optimizer, criterion, metrics)
        training_results = {'loss': train_loss}
        if metrics:
            for m in range(len(train_metrics)):
                training_results[metrics[m].__name__] = train_metrics[m]
        training_metrics.update(**training_results)


    for data, label in val_loader:
        hypervariables = data[:, :-1] # mu, l, k TODO: SOSTITUIRE CON UN DIZIONARIO UNA VOLTA IMPLEMENTATO UN DATALOADER
        x = data[:, -1:] # x
        val_loss, val_metrics = train_step(hypernetwork, targetnetwork, hypervariables, x, label, optimizer, criterion, metrics, evaluation=True)
        validation_results = {'loss': val_loss}
        if metrics:
            for m in range(len(val_metrics)):
                validation_results[metrics[m].__name__] = val_metrics[m]
        validation_metrics.update(**validation_results)
    return training_metrics, validation_metrics
# TODO: provare a jittare anche questa funzione, vedere se conviene

def train_and_evaluate(
        hypernetwork: nnx.Module,
        targetnetwork: nnx.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: nnx.Optimizer,
        num_epochs: int,
        criterion: Callable = optax.l2_loss,
        metrics: Union[List[Callable], None] = None,
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
    Returns:
        history (dict): A dictionary containing training and validation losses for each epoch.
    """
    history = {'train_metrics': [], 'val_metrics': []}

    pbar = tqdm(range(num_epochs))
    for epoch in pbar:
        train_metrics, val_metrics = train_epoch(hypernetwork, targetnetwork, train_loader, val_loader, optimizer, criterion, metrics)
        history['train_metrics'].append(train_metrics.compute())
        history['val_metrics'].append(val_metrics.compute())
        pbar.set_description(
            f"Epoch {epoch+1}/{num_epochs} - "
            f"Train: {', '.join(f'{k}: {v.item():.4f}' for k,v in train_metrics.compute().items())} - "
            f"Val: {', '.join(f'{k}: {v.item():.4f}' for k,v in val_metrics.compute().items())}"
        )
        # TODO: stampare log su file contenente i valori di tutte le metriche del train e del val
        # with open("training_log.txt", "a") as f:
        #     f.write(f"Epoch {epoch+1}/{num_epochs} - Train Metrics: {train_metrics} - Val Metrics: {val_metrics}\n")
        # TODO: aggiungere scheduler e early stopping
        # TODO: salvare il modello con i pesi migliori sul val
        # TODO: controllare se funziona il tutto (probabilmente no)
        # TODO: decidere come passare le metriche: devono essere passate come tupla ala funzione train_step perchè è jittata,
        #       ma la tupla può essere creata qui dentro, in modo da passare da fuori una lista o un dizionario

    return history