import jax.numpy as jnp
from flax import nnx
from flax.nnx.training.metrics import Average, MultiMetric, Metric
from typing import Callable, Optional, List, Tuple, Union, Dict
from data import JaxDataLoader
from tqdm import tqdm
from training import build_state_from_parameters


def test_model(
        hypernetwork: nnx.Module,
        targetnetwork: nnx.Module,
        loader: JaxDataLoader,
        metrics: Optional[Union[MultiMetric, Dict[str, Metric]]] = None,
        ) -> tuple:
    """
    Trains and evaluates the model for one epoch.
    Args:
        hypernetwork (nnx.Module): The hypernetwork that generates the parameters for the target network.
        targetnetwork (nnx.Module): The target network that will be modified by the hypernetwork.
        loader (DataLoader): DataLoader for testing data.
        metrics (Optional[Union[MultiMetric, Dict[str, Metric]]]): Metrics to compute during testing. Default: None.
    Returns:
        MultiMetric: Computed metrics.
    """
    if not isinstance(metrics, MultiMetric):
        if isinstance(metrics, dict):
            metrics = MultiMetric(**metrics)
        elif metrics is None:
            metrics = MultiMetric()
        else:
            raise ValueError("metrics must be either a MultiMetric instance, a dictionary of metrics, or None.")

    replace = getattr(targetnetwork, 'replace_weights', True)
    for data in tqdm(loader, desc="Testing"):
        y = data['labels'] # y
        hypervariables = data['hypervars'] # mu, l, k
        x = data['vars'] # x
        w = hypernetwork(hypervariables)
        graphdef, template_state = nnx.split(targetnetwork)
        state = nnx.vmap(build_state_from_parameters, in_axes=(None, 0, None), out_axes=0)(template_state, w, replace)
        modified_targetnetwork = nnx.merge(graphdef, state)
        pred = nnx.vmap(type(modified_targetnetwork).__call__)(modified_targetnetwork, x)
        metrics.update(predictions=pred, targets=y)
    return metrics.compute()
