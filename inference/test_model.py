import jax.numpy as jnp
from flax import nnx
from flax.nnx.training.metrics import Average, MultiMetric
from typing import Callable, Optional, List, Tuple
from data import JaxDataLoader
from tqdm import tqdm
from training import build_state_from_parameters


def test_model(
        hypernetwork: nnx.Module,
        targetnetwork: nnx.Module,
        loader: JaxDataLoader,
        metrics: Tuple[Callable, ...] = (),
        metrics_names: Optional[List[str]] = None,
        ) -> tuple:
    """
    Trains and evaluates the model for one epoch.
    Args:
        hypernetwork (nnx.Module): The hypernetwork that generates the parameters for the target network.
        targetnetwork (nnx.Module): The target network that will be modified by the hypernetwork.
        loader (DataLoader): DataLoader for testing data.
        metrics (Union[Callable, dict, None]): Metrics to compute during training and evaluation. Default: None.
        metrics_names (Optional[List[str]]): Optional list of names for the metrics. If provided, its length must match the number of metrics. Default: None.
    Returns:
        MultiMetric: Computed metrics.
    """
    if metrics_names:
        assert len(metrics_names) == len(metrics), "Length of metrics_names must be equal to length of metrics."
    else:
        metrics_names = [m.__name__ for m in metrics]

    metrics_collector = MultiMetric(**{k: Average(argname=k) for k in metrics_names})

    for data in tqdm(loader, desc="Testing"):
        y = data['labels'] # y
        hypervariables = data['hypervars'] # mu, l, k
        x = data['vars'] # x
        w = hypernetwork(hypervariables)
        graphdef, template_state = nnx.split(targetnetwork)
        state = nnx.vmap(build_state_from_parameters, in_axes=(None, 0), out_axes=0)(template_state, w)
        modified_targetnetwork = nnx.merge(graphdef, state)
        pred = nnx.vmap(type(modified_targetnetwork).__call__)(modified_targetnetwork, x)
        modified_targetnetwork = nnx.merge(graphdef, state)
        metrics_vals = {metrics_names[i]: jnp.mean(m(pred, y)) for i, m in enumerate(metrics)} if metrics else {}
        metrics_collector.update(**metrics_vals)
    return metrics_collector.compute()

#test_model = nnx.jit(test_model, static_argnames=("hypernetwork", "targetnetwork", "loader", "metrics"))
# TODO: jitting