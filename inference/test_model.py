import jax.numpy as jnp
from flax import nnx
from flax.nnx.training.metrics import Average, MultiMetric, Metric
from typing import Callable, Optional, List, Tuple, Union, Dict
from tqdm import tqdm
import grain.python as grain
from training import build_state_from_parameters
from data_processing.grain_dataset import build_dataset


def test_model(
        model: nnx.Module,
        test_source: grain.RandomAccessDataSource,
        batch_size: int = 32,
        metrics: Optional[Union[MultiMetric, Dict[str, Metric]]] = None,
        ) -> tuple:
    """
    Trains and evaluates the model for one epoch.
    Args:
        model (nnx.Module): The model to test.
        test_source (grain.RandomAccessDataSource): DataSource for testing data.
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

    test_iter = build_dataset(test_source, is_training=False, batch_size=batch_size)
    for data, labels in tqdm(test_iter, desc="Testing"):
        pred = model(data)  # Forward pass to get predictions for metrics
        metrics.update(predictions=pred, targets=labels)
    return metrics.compute()
