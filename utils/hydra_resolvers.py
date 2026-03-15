from omegaconf import DictConfig, OmegaConf, ListConfig
from typing import Optional

def compute_train_steps(num_epochs: int, N: int, batch_size: int, n_realizations: Optional[int] = None) -> int:
    if n_realizations is None:
        n_realizations = 1
    if batch_size == 0:
        batch_size = 1
    return int((num_epochs * N * n_realizations) / min(batch_size, N * n_realizations))

def product(lst):
    result = 1
    for item in lst:
        result *= item
    return result

def register_resolvers():
    OmegaConf.register_new_resolver("compute_train_steps", compute_train_steps)
    OmegaConf.register_new_resolver("product", product)