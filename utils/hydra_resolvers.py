from omegaconf import DictConfig, OmegaConf, ListConfig
from omegaconf.errors import ConfigValueError
import hydra
from typing import Optional
import math

def compute_train_steps(num_epochs: int, N: int, batch_size: int) -> int:
    if batch_size == 0:
        batch_size = 1
    return int((num_epochs * N) / min(batch_size, N))

def as_tuple(*args):
    return tuple(args)

def resolve_higher_order(wrapper_path: str, inner_path: str):
    """
    Resolves higher-order functions.
    Example: wrapper(inner) -> optax.inject_hyperparams(optax.adam)
    """
    try:
        wrapper_func = hydra.utils.get_method(wrapper_path)
        inner_func = hydra.utils.get_method(inner_path)
        return wrapper_func(inner_func)
    except Exception as e:
        raise ConfigValueError(f"Failed to resolve higher-order function: {e}")

def register_resolvers():
    OmegaConf.register_new_resolver("compute_train_steps", compute_train_steps)
    OmegaConf.register_new_resolver("product", math.prod)
    OmegaConf.register_new_resolver("sum", sum)
    OmegaConf.register_new_resolver("as_tuple", as_tuple)
    OmegaConf.register_new_resolver("len", len)
    OmegaConf.register_new_resolver("ho_func", resolve_higher_order)
    OmegaConf.register_new_resolver("int_product", lambda x, y: int(x * y))