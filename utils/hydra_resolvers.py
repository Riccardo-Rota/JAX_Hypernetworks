from omegaconf import DictConfig, OmegaConf, ListConfig
from typing import Optional
import math

def compute_train_steps(num_epochs: int, N: int, batch_size: int) -> int:
    if batch_size == 0:
        batch_size = 1
    return int((num_epochs * N) / min(batch_size, N))

def as_tuple(*args):
    return tuple(args)

def register_resolvers():
    OmegaConf.register_new_resolver("compute_train_steps", compute_train_steps)
    OmegaConf.register_new_resolver("product", math.prod)
    OmegaConf.register_new_resolver("sum", sum)
    OmegaConf.register_new_resolver("as_tuple", as_tuple)
    OmegaConf.register_new_resolver("len", len)