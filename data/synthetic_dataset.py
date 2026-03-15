import simpleeval
from typing import List, Optional, Dict, Sequence, Union, Tuple
from PACS.PACSproject.data import dataset
from PACS.PACSproject.data import dataset
from utils import get_function_from_string
import jax
import jax.numpy as jnp
from jax import random


class SyntheticDataset():
    """
    Dataset container for variables, hypervariables, and labels. Data are stored in a dictionary, divided in
       "hypervariables", "variables" and "labels".
    """

    def __init__(self,
                N: int,
                n_realizations: int,
                n_vars: int,
                n_hypervars: int,
                f_to_learn: Union[str, callable],
                vars_domain: Sequence[Tuple[float, float]],
                hypervars_domain: Sequence[Tuple[float, float]],
                seed: int = 0,
                ):
        self.N = N
        self.n_realizations = n_realizations
        self.n_vars = n_vars
        self.n_hypervars = n_hypervars
        if len(vars_domain) != n_vars:
            raise ValueError("Length of vars_domain must match n_vars.")
        if len(hypervars_domain) != n_hypervars:
            raise ValueError("Length of hypervars_domain must match n_hypervars.")
        self.vars_domain = vars_domain
        self.hypervars_domain = hypervars_domain
        self.key = random.key(seed)  # check if this is correct or I have to use PRNGKey
        if isinstance(f_to_learn, str):
            self.f_to_learn = get_function_from_string(f_to_learn)
        if callable(f_to_learn):  
            self.f_to_learn = f_to_learn
        else:
            raise ValueError("f_to_learn must be either a string or a callable function.")

    
    def __len__(self) -> int:
        return self.N*self.n_realizations


    def __getitem__(self, idx: Union[int, ArrayLike]) -> Dict[str, ArrayLike]:  #TODO: Cjeck how collate function is defined in dataloader
        idx = self._normalize_index(idx)

        if self._mode == "separated":
            hypervars = self.hypervars[idx]
            vars = self.vars[idx]
            labels = self.labels[idx]

        else:  # combined mode
            rows = self.data[idx]
            hypervars = rows[..., self._hypervars_idx]
            vars = rows[..., self._vars_idx]
            labels = rows[..., self._labels_idx]

        if hypervars.ndim == 1:
            hypervars = hypervars[:, None]
        if vars.ndim == 1:
            vars = vars[:, None]
        if labels.ndim == 1:
            labels = labels[:, None]

        return {
            "hypervars": hypervars,
            "vars": vars,
            "labels": labels,
        }
    

    def _generate_data(self):
        """
        """
        
        hypervariables = []
        for domain in self.hypervars_domain:
            assert len(domain) == 2, "Each hypervariable domain must be a tuple of (min, max)."
            self.key, subkey = random.split(self.key)
            sample = random.uniform(subkey, (self.N,), minval=domain[0], maxval=domain[1])
            sample = sample.repeat(self.n_realizations)
            hypervariables.append(sample)
        hypervariables = jnp.stack(hypervariables, axis=-1)  # Shape: (N*n_realizations, n_hypervars)
        
        variables = []
        for domain in self.vars_domain:
            assert len(domain) == 2, "Each variable domain must be a tuple of (min, max)."
            self.key, subkey = random.split(self.key)
            sample = random.uniform(subkey, (self.N*self.n_realizations,), minval=domain[0], maxval=domain[1])
            variables.append(sample)
        variables = jnp.stack(variables, axis=-1)  # Shape: (N*n_realizations, n_vars)

        labels = []
        for i in range(self.N*self.n_realizations):
            theta = hypervariables[i, :]
            x = variables[i, :]
            y = self.f_to_learn(theta, x)
            labels.append(y)
        labels = jnp.stack(labels, axis=0)  # Shape: (N*n_realizations, )

        self.hypervariables = hypervariables
        self.variables = variables
        self.labels = labels