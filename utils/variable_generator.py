from jax import random
from typing import Optional

def variables_generator_beta(N: int, domains: list, key: random.PRNGKey = random.key(0)) -> list:
    """
    Generates N random samples of an arbitrary number of variables from the specified domains.
    Each domain is a tuple (min, max) representing the range of one variable.
    Args:
        N (int): Number of samples to generate.
        domains (list): List of tuples, where each tuple contains the min and max values for a variable.
        key (random.PRNGKey): JAX random key for reproducibility.
    Returns:
        list: A list of JAX arrays, each containing N random variables from the corresponding domain
    """
    variables = []
    for domain in domains:
        key, subkey = random.split(key)
        var = random.uniform(subkey, (N,), minval=domain[0], maxval=domain[1])
        variables.append(var)
    return variables

def variables_generator(
        N: int, 
        var_domains: list,
        hypervar_domains: Optional[list] = None, 
        var_names: Optional[list] = None,
        hypervar_names: Optional[list] = None,
        n_realizations: int = 1,
        key: random.PRNGKey = random.key(0)) -> dict:
    #TODO: We have to decide wether we prefer to return a dictionary (with variable names as keys) or a list, or a tensor.
    # This also depends on the way we define the dataset class.
    """
    Generates a dataset of variables each sampled from specified domains.
    If hypervariables are provided, they are sampled N times, and, for each sample, n_realizations sets of variables are generated.
    Note: if hypervariables are not provided, the function generates N*n_realizations samples of the variables.
    Args:
        N (int): Number of samples to generate for each hypervariable.
        var_domains (list): List of tuples, where each tuple contains the min and max values for a variable.
        hypervar_domains (Optional[list]): List of tuples for hypervariables, same format as var_domains.
        var_names (Optional[list]): Names for the variables. If None, defaults to 'var_0', 'var_1', etc.
        hypervar_names (Optional[list]): Names for the hypervariables. If None, defaults to 'hypervar_0', 'hypervar_1', etc.
        n_realizations (int): Number of variable realizations for each sample of hypervariables.
        key (random.PRNGKey): JAX random key for reproducibility.
    Returns:
        dict: A dictionary where keys are variable names and values are JAX arrays of sampled values.
    Raises:
        AssertionError: If the number of variable names does not match the number of variable domains,
                        or if the number of hypervariable names does not match the number of hypervariable domains.
    """

    if hypervar_domains is None:
        hypervar_domains = []

    if var_names:
        assert len(var_names) == len(var_domains), "Number of variable names must match number of variable domains."
    else:
        var_names = [f"var_{i}" for i in range(len(var_domains))]

    if hypervar_names:
        assert len(hypervar_names) == len(hypervar_domains), "Number of hypervariable names must match number of hypervariable domains."
    else:
        hypervar_names = [f"hypervar_{i}" for i in range(len(hypervar_domains))]

    dataset = {}
    
    for domain, name in zip(hypervar_domains, hypervar_names):
        assert len(domain) == 2, "Each hypervariable domain must be a tuple of (min, max)."
        key, subkey = random.split(key)
        sample = random.uniform(subkey, (N,), minval=domain[0], maxval=domain[1])
        sample = sample.repeat(n_realizations)
        dataset[name] = sample
    
    for domain, name in zip(var_domains, var_names):
        assert len(domain) == 2, "Each variable domain must be a tuple of (min, max)."
        key, subkey = random.split(key)
        sample = random.uniform(subkey, (N*n_realizations,), minval=domain[0], maxval=domain[1])
        dataset[name] = sample

    return dataset