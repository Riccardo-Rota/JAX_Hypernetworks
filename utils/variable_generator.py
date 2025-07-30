from jax import random

def variables_generator(N: int, domains: list, key: random.PRNGKey = random.key(0)) -> list:
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