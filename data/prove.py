from grain_dataset import get_pipeline
from grain_dataset import ToyDataSource
import jax.numpy as jnp


toy_lambda = lambda h, v: h[0] * jnp.sin(h[1] * v[0]) * jnp.exp(-h[2] * v[0])

# Define the domains for the 3 hypervariables (mu, k, lambda)
hyper_domains = [
    (0.5, 2.0),  # mu domain
    (1.0, 5.0),  # k domain
    (0.1, 0.5)   # lambda domain
]

# Define the domain for the 1 variable (x)
var_domains = [
    (0.0, 10.0)  # x domain
]

# Instantiate the dataset
# Total samples N=100000. Each unique set of (mu, k, lambda) gets 100 varying 'x' realizations.
toy_source = ToyDataSource(
    f=toy_lambda,
    hyper_domains=hyper_domains,
    var_domains=var_domains,
    N=100000,
    n_realizations=100,
    seed=42
)

# 1. Initialize the pipeline for a trial run
# (Assuming toy_source is already instantiated)
train_iterator = get_pipeline(
    source=toy_source, 
    is_training=False, 
    batch_size=32,
    seed = 10,
    in_memory=True
)

# 2. Pull a single batch using Python's built-in next()
# This triggers the Grain workers to fetch, batch, and return the data.
first_batch = next(train_iterator)

print(first_batch["hypervars"][0])

# 3. Inspect the batch structure
print("Type of batch:", type(first_batch))
print("Available keys:", first_batch.keys())

# 4. Verify the shapes and datatypes of the NumPy arrays
for key, array in first_batch.items():
    print(f"Key: '{key}' | Shape: {array.shape} | Dtype: {array.dtype}")