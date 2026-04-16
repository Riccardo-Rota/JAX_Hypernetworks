import hydra
from omegaconf import DictConfig, OmegaConf
import flax.nnx as nnx
import os
if 'JAX_PLATFORMS' not in os.environ:
    try:
        from jax import devices
        if not any(d.platform == 'gpu' for d in devices()): os.environ['JAX_PLATFORMS'] = 'cpu'
    except Exception:
        os.environ['JAX_PLATFORMS'] = 'cpu'
from models import HypernetworkManager, TargetNetwork, ProjectionHead, Hypernetwork
from models import HypernetworkManager
from collections.abc import Mapping, Sequence
import yaml

import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax
# Assume HypernetworkManager, TargetNetwork, ProjectionHead, Hypernetwork are imported or defined above
def train_dummy_model():
    rngs = nnx.Rngs(42)

    # 1. Initialize the Modules
    base_target = nnx.Linear(in_features=3, out_features=2, rngs=rngs)
    base_hyper = nnx.Linear(in_features=4, out_features=5, rngs=rngs)

    # FIX: Use 'input' and 'output' (matches your updated NeuralNetwork __init__)
    # FIX: Pass strings so they map to positional *args for nnx.Linear
    hyper_block = Hypernetwork(
        network=base_hyper, 
        input='latent_z', 
        output='hyper_features'
    )
    
    proj_block = ProjectionHead(
        in_features=5, 
        input='hyper_features', 
        output='predicted_weights', 
        rngs=rngs
    )
    
    # FIX: Use 'weights' instead of 'weights_mapping' (matches TargetNetwork __init__)
    target_block = TargetNetwork(
        network=base_target, 
        input='target_input', 
        weights={'all': 'predicted_weights'}, 
        output='final_output'
    )

    manager = HypernetworkManager([target_block, proj_block, hyper_block], output='final_output')

    # 2. Setup the Optimizer
    learning_rate = 0.01
    optimizer = nnx.Optimizer(manager, optax.adam(learning_rate))

    # 3. Generate Dummy Dataset
    num_samples = 100
    batch_size = 10
    
    key = jax.random.key(0)
    k1, k2, k3 = jax.random.split(key, 3)
    X_data = jax.random.normal(k1, (num_samples, 3))  # Target network inputs
    Z_data = jax.random.normal(k2, (num_samples, 4))  # Hypernetwork latent inputs
    Y_data = jax.random.normal(k3, (num_samples, 2))  # Expected output

    # 4. Define Loss and Train Step
    def loss_fn(model, batch):
        outputs = model(batch)
        preds = outputs
        loss = jnp.mean((preds - batch['y_true']) ** 2)
        return loss

    @nnx.jit
    def train_step(model, optim, batch):
        loss, grads = nnx.value_and_grad(loss_fn)(model, batch)
        optim.update(grads)
        return loss

    # 5. Training Loop
    epochs = 500
    print("\n" + "="*50)
    print("STARTING DUMMY TRAINING (Overfitting random data)")
    print("="*50)
    
    # Fallback if tqdm is missing
    try:
        from tqdm import tqdm
        iterator = tqdm(range(epochs))
    except ImportError:
        iterator = range(epochs)
        
    for epoch in iterator:
        epoch_loss = 0.0
        num_batches = num_samples // batch_size
        
        for i in range(num_batches):
            start = i * batch_size
            end = start + batch_size
            
            batch = {
                'target_input': X_data[start:end],
                'latent_z': Z_data[start:end],
                'y_true': Y_data[start:end]
            }
            
            loss = train_step(manager, optimizer, batch)
            epoch_loss += loss.item()
            
        if (epoch + 1) % 50 == 0 or epoch == 0:
            avg_loss = epoch_loss / num_batches
            if not type(iterator).__name__ == 'tqdm':
                print(f"Epoch {epoch + 1:03d}/{epochs} - Loss: {avg_loss:.4f}")
            else:
                iterator.set_postfix({'Loss': f"{avg_loss:.4f}"})

if __name__ == "__main__":
    train_dummy_model()