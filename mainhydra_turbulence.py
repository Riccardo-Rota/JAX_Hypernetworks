import os

# 1. Force XLA to use deterministic algorithms on GPU
# os.environ["XLA_FLAGS"] = "--xla_gpu_deterministic_ops=true"

# CPU fallback
if 'JAX_PLATFORMS' not in os.environ:
    try:
        from jax import devices
        if not any(d.platform == 'gpu' for d in devices()): os.environ['JAX_PLATFORMS'] = 'cpu'
    except Exception:
        os.environ['JAX_PLATFORMS'] = 'cpu'


import hydra
from omegaconf import DictConfig, OmegaConf, ListConfig
import jax.numpy as jnp
import jax.random as random
import matplotlib.pyplot as plt
from data import Dataset, JaxDataLoader, create_dataset_turbulence
from training import train_model, assign_parameters, EarlyStopping
from inference import test_model
from utils import variables_generator
from flax import nnx
from typing import Optional
import optax
from losses import *
from metrics import *
import datetime, time
from utils import save_model
import json

def compute_train_steps(num_epochs: int, N: int, batch_size: int) -> int:
    N=(N)
    batch_size=(batch_size)
    num_epochs=(num_epochs)
    if batch_size == 0:
        batch_size = 1
    return int((num_epochs * N) / min(batch_size, N))

def product(lst):
    result = 1
    for item in lst:
        result *= item
    return result

OmegaConf.register_new_resolver("compute_train_steps", compute_train_steps)
OmegaConf.register_new_resolver("product", product)

@hydra.main(config_path="config_turbulence", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:

    run_path = os.getcwd()
    print(f"Results will be saved in: {run_path}")

    dataset_train, dataset_val, dataset_test = hydra.utils.instantiate(cfg.data)

    train_loader = JaxDataLoader(dataset_train, batch_size=cfg.training.batch_size, shuffle=True, seed=cfg.seed)
    val_loader = JaxDataLoader(dataset_val, batch_size=cfg.training.batch_size, shuffle=False, seed=cfg.seed)
    test_loader = JaxDataLoader(dataset_test, batch_size=cfg.training.batch_size, shuffle=False, seed=cfg.seed)

    N = len(dataset_train)
    cfg.N = N

    # Instantiate Models using Hydra
    cfg.targetnetwork.num_neurons[0] = dataset_train.dim_vars()
    cfg.targetnetwork.num_neurons[-1] = dataset_train.dim_labels()
    targetnetwork = hydra.utils.instantiate(cfg.targetnetwork)
    num_params = targetnetwork.num_parameters()
    print(f"Target network '{type(targetnetwork).__name__}' instantiated with {num_params} parameters.")

    cfg.hypernetwork.num_neurons[0] = dataset_train.dim_hypervars()
    OmegaConf.update(cfg.hypernetwork, "num_neurons.-1", num_params, merge=False) # Set output layer size to match target network parameters
    hypernetwork = hydra.utils.instantiate(cfg.hypernetwork)
    print(f"Hypernetwork '{type(hypernetwork).__name__}' instantiated.")
    
    # Instantiate Training Components
    criterion = hydra.utils.instantiate(cfg.training.criterion)
    metrics = {name: hydra.utils.instantiate(metric_cfg) for name, metric_cfg in cfg.training.metrics.items()}
    early_stopping = hydra.utils.instantiate(cfg.training.early_stopping)
    log_path = os.path.join(run_path, 'training_log.txt')

    optimizer = hydra.utils.instantiate(cfg.optimizer, model=hypernetwork)

    print("Configuration being used:")
    print(OmegaConf.to_yaml(cfg))

    print("Starting training...")
    # Run Training
    start_time = time.time()
    history = train_model(
        hypernetwork=hypernetwork,
        targetnetwork=targetnetwork,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        num_epochs=cfg.training.epochs,
        criterion=criterion,
        metrics=metrics,
        early_stopping=early_stopping,
        log_file_path=log_path
    )
    end_time = time.time()
    print("Training completed.")
    # Run Testing
    test_metrics = test_model(
        hypernetwork=hypernetwork,
        targetnetwork=targetnetwork,
        loader=test_loader,
        metrics=metrics,
    )
    print(f"Test Metrics: {test_metrics}")

    # Save plots
    training_loss_history = [m['loss'] for m in history['train_results']]
    val_loss_history = [m['loss'] for m in history['val_results']]

    plt.figure()
    plt.loglog(range(len(training_loss_history)), training_loss_history, label='Training Loss')
    plt.loglog(range(len(val_loss_history)), val_loss_history, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.legend()
    loss_plot_path = 'loss_plot_loglog.png'
    plt.savefig(loss_plot_path)
    plt.close()
    plt.figure()
    plt.semilogx(range(len(training_loss_history)), training_loss_history, label='Training Loss')
    plt.semilogx(range(len(val_loss_history)), val_loss_history, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.legend()
    loss_plot_path = 'loss_plot_semilogx.png'
    plt.savefig(loss_plot_path)
    plt.close()

    # Save model parameters
    hypernetwork_path = 'hypernetwork_params'
    #save_model(hypernetwork, hypernetwork_path) TODO: Implement model saving

    # Save JSON with all info
    run_data = {
        'test_metrics': test_metrics,
        'N': N,
        'val_metrics': history['val_results'][early_stopping.best_epoch],
        'train_metrics': history['train_results'][early_stopping.best_epoch],
        'early_stopping_triggered': early_stopping.should_stop,
        'num_epochs': len(history['train_results']),
        'best_epoch': early_stopping.best_epoch,
        'training_time_seconds': end_time - start_time,
        'time_per_epoch_seconds': (end_time - start_time) / len(history['train_results']),
        'train_dataset_size': len(dataset_train),
        'hypernetwork': {
            'type': type(hypernetwork).__name__,
            'num_parameters': hypernetwork.num_parameters(),
            'num_neurons': cfg.hypernetwork.num_neurons,
        },
        'targetnetwork': {
            'type': type(targetnetwork).__name__,
            'num_parameters': targetnetwork.num_parameters(),
            'num_neurons': cfg.targetnetwork.num_neurons,
        },
        'training_history': {
            'train_results': history['train_results'],
            'val_results': history['val_results']
        }
    }

    # Convert JAX arrays to list for JSON
    def convert(obj):
        if isinstance(obj, jnp.ndarray):
            return obj.tolist()
        if isinstance(obj, (list, tuple, ListConfig)):
            return [convert(o) for o in obj]
        if isinstance(obj, DictConfig):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        return obj

    json_path = 'run_data.json'
    with open(json_path, 'w') as f:
        json.dump(convert(run_data), f, indent=4)

if __name__ == "__main__":
    main()