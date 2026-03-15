import os

#1. Force XLA to use deterministic algorithms on GPU
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
from data import Dataset, JaxDataLoader
from training import train_model, assign_parameters, EarlyStopping
from inference import test_model
from utils import variables_generator, to_basic_types
from flax import nnx
from typing import Optional
import optax
from losses import *
from metrics import *
import datetime, time
from utils import save_model, register_resolvers
import json

register_resolvers()

@hydra.main(config_path="config", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:

    run_path = os.getcwd()
    print(f"Results will be saved in: {run_path}")

    key = random.key(cfg.seed)
    data_cfg = cfg.data
    f_to_learn = eval(cfg.data.f_to_learn, {"__builtins__": None, "jnp": jnp})
    mu_domain, l_domain, k_domain, x_domain = map(tuple, [data_cfg.mu_domain, data_cfg.l_domain, data_cfg.k_domain, data_cfg.x_domain])

    dataset_train, dataset_val, dataset_test = hydra.utils.instantiate(cfg.data)

    train_loader = JaxDataLoader(dataset_train, batch_size=cfg.training.batch_size, shuffle=True)
    val_loader = JaxDataLoader(dataset_val, batch_size=cfg.training.batch_size, shuffle=False)
    test_loader = JaxDataLoader(dataset_test, batch_size=cfg.training.batch_size, shuffle=False)
    
    #### TODO: remove this if possible
    N = len(dataset_train)
    cfg.data.N = N
    cfg.targetnetwork.num_neurons[0] = dataset_train.dim_vars()
    cfg.targetnetwork.num_neurons[-1] = dataset_train.dim_labels()
    #####
    
    # Instantiate Models using Hydra
    targetnetwork = hydra.utils.instantiate(cfg.targetnetwork)
    num_params = targetnetwork.num_parameters()
    print(f"Target network '{type(targetnetwork).__name__}' instantiated with {num_params} parameters.")

    OmegaConf.update(cfg.hypernetwork, "num_neurons.-1", num_params, merge=False) # Set output layer size to match target network parameters
    hypernetwork = hydra.utils.instantiate(cfg.hypernetwork)
    print(f"Hypernetwork '{type(hypernetwork).__name__}' instantiated.")
    
    # Instantiate Training Components
    criterion = hydra.utils.instantiate(cfg.training.criterion)
    metrics = {name: hydra.utils.instantiate(metric_cfg) for name, metric_cfg in cfg.training.metrics.items()}
    early_stopping = hydra.utils.instantiate(cfg.training.early_stopping)
    log_path = os.path.join(run_path, 'training_log.txt')

    optimizer = hydra.utils.instantiate(cfg.optimizer, model=hypernetwork)

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

    # Save model parameters
    hypernetwork_path = 'hypernetwork_params'
    #save_model(hypernetwork, hypernetwork_path) TODO: Implement model saving

    
    train_history = history['train_results']
    val_history = history['val_results']

    if "postprocessing" in cfg and ("output_plots" in cfg.postprocessing or "loss_plots" in cfg.postprocessing):
        plots_path = 'plots'
        os.makedirs(plots_path, exist_ok=True)
        print("\n--- Generating Plots ---")
        
        if "output_plots" in cfg.postprocessing:
            for name, config in cfg.postprocessing.output_plots.items():
                hydra.utils.call(config, hypernetwork=hypernetwork, targetnetwork=targetnetwork, save_path=plots_path)
                
        if "loss_plots" in cfg.postprocessing:
            for name, config in cfg.postprocessing.loss_plots.items():
                save_path = os.path.join(plots_path, f"{name}.png")
                hydra.utils.call(config, train_history=train_history, val_history=val_history, save_path=save_path)

    # Save JSON with all info
    run_data = {
        'f_to_learn': cfg.data.f_to_learn if 'f_to_learn' in cfg.data else 'other dataset',
        'test_metrics': test_metrics,
        'val_metrics': history['val_results'][early_stopping.best_epoch],
        'train_metrics': history['train_results'][early_stopping.best_epoch],
        'early_stopping_triggered': early_stopping.should_stop,
        'num_epochs': len(history['train_results']),
        'best_epoch': early_stopping.best_epoch,
        'training_time_seconds': end_time - start_time,
        'time_per_epoch_seconds': (end_time - start_time) / len(history['train_results']),
        'train_dataset_size': len(dataset_train),
        'N': cfg.data.N,
        'n_realizations': cfg.data.n_realizations,
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

    json_path = 'run_data.json'
    with open(json_path, 'w') as f:
        json.dump(to_basic_types(run_data), f, indent=4)

if __name__ == "__main__":
    main()