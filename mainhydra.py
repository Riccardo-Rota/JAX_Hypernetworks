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
from training import train_model
from inference import test_model
from utils import to_basic_types
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

    train_source = hydra.utils.instantiate(cfg.data_source.train)
    val_source = hydra.utils.instantiate(cfg.data_source.val) # Use different seed for validation set
    test_source = hydra.utils.instantiate(cfg.data_source.test) # Use different seed for test set

    train_dataset_len = len(train_source)
    OmegaConf.set_struct(cfg, False)
    cfg.runtime.N = train_dataset_len
    OmegaConf.set_struct(cfg, True)
    
    # Instantiate Models using Hydra
    model = hydra.utils.instantiate(cfg.model.manager)
    
    # Instantiate Training Components
    criterion = hydra.utils.instantiate(cfg.training.criterion)
    metrics = {name: hydra.utils.instantiate(metric_cfg) for name, metric_cfg in cfg.training.metrics.items()}

    early_stopping = None
    if 'early_stopping' in cfg.training and cfg.training.early_stopping:
        early_stopping = hydra.utils.instantiate(cfg.training.early_stopping, best_metric=float('inf'))

    log_path = os.path.join(run_path, 'training_log.txt')

    optimizer = hydra.utils.instantiate(cfg.optimizer, model=model)

    print("Starting training...")
    # Run Training
    start_time = time.time()
    history, final_early_stopping, best_epoch = train_model(
        model=model,
        train_source=train_source,
        val_source=val_source,
        optimizer=optimizer,
        num_epochs=cfg.training.epochs,
        batch_size=cfg.training.batch_size,
        criterion=criterion,
        metrics=metrics,
        early_stopping=early_stopping,
        log_file_path=log_path
    )
    end_time = time.time()
    print("Training completed.")
    # Run Testing
    test_metrics = test_model(
        model=model,
        test_source=test_source,
        batch_size=cfg.training.batch_size,
        metrics=metrics,
    )
    print(f"Test Metrics: {test_metrics}")  

    # Save model parameters
    saved_model_path = 'model_weights'
    #save_model(model, saved_model_path) TODO: Implement model saving

    
    train_history = history['train_results']
    val_history = history['val_results']

    if "postprocessing" in cfg and ("output_plots" in cfg.postprocessing or "loss_plots" in cfg.postprocessing):
        plots_path = 'figures'
        os.makedirs(plots_path, exist_ok=True)
        print("\n--- Generating Plots ---")
        
        if "output_plots" in cfg.postprocessing:
            for name, config in cfg.postprocessing.output_plots.items():
                hydra.utils.call(config, model=model)
                
        if "loss_plots" in cfg.postprocessing:
            for name, config in cfg.postprocessing.loss_plots.items():
                save_path = os.path.join(plots_path, f"{name}.png")
                hydra.utils.call(config, train_history=train_history, val_history=val_history, save_path=save_path)

    num_epochs_run = len(history['train_results'])
    if best_epoch is None:
        # If early stopping was not used, the best epoch is the last one.
        best_epoch = num_epochs_run - 1

    es_triggered = final_early_stopping.should_stop if final_early_stopping else False

    # Save JSON with all info
    run_data = {
        'test_metrics': test_metrics,
        'val_metrics': history['val_results'][best_epoch],
        'train_metrics': history['train_results'][best_epoch],
        'early_stopping_triggered': es_triggered,
        'num_epochs': num_epochs_run,
        'best_epoch': best_epoch,
        'training_time_seconds': end_time - start_time,
        'time_per_epoch_seconds': (end_time - start_time) / num_epochs_run,
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