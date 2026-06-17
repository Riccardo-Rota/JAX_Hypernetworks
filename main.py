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
import time
from utils import load_model, register_resolvers
import json
import logging
import glob
import wandb

register_resolvers()

log = logging.getLogger(__name__)

@hydra.main(config_path="config", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:

    run_path = os.getcwd()
    plots_path = os.path.join(run_path, 'figures')
    log.info(f"Results will be saved in: {run_path}")
    use_wandb = cfg.get("use_wandb", False)
    train_flag = cfg.get("train_model", True)
    test_flag = cfg.get("test_model", True)
    inference_flag = cfg.get("plot_inference", True)
    checkpoint_path = cfg.get("checkpoint", None)
    load_path = hydra.utils.to_absolute_path(checkpoint_path) if checkpoint_path else None

    if train_flag == False and load_path is None:
        log.warning("Training is disabled and no checkpoint path provided. The model will be initialized with random weights.")

    try:
        # Instantiate Data Sources
        if train_flag:
            train_source = hydra.utils.instantiate(cfg.data_source.train)
            val_source = hydra.utils.instantiate(cfg.data_source.val)
            train_dataset_len = len(train_source)
            OmegaConf.set_struct(cfg, False)
            cfg.runtime.N = train_dataset_len
            OmegaConf.set_struct(cfg, True)
        if test_flag or inference_flag:
            test_source = hydra.utils.instantiate(cfg.data_source.test)

        # Initialize W&B if enabled
        if use_wandb:
            dir_time = os.path.basename(run_path)
            dir_day = os.path.basename(os.path.dirname(run_path)) 
            dir_problem = os.path.basename(os.path.dirname(os.path.dirname(run_path)))
            custom_name = f"{dir_problem}_{dir_day}_{dir_time}"
            config_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
            project_name = cfg.wandb_settings.project
            entity_name = cfg.wandb_settings.entity
            try:
                wandb.init(project=project_name, entity=entity_name, config=config_dict, dir=run_path, tags=["hydra-run"], name=custom_name)
            except Exception as e:
                log.error(f"Error initializing W&B: {e}")
                use_wandb = False

        # Instantiate Model
        model = hydra.utils.instantiate(cfg.model.manager)
        metrics = {name: hydra.utils.instantiate(metric_cfg) for name, metric_cfg in cfg.training.metrics.items()}

        # Single load point: if a checkpoint path is given, load the model weights from it
        # otherwise the model is initialized as usual.
        if load_path is not None:
            load_model(model, load_path)
            log.info(f"Loaded model weights from: {load_path}")
        else:
            log.info("No checkpoint path provided; using freshly initialized model.")

        # Instantiate Training Components
        if train_flag:
            criterion = hydra.utils.instantiate(cfg.training.criterion)
            early_stopping = None
            if 'early_stopping' in cfg.training and cfg.training.early_stopping:
                early_stopping = hydra.utils.instantiate(cfg.training.early_stopping, best_metric=float('inf'))
            log_path = os.path.join(run_path, 'training_log.txt')
            checkpoint_path = os.path.join(run_path, 'checkpoints')
            optimizer = hydra.utils.instantiate(cfg.optimizer, model=model)

            log.info("Starting training...")
            # Run Training (continues from the weights loaded above, if any)
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
                log_file_path=log_path,
                checkpoint_path=checkpoint_path,
                use_wandb=use_wandb
            )
            end_time = time.time()
            log.info("Training completed.")

            if "postprocessing" in cfg and ("loss_plots" in cfg.postprocessing):
                os.makedirs(plots_path, exist_ok=True)
                log.info("\n--- Generating Loss Plots ---")
                for name, config in cfg.postprocessing.loss_plots.items():
                    save_path = os.path.join(plots_path, f"{name}.png")
                    hydra.utils.call(config, train_history=history['train_results'], val_history=history['val_results'], save_path=save_path)
            
            best_idx = best_epoch if best_epoch is not None else len(history['train_results']) - 1
            train_data = {
                'train_metrics': history['train_results'][best_idx],
                'val_metrics': history['val_results'][best_idx],
                'early_stopping_triggered': final_early_stopping.should_stop if final_early_stopping else False,
                'num_epochs': len(history['train_results']),
                'best_epoch': best_idx,
                'training_time_seconds': end_time - start_time,
                'time_per_epoch_seconds': (end_time - start_time) / len(history['train_results']) if history['train_results'] else 0,
                'training_history': {
                    'train_results': history['train_results'],
                    'val_results': history['val_results']
                }
            }

        # Compute metrics on test set
        if test_flag:
            test_metrics = test_model(
                model=model,
                test_source=test_source,
                batch_size=cfg.training.batch_size,
                metrics=metrics,
            )
            log.info(f"Test Metrics: {test_metrics}")  
            if use_wandb:
                wandb.log({f"test/{k}": float(v) for k, v in test_metrics.items()})
        
        # Run Inference on test set and generate plots
        if inference_flag and "postprocessing" in cfg and "output_plots" in cfg.postprocessing:
            os.makedirs(plots_path, exist_ok=True)
            log.info("\n--- Generating Inference Plots ---")
            for name, config in cfg.postprocessing.output_plots.items():
                hydra.utils.call(config, model=model)

        # Save JSON with all info
        run_data = {}
        if test_flag:
            run_data['test_metrics'] = test_metrics
        if train_flag:
            run_data.update(train_data)
        json_path = 'run_data.json'
        with open(json_path, 'w') as f:
            json.dump(to_basic_types(run_data), f, indent=4)

        # Upload plots and summary to W&B if enabled
        if use_wandb and os.path.exists(plots_path):
            plot_files = glob.glob(os.path.join(plots_path, "*.png"))
            if plot_files:
                wandb_images = {}
                for file_path in plot_files:
                    base_name = os.path.basename(file_path)
                    plot_name = os.path.splitext(base_name)[0]
                    wandb_images[f"plots/outputs/{plot_name}"] = wandb.Image(file_path)
                wandb.log(wandb_images)
                log.info(f"Uploaded {len(plot_files)} plots to W&B.")
            summary_data = {k: v for k, v in run_data.items() if k != 'training_history'}
            wandb.summary.update(to_basic_types(summary_data))
    
    finally: # Ensure W&B session is closed properly even if an error occurs
        if use_wandb:
            wandb.finish()

if __name__ == "__main__":
    main()