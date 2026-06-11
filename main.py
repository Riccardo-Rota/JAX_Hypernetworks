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
from utils import to_basic_types, load_training_checkpoint
from flax import nnx
from typing import Optional
import optax
from losses import *
from metrics import *
import time
from utils import save_model, register_resolvers
import json
import logging
import glob
import wandb

register_resolvers()

log = logging.getLogger(__name__)

@hydra.main(config_path="config", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:

    run_path = os.getcwd()
    log.info(f"Results will be saved in: {run_path}")

    try:
        train_source = hydra.utils.instantiate(cfg.data_source.train)
        val_source = hydra.utils.instantiate(cfg.data_source.val) # Use different seed for validation set
        test_source = hydra.utils.instantiate(cfg.data_source.test) # Use different seed for test set

        train_dataset_len = len(train_source)
        OmegaConf.set_struct(cfg, False)
        cfg.runtime.N = train_dataset_len
        OmegaConf.set_struct(cfg, True)
        
        # Initialize W&B if enabled
        use_wandb = cfg.get("use_wandb", False) 
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

        # Instantiate Models using Hydra
        model = hydra.utils.instantiate(cfg.model.manager)
        
        # Instantiate Training Components
        criterion = hydra.utils.instantiate(cfg.training.criterion)
        metrics = {name: hydra.utils.instantiate(metric_cfg) for name, metric_cfg in cfg.training.metrics.items()}

        early_stopping = None
        if 'early_stopping' in cfg.training and cfg.training.early_stopping:
            early_stopping = hydra.utils.instantiate(cfg.training.early_stopping, best_metric=float('inf'))

        log_path = os.path.join(run_path, 'training_log.txt')
        checkpoint_path = os.path.join(run_path, 'checkpoints')
        optimizer = hydra.utils.instantiate(cfg.optimizer, model=model)
        checkpoint_manager = load_training_checkpoint(
            save_path=checkpoint_path,
            checkpoint_frequency=cfg.training.get('checkpointing_frequency', None),
            model=model,
            optimizer=optimizer,
            resume_path=cfg.training.get('resume_from_checkpoint', None)
        )

        log.info("Starting training...")
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
            log_file_path=log_path,
            checkpoint_manager=checkpoint_manager,
            use_wandb=use_wandb
        )
        end_time = time.time()
        log.info("Training completed.")
        # Run Testing
        test_metrics = test_model(
            model=model,
            test_source=test_source,
            batch_size=cfg.training.batch_size,
            metrics=metrics,
        )
        log.info(f"Test Metrics: {test_metrics}")  
        if use_wandb:
            wandb.log({f"test/{k}": float(v) for k, v in test_metrics.items()})
        
        train_history = history['train_results']
        val_history = history['val_results']

        plots_path = os.path.join(run_path, 'figures')
        if "postprocessing" in cfg and ("output_plots" in cfg.postprocessing or "loss_plots" in cfg.postprocessing):
            os.makedirs(plots_path, exist_ok=True)
            log.info("\n--- Generating Plots ---")
            
            if "output_plots" in cfg.postprocessing:
                for name, config in cfg.postprocessing.output_plots.items():
                    hydra.utils.call(config, model=model)
                    
            if "loss_plots" in cfg.postprocessing:
                for name, config in cfg.postprocessing.loss_plots.items():
                    save_path = os.path.join(plots_path, f"{name}.png")
                    hydra.utils.call(config, train_history=train_history, val_history=val_history, save_path=save_path)

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

        num_epochs_run = len(history['train_results'])
        if best_epoch is None:
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
        if use_wandb:
            summary_data = {k: v for k, v in run_data.items() if k != 'training_history'}
            wandb.summary.update(to_basic_types(summary_data))
    
    except Exception as e:
        raise e
    
    finally:
        if use_wandb:
            wandb.finish()

if __name__ == "__main__":
    main()