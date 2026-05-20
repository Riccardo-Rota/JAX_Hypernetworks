from flax import nnx
import orbax.checkpoint as ocp
from absl import logging
from flax import nnx
from typing import Tuple, Optional
import jax
import os

logging.set_verbosity(logging.WARNING) # suppress verbose logging from Orbax

def save_model(model: nnx.Module, path: str):
    """
    Saves only the learnable parameters of the nnx model.
    """
    _, params, _ = nnx.split(model, nnx.Param, ...)
    checkpointer = ocp.StandardCheckpointer()
    checkpointer.save(path, params)
    checkpointer.wait_until_finished()

def load_model(model: nnx.Module, path: str, abstract_params):
    """
    Restores the learnable parameters and updates the existing model in-place.
    """
    checkpointer = ocp.StandardCheckpointer()
    restored_params = checkpointer.restore(path, abstract_params)
    nnx.update(model, restored_params)


def load_training_checkpoint(
    save_path: str,
    checkpoint_frequency: Optional[int],
    model: nnx.Module,
    optimizer: nnx.Optimizer,
    resume_path: Optional[str] = None
) -> ocp.CheckpointManager:
    """
    Initializes the CheckpointManager for saving new checkpoints, and restores 
    active model/optimizer weights if a checkpoint exists in the load directory.
    """
    options = ocp.CheckpointManagerOptions(
        max_to_keep=3,
        save_interval_steps=checkpoint_frequency,
        create=True
    )
    manager = ocp.CheckpointManager(os.path.abspath(save_path), options=options)

    load_dir = resume_path if resume_path else save_path
    
    if load_dir == save_path:
        load_manager = manager 
    else:
        load_manager = ocp.CheckpointManager(os.path.abspath(load_dir))
    latest_step = load_manager.latest_step()
    
    if latest_step is not None:
        print(f"Found active checkpoint at epoch {latest_step} in '{load_dir}'. Restoring active weights...")
        
        _, original_params, _ = nnx.split(model, nnx.Param, ...)
        _, original_opt_state = nnx.split(optimizer)

        abstract_params = jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), original_params)
        abstract_opt_state = jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), original_opt_state)

        abstract_payload = {
            'model': abstract_params,
            'optimizer': abstract_opt_state,
            'epoch': 0
        }

        restored_payload = load_manager.restore(
            latest_step, 
            args=ocp.args.StandardRestore(abstract_payload)
        )

        nnx.update(model, restored_payload['model'])
        nnx.update(optimizer, restored_payload['optimizer'])

    return manager