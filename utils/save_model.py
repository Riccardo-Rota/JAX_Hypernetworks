import os
import jax
from flax import nnx
import orbax.checkpoint as ocp
from typing import Optional
from absl import logging

logging.set_verbosity(logging.WARNING)

def get_checkpoint_manager(
    save_path: str,
    checkpoint_frequency: Optional[int] = None,
    max_to_keep: int = 3
) -> ocp.CheckpointManager:
    """
    Creates and returns an Orbax CheckpointManager to handle directory routing and retention.
    """
    options = ocp.CheckpointManagerOptions(
        max_to_keep=max_to_keep,
        save_interval_steps=checkpoint_frequency,
        create=True
    )
    return ocp.CheckpointManager(os.path.abspath(save_path), options=options)


def restore_checkpoint(
    manager: ocp.CheckpointManager,
    model: nnx.Module,
    optimizer: Optional[nnx.Optimizer] = None,
    step: Optional[int] = None
) -> int:
    """
    Restores model parameters (and optionally optimizer) weights from disk. 
    Mutates the `model` and `optimizer` in-place.
    """
    target_step = step if step is not None else manager.latest_step()
    
    if target_step is None:
        return 0 

    print(f"Restoring checkpoint from step {target_step}...")

    _, original_params, _ = nnx.split(model, nnx.Param, ...)
    
    abstract_params = jax.tree.map(
        lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype) if hasattr(x, 'shape') else x, 
        original_params
    )
    abstract_payload = {'model': abstract_params}

    if optimizer is not None:
        _, original_opt_state = nnx.split(optimizer)
        abstract_opt_state = jax.tree.map(
            lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype) if hasattr(x, 'shape') else x, 
            original_opt_state
        )
        abstract_payload['optimizer'] = abstract_opt_state

    restore_args = ocp.args.StandardRestore(abstract_payload)
    restored_payload = manager.restore(target_step, args=restore_args)

    nnx.update(model, restored_payload['model'])
    if optimizer is not None and 'optimizer' in restored_payload:
        nnx.update(optimizer, restored_payload['optimizer'])

    return target_step

def save_checkpoint(
    manager: ocp.CheckpointManager,
    step: int,
    model: nnx.Module,
    optimizer: Optional[nnx.Optimizer] = None
):
    """
    Extracts the state from the model and optimizer and saves it to disk.
    """
    _, model_state, _ = nnx.split(model, nnx.Param, ...)
    payload = {'model': model_state}
    
    if optimizer is not None:
        _, opt_state = nnx.split(optimizer)
        payload['optimizer'] = opt_state

    save_args = ocp.args.StandardSave(payload)
    manager.save(step, args=save_args)