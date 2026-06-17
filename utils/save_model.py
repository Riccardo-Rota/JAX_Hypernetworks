import os
from flax import nnx
import orbax.checkpoint as ocp
from absl import logging

logging.set_verbosity(logging.WARNING)


def save_model(model: nnx.Module, path: str):
    """
    Saves only the learnable parameters of the nnx model to `path` using an Orbax
    StandardCheckpointer. Overwrites any existing checkpoint at that location.
    """
    _, params, _ = nnx.split(model, nnx.Param, ...)
    checkpointer = ocp.StandardCheckpointer()
    checkpointer.save(os.path.abspath(path), params, force=True)
    checkpointer.wait_until_finished()


def load_model(model: nnx.Module, path: str):
    """
    Restores the learnable parameters from `path` and updates `model` in-place.
    Retro-compatible with checkpoints written by `save_model`. The current model's
    parameters are used as the restore target, so the weights are placed on the
    active device regardless of which device the checkpoint was saved on.
    """
    _, params, _ = nnx.split(model, nnx.Param, ...)
    checkpointer = ocp.StandardCheckpointer()
    restored_params = checkpointer.restore(os.path.abspath(path), params)
    nnx.update(model, restored_params)
