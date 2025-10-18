import orbax.checkpoint as ocp
from flax import nnx

# TODO: CHECK IF THIS WORKS

def save_model(model: nnx.Module, path: str):
    """
    Saves an nnx model to an Orbax checkpoint.
    Args:
        model: nnx model to be saved.
        path: Path where the model will be saved.
    """
    graphdef, state = nnx.split(model)
    save_data = {'graphdef': graphdef, 'state': state}
    checkpointer = ocp.StandardCheckpointer()
    checkpointer.save(path, state=state)

def load_model(path: str):
    """
    Loads an nnx model from an Orbax checkpoint.
    Args:
        path: Path from which the model will be loaded.
    """
    checkpointer = ocp.StandardCheckpointer()
    restored_data = checkpointer.restore(path)
    graphdef = restored_data['graphdef']
    state = restored_data['state']
    model = nnx.merge(graphdef, state)

    return model