import os
if 'JAX_PLATFORMS' not in os.environ:
    try:
        from jax import devices
        if not any(d.platform == 'gpu' for d in devices()): os.environ['JAX_PLATFORMS'] = 'cpu'
    except Exception:
        os.environ['JAX_PLATFORMS'] = 'cpu'
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


# ================================================================ #

from flax import nnx
import orbax.checkpoint as ocp
import jax
from jax import numpy as jnp
import numpy as np

class TwoLayerMLP(nnx.Module):
    def __init__(self, dim, rngs: nnx.Rngs):
        self.linear1 = nnx.Linear(dim, dim, rngs=rngs, use_bias=False)
        self.linear2 = nnx.Linear(dim, dim, rngs=rngs, use_bias=False)

    def __call__(self, x):
        x = self.linear1(x)
        return self.linear2(x)
        
def main():
    import pathlib
    ckpt_dir = ocp.test_utils.erase_and_create_empty('my-checkpoints/')
    ckpt_dir = pathlib.Path(ckpt_dir).absolute()
    # Instantiate the model and show we can run it.
    model = TwoLayerMLP(4, rngs=nnx.Rngs(0))
    x = jax.random.normal(jax.random.key(42), (3, 4))
    assert model(x).shape == (3, 4)
    _, state = nnx.split(model)
    nnx.display(state)
    checkpointer = ocp.StandardCheckpointer()
    checkpointer.save(ckpt_dir / 'state', state)
    checkpointer.wait_until_finished()

    abstract_model = nnx.eval_shape(lambda: TwoLayerMLP(4, rngs=nnx.Rngs(0)))
    graphdef, abstract_state = nnx.split(abstract_model)
    print('The abstract NNX state (all leaves are abstract arrays):')
    nnx.display(abstract_state)

    state_restored = checkpointer.restore(ckpt_dir / 'state', abstract_state)
    jax.tree.map(np.testing.assert_array_equal, state, state_restored)
    print('NNX State restored: ')
    nnx.display(state_restored)

    # The model is now good to use!
    model = nnx.merge(graphdef, state_restored)
    assert model(x).shape == (3, 4)

if __name__ == "__main__":
    main()