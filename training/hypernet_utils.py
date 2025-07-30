from flax import nnx
import jax

@nnx.jit
#@nnx.vmap(in_axes=(None, 0), out_axes=0) mi sembra più comodo tenere come base la funzione senza vmap, e poi usare nnx.vmap quando serve
def build_state_from_parameters(template_state: nnx.statelib.State, parameters: jax.Array) -> nnx.statelib.State:
    """
    Builds a state from the parameters, reshaping them according to the template state.
    Args:
        template_state (nnx.statelib.State): The template state that defines the structure of the parameters.
        parameters (jax.Array): The parameters to be reshaped and assigned to the template state.
    Returns:
        nnx.statelib.State: The state with the parameters reshaped according to the template.
    """
    treedef = jax.tree.structure(template_state)
    reshaped_parameters = []
    shapes = []
    sizes = []
    for _, param in nnx.to_flat_state(template_state):
        shapes.append(param.value.shape)
        sizes.append(param.value.size)
    i = 0
    for shape, size in zip(shapes, sizes):
        reshaped_parameters.append(parameters[i:i+size].reshape(shape))
        i += size
    state = jax.tree.unflatten(treedef, reshaped_parameters)
    return state

@nnx.jit
def assign_parameters(model: nnx.Module, parameters: jax.Array) -> nnx.Module:
    """
    Assigns the parameters from an array to the state of the model.
    Args:
        model (nnx.Module): The model whose state will be updated with the new parameters.
        parameters (jax.Array): The parameters to assign to the model's state.
    Returns:
        nnx.Module: The model with the updated state.
    """
    
    graphdef, template_state = nnx.split(model)
    state = build_state_from_parameters(template_state = template_state, parameters = parameters)
    return nnx.merge(graphdef, state)


@nnx.jit
def apply(model: nnx.Module, parameters: jax.Array, x: jax.Array) -> jax.Array:
    """
    Applies the model to the input data using the specified parameters.
    Args:
        model (nnx.Module): The model to apply.
        parameters (jax.Array): The parameters to use for the model.
        x (jax.Array): Input data to the model.
    Returns:
        jax.Array: The output of the model after applying it to the input data.
    """
    return assign_parameters(model, parameters)(x)