import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing import Dict, List, Optional, Callable, Union, Any
import networkx
import warnings
import math 
from flax.traverse_util import flatten_dict, unflatten_dict


def state_to_dict(state: Any) -> Any:
    """Recursively converts an nnx.State mapping into a standard Python dictionary."""
    if not isinstance(state, nnx.State):
        return state
    return {k: state_to_dict(v) for k, v in state.items()}


class OutputsNumberWarning(UserWarning):
    pass


class NeuralNetwork(nnx.Module):
    """
    Base class for neural network modules that includes metadata about input and output mappings for use in the HypernetworkManager."""
    def __init__(self, network: Optional[nnx.Module], input_mapping: Optional[Dict[str, str]], output_mapping: Optional[Union[List[str], str]]):
        self.network = network
        self.input_mapping = input_mapping #(e.g., {'inputs': 'variables', 'theta': 'hypervariables'})
        if isinstance(output_mapping, str):
            output_mapping = [output_mapping]
        self.output_mapping = output_mapping # (e.g., ['features1', 'features2'])

    def __call__(self, *args, **kwargs):
        if self.network is None:
            raise RuntimeError(f"Network is None in {self.__class__.__name__}. Please build it first.")
        return self.network(*args, **kwargs)


class TargetNetwork(NeuralNetwork):
    def __init__(self, network: nnx.Module, input_mapping: Optional[Dict[str, str]], output_mapping: Optional[Union[List[str], str]], weights_mapping: Optional[Dict[str, str]] = None, replace_weights: bool = True):
        super().__init__(network, input_mapping, output_mapping)
        self.weights_mapping = weights_mapping # (e.g., {'layer1.weight': 'features1', 'layer2.bias': 'features2'}, or {'all': 'features1'} for a single signal containing all weights)
        self.replace_weights = replace_weights

        # Pre-compute shapes and total sizes needed for each signal safely using state_to_dict
        target_state = nnx.state(network)
        state_dict = state_to_dict(target_state) 
        
        self.flat_state_shapes = {
            k: (v.value.shape if hasattr(v, 'value') else v.shape)
            for k, v in flatten_dict(state_dict, sep='.').items()
        }

        self.signal_to_weight_keys = {}  # Maps signal_name -> list of weight keys it must fill
        self.signal_to_weight_size = {}  # Maps signal_name -> total flat elements that the signal must provide

        if self.weights_mapping:
            if 'all' in self.weights_mapping:
                signal = self.weights_mapping['all']
                self.signal_to_weight_keys[signal] = list(self.flat_state_shapes.keys())
                self.signal_to_weight_size[signal] = sum(math.prod(s) for s in self.flat_state_shapes.values())
            else:
                for weight_key, signal in self.weights_mapping.items():
                    if weight_key not in self.flat_state_shapes:
                        raise ValueError(f"Weight key '{weight_key}' not found in TargetNetwork.")
                    self.signal_to_weight_keys.setdefault(signal, []).append(weight_key)
                    shape = self.flat_state_shapes[weight_key]
                    self.signal_to_weight_size[signal] = self.signal_to_weight_size.get(signal, 0) + math.prod(shape)

    def _inject_weights(self, state_dict: dict, weights: dict[str, jnp.ndarray]) -> dict:
        flat_state = flatten_dict(state_dict, sep='.')
        
        for signal, flat_array in weights.items():
            if signal not in self.signal_to_weight_keys:
                continue
            
            weight_keys = self.signal_to_weight_keys[signal]
            is_batched = flat_array.ndim > 1
            batch_size = flat_array.shape[0] if is_batched else None

            current_idx = 0
            for weight_key in weight_keys:
                shape = self.flat_state_shapes[weight_key]
                num_elements = math.prod(shape)

                if is_batched:
                    flat_slice = flat_array[:, current_idx : current_idx + num_elements]
                    target_shape = (batch_size,) + shape
                else:
                    flat_slice = flat_array[current_idx : current_idx + num_elements]
                    target_shape = shape

                reshaped_weight = jnp.reshape(flat_slice, target_shape)
                
                old_var = flat_state[weight_key]
                
                # Check if it's a VariableState container (has both 'type' and 'value')
                if hasattr(old_var, 'value') and hasattr(old_var, 'type'):
                    new_val = reshaped_weight if self.replace_weights else (old_var.value + reshaped_weight)
                    # Reconstruct correctly: VariableState(type, value)
                    flat_state[weight_key] = type(old_var)(old_var.type, new_val)
                    
                # Fallback for other potential variable wrappers
                elif hasattr(old_var, 'value'):
                    new_val = reshaped_weight if self.replace_weights else (old_var.value + reshaped_weight)
                    flat_state[weight_key] = type(old_var)(value=new_val)
                    
                # Standard raw array
                else:
                    if self.replace_weights:
                        flat_state[weight_key] = reshaped_weight
                    else:
                        flat_state[weight_key] += reshaped_weight
                        
                current_idx += num_elements
                
        return unflatten_dict(flat_state, sep='.')

    def __call__(self, *args, weights: Optional[Dict[str, Any]] = None, **kwargs):
        if weights is not None:
            graphdef, state = nnx.split(self.network)
            state_dict = state_to_dict(state) 
            state_dict = self._inject_weights(state_dict, weights)
            new_state = nnx.State(state_dict)
            
            is_batched = any(w.ndim > 1 for w in weights.values())
            
            if is_batched:
                # Map across axis 0 for both the new state and the kwargs PyTrees.
                def apply_fn(state_to_apply, fn_kwargs):
                    modified_network = nnx.merge(graphdef, state_to_apply)
                    return modified_network(**fn_kwargs)
                
                vmap_forward = jax.vmap(apply_fn, in_axes=(0, 0))
                return vmap_forward(new_state, kwargs)
            else:
                modified_network = nnx.merge(graphdef, new_state) 
                return modified_network(*args, **kwargs)
        else:
            return self.network(*args, **kwargs)


class Hypernetwork(NeuralNetwork):
    "Hypernetwork class that generates a output representing a latent space, to be used by a ProjectionHead."
    pass


class ProjectionHead(NeuralNetwork):
    def __init__(self, 
                 in_features: int, 
                 input_mapping: Optional[Dict[str, str]], 
                 output_mapping: str,
                 rngs: Optional[nnx.Rngs] = None,
                 kernel_init: Callable = nnx.initializers.lecun_normal(),
                 bias_init: Callable = nnx.initializers.zeros_init()):
        
        super().__init__(network=None, input_mapping=input_mapping, output_mapping=output_mapping)
        self.in_features = in_features
        # Instantiate inside the init to avoid eager JAX evaluation at import time
        self.rngs = rngs if rngs is not None else nnx.Rngs(0) 
        self.kernel_init = kernel_init
        self.bias_init = bias_init

    def build(self, out_features: int):
        self.network = nnx.Linear(
            in_features=self.in_features, 
            out_features=out_features, 
            kernel_init=self.kernel_init, 
            bias_init=self.bias_init,
            rngs=self.rngs
        )


class HypernetworkManager(nnx.Module):
    def __init__(self, blocks: List[NeuralNetwork]):
        
        signal_to_weight_size = {}
        for block in blocks:
            if isinstance(block, TargetNetwork) and block.weights_mapping:
                for signal, size in block.signal_to_weight_size.items():
                    if signal in signal_to_weight_size and signal_to_weight_size[signal] != size:
                        raise ValueError(f"Signal '{signal}' is used by two different weight sets with conflicting sizes.")
                    signal_to_weight_size[signal] = size

        for block in blocks:
            if isinstance(block, ProjectionHead):
                if not block.output_mapping or len(block.output_mapping) != 1:
                    raise ValueError("ProjectionHead must have exactly one output mapped.")
                
                out_signal = block.output_mapping[0]
                if out_signal in signal_to_weight_size:
                    block.build(out_features=signal_to_weight_size[out_signal])
                else:
                    warnings.warn(f"Head output '{out_signal}' is unused. Building with size 1.")
                    block.build(out_features=1)

        graph = networkx.DiGraph() 
        outputs_producers = {} 

        for i, block in enumerate(blocks): 
            graph.add_node(i) 
            if block.output_mapping:
                for output_node in block.output_mapping:
                    outputs_producers[output_node] = i

        for i, block in enumerate(blocks): 
            if block.input_mapping:
                for input_node in block.input_mapping.values():
                    if input_node in outputs_producers:
                        graph.add_edge(outputs_producers[input_node], i)
            if isinstance(block, TargetNetwork) and block.weights_mapping:
                for weight_signal in block.weights_mapping.values():
                    if weight_signal in outputs_producers:
                        graph.add_edge(outputs_producers[weight_signal], i)

        try: 
            sorted_indices = list(networkx.topological_sort(graph))
        except networkx.NetworkXUnfeasible:
            raise ValueError("A circular dependency was detected in the graph.")

        self.execution_order = [blocks[i] for i in sorted_indices]

    def __call__(self, inputs: dict[str, jnp.ndarray]) -> Dict[str, Any]:
        objects = inputs.copy() 
        
        for block in self.execution_order:
            input_kwargs = {name: objects[key] for name, key in (block.input_mapping or {}).items()}
            
            if isinstance(block, TargetNetwork) and block.weights_mapping:
                unique_signals = set(block.weights_mapping.values())
                if 'weights' in input_kwargs:
                    raise ValueError("The reserved keyword 'weights' cannot be used as an input mapping key for a TargetNetwork block.")
                input_kwargs['weights'] = {signal: objects[signal] for signal in unique_signals}

            y = block(**input_kwargs)
            output_names = block.output_mapping or [] 
            if not isinstance(y, tuple):
                y = (y,)
            
            if len(output_names) != len(y):
                warnings.warn(f"Block '{type(block).__name__}' produced {len(y)} outputs but has {len(output_names)} output names defined.")
            
            for i in range(min(len(output_names), len(y))):
                objects[output_names[i]] = y[i]

        return objects
    

# def run_dummy_test():
#     rngs = nnx.Rngs(42)

#     base_target = nnx.Linear(in_features=3, out_features=2, rngs=rngs)
#     base_hyper = nnx.Linear(in_features=4, out_features=5, rngs=rngs)

#     # Note the kwarg key is 'inputs' for all mapping configurations
#     # because that is the exact parameter name flax.nnx.Linear expects.
#     hyper_block = Hypernetwork(
#         network=base_hyper,
#         input_mapping={'inputs': 'latent_z'},  
#         output_mapping='hyper_features'        
#     )

#     proj_block = ProjectionHead(
#         in_features=5,                         
#         input_mapping={'inputs': 'hyper_features'}, 
#         output_mapping='predicted_weights',    
#         rngs=rngs  # Will default internally if omitted, but passed here safely                            
#     )

#     target_block = TargetNetwork(
#         network=base_target,
#         input_mapping={'inputs': 'target_input'},   
#         weights_mapping={'all': 'predicted_weights'}, 
#         output_mapping='final_output'
#     )

#     manager = HypernetworkManager([target_block, proj_block, hyper_block])

#     print("\n" + "="*50)
#     print("TEST 1: UNBATCHED FORWARD PASS (Single item)")
#     print("="*50)
    
#     inputs_unbatched = {
#         'latent_z': jnp.ones((4,)),        
#         'target_input': jnp.ones((3,))     
#     }
    
#     outputs_unbatched = manager(inputs_unbatched)
    
#     print(f"Provided latent: {inputs_unbatched['latent_z'].shape}")
#     print(f"Provided data input: {inputs_unbatched['target_input'].shape}")
#     print(f"Predicted weights shape (ProjectionHead): {outputs_unbatched['predicted_weights'].shape}")
#     print(f"Final output (TargetNetwork): {outputs_unbatched['final_output'].shape} -> Expected: (2,)")

#     print("\n" + "="*50)
#     print("TEST 2: BATCHED FORWARD PASS (vmap active)")
#     print("="*50)
    
#     batch_size = 10
#     inputs_batched = {
#         'latent_z': jnp.ones((batch_size, 4)),       
#         'target_input': jnp.ones((batch_size, 3))    
#     }
    
#     outputs_batched = manager(inputs_batched)
    
#     print(f"Provided latents: {inputs_batched['latent_z'].shape}")
#     print(f"Provided data input: {inputs_batched['target_input'].shape}")
#     print(f"Predicted weights shape (ProjectionHead): {outputs_batched['predicted_weights'].shape}")
#     print(f"Final output (TargetNetwork): {outputs_batched['final_output'].shape} -> Expected: (10, 2)")
#     print("==================================================\n")


# if __name__ == "__main__":
#     import os
#     if 'JAX_PLATFORMS' not in os.environ:
#         try:
#             from jax import devices
#             if not any(d.platform == 'gpu' for d in devices()): os.environ['JAX_PLATFORMS'] = 'cpu'
#         except Exception:
#             os.environ['JAX_PLATFORMS'] = 'cpu'
#     run_dummy_test()

import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax
# Assume HypernetworkManager, TargetNetwork, ProjectionHead, Hypernetwork are imported or defined above

def train_dummy_model():
    rngs = nnx.Rngs(42)

    # 1. Initialize the Modules
    base_target = nnx.Linear(in_features=3, out_features=2, rngs=rngs)
    base_hyper = nnx.Linear(in_features=4, out_features=5, rngs=rngs)

    hyper_block = Hypernetwork(
        network=base_hyper, 
        input_mapping={'inputs': 'latent_z'}, 
        output_mapping='hyper_features'
    )
    proj_block = ProjectionHead(
        in_features=5, 
        input_mapping={'inputs': 'hyper_features'}, 
        output_mapping='predicted_weights', 
        rngs=rngs
    )
    target_block = TargetNetwork(
        network=base_target, 
        input_mapping={'inputs': 'target_input'}, 
        weights_mapping={'all': 'predicted_weights'}, 
        output_mapping='final_output'
    )

    manager = HypernetworkManager([target_block, proj_block, hyper_block])

    # 2. Setup the Optimizer
    # nnx.Optimizer automatically traverses the manager PyTree and registers all nnx.Param instances
    learning_rate = 0.01
    optimizer = nnx.Optimizer(manager, optax.adam(learning_rate))

    # 3. Generate Dummy Dataset
    num_samples = 100
    batch_size = 10
    
    key = jax.random.key(0)
    k1, k2, k3 = jax.random.split(key, 3)
    X_data = jax.random.normal(k1, (num_samples, 3))  # Target network inputs
    Z_data = jax.random.normal(k2, (num_samples, 4))  # Hypernetwork latent inputs
    Y_data = jax.random.normal(k3, (num_samples, 2))  # Expected output

    # 4. Define Loss and Train Step
    def loss_fn(model, batch):
        outputs = model(batch)
        preds = outputs['final_output']
        # Simple Mean Squared Error
        loss = jnp.mean((preds - batch['y_true']) ** 2)
        return loss

    @nnx.jit
    def train_step(model, optim, batch):
        # nnx.value_and_grad extracts the state, computes the forward pass, 
        # and traces the gradients back to the respective nnx.Params.
        loss, grads = nnx.value_and_grad(loss_fn)(model, batch)
        optim.update(grads)
        return loss

    # 5. Training Loop
    epochs = 50
    print("\n" + "="*50)
    print("STARTING DUMMY TRAINING (Overfitting random data)")
    print("="*50)

    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = num_samples // batch_size
        
        for i in range(num_batches):
            start = i * batch_size
            end = start + batch_size
            
            # Construct the batched dictionary expected by the Manager
            batch = {
                'target_input': X_data[start:end],
                'latent_z': Z_data[start:end],
                'y_true': Y_data[start:end]
            }
            
            loss = train_step(manager, optimizer, batch)
            epoch_loss += loss.item()
            
        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0:
            avg_loss = epoch_loss / num_batches
            print(f"Epoch {epoch + 1:02d}/{epochs} - Loss: {avg_loss:.4f}")

if __name__ == "__main__":
    import os
    # CPU fallback for local testing
    if 'JAX_PLATFORMS' not in os.environ:
        try:
            from jax import devices
            if not any(d.platform == 'gpu' for d in devices()): 
                os.environ['JAX_PLATFORMS'] = 'cpu'
        except Exception:
            os.environ['JAX_PLATFORMS'] = 'cpu'
            
    train_dummy_model()