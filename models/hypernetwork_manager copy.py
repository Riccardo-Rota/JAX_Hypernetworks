import flax.nnx as nnx
import jax.numpy as jnp
from typing import Dict, List, Optional, Callable, Sequence, Union, Any
import networkx
import warnings
import math 
from flax.traverse_util import flatten_dict, unflatten_dict 


# THIS IS THE "SMARTER" VERSION. CHECK IF IT MAKES SENSE

def set_nested_dict(d: dict, path: str, value: Any): 
    """
    Sets a value in a nested dictionary given a dot-separated path.
    Args:
    d (dict): The dictionary to modify.
    path (str): The dot-separated path to the key to set (e.g., 'layer1.weight').
    value (Any): The value to set at the specified path.
    """
    keys = path.split('.') 
    for key in keys[:-1]: 
        if key not in d:
            raise KeyError(f"Failed to set weight: '{key}' not found in path '{path}'. Check your config.")
        d = d[key] 
    if keys[-1] not in d:
        raise KeyError(f"Failed to set weight: '{keys[-1]}' not found in path '{path}'. Check your config.")
    d[keys[-1]] = value

class OutputsNumberWarning(UserWarning):
    pass

class NeuralNetwork(nnx.Module):
    """
    Base class for neural network modules that includes metadata about input and output mappings for use in the HypernetworkManager."""
    def __init__(self, network: nnx.Module, input_mapping: Optional[Dict[str, str]], output_mapping: Optional[List[str]]):
        """
        Args:
            network (nnx.Module): The neural network module to be wrapped.
            input_mapping (Optional[Dict[str, str]]): A dictionary mapping argument names of the network's forward method to the keys in the input dictionary that will be passed to the HypernetworkManager. For example, if the network's forward method has an argument 'x', and we want to pass the value from the input dictionary with key 'variables', then we would have {'x': 'variables'}.
            output_mapping (Optional[List[str]]): A list of strings representing the keys under which the outputs of this network will be stored in the HypernetworkManager's output dictionary. The order of this list should match the order of outputs returned by the network's forward method.
        """
        self.network = network
        self.input_mapping = input_mapping
        self.output_mapping = output_mapping

    def __call__(self, x):
        return self.network(x)
    

class TargetNetwork(NeuralNetwork):
    def __init__(self, network: nnx.Module, input_mapping: Optional[Dict[str, str]], output_mapping: List[str], weights: Optional[Dict[str, str]] = None, replace_weights: bool = True):
        super().__init__(network, input_mapping, output_mapping)
        self.weights_mapping = weights
        self.replace_weights = replace_weights

        # Pre-compute shapes and total sizes needed for each signal
        target_state = nnx.state(network)
        self.flat_state_shapes = {
            k: (v.value.shape if hasattr(v, 'value') else v.shape)
            for k, v in flatten_dict(target_state.to_dict(), sep='.').items()
        }

        self.signal_to_paths = {}  # Maps signal_name -> list of paths it must fill
        self.required_elements = {} # Maps signal_name -> total flat elements needed

        if self.weights_mapping:
            if 'all' in self.weights_mapping:
                signal = self.weights_mapping['all']
                self.signal_to_paths[signal] = list(self.flat_state_shapes.keys())
                self.required_elements[signal] = sum(math.prod(s) for s in self.flat_state_shapes.values())
            else:
                for path, signal in self.weights_mapping.items():
                    if path not in self.flat_state_shapes:
                        raise ValueError(f"Path '{path}' not found in TargetNetwork.")
                    self.signal_to_paths.setdefault(signal, []).append(path)
                    shape = self.flat_state_shapes[path]
                    self.required_elements[signal] = self.required_elements.get(signal, 0) + math.prod(shape)

    def __call__(self, x: jnp.ndarray, weights: Optional[Dict[str, Any]] = None):
        if weights is not None and self.replace_weights:
            graphdef, state = nnx.split(self.network)
            state_dict = state.to_dict()

            # The manager now passes {signal_name: massive_flat_tensor}
            for signal, flat_array in weights.items():
                if signal not in self.signal_to_paths:
                    continue
                
                paths = self.signal_to_paths[signal]
                is_batched = flat_array.ndim > 1
                batch_size = flat_array.shape[0] if is_batched else None

                current_idx = 0
                for path in paths:
                    shape = self.flat_state_shapes[path]
                    num_elements = math.prod(shape)

                    # Slice the massive array chunk by chunk
                    if is_batched:
                        flat_slice = flat_array[:, current_idx : current_idx + num_elements]
                        target_shape = (batch_size,) + shape
                    else:
                        flat_slice = flat_array[current_idx : current_idx + num_elements]
                        target_shape = shape

                    # Reshape and inject into the dictionary
                    reshaped_weight = jnp.reshape(flat_slice, target_shape)
                    set_nested_dict(state_dict, path, reshaped_weight)
                    current_idx += num_elements

            new_state = nnx.State(state_dict)
            modified_network = nnx.merge(graphdef, new_state) 
            return modified_network(x)
            
        if weights is not None and not self.replace_weights:
            raise NotImplementedError("Currently only weight replacement is implemented.")
        return self.network(x)
            

class Hypernetwork(NeuralNetwork):
    "Hypernetwork class that generates a output representing a latent space, to be used by a ProjectionHead to produce weights for the TargetNetwork."
    pass

class ProjectionHead(NeuralNetwork):
    def __init__(self, 
                 in_features: int, 
                 input_mapping: Optional[Dict[str, str]] = None, 
                 output_mapping: Optional[List[str]] = None,
                 kernel_init: Callable = nnx.initializers.lecun_normal(),
                 bias_init: Callable = nnx.initializers.zeros_init()):
        
        # We don't initialize the nnx.Linear layer yet!
        super().__init__(network=None, input_mapping=input_mapping, output_mapping=output_mapping)
        self.in_features = in_features
        self.kernel_init = kernel_init
        self.bias_init = bias_init

    def build(self, out_features: int):
        """Called by the Manager to finalize the layer once dimensions are known."""
        self.network = nnx.Linear(
            in_features=self.in_features, 
            out_features=out_features, 
            kernel_init=self.kernel_init, 
            bias_init=self.bias_init
        )

    def __call__(self, x: jnp.ndarray):
        if self.network is None:
            raise RuntimeError("ProjectionHead was executed before being built by the Manager.")
        return self.network(x)



class HypernetworkManager(nnx.Module):
    def __init__(self, blocks: List[NeuralNetwork]):
        # 1. CALCULATE REQUIRED DIMENSIONS & BUILD HEADS
        signal_requirements = {}
        for block in blocks:
            if isinstance(block, TargetNetwork) and block.weights_mapping:
                for signal, req_size in block.required_elements.items():
                    if signal in signal_requirements and signal_requirements[signal] != req_size:
                        raise ValueError(f"Signal '{signal}' is requested with conflicting sizes.")
                    signal_requirements[signal] = req_size

        for block in blocks:
            if isinstance(block, ProjectionHead):
                if not block.output_mapping or len(block.output_mapping) != 1:
                    raise ValueError("ProjectionHead must have exactly one output mapped.")
                
                out_signal = block.output_mapping[0]
                if out_signal in signal_requirements:
                    block.build(out_features=signal_requirements[out_signal])
                else:
                    warnings.warn(f"Head output '{out_signal}' is unused. Building with size 1.")
                    block.build(out_features=1)

        # 2. STANDARD GRAPH BUILDING
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
                # We now pass unique signals directly, e.g., {'head_A_out': <massive_tensor>}
                unique_signals = set(block.weights_mapping.values())
                input_kwargs['weights'] = {signal: objects[signal] for signal in unique_signals}

            y = block(**input_kwargs)
            output_names = block.output_mapping or [] 
            if not isinstance(y, tuple):
                y = (y,)
            
            for i in range(min(len(output_names), len(y))):
                objects[output_names[i]] = y[i]

        return objects