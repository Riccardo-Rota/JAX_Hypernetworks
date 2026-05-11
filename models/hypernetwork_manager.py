import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing import Dict, List, Optional, Callable, Union, Any, Tuple
import networkx
import warnings
import math 
from flax.traverse_util import flatten_dict, unflatten_dict
from utils import state_to_dict
from collections.abc import Mapping, Sequence

# TODO: ADD COMMENTS

class TargetNetworkWeight(nnx.Variable):
    pass

class OutputsNumberWarning(UserWarning):
    pass

class NeuralNetwork(nnx.Module):
    """
    """
    def __init__(self, network: Optional[nnx.Module], input: Optional[List[Union[str, Dict[str, str]]]] = None, output: Optional[Union[List[str], str]] = None):
        self.network = network
        
        self.input_args, self.input_kwargs = self._get_input(input)
        self.output = self._get_output(output)

    def __call__(self, *args, **kwargs):
        if self.network is None:
            raise RuntimeError(f"Network is None in {self.__class__.__name__}. Please build it first.")
        return self.network(*args, **kwargs)

    def _get_input(self, input: Any) -> Tuple[List[str], Dict[str, str]]:
        input_args = []
        input_kwargs = {}
        if isinstance(input, str):
            input_args = [input]
        elif isinstance(input, Mapping):
            input_kwargs = dict(input)        
        elif isinstance(input, Sequence):
            for item in input:
                if isinstance(item, str):
                    input_args.append(item)
                elif isinstance(item, Mapping):
                    input_kwargs.update(dict(item))
                else:
                    raise ValueError(f"Invalid item type in input list: {type(item)}. Must be string or dict.")
        elif input is not None:
            raise ValueError("Input must be a string, dict, or list of mixed args/kwargs.")
        return input_args, input_kwargs

    def _get_output(self, output: Optional[Union[List[str], str]]) -> List[str]:
        if isinstance(output, str):
            output = [output]
        return list(output) if output else []

class TargetNetwork(NeuralNetwork):
    def __init__(self, network: nnx.Module, input: Optional[List[Union[str, Dict[str, str]]]] = None, output: Optional[Union[List[str], str]] = None, weights: Optional[Dict[str, str]] = None, replace_weights: bool = True, name: Optional[str] = None):
        super().__init__(network, input, output)
        self.weights_mapping = weights
        self.replace_weights = replace_weights
        self.name = name
        self.freeze_weights() 
        self.target_graphdef, initial_state = nnx.split(self.network)
        self.base_state_dict = state_to_dict(initial_state)
        self._map_signals() 

    def freeze_weights(self):
        graphdef, state = nnx.split(self.network)
        frozen_state = jax.tree.map(
            lambda v: TargetNetworkWeight(v.value) if isinstance(v, nnx.Param) else v,
            state,
            is_leaf=lambda x: isinstance(x, nnx.Variable)
        )
        self.network = nnx.merge(graphdef, frozen_state)

    def unfreeze_weights(self):
        graphdef, state = nnx.split(self.network)
        unfrozen_state = jax.tree.map(
            lambda v: nnx.Param(v.value) if isinstance(v, TargetNetworkWeight) else v,
            state,
            is_leaf=lambda x: isinstance(x, nnx.Variable)
        )
        self.network = nnx.merge(graphdef, unfrozen_state)
        self.target_graphdef, new_initial_state = nnx.split(self.network)
        self.base_state_dict = state_to_dict(new_initial_state)

    def _map_signals(self):
        leaves_with_paths, self.treedef = jax.tree_util.tree_flatten_with_path(
            self.base_state_dict, 
            is_leaf=lambda x: isinstance(x, nnx.Variable)
        )
        
        self.leaf_shapes = []
        self.signal_to_leaf_indices = {}
        self.signal_to_weight_size = {}
        
        for i, (path, leaf) in enumerate(leaves_with_paths):
            shape = leaf.value.shape if hasattr(leaf, 'value') else leaf.shape
            self.leaf_shapes.append(shape)
            
            if self.weights_mapping:
                if 'all' in self.weights_mapping:
                    signal = self.weights_mapping['all']
                    self.signal_to_leaf_indices.setdefault(signal, []).append(i)
                    self.signal_to_weight_size[signal] = self.signal_to_weight_size.get(signal, 0) + math.prod(shape)
                else:
                    path_keys = [str(getattr(p, 'key', getattr(p, 'idx', p))) for p in path]
                    
                    for weight_key, signal in self.weights_mapping.items():
                        if weight_key in path_keys:
                            self.signal_to_leaf_indices.setdefault(signal, []).append(i)
                            self.signal_to_weight_size[signal] = self.signal_to_weight_size.get(signal, 0) + math.prod(shape)

    def _inject_weights(self, state_dict: dict, weights: dict[str, jnp.ndarray]) -> dict:
        leaves = jax.tree_util.tree_leaves(
            state_dict, 
            is_leaf=lambda x: isinstance(x, nnx.Variable)
        )
        
        new_leaves = list(leaves)
        
        for signal, flat_array in weights.items():
            if signal not in self.signal_to_leaf_indices:
                continue
                
            indices = self.signal_to_leaf_indices[signal]
            is_batched = flat_array.ndim > 1
            batch_size = flat_array.shape[0] if is_batched else None
            
            current_idx = 0
            for i in indices:
                shape = self.leaf_shapes[i]
                num_elements = math.prod(shape)
                
                if is_batched:
                    flat_slice = flat_array[:, current_idx : current_idx + num_elements]
                    target_shape = (batch_size,) + shape
                else:
                    flat_slice = flat_array[current_idx : current_idx + num_elements]
                    target_shape = shape
                    
                reshaped_weight = jnp.reshape(flat_slice, target_shape)
                old_var = leaves[i]
                
                if hasattr(old_var, 'value'): 
                    new_val = reshaped_weight if self.replace_weights else (old_var.value + reshaped_weight)
                    if hasattr(old_var, 'type'):
                        new_leaves[i] = type(old_var)(old_var.type, new_val)
                    else:
                        new_leaves[i] = type(old_var)(new_val)
                else: 
                    new_val = reshaped_weight if self.replace_weights else (old_var + reshaped_weight)
                    new_leaves[i] = new_val
                    
                current_idx += num_elements
                
        return jax.tree_util.tree_unflatten(self.treedef, new_leaves)
    
    def __call__(self, *args, weights: Optional[Dict[str, Any]] = None, **kwargs):
        if weights is not None:
            state_dict = self._inject_weights(self.base_state_dict, weights)
            new_state = nnx.State(state_dict)
            
            is_batched = any(w.ndim > 1 for w in weights.values())
            
            if is_batched:
                modified_network = nnx.merge(self.target_graphdef, new_state)
                vmap_forward = nnx.vmap(
                    lambda net, a, kw: net(*a, **kw), 
                    in_axes=(nnx.StateAxes({nnx.Variable: 0}), 0, 0)
                )
                return vmap_forward(modified_network, args, kwargs)
            else:
                modified_network = nnx.merge(self.target_graphdef, new_state) 
                return modified_network(*args, **kwargs)
        else:
            return self.network(*args, **kwargs)

class Hypernetwork(NeuralNetwork):
    "Hypernetwork class that generates a output representing a latent space, to be used by a ProjectionHead."
    pass


class ProjectionHead(NeuralNetwork):
    def __init__(self, 
                 in_features: int, 
                 input: Optional[List[Union[str, Dict[str, str]]]],
                 output: str,
                 rngs: Optional[nnx.Rngs] = None,
                 kernel_init: Callable = nnx.initializers.lecun_normal(),
                 bias_init: Callable = nnx.initializers.zeros_init()):
        
        super().__init__(network=None, input=input, output=output)
        self.in_features = in_features
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
    def __init__(self, blocks: List[NeuralNetwork], output: Union[List[str], str]):
        blocks = self._build_projection_heads(blocks) 
        self.blocks = self._determine_execution_order(blocks)
        self.output = [output] if isinstance(output, str) else (output or [])

    def __call__(self, inputs: dict[str, jnp.ndarray]) -> Dict[str, Any]:
        objects = inputs.copy() 
        
        for block in self.blocks:
            input_args = [objects[signal] for signal in block.input_args]
            input_kwargs = {name: objects[key] for name, key in block.input_kwargs.items()}
            
            if isinstance(block, TargetNetwork) and block.weights_mapping:
                unique_signals = set(block.weights_mapping.values())
                if 'weights' in input_kwargs:
                    raise ValueError("The reserved keyword 'weights' cannot be used as an input mapping key for a TargetNetwork block.")
                input_kwargs['weights'] = {signal: objects[signal] for signal in unique_signals}

            y = block(*input_args, **input_kwargs)
            block_output_names = block.output or [] 
            if not isinstance(y, tuple):
                y = (y,)
            
            if len(block_output_names) != len(y):
                warnings.warn(f"Block '{type(block).__name__}' produced {len(y)} outputs but has {len(block_output_names)} output names defined.")
            
            for i in range(min(len(block_output_names), len(y))):
                objects[block_output_names[i]] = y[i]

        out_values = [objects[name] for name in self.output]
        return out_values[0] if len(out_values) == 1 else tuple(out_values)
    
    def _build_projection_heads(self, blocks: List[NeuralNetwork]):
        signal_to_weight_size = {}
        for block in blocks:
            if isinstance(block, TargetNetwork) and block.weights_mapping:
                for signal, size in block.signal_to_weight_size.items():
                    if signal in signal_to_weight_size and signal_to_weight_size[signal] != size:
                        raise ValueError(f"Signal '{signal}' is used by two different weight sets with conflicting sizes.")
                    signal_to_weight_size[signal] = size
        for block in blocks:
            if isinstance(block, ProjectionHead):
                if not block.output or len(block.output) != 1:
                    raise ValueError("ProjectionHead must have exactly one output mapped.")
                
                out_signal = block.output[0]
                if out_signal in signal_to_weight_size:
                    block.build(out_features=signal_to_weight_size[out_signal])
                else:
                    warnings.warn(f"Head output '{out_signal}' is unused. Building with size 1.")
                    block.build(out_features=1)
        return blocks
    
    def _determine_execution_order(self, blocks: List[NeuralNetwork]) -> List[NeuralNetwork]:
        graph = networkx.DiGraph() 
        outputs_producers = {} 
        for i, block in enumerate(blocks): 
            graph.add_node(i) 
            if block.output:
                for output_node in block.output:
                    outputs_producers[output_node] = i
        for i, block in enumerate(blocks): 
            deps = []
            deps.extend(block.input_args) 
            deps.extend(block.input_kwargs.values()) 
            if isinstance(block, TargetNetwork) and block.weights_mapping:
                deps.extend(block.weights_mapping.values())
            for d in deps:
                if d in outputs_producers:
                    graph.add_edge(outputs_producers[d], i)

        try: 
            sorted_indices = list(networkx.topological_sort(graph))
        except networkx.NetworkXUnfeasible:
            raise ValueError("A circular dependency was detected in the graph.")

        ordered_blocks = [blocks[i] for i in sorted_indices]
        return ordered_blocks
    
    def extract_target_network(self, inputs: dict[str, jnp.ndarray]) -> Dict[str, nnx.Module]:
        """
        Executes the manager to extract the TargetNetwork with injected weights as a standalone nnx.Module.
        """
        # determine target networks and the signals they require, to only execute the necessary blocks
        target_networks = [b for b in self.blocks if isinstance(b, TargetNetwork)]
        if not target_networks:
            raise ValueError("No TargetNetwork blocks found in the manager.")
        all_required_signals = set()
        for tn in target_networks:
            if tn.weights_mapping:
                all_required_signals.update(tn.weights_mapping.values())

        # execute blocks to get the needed signals
        objects = inputs.copy()
        for block in self.blocks:
            if all_required_signals and all_required_signals.issubset(objects.keys()): # ends the loop early if all signals needed have been generated
                break
            if isinstance(block, TargetNetwork): # Only execute if needed to generate weight for another TargetNetwork
                is_generator = any(out in all_required_signals for out in (block.output or []))
                if not is_generator:
                    continue 
                required_signals = set(block.weights_mapping.values()) if block.weights_mapping else set()                
                block_weights = {signal: objects[signal] for signal in required_signals}
                input_args = [objects[s] for s in block.input_args]
                input_kwargs = {k: objects[v] for k, v in block.input_kwargs.items()}
                y = block(*input_args, weights=block_weights if block_weights else None, **input_kwargs)
            else: # Standard block execution (Hypernetwork, ProjectionHead, etc.)
                input_args = [objects[s] for s in block.input_args]
                input_kwargs = {k: objects[v] for k, v in block.input_kwargs.items()}
                y = block(*input_args, **input_kwargs)
            y = (y,) if not isinstance(y, tuple) else y
            for i, name in enumerate(block.output or []): # Store signals
                if i < len(y):
                    objects[name] = y[i]

        extracted_modules = {}
        for i, target_block in enumerate(target_networks): # inject weights and unfeeze them to make them trainable
            required_signals = set(target_block.weights_mapping.values()) if target_block.weights_mapping else set()
            weights = {signal: objects[signal] for signal in required_signals}
            new_state = nnx.State(target_block._inject_weights(target_block.base_state_dict, weights))
            unfrozen_state = jax.tree.map(
                lambda v: nnx.Param(v.value) if isinstance(v, TargetNetworkWeight) else v,
                new_state,
                is_leaf=lambda x: isinstance(x, nnx.Variable)
            )
            network = nnx.merge(target_block.target_graphdef, unfrozen_state)            
            
            block_name = getattr(target_block, 'name', None)
            if not block_name: # If no name is provided, generate a unique one
                block_name = f"target_network_{i}"
            while block_name in extracted_modules: # Ensure unique block names
                block_name += "_duplicate"
                
            extracted_modules[block_name] = network
            
        return extracted_modules