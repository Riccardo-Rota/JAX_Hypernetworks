import re

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing import Dict, List, Optional, Callable, Union, Any, Tuple, Set
import networkx
import warnings
import math
from utils import state_to_dict
from collections.abc import Mapping, Sequence


class TargetNetworkWeight(nnx.Variable):
    """
    Wrapper class for target network weights to distinguish them from regular trainable parameters.
    """
    pass

class InjectedBatchedWeights(nnx.Variable):
    """
    Wrapper class for injected weights that are batched, to distinguish them from regular trainable parameters and unbatched injected weights.
    This is useful for the forward method of TargetNetwork to determine whether to apply vmap for batch processing during weight injection.
    """
    pass

class InjectedUnbatchedWeights(nnx.Variable):
    """
    Wrapper class for injected weights that are unbatched, to distinguish them from regular trainable parameters and batched injected weights.
    This is useful for the forward method of TargetNetwork to determine whether to apply vmap for batch processing during weight injection.
    """
    pass


class OutputsNumberWarning(UserWarning):
    """
    Warning raised when number of outputs produced by a block does not match the number of output keys defined for that block.
    """
    pass


class NeuralNetwork(nnx.Module):
    """
    Base class for Neural Networks managed by HypernetworkManager.
    Each network can have a mapping of its inputs and outputs to object keys, which are used by the manager to route data between blocks.
    """
    def __init__(self, network: Optional[nnx.Module], input: Optional[Union[str, Dict[str, str], List[Union[str, Dict[str, str]]]]] = None, output: Optional[Union[List[str], str]] = None):
        """
        Args:
        network: nnx.Module representing the neural network architecture.
        input: mapping of objects to be passed as input to the network.
            If objects are provided as strings, they will be passed as positional arguments
            If objects are provided as dict, they will be passed as keyword arguments, with the dict key being the argument keyword
        output: list of object keys or single object key that the network outputs. The order of the list should match the order of the outputs produced by the network's forward pass.
        """
        self.network = network
        self.input_args, self.input_kwargs = self._get_input(input)
        self.output = self._get_output(output)

    def __call__(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Forward pass through the network.
        Returns the output(s) of the network in a dict mapping output keys to corresponding values.
        """
        if self.network is None:
            raise RuntimeError(f"Network is None in {self.__class__.__name__}. Please build it first.")
        return self.network(*args, **kwargs)

    def _get_input(self, input: Any) -> Tuple[List[str], Dict[str, str]]:
        """
        Format the input object(s) as a list of positional arguments and a dictionary of keyword arguments.
        This implementation is flexible to allow for different input formats, and specifically hydra friendly.
        The input formatting mimics the args, kwargs structure of python functions.
        """
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
        """
        Format the output key(s) as a list of strings
        """
        if isinstance(output, str):
            output = [output]
        return list(output) if output else []


class TargetNetwork(NeuralNetwork):
    """
    TargetNetwork class that allows weights injection, based on a mapping provided at initialization.
    The weights are set to non-trainable variables, because they are meant to be optimized by a hypernetwork and not directly. 
    The extract_target_network method in the manager allows to extract the TargetNetwork with injected weights as a standalone nnx.Module, converting the weights back to trainable parameters for fine-tuning.
    """
    def __init__(self, network: nnx.Module, input:Optional[Union[str, Dict[str, str], List[Union[str, Dict[str, str]]]]] = None, output: Optional[Union[List[str], str]] = None, weights_mapping: Optional[Dict[Union[str, Tuple[str]], str]] = None, name: Optional[str] = None):
        """
        Args:
        network: nnx.Module representing the neural network architecture.
        input: mapping of objects to be passed as input to the network.
            If inputs are provided as strings, they will be passed as positional arguments
            If inputs are provided as dict, they will be passed as keyword arguments, with the dict key being the argument keyword
        output: list of output keys or single output key that the network outputs. The order of the list should match the order of the outputs produced by the network's forward pass.
        weights_mapping: Dict mapping state variable identifiers (e.g. layer keys, parameter types) to object keys for weight injection. 
            Use 'all' as key if a single object represents all weights of the network (the dimension of the object will be inferred as the total number of parameters in the network).
            Use tuples as keys if a single object represents weights for multiple layers (the dimension of the object will be inferred as the cumulative number of parameters in the layers).
            Use single strings as keys to map each object to a layer.
        name: Optional name for the TargetNetwork block, used for identification when extracting the target network from the manager.
        """
        super().__init__(network, input, output)
        self.weights_mapping = weights_mapping # mapping weight keys to the objects that will be injected in them
        self.name = name
        self.graphdef, self.state = nnx.split(self.network)
        self.state_dict = state_to_dict(self.state)
        self.freeze_weights() 
        # information about flat tree structure and leaf keys
        self.pytree_layout, self.flat_leaves, self.leaf_registry = self._build_leaf_registry()
        # mapping of objects to the corresponding weight paths and sizes for injection
        self.object_registry = self._build_object_registry()
        
    def freeze_weights(self):
        """ 
        Converts the network state weights into TargetNetworkWeight to prevent them from being updated during training. 
        """
        frozen_state = jax.tree.map(
            lambda v: TargetNetworkWeight(v.value) if isinstance(v, nnx.Param) else v,
            self.state,
            is_leaf=lambda x: isinstance(x, nnx.Variable)
        )
        self.state = frozen_state
        self.state_dict = state_to_dict(self.state)
        self.network = nnx.merge(self.graphdef, self.state)

    def unfreeze_weights(self):
        """ 
        Converts the network state weights back into nnx.Param to allow them to be updated during training.
        Usecase: after optimizing the hypernetwork, we want to extract the target network and fine tune it 
        """
        unfrozen_state = jax.tree.map(
            lambda v: nnx.Param(v.value) if isinstance(v, (TargetNetworkWeight, InjectedBatchedWeights, InjectedUnbatchedWeights)) else v,
            self.state,
            is_leaf=lambda x: isinstance(x, nnx.Variable)
        )
        self.state = unfrozen_state
        self.state_dict = state_to_dict(self.state)
        self.network = nnx.merge(self.graphdef, self.state)

    def _build_leaf_registry(self):
        """
        Builds a mapping of the flattened state leaves to their corresponding paths in the pytree, for easy access during weight injection.
        This is necessary because the weights to be injected are provided as flat arrays, so we need to know the order and shape of the weights in the state to correctly inject them.
        """
        leaves_with_paths, pytree_layout = jax.tree_util.tree_flatten_with_path(
            self.state_dict, 
            is_leaf=lambda x: isinstance(x, nnx.Variable)
        )
        flat_leaves = []
        leaf_registry = {}
        for i, (path, leaf) in enumerate(leaves_with_paths):
            shape = leaf.value.shape if hasattr(leaf, 'value') else leaf.shape
            path_keys = [str(getattr(p, 'key', getattr(p, 'name', getattr(p, 'idx', p)))) for p in path]
            path_string = "/".join(path_keys)
            leaf_registry[path_string] = {"leaf": leaf, "shape": shape, "idx": i}
            flat_leaves.append(leaf)
        
        return pytree_layout, flat_leaves, leaf_registry

    def _build_object_registry(self):
        """
        Maps the objects to the corresponding state variables and sizes for weight injection.
        Also stores the flattened state leaves and their paths for easy access during injection, and the pytree layout for reconstructing the state after injection.
        """
        object_registry = {} # mapping of object keys to their corresponding weight paths and sizes
        mapped_paths = set() # keeps track of paths that have already been mapped to an object, to avoid conflicts in the mapping
        
        if self.weights_mapping is None:
            return

        if 'all' in self.weights_mapping:
            # replace 'all' with the tuple containing all paths
            object_key = self.weights_mapping['all']
            all_paths = []
            for path_string, values in self.leaf_registry.items():
                all_paths.append(path_string)
            self.weights_mapping = {tuple(all_paths): object_key}

        for weight_keys, object_key in self.weights_mapping.items():
            if not isinstance(weight_keys, tuple): # ensure weight_keys is a tuple for consistent processing
                weight_keys = (weight_keys,)
            # find all paths that match the weight keys, and check for conflicts in the mapping
            matching_paths = []
            for weight_key in weight_keys:
                pattern = re.compile(rf'(?:^|/){re.escape(weight_key)}(?:/|$)') # Example: 'layer1' matches 'layer1/...' or '.../layer1/...' or '.../layer1' but not 'layer10'
                for path_string in self.leaf_registry.keys():
                    if pattern.search(path_string):
                        if path_string in mapped_paths:
                            raise ValueError(f"Overlapping keys: '{path_string}' is mapped more than once in weights_mapping. Please check the mapping to avoid conflicts.")
                        matching_paths.append(path_string)
                        mapped_paths.add(path_string)
            current_idx = 0
            object_metadata = {'size': 0,
                                'leaf_ids': [],
                                'shapes': [],
                                'slice_ids': []}
            if not matching_paths:
                raise ValueError(f"No matching paths found for weight key(s) '{weight_keys}' in the network state. Please check the mapping.")
            for path_string in matching_paths:
                leaf_idx = self.leaf_registry[path_string]['idx']
                shape = self.leaf_registry[path_string]['shape']
                size = math.prod(shape)
                object_metadata['size'] += size
                object_metadata['leaf_ids'].append(leaf_idx)
                object_metadata['shapes'].append(shape)
                object_metadata['slice_ids'].append((current_idx, current_idx + size))
                current_idx += size
            if object_key in object_registry:
                # check if the weight key already exists in the leaf registry. In this case, we only accept it if the size corresponds to the existing one
                if object_registry[object_key]['size'] != object_metadata['size']:
                    raise ValueError(f"'{object_key}' is mapped to multiple weight sets with different sizes. Please check the mapping to avoid conflicts.")
                # if size matches, we allow extending the existing metadata for this object key, to have the same object injected in multiple sets of weights (e.g. for weight sharing)
                object_registry[object_key]['leaf_ids'].extend(object_metadata['leaf_ids'])
                object_registry[object_key]['shapes'].extend(object_metadata['shapes'])
                object_registry[object_key]['slice_ids'].extend(object_metadata['slice_ids'])
            else:
                # add the mapping for the object key if it does not exist yet
                object_registry[object_key] = object_metadata
        
        return object_registry
    
    def _inject_weights(self, weights: dict[str, jnp.ndarray]) -> dict:
        """
        Injects the weights from the provided objects into the network state according to the mapping defined in self.object_destinations.
        Args:
            weights: dict mapping object keys to their corresponding weight arrays. The arrays can be either 2D (batched) or 1D (unbatched).
        Returns:
            new_state_dict: dict representing the new state with injected weights.
        """
        new_leaves = list(self.flat_leaves)
        
        for object_key, flat_array in weights.items():
            if object_key not in self.object_registry:
                # HypernetworkManager ensures we never get this error. We keep it as a safety check
                raise ValueError(f"Object key '{object_key}' not found in registry.")
                
            object_metadata = self.object_registry[object_key]
            is_batched = flat_array.ndim > 1
            batch_size = flat_array.shape[0] if is_batched else None
            
            for leaf_idx, shape, (start, end) in zip(
                object_metadata['leaf_ids'], 
                object_metadata['shapes'], 
                object_metadata['slice_ids']
            ):
                if is_batched:
                    flat_slice = flat_array[:, start:end]
                    target_shape = (batch_size,) + shape
                    new_weights = jnp.reshape(flat_slice, target_shape)
                    new_leaves[leaf_idx] = InjectedBatchedWeights(new_weights)
                else:
                    flat_slice = flat_array[start:end]
                    target_shape = shape
                    new_weights = jnp.reshape(flat_slice, target_shape)
                    new_leaves[leaf_idx] = InjectedUnbatchedWeights(new_weights)

        return jax.tree_util.tree_unflatten(self.pytree_layout, new_leaves)
    
    def __call__(self, *args, weights: Optional[Dict[str, Any]] = None, unbatched_keys: Optional[Tuple[str, ...]] = None, **kwargs) -> Dict[str, Any]:
        """
        Forward pass with optional weight injection. 
        If weights are provided, they are injected into the network state according to the mapping before executing the forward pass. 
        Args:
            weights: dict mapping object keys to their corresponding weight arrays for injection. The arrays can be either 2D (batched) or 1D (unbatched).
            unbatched_keys: tuple of object keys that should be treated as unbatched.
        Returns:
            A dictionary mapping output keys to their corresponding values.
        """
        if weights is not None:
            state_dict = self._inject_weights(weights)
            new_state = nnx.State(state_dict)
            is_batched = any(w.ndim > 1 for w in weights.values())

            if is_batched:
                state_axes = jax.tree.map(
                    lambda v: 0 if isinstance(v, InjectedBatchedWeights) else None,
                    new_state,
                    is_leaf=lambda x: isinstance(x, nnx.Variable)
                )
                
                if unbatched_keys:
                    args_axes = tuple(
                        None if name in unbatched_keys else 0
                        for name in self.input_args
                    )
                    
                    kwargs_axes = {
                        kw: (None if object_key in unbatched_keys else 0)
                        for kw, object_key in self.input_kwargs.items()
                    }
                else:
                    args_axes = 0
                    kwargs_axes = 0
                    
                @nnx.vmap(in_axes=(state_axes, args_axes, kwargs_axes)) 
                def vmap_forward(state, args, kwargs):
                    modified_network = nnx.merge(self.graphdef, state)
                    return modified_network(*args, **kwargs)
                    
                return vmap_forward(new_state, args, kwargs)
            
            else:
                modified_network = nnx.merge(self.graphdef, new_state)
                return modified_network(*args, **kwargs)
        else:
            return self.network(*args, **kwargs)
        

class Hypernetwork(NeuralNetwork):
    """
    Hypernetwork class that generates a output representing a latent representation, to be used by a ProjectionHead to generate weights.
    Args:
        network: nnx.Module representing the neural network architecture.
        input: mapping of objects to be passed as input to the network.
            If objects are provided as strings, they will be passed as positional arguments
            If objects are provided as dict, they will be passed as keyword arguments, with the dict key being the argument keyword
        output: list of object keys or single object key that the network outputs. The order of the list should match the order of the outputs produced by the network's forward pass.
    """
    # Hypernetwork is a standard neural network, it does not require any special handling beyond the base NeuralNetwork class.
    # We define it for semantic clarity and consistency in the HypernetworkManager code.
    pass


class ProjectionHead(NeuralNetwork):
    """
    ProjectionHead class that generates weights for the TargetNetwork based on the output of the Hypernetwork, defined as a single linear layer.
    The head is instantiated only when build method is called, which allows to determine the output size based on the size of the weights it needs to generate.
    """
    def __init__(self, 
                 in_features: int, 
                 input: Optional[List[Union[str, Dict[str, str]]]],
                 output: str,
                 rngs: Optional[nnx.Rngs] = None,
                 kernel_init: Callable = nnx.initializers.lecun_normal(),
                 bias_init: Callable = nnx.initializers.zeros_init()):
        """
        Args:
        in_features: number of input features for the linear layer. This should match the output size of the corresponding Hypernetwork.
        input: Single input object key
        output: Single output object key that this head generates
        rngs: Optional nnx.Rngs for initializing the head weights. If not provided, a default rng will be used.
        kernel_init: Optional weight initialization function for the linear layer kernel. Defaults to lecun_normal initializer.
        bias_init: Optional weight initialization function for the linear layer bias. Defaults to zeros initializer.
        """
        super().__init__(network=None, input=input, output=output)
        self.in_features = in_features
        self.rngs = rngs if rngs is not None else nnx.Rngs(0) 
        self.kernel_init = kernel_init
        self.bias_init = bias_init

    def build(self, out_features: int):
        """"
        Builds the linear layer of the head based on the provided output size.
        """
        self.network = nnx.Linear(
            in_features=self.in_features, 
            out_features=out_features, 
            kernel_init=self.kernel_init, 
            bias_init=self.bias_init,
            rngs=self.rngs
        )
    
    def clone(self) -> 'ProjectionHead':
        """ 
        Return an unbuilt copy of the head 
        """
        return ProjectionHead(
            in_features=self.in_features,
            input=self.input_args.copy(),
            output=self.output.copy(),
            rngs=self.rngs,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init
        )


class HypernetworkManager(nnx.Module):
    """
    HypernetworkManager class that manages the execution of multiple blocks (Hypernetworks, ProjectionHeads, TargetNetworks) in a directed acyclic graph (DAG) structure, based on their input and output dependencies.
    The manager determines the execution order of the blocks, routes the data through them, and collects the final outputs based on the defined output keys.
    """
    def __init__(self, blocks: List[NeuralNetwork], output: Union[List[str], str]):
        """
        Args:
        blocks: List of NeuralNetwork blocks, including Hypernetworks, TargetNetworks, ProjectionHeads
        output: list of output keys or single output key that the manager outputs. The order of the list should match the order of the outputs produced by the manager's forward pass.
        """
        blocks = self._build_projection_heads(blocks) # initialize all the heads 
        self.blocks, self.external_inputs = self._determine_execution_order(blocks)
        self.output = [output] if isinstance(output, str) else output

    def __call__(self, inputs: dict[str, jnp.ndarray], unbatched_keys: Optional[Tuple[str, ...]] = None) -> Union[jnp.ndarray, Tuple[jnp.ndarray, ...]]:
        """
        Executes the forward pass through the manager by routing the data through the blocks in the correct order, based on the defined inputs and outputs of each block.
        Args:            
            inputs: dict mapping input keys to their corresponding input tensors.
            unbatched_keys: Optional tuple of input keys that should not be considered as batched.
        Returns:            
            Union[jnp.ndarray, Tuple[jnp.ndarray, ...]]: The final output tensor(s) based on the defined output key(s).
        """
        objects = inputs.copy() 

        # check all required external inputs are provided
        if not set(self.external_inputs).issubset(objects.keys()):
            missing_inputs = self.external_inputs - set(objects.keys())
            raise ValueError(f"Missing external inputs for the manager: {missing_inputs}")
        
        # execute blocks in order
        for block in self.blocks:
            # get input args and kwargs
            input_args = [objects[object_key] for object_key in block.input_args]
            input_kwargs = {kw: objects[object_key] for kw, object_key in block.input_kwargs.items()}
            
            # for target networks, build the weights dict for injection
            if isinstance(block, TargetNetwork) and block.weights_mapping is not None:
                weight_objects = set(block.weights_mapping.values())
                if 'weights' in input_kwargs:
                    raise ValueError("The reserved keyword 'weights' cannot be used as an input mapping key for a TargetNetwork block.")
                input_kwargs['weights'] = {object_key: objects[object_key] for object_key in weight_objects}
                input_kwargs['unbatched_keys'] = unbatched_keys


            # run the forward pass of the block (for target networks, the weights will be injected in the forward method)
            y = block(*input_args, **input_kwargs)

            # store outputs in objects dict for routing to next blocks
            block_output_keys = block.output or [] 
            if not isinstance(y, tuple):
                y = (y,)
            if len(block_output_keys) != len(y):
                warnings.warn(f"Block '{type(block).__name__}' produced {len(y)} outputs but has {len(block_output_keys)} output keys defined.")
            for i in range(min(len(block_output_keys), len(y))):
                objects[block_output_keys[i]] = y[i]

        # gather and return the final outputs to be returned
        out_values = [objects[output_key] for output_key in self.output]
        return out_values[0] if len(out_values) == 1 else tuple(out_values)
    
    def _build_projection_heads(self, blocks: List[NeuralNetwork]) -> List[NeuralNetwork]:
        """
        Builds all the ProjectionHead blocks based on the sizes of the weights they need to generate.
        Returns a new list of blocks with the built ProjectionHeads, ready for execution.
        Args:
            blocks: List of NeuralNetwork blocks, including Hypernetworks, TargetNetworks, ProjectionHeads
        Returns:
            List of NeuralNetwork blocks with built ProjectionHeads.
        """
        weight_sizes = {}
        # populate weight_sizes to determine the output size of each projection head to be built
        for block in blocks:
            if isinstance(block, TargetNetwork) and block.weights_mapping is not None:
                for object_key, metadata in block.object_registry.items():
                    if object_key in weight_sizes and weight_sizes[object_key] != metadata['size']: # check for conflicting sizes for the same object
                        raise ValueError(f"object '{object_key}' is used by two different weight sets with conflicting sizes.")
                    weight_sizes[object_key] = metadata['size']

        built_blocks = []
        # collect the blocks and store them in built_blocks list, calling the build method for each ProjectionHead with the correct output size
        for b in blocks:
            if isinstance(b, ProjectionHead):
                if not b.output or len(b.output) != 1:
                    raise ValueError("ProjectionHead must have exactly one output mapped.")
                #block = b.clone()
                out_object = b.output[0]
                if out_object in weight_sizes:
                    b.build(out_features=weight_sizes[out_object])
                else: # dangling projection heads with no corresponding weights fail loudly, to avoid silently accepting misconfigurations
                    raise ValueError(f"ProjectionHead output object '{out_object}' does not correspond to any weights in the TargetNetworks. Please check the mappings.")
                built_blocks.append(b)
            else:
                built_blocks.append(b)

        return built_blocks
    
    def _determine_execution_order(self, blocks: List[NeuralNetwork]) -> Tuple[List[NeuralNetwork], Set[str]]:
        """
        Determines the execution order of the blocks based on their input and output dependencies, using topological sorting.
        Also identifies the external inputs that need to be provided to the manager for execution.
        Args:
            blocks: List of NeuralNetwork blocks, including Hypernetworks, TargetNetworks, ProjectionHeads
        Returns:
            ordered_blocks: List of NeuralNetwork blocks in the order they should be executed.
            external_inputs: Set of input object keys that need to be provided to the manager for execution, which are not produced by any block in the manager.
        """
        graph = networkx.DiGraph() 

        # build a dict mapping each object to the block that produces it
        object_producers = {} 
        for i, block in enumerate(blocks): 
            graph.add_node(i) 
            if block.output:
                for output_node in block.output:
                    object_producers[output_node] = i
        # add edges connecting each block to the blocks using its outputs and track all needed objects
        consumed_objects = set()
        for i, block in enumerate(blocks): 
            deps = []
            # collect all dependencies of the block
            deps.extend(block.input_args) 
            deps.extend(block.input_kwargs.values()) 
            if isinstance(block, TargetNetwork) and block.weights_mapping is not None:
                deps.extend(block.weights_mapping.values())
            # add edges in the graph based on the dependencies and keep track of needed objects
            for d in deps:
                consumed_objects.add(d)
                if d in object_producers:
                    graph.add_edge(object_producers[d], i)

        # order the blocks based on the graph, and raise an error if a circular dependecy is detected
        try: 
            sorted_indices = list(networkx.topological_sort(graph))
        except networkx.NetworkXUnfeasible:
            raise ValueError("A circular dependency was detected in the graph.")

        # identify objects to be passed as external inputs
        external_inputs = consumed_objects - set(object_producers.keys())
        ordered_blocks = [blocks[i] for i in sorted_indices]

        return ordered_blocks, external_inputs
    
    def extract_target_network(self, inputs: dict[str, jnp.ndarray]) -> Dict[str, nnx.Module]:
        """
        Executes the manager to extract the TargetNetwork with injected weights as a standalone nnx.Module.
        """
        # determine target networks and the objects they require, to only execute the necessary blocks
        target_networks = [b for b in self.blocks if isinstance(b, TargetNetwork)]
        if not target_networks:
            raise ValueError("No TargetNetwork blocks found in the manager.")
        all_required_objects = set()
        for tn in target_networks:
            if tn.weights_mapping is not None:
                all_required_objects.update(tn.weights_mapping.values())

        # execute blocks to get the needed objects
        objects = inputs.copy()
        for block in self.blocks:
            if all_required_objects and all_required_objects.issubset(objects.keys()): # ends the loop early if all objects needed have been generated
                break
            if isinstance(block, TargetNetwork): # Only execute if needed to generate weight for another TargetNetwork
                is_generator = any(out in all_required_objects for out in (block.output or []))
                if not is_generator:
                    continue 
                required_objects = set(block.weights_mapping.values()) if block.weights_mapping is not None else set()                
                block_weights = {object_key: objects[object_key] for object_key in required_objects}
                input_args = [objects[s] for s in block.input_args]
                input_kwargs = {k: objects[v] for k, v in block.input_kwargs.items()}
                y = block(*input_args, weights=block_weights if block_weights else None, **input_kwargs)
            else: # Standard block execution (Hypernetwork, ProjectionHead, etc.)
                input_args = [objects[s] for s in block.input_args]
                input_kwargs = {k: objects[v] for k, v in block.input_kwargs.items()}
                y = block(*input_args, **input_kwargs)
            y = (y,) if not isinstance(y, tuple) else y
            for i, output_key in enumerate(block.output or []): # Store objects
                if i < len(y):
                    objects[output_key] = y[i]

        extracted_modules = {}
        for i, target_block in enumerate(target_networks): # inject weights and unfeeze them to make them trainable
            required_objects = set(target_block.weights_mapping.values()) if target_block.weights_mapping is not None else set()
            weights = {object_key: objects[object_key] for object_key in required_objects}
            # A standalone target network must have a single set of (unbatched) weights. Batched
            # weights (produced from batched hypervariable input) would create params with a leading
            # batch dimension that do not form a usable network, so reject them with a clear error.
            batched = {object_key: w.shape for object_key, w in weights.items() if getattr(w, "ndim", 1) > 1}
            if batched:
                raise ValueError(
                    "extract_target_network expects a single (unbatched) hypervariable configuration, "
                    f"but the generated weights are batched {batched}. Pass one configuration at a time."
                )
            new_state = nnx.State(target_block._inject_weights(weights))
            unfrozen_state = jax.tree.map(
                lambda v: nnx.Param(v.value) if isinstance(v, (TargetNetworkWeight, InjectedBatchedWeights, InjectedUnbatchedWeights)) else v,
                new_state,
                is_leaf=lambda x: isinstance(x, nnx.Variable)
            )
            network = nnx.merge(target_block.graphdef, unfrozen_state)            
            
            block_name = getattr(target_block, 'name', None)
            if not block_name: # If no name is provided, generate a unique one
                block_name = f"target_network_{i}"
            while block_name in extracted_modules: # Ensure unique block names
                block_name += "_duplicate"
                
            extracted_modules[block_name] = network
            
        return extracted_modules