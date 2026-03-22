import flax.nnx as nnx
import jax.numpy as jnp
from typing import Dict, List, Optional, Callable, Sequence, Union, Any
import networkx
import warnings
import math 
from flax.traverse_util import flatten_dict, unflatten_dict 


# THIS SHOULD WORK, BUT I WOULD LIKE THE MANAGER TO AUTOMATICALLY INSTANTIATE THE PROJECTION HEADS BASED ON THE 
# NUMBER OF REQUIRED PARAMETERS. IS IT DOABLE?

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
    "Target network class that can optionally take weights as input and assign them before the forward pass."
    def __init__(self, network: nnx.Module, input_mapping: Optional[Dict[str, str]], output_mapping: List[str], weights_mapping: Optional[Dict[str, str]] = None, replace_weights: bool = True):
        """
        Args:
            network (nnx.Module): The neural network module to be wrapped.
            input_mapping (Optional[Dict[str, str]]): A dictionary mapping argument names of the network's forward method to the keys in the input dictionary that will be passed to the HypernetworkManager. For example, if the network's forward method has an argument 'x', and we want to pass the value from the input dictionary with key 'variables', then we would have {'x': 'variables'}.
            output_mapping (Optional[List[str]]): A list of strings representing the keys under which the outputs of this network will be stored in the HypernetworkManager's output dictionary. The order of this list should match the order of outputs returned by the network's forward method.
            weights_mapping (Optional[Union[Dict[str, str], str]]): A dictionary mapping weight paths to their values. Each weight path is a dot-separated string that specifies the location of the weight in the target network's state dictionary (e.g., 'layer1.weight'). Alternatively, if set to 'all', it indicates that the hypernetwork will generate all weights for the target network.
            replace_weights (bool): Whether to replace existing weights with the provided ones. If False, the provided weights will be summed with existing ones.
        """
        super().__init__(network, input_mapping, output_mapping)
        self.weights_mapping = weights_mapping
        self.replace_weights = replace_weights

    def __call__(self, x: jnp.ndarray, weights: Optional[Dict[str, Any]] = None):
        if weights is not None and self.replace_weights:
            graphdef, state = nnx.split(self.network)
            if 'all' in weights:
                if len(weights) > 1:
                    warnings.warn("weights_mapping contains 'all' but also specific paths. Replacing all weights and ignoring the specific paths.")
                # We extract the actual PyTree using the 'all' key
                new_state = nnx.State(weights['all'])
            else:
                state_dict = state.to_dict()
                for target_path, generated_weight in weights.items():
                    set_nested_dict(state_dict, target_path, generated_weight)
                new_state = nnx.State(state_dict)
            modified_network = nnx.merge(graphdef, new_state) 
            return modified_network(x)
        if weights is not None and not self.replace_weights:
            raise NotImplementedError("Currently only weight replacement is implemented. Merging with existing weights is not yet supported.")
        return self.network(x)
            

class Hypernetwork(NeuralNetwork):
    "Hypernetwork class that generates a output representing a latent space, to be used by a ProjectionHead to produce weights for the TargetNetwork."
    pass

class ProjectionHead(NeuralNetwork):
    "Projection head class that takes the output of a Hypernetwork and produces weights for the TargetNetwork."
    def __init__(self, 
                 in_features: int, 
                 target_network: nnx.Module, 
                 weight_paths: Union[str, List[str]], 
                 input_mapping: Optional[Dict[str, str]] = None, 
                 output_mapping: Optional[List[str]] = None,
                 kernel_init: Callable = nnx.initializers.lecun_normal(),
                 bias_init: Callable = nnx.initializers.zeros_init()):
        """
        Args:
            in_features (int): The number of input features to the projection head (i.e., the size of the hypernetwork's output).
            target_network (nnx.Module): The target network for which this projection head will generate weights.
            weight_paths (Union[str, List[str]]): A list of dot-separated strings representing the paths to the weights in the target network's state dictionary that will be generated by the projection head. Alternatively, if set to 'all', the projection head will generate all weights for the target network.
            input_mapping (Optional[Dict[str, str]]): A dictionary mapping argument names of the projection head's forward method to the keys in the input dictionary that will be passed to the HypernetworkManager.
            output_mapping (Optional[List[str]]): A list of strings representing the keys under which the outputs of this projection head will be stored in the HypernetworkManager's output dictionary. The order of this list should match the order of outputs returned by the projection head's forward method.
            kernel_init (Callable): The initialization function for the weights of the linear layer in the projection head.
            bias_init (Callable): The initialization function for the biases of the linear layer in the projection head.
        """
        
        # Extract and flatten the target network's state dictionary
        target_state = nnx.state(target_network)
        flat_state = flatten_dict(target_state.to_dict(), sep='.')
        
        # Determine which paths we are predicting
        if weight_paths == 'all':
            self.weight_paths = list(flat_state.keys())
        else:
            self.weight_paths = weight_paths

        # Calculate shapes and total required output features
        self.weight_shapes = {}
        out_features = 0
        
        for path in self.weight_paths:
            if path not in flat_state:
                raise ValueError(f"Target path '{path}' not found in the target network.")
            
            param = flat_state[path]
            shape = param.value.shape if hasattr(param, 'value') else param.shape
            
            self.weight_shapes[path] = shape
            out_features += math.prod(shape)
            
        # Create the linear layer with customizable initialization
        network = nnx.Linear(
            in_features, 
            out_features, 
            kernel_init=kernel_init, 
            bias_init=bias_init
        )
        super().__init__(network, input_mapping, output_mapping)

    def __call__(self, x: jnp.ndarray): # TODO: CHECK BATCHING LOGIC
        flat_predictions = self.network(x)
        
        is_batched = flat_predictions.ndim > 1
        batch_size = flat_predictions.shape[0] if is_batched else None
        
        generated_flat_dict = {}
        current_idx = 0
        
        for path in self.weight_paths:
            shape = self.weight_shapes[path]
            num_elements = math.prod(shape)
            
            if is_batched:
                flat_slice = flat_predictions[:, current_idx : current_idx + num_elements]
                target_shape = (batch_size,) + shape
            else:
                flat_slice = flat_predictions[current_idx : current_idx + num_elements]
                target_shape = shape
                
            generated_flat_dict[path] = jnp.reshape(flat_slice, target_shape)
            current_idx += num_elements
            
        nested_weights = unflatten_dict(generated_flat_dict, sep='.')
        return nested_weights

class HypernetworkManager(nnx.Module):
    """
    Class that manages the execution of a graph of NeuralNetwork blocks (TargetNetworks, Hypernetworks, ProjectionHeads) based on their input and output dependencies.
    """
    def __init__(self, blocks: List[NeuralNetwork]):
        """
        Initializes the HypernetworkManager with a list of NeuralNetwork blocks (which can be TargetNetworks, Hypernetworks, or ProjectionHeads).
        The manager constructs a directed acyclic graph based on the input and output, to determine the execution order of the blocks.
        Args:
            blocks (List[NeuralNetwork]): A list of NeuralNetwork instances.
        """

        graph = networkx.DiGraph() # Create a directed graph to represent dependencies between blocks
        outputs_producers = {} # Map output keys to the index of the block that produces them

        # Populate graph and outputs_producers
        for i, block in enumerate(blocks):
            graph.add_node(i) 
            if block.output_mapping:
                for output_node in block.output_mapping:
                    outputs_producers[output_node] = i

        # Create edges between blocks based on dependencies
        for i, block in enumerate(blocks):
            if block.input_mapping:
                for input_node in block.input_mapping.values():
                    # If this input is produced by another block, draw an edge
                    if input_node in outputs_producers:
                        producer_index = outputs_producers[input_node]
                        graph.add_edge(producer_index, i)
            if isinstance(block, TargetNetwork) and block.weights_mapping:
                for weight_node in block.weights_mapping.values():
                    # If this weight is produced by another block, draw an edge
                    if weight_node in outputs_producers:
                        graph.add_edge(outputs_producers[weight_node], i)

        # Sort the graph and build the execution order
        try:
            sorted_indices = list(networkx.topological_sort(graph))
        except networkx.NetworkXUnfeasible:
            raise ValueError("A circular dependency was detected in the graph.")

        self.execution_order = [blocks[i] for i in sorted_indices]

    def __call__(self, inputs: dict[str, jnp.ndarray]) -> Dict[str, Any]:
        """
        Executes the entire processing chain of the graph.
        Args:
            inputs (dict[str, jnp.ndarray]): A dictionary containing the initial inputs, usually with keys 'variables' and 'hypervariables' and their corresponding values as jnp arrays.
        Returns:
            Dict[str, Any]: The final dictionary containing all signals,
                including initial ones and those computed by every node.
        """
        objects = inputs.copy()  # Start with the initial inputs as the objects we have available
        for block in self.execution_order:
            input_kwargs = {input_name: objects[input_key] for input_name, input_key in (block.input_mapping or {}).items()}
            if isinstance(block, TargetNetwork) and block.weights_mapping:
                input_kwargs['weights'] = {weight_path: objects[weight_key] for weight_path, weight_key in block.weights_mapping.items()}

            y = block(**input_kwargs)
            output_names = block.output_mapping or [] # MODIFIED
            if not isinstance(y, tuple):
                y = (y,)
            if len(output_names) != len(y):
                warnings.warn(OutputsNumberWarning(f"Node '{block.network.__class__.__name__}' output mapping has {len(output_names)} keys, but the module returned {len(y)} values. Ignoring extra outputs."))
            for i in range(min(len(output_names), len(y))):
                objects[output_names[i]] = y[i]

        return objects