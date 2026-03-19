import flax.nnx as nnx
import jax.numpy as jnp
from typing import Dict, List, Optional, Callable, Sequence, Union, Any
import networkx
import warnings

class OutputsNumberWarning(UserWarning):
    pass

class NeuralNetwork(nnx.Module):
    def __init__(self, network: nnx.Module, inputs: Optional[Dict[str, str]], outputs: Optional[List[str]]):
        self.network = network
        self.inputs = inputs
        self.outputs = outputs

    def __call__(self, x):
        return self.network(x)

class TargetNetwork(NeuralNetwork):
    def __init__(self, network: nnx.Module, inputs: Optional[Dict[str, str]], outputs: List[str], weights: Optional[Dict[str, str]] = None, replace_weights: bool = True):
        super().__init__(network, inputs, outputs)
        self.weights = weights
        self.replace_weights = replace_weights

    def __call__(self, x: jnp.ndarray, weights: Optional[Dict[str, jnp.ndarray]] = None):
        if weights is not None:
            # If weights are provided, we need to assign them to the network before calling it
            graphdef, previous_state = nnx.split(self.network) 
            if self.replace_weights:
                modified_network = nnx.merge(graphdef, weights)
            else:
                #TODO
                raise NotImplementedError("Currently only weight replacement is implemented. Merging with existing weights is not yet supported.")
            output = nnx.vmap(type(modified_network).__call__)(modified_network, x) #TODO: CHECK THIS
        return output

class Hypernetwork(NeuralNetwork):
    pass

class ProjectionHead(NeuralNetwork):

    def __call__(self, x: jnp.ndarray):
        w = super().__call__(x)  # Get the new weights from the hypernetwork
        state = nnx.vmap(build_state_from_parameters, in_axes=(None, 0, None), out_axes=0)(previous_state, w, replace=self.replace_weights) 
        # TODO: REWRITE build_state_from_parameters

class HypernetworkManager(nnx.Module):
    def __init__(self, networks: List[NeuralNetwork]):
        graph = networkx.DiGraph()
        self.network_map = {i: network for i, network in enumerate(networks)}
        network_outputs = {}
        for i, network in enumerate(networks):
            graph.add_node(i)  # Ensure all networks are added as nodes
            for output_node in network.outputs:
                network_outputs[output_node] = i

        # 2. Create edges directly between networks based on dependencies
        for i, network in enumerate(networks):
            for input_node in network.inputs.values():
                # If this input is produced by another network in our list, draw an edge
                if input_node in network_outputs:
                    producer_index = network_outputs[input_node]
                    graph.add_edge(producer_index, i)

        # 3. Sort and build the execution order
        try:
            sorted_indices = list(networkx.topological_sort(graph))
        except networkx.NetworkXUnfeasible:
            raise ValueError("A circular dependency was detected in the graph.")

        self.execution_order = [networks[i] for i in sorted_indices]

    def __call__(self, inputs: dict[str, jnp.ndarray]) -> Dict[str, Any]:
        """
        Executes the entire processing chain of the graph.
        Args:
            inputs (dict[str, jnp.ndarray]): A dictionary containing the initial inputs, usually with keys 'variables' and 'hypervariables' and their corresponding values as jnp arrays.
        Returns:
            Dict[str, Any]: The final dictionary containing all signals,
                including initial ones and those computed by every node.
        """
        outputs = inputs.copy()  # Start with the initial inputs as the initial signals
        for block in self.execution_order:
            kwargs = {arg_name: outputs[output_key] for arg_name, output_key in block.inputs.items()}
            y = block(**kwargs)
            output_names = block.outputs
            if not isinstance(y, tuple):
                y = (y,)
            if len(output_names) != len(y):
                warnings.warn(OutputsNumberWarning(f"Node '{block.network.__class__.__name__}' output mapping has {len(output_names)} keys, but the module returned {len(y)} values. Ignoring extra outputs."))
            for i in range(min(len(output_names), len(y))):
                outputs[output_names[i]] = y[i]

        return outputs