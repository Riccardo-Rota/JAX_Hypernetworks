from typing import Optional, Union, Callable, Sequence, List
import jax
from flax import nnx
from flax.typing import Initializer

# Define possible initializers
initializers = {
    "lecun_normal": nnx.initializers.lecun_normal(),
    "lecun_uniform": nnx.initializers.lecun_uniform(),
    "glorot_normal": nnx.initializers.glorot_normal(),
    "glorot_uniform": nnx.initializers.glorot_uniform(),
    "he_normal": nnx.initializers.he_normal(),
    "he_uniform": nnx.initializers.he_uniform()
}

def get_initializer(name):
    if name in initializers:
        return initializers[name]
    else:
        raise ValueError(f"Initializer '{name}' not found. Available options: {list(initializers.keys())}")

class MLP(nnx.Module):
    """
    A flexible MLP class that allows for a variable number of hidden layers and dimensions.
    """
    def __init__(
        self,
        num_neurons: List[int],
        *,
        activation_functions: Union[Callable, Sequence[Callable]] = nnx.relu,
        kernel_init: Initializer = nnx.initializers.lecun_normal(),
        bias_init: Initializer = nnx.initializers.zeros_init(),
        rngs: nnx.Rngs = nnx.Rngs(0)
    ):
        """
        Initializes the MLP with the specified parameters.
        Args:
            num_neurons (List[int]): List containing the number of neurons in each layer.
            activation_functions (Callable or Sequence[Callable], optional): Activation function(s)
                for the hidden layers. If a single callable is provided, it is used for all
                hidden layers. If a sequence is provided, its length must match the number of
                hidden layers. Default: nnx.relu.
            kernel_init (Initializer, optional): Initializer for the weights. Default: nnx.initializers.lecun_normal().
            bias_init (Initializer, optional): Initializer for the biases. Default: nnx.initializers.zeros_init().
            rngs (nnx.Rngs): Random number generators used to initialize the network. Default: nnx.Rngs(0).
        """

        if type(kernel_init) == str:
            kernel_init = get_initializer(kernel_init)

        self.layers = [
            nnx.Linear(num_neurons[i], num_neurons[i+1], rngs=rngs, kernel_init = kernel_init, bias_init = bias_init)
            for i in range(len(num_neurons) - 1)]
        
        if isinstance(activation_functions, Sequence):
            self.activation_functions = activation_functions
        else:
            self.activation_functions = [activation_functions] * (len(num_neurons) - 2)
        
        
    def __call__(self, x: jax.Array):
        """
        Forward pass through the network.
        Args:
            x (jax.Array): Input data.
        Returns:
            jax.Array: Output of the network.
        """
        for i_lay, lay in enumerate(self.layers):
            x = lay(x)
            if i_lay < len(self.layers) - 1:
                x = self.activation_functions[i_lay](x)
        return x
    
    def num_parameters(self):
        """
        Computes the total number of parameters in the network.
        Returns:
            int: Total number of parameters.
        """
        num_par = 0
        for lay in self.layers:
            num_par += lay.kernel.value.size
            if lay.use_bias:
                num_par += lay.bias.value.size
        return num_par