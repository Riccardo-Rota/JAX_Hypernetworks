import jax
import jax.numpy as jnp
from flax import nnx
from typing import List, Literal, Union, Any
from flax.typing import Initializer
from .hypernetwork_manager import ProjectionHead
from typing import Callable, Sequence, Optional, Dict
from .activation_functions import uniform_init

Dtype = Union[jax.typing.DTypeLike, Any]

# Define a dictionary of initializers for easy access
initializers = {
    "lecun_normal": nnx.initializers.lecun_normal(),
    "lecun_uniform": nnx.initializers.lecun_uniform(),
    "glorot_normal": nnx.initializers.glorot_normal(),
    "glorot_uniform": nnx.initializers.glorot_uniform(),
    "he_normal": nnx.initializers.he_normal(),
    "he_uniform": nnx.initializers.he_uniform(),
    "close_to_zero": nnx.initializers.normal(stddev=1e-5)
}

def get_initializer(name: str) -> Initializer:
    """
    Get an initializer by name.

    Args:
        name (str): The name of the initializer to retrieve.

    Returns:
        The requested initializer.
    """
    if name in initializers:
        return initializers[name]
    else:
        raise ValueError(f"Initializer '{name}' not found. Available options: {list(initializers.keys())}")


class SirenLayer(nnx.Module):
    """
    Single layer of a Siren network, consisting of a linear transformation followed by a sine activation function.
    The weights are initialized according to the Siren paper.
    """
    
    def __init__(self,
        in_features: int,
        out_features: int,
        *,
        use_bias: bool = True,
        use_activation: bool = True,
        w0: float = 1.0,
        kernel_init_first: bool = False,
        rngs: nnx.Rngs,
        ):
        """
        Initialize the SirenLayer.

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
            use_bias (bool): Whether to include a bias term.
            use_activation (bool): Whether to apply the sine activation function.
            w0 (float): The frequency factor for the sine activation function.
            kernel_init_first (bool): Whether to use the first layer initialization for the kernel.
            rngs (nnx.Rngs): The random number generator state.
        """

        self.use_bias = use_bias
        self.use_activation = use_activation
        self.w0 = w0

        # kernel initialization
        kernel_key = rngs.params()
        kernel_init_base = jax.random.uniform(kernel_key, (in_features, out_features), minval = -1, maxval = 1)
        if kernel_init_first:
            kernel_init_scaled = kernel_init_base / in_features
        else:
            kernel_init_scaled = kernel_init_base * jnp.sqrt(6/in_features) / w0
        self.kernel = nnx.Param(kernel_init_scaled)

        # bias initialization
        if use_bias:
            self.bias = nnx.Param(jnp.zeros((out_features,)))
        else:
            self.bias = nnx.Param(None)

    def apply_linear(self, inputs: jnp.ndarray) -> jnp.ndarray:
        """
        Apply the linear transformation of the layer.
        
        Args:
            inputs (jnp.ndarray): The input tensor of shape (batch_size, in_features).
        Returns:
            The output tensor of shape (batch_size, out_features) after applying the linear transformation.
        """
        y = inputs @ self.kernel

        if self.use_bias:
            y = y + self.bias
        return y

    def activation(self, inputs: jnp.ndarray) -> jnp.ndarray:
        return jnp.sin(self.w0 * inputs)

    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        y = self.apply_linear(inputs)
        if self.use_activation:
            y = self.activation(y)
        return y
        
class Siren(nnx.Module):
    """
    A Siren network consisting of multiple Siren layers. The weights of each layer are initialized according to the Siren paper.
    """
    def __init__(self, 
        num_neurons: List[int], 
        *,
        w0_first: float = 30,
        w0_last: float = 30,
        w0_other: float = 30,
        rngs: nnx.Rngs
        ):
        """
        Initialize the Siren network.
        Args:
            num_neurons (List[int]): A list specifying the number of neurons in each layer.
            w0_first (float): The frequency factor for the sine activation function in the first layer.
            w0_last (float): The frequency factor for the sine activation function in the last layer
            w0_other (float): The frequency factor for the sine activation function in the intermediate layers.
            rngs (nnx.Rngs): The random number generator state.
        """
        
        self.layers = list()
        for i_layer in range(len(num_neurons)-1):
            if i_layer == 0:
                w0 = w0_first
            elif i_layer == len(num_neurons) - 2:
                w0 = w0_last
            else:
                w0 = w0_other

            self.layers.append(SirenLayer(
                in_features = num_neurons[i_layer], 
                out_features = num_neurons[i_layer+1],
                use_bias = True,
                use_activation = i_layer < len(num_neurons) - 2,
                w0 = w0,
                kernel_init_first = i_layer == 0,
                rngs = rngs))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for lay in self.layers:
            x = lay(x)
        return x
    
    def num_parameters(self):
        """
        Calculate the total number of parameters in the Siren network.
            Returns:
            int: The total number of parameters in the network.
        """
        total_params = 0
        for lay in self.layers:
            total_params += lay.kernel.value.size
            if lay.use_bias:
                total_params += lay.bias.value.size
        return total_params

SirenInitMode = Literal['first_kernel', 'kernel', 'bias']

class SirenHead(ProjectionHead):
    """
    Module to take a latent space outputed from a hypernetwork and map it to the weights of a Siren network.
    The layer is initialized accordingly to Siren initialization.
    """
    def __init__(self, 
                 in_features: int, 
                 input: Optional[List[Union[str, Dict[str, str]]]],
                 output: str,
                 rngs: Optional[nnx.Rngs] = None,
                 mode: SirenInitMode = 'kernel',
                 siren_in_features: Optional[int] = None,
                 w0: float = 30.0
                 ):
        """
        Initialize the SirenHead.
        Args:
            in_features (int): Number of input features.
            input (Optional[List[Union[str, Dict[str, str]]]]): Input specification for the hypernetwork.
            output (str): Output specification for the hypernetwork.
            rngs (Optional[nnx.Rngs]): Random number generator state.
            mode (SirenInitMode): Mode for initializing the Siren weights. Can be 'first_kernel', 'kernel', or 'bias'.
            siren_in_features (Optional[int]): Number of input features for the Siren layer. Required for 'first_kernel' and 'kernel' modes.
            w0 (float): Frequency factor for the sine activation function. Required for 'kernel' mode.
        """
        kernel_init = nnx.initializers.normal(stddev=1e-5)
        if mode == 'first_kernel':
            if siren_in_features is None:
                raise ValueError("siren_in_features must be provided when mode is 'first_kernel'")
            limit = 1.0 / siren_in_features
            bias_init = uniform_init(limit)
        elif mode == 'kernel':
            if siren_in_features is None or w0 is None:
                raise ValueError("siren_in_features and w0 must be provided when mode is 'kernel'")
            limit = jnp.sqrt(6 / siren_in_features) / w0
            bias_init = uniform_init(limit)
        elif mode == 'bias':
            bias_init = nnx.initializers.zeros_init()
        else:
            raise ValueError(f"Invalid mode '{mode}'. Expected one of: 'first_kernel', 'kernel', 'bias'.")
        super().__init__(in_features, input, output, rngs, kernel_init, bias_init)