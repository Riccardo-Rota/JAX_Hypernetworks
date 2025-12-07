import jax
import jax.numpy as jnp
from flax import nnx
from typing import List, Union, Any
from flax.typing import Initializer

Dtype = Union[jax.typing.DTypeLike, Any]

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


class FCNN(nnx.Module):
    """
    Fully connected neural network.
    """

    def __init__(self, 
        num_neurons: List[int], 
        *,
        kernel_init: Initializer = nnx.initializers.lecun_normal(),
        bias_init: Initializer = nnx.initializers.zeros_init(),
        rngs: nnx.Rngs,
        ):
        if type(kernel_init) == str:
            kernel_init = get_initializer(kernel_init)
        self.layers = [
            nnx.Linear(num_neurons[i], num_neurons[i+1], rngs=rngs, kernel_init = kernel_init, bias_init = bias_init)
            for i in range(len(num_neurons) - 1)]

    def __call__(self, x):
        for i_lay, lay in enumerate(self.layers):
            x = lay(x)
            if i_lay < len(self.layers) - 1:
                x = jnp.tanh(x)
        return x

    # ASK REGAZZONI ABOUT THIS FUNCTION
    # def get_RMS_kernels(self):
    #     kernels = [jnp.square(lay.kernel.value) for lay in self.layers]
    #     return utils.global_average(kernels)

class SirenLayer(nnx.Module):
    
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

    def apply_linear(self, inputs):
        kernel = self.kernel.value
        bias = self.bias.value

        y = jax.lax.dot_general(
            inputs,
            kernel,
            (((inputs.ndim - 1,), (0,)), ((), ()))
            ) # in the end... simple dot product
        if self.use_bias:
            y += jnp.reshape(bias, (1,) * (y.ndim - 1) + (-1,))     # in practice we force bias with dim (1, 1, ..., 1, out_features) with the same number of leading 1s as y.ndim - 1 (batch dimensions)
        return y

    def activation(self, inputs):
        return jnp.sin(self.w0 * inputs)

    def __call__(self, inputs):
        y = self.apply_linear(inputs)
        if self.use_activation:
            y = self.activation(y)
        return y
        
class Siren(nnx.Module):
    def __init__(self, 
        num_neurons: List[int], 
        *,
        w0_first: float = 30,
        w0_last: float = 30,
        w0_other: float = 30,
        rngs: nnx.Rngs,
        replace_weights: bool = False
        ):
        
        self.replace_weights = replace_weights
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

    def __call__(self, x):
        for lay in self.layers:
            x = lay(x)
        return x
    
    def num_parameters(self):
        total_params = 0
        for lay in self.layers:
            total_params += lay.kernel.value.size
            if lay.use_bias:
                total_params += lay.bias.value.size
        return total_params
    
# Briefly: Siren is a basic FCNN with:
# - sinusoidal activation functions taken after a rescaling (line 103)
# - weights initialized from uniform [-1,1] amd rescaled with a specific formula (line 78 for 1st layer, line 80 for others)
# bias initialized with zeros (nothing special, but I say it for the sake of completeness)

class SirenHead(nnx.Module):
    """
    Module to take a latent space outputed from a hypernetwork and map it to the weights of a Siren network.
    Outputs are rescaled accordingly to Siren initialization.
    """
    def __init__(self,
        siren: Siren,
        latent_dim: int,
        *,
        rngs: nnx.Rngs,
        ):
        
        self.siren = siren
        self.latent_dim = latent_dim
        self.total_params = siren.num_parameters()

        # Define a simple FCNN to map latent space to siren weights
        self.mapper = FCNN(
            num_neurons = [latent_dim, 128, 256, self.total_params],
            kernel_init = "he_uniform",
            bias_init = nnx.initializers.zeros_init(),
            rngs = rngs
        )
