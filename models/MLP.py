from typing import Optional, Union, Callable, Sequence
import jax
from flax import nnx

class MLP(nnx.Module):
    """
    A flexible MLP class that allows for a variable number of hidden layers and dimensions.
    Attributes:
        input_dim (int): Dimension of the input layer.
        output_dim (int): Dimension of the output layer.
        num_hidden_layers (int): Number of hidden layers.
        hidden_dims (list): Dimensions of the hidden layers.
        activation_functions (list): Activation functions for the hidden layers.
        num_parameters (int): Total number of parameters in the network.
    Methods:
        __call__(x): Forward pass through the network.
    """
   
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_hidden_layers: Optional[int] = None,
        hidden_dims: Union[int, Sequence[int]] = 8,
        activation_functions: Union[Callable, Sequence[Callable]] = nnx.relu,
        rngs: nnx.Rngs = nnx.Rngs(0)
    ):
        """
        Initializes the MLP with the specified parameters.
        Args:
            input_dim (int): Dimension of the input layer.
            output_dim (int): Dimension of the output layer.
            num_hidden_layers (int, optional): Number of hidden layers. If None, it is inferred
                from the length of hidden_dims. Default: None.
            hidden_dims (int or Sequence[int], optional): Dimension(s) of the hidden layers.
                If an int is provided, it is used for all hidden layers. If a sequence is 
                provided, its length must match `num_hidden_layers` if specified. Default: 8.
            activation_functions (Callable or Sequence[Callable], optional): Activation function(s)
                for the hidden layers. If a single callable is provided, it is used for all
                hidden layers. If a sequence is provided, its length must match the number of
                hidden layers. Default: nnx.relu.
            rngs (nnx.Rngs): Random number generators used to initialize the network. Default: nnx.Rngs(0).
        """

        if isinstance(hidden_dims,Sequence):
            if not all(isinstance(dim, int) for dim in hidden_dims):
                raise TypeError("hidden_dims must be an int or a sequence of ints")
        elif not isinstance(hidden_dims, int):
            raise TypeError("hidden_dims must be an int or a sequence of ints")
        if isinstance(activation_functions, Sequence):
            if not all(callable(func) for func in activation_functions):
                raise TypeError("activation_functions must be a callable or a sequence of callables")
        elif not callable(activation_functions):
            raise TypeError("activation_functions must be a callable or a sequence of callables")
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_parameters = 0

        if num_hidden_layers:
            if isinstance(hidden_dims, int):
                self.hidden_dims = [hidden_dims] * num_hidden_layers
                self.num_hidden_layers = num_hidden_layers
            elif len(hidden_dims) == num_hidden_layers:
                self.hidden_dims = list(hidden_dims)
                self.num_hidden_layers = len(hidden_dims)
            else:
                raise ValueError("Length of hidden_dims must match num_hidden_layers")
            
            if isinstance(activation_functions, Callable):
                self.activation_functions = [activation_functions] * self.num_hidden_layers
            elif len(activation_functions) == self.num_hidden_layers:
                self.activation_functions = list(activation_functions)
            else:
                raise ValueError("Length of activation_functions must match num_hidden_layers")
            
        else:
            self.hidden_dims = [hidden_dims] if isinstance(hidden_dims, int) else list(hidden_dims)
            self.num_hidden_layers = len(self.hidden_dims)
            self.activation_functions = [activation_functions] * self.num_hidden_layers if isinstance(activation_functions, Callable) else list(activation_functions)
        
        layers = []
        
        if self.num_hidden_layers > 0:
            layers.append(nnx.Linear(input_dim, self.hidden_dims[0], rngs=rngs))  # First hidden layer
            layers.append(self.activation_functions[0])
            self.num_parameters = self.hidden_dims[0] * (input_dim + 1)  # +1 for bias

            for i in range(self.num_hidden_layers - 1): # Additional hidden layers
                layers.append(nnx.Linear(self.hidden_dims[i], self.hidden_dims[i+1], rngs=rngs))
                layers.append(self.activation_functions[i+1])
                self.num_parameters += self.hidden_dims[i+1] * (self.hidden_dims[i] + 1)

            layers.append(nnx.Linear(self.hidden_dims[-1], output_dim, rngs=rngs)) # Output layer
            self.num_parameters += output_dim * (self.hidden_dims[-1] + 1)

        else: # Direct input to output
            layers.append(nnx.Linear(input_dim, output_dim, rngs=rngs))
            self.num_parameters = output_dim * (input_dim + 1)

        self.layers = nnx.Sequential(*layers)

    def __call__(self, x: jax.Array):
        """
        Forward pass through the network.
        Args:
            x (jax.Array): Input data.
        Returns:
            jax.Array: Output of the network.
        """
        return self.layers(x)