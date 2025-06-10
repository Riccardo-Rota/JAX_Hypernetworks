# PARE CHE DOBBIAMO CREARCI UN PO TUTTO, DUNQUE: (POI SISTEMEREMO TUTTO IN FILE ORGANIZZATI BENE)

import jax
import jax.numpy as jnp
from flax import linen as nn

class DataLoader:
    """
    DataLoader for JAX that supports batching and shuffling.
    The data and labels are stored as squeezed JAX arrays.
    When iterating, batches of data and labels are returned as arrays with an additional dimension.
    """

    def __init__(self, data, labels, batch_size=32, shuffle=True, seed=0):
        """
        Initialize the DataLoader.
        Parameters:
            data (array-like): Input data.
            labels (array-like): Corresponding labels.
            batch_size (int): Size of each batch.
            shuffle (bool): Whether to shuffle the data.
            seed (int): Random seed for shuffling.
        """
        self.data = jnp.array(data).squeeze()
        self.labels = jnp.array(labels).squeeze()
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.n_samples = self.data.shape[0]
        self._reset_indices()
    
    def _reset_indices(self):
        self.indices = jnp.arange(self.n_samples)
        if self.shuffle:
            key = jax.random.PRNGKey(self.seed)
            self.indices = jax.random.permutation(key, self.indices)
    
    def __iter__(self):
        self._current_idx = 0
        return self
    
    def __next__(self):
        """Return the next batch of data and labels."""
        if self._current_idx >= self.n_samples:
            raise StopIteration
        
        start = self._current_idx
        end = start + self.batch_size
        batch_indices = self.indices[start:end]
        
        batch_data = jnp.expand_dims(self.data[batch_indices], axis=0)
        batch_labels = jnp.expand_dims(self.labels[batch_indices], axis=0)
        
        self._current_idx += self.batch_size
        return batch_data, batch_labels
    
    def __len__(self):
        """Return the number of batches."""
        return (self.n_samples + self.batch_size - 1) // self.batch_size


class MLP(nn.Module):
    """
    Author: RICCARDO ROTA, ASSOLUTAMENTE LEONARDO BOCCHIERI NON HA CONTRIBUITO.
    """
    # attributi che vengono inizializzati nell'init di nn.Module, quindi andranno passati in input quando inizializziamo il modello, 
    # tipo model = MLP(output_dim=1, hidden_dim=8, num_hidden_layers=2), oppure semplicemente MLP() perchè ho messo default
    output_dim: int = 1
    hidden_dim: int = 8
    num_hidden_layers: int = 2

    @nn.compact
    def __call__(self, x):
        for _ in range(self.num_hidden_layers):
            x = nn.Dense(self.hidden_dim)(x) # strato dense (fully connected), capisce la dimensione dell'input 
                                             # da solo, gli dobbiamo specificare quella dell'output
            x = nn.relu(x) # vedi di capire da solo cos'è questo
        x = nn.Dense(self.output_dim)(x) 
        return x