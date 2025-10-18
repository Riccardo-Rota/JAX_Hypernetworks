import os

# CPU fallback (if GPU is not available)
if 'JAX_PLATFORMS' not in os.environ:
    try:
        from jax import devices
        if not any(d.platform == 'gpu' for d in devices()):
            os.environ['JAX_PLATFORMS'] = 'cpu'
    except Exception:
        os.environ['JAX_PLATFORMS'] = 'cpu'

from datetime import datetime
import yaml
import json
import jax.numpy as jnp
import jax.random as random
import matplotlib.pyplot as plt
from flax import nnx
import optax
import orbax.checkpoint as ocp
from flax import serialization
import copy
from data import Dataset, JaxDataLoader
from models import MLP
from training import assign_parameters, EarlyStopping, train_model
from inference import test_model
from utils import variables_generator
from losses import CustomLoss, L2Loss
from metrics import RRMSE, MAE, RMSE, MSE
from models import Siren


model_class_map = {
    'MLP': MLP,
    'SIREN': Siren
}

activation_map = {
    'tanh': nnx.tanh,
    'relu': nnx.relu,
    'sine': jnp.sin
}

def main():
    # -------------------------------
    # Load configuration
    # -------------------------------
    with open('config.yaml', 'r') as f:
        cfg = yaml.safe_load(f)

    # Function to learn (can be adapted or made parametric)
    f_to_learn = lambda mu, l, k, x: jnp.exp(cfg['function']['l']*x) + cfg['function']['mu']*x + cfg['function']['k']*x**2

    # Dataset parameters
    mu_domain = tuple(cfg['dataset']['mu_domain'])
    l_domain = tuple(cfg['dataset']['l_domain'])
    k_domain = tuple(cfg['dataset']['k_domain'])
    x_domain = tuple(cfg['dataset']['x_domain'])
    N = cfg['dataset']['N']
    n_realizations = cfg['dataset']['n_realizations']

    batch_size = cfg['dataloader']['batch_size']

    criterion_map = {'l2_loss': L2Loss(), 'CustomLoss': CustomLoss()}
    criterion = criterion_map[cfg['training']['criterion']]

    metrics_map = {'RRMSE': RRMSE(), 'MAE': MAE(), 'RMSE': RMSE(), 'MSE': MSE()}
    if cfg['training']['metrics'] is None:
        metrics = None
    else:
        metrics = {k: metrics_map[k] for k in cfg['training']['metrics']}

    epochs = cfg['training']['epochs']

    early_stopping = EarlyStopping(
        patience=cfg['training']['early_stopping']['patience'],
        min_delta=cfg['training']['early_stopping']['min_delta']
    )

    scheduler_type = cfg['training']['scheduler']['type']
    cosine_decay_parameters = cfg['training']['scheduler'].get('cosine_decay', {})
    plateau_scheduler_parameters = cfg['training']['scheduler'].get('reduce_on_plateau', {})

    # -------------------------------
    # Create run folder
    # -------------------------------
    run_name = f"runs/run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(run_name, exist_ok=True)

    # -------------------------------
    # Dataset generation
    # -------------------------------
    key = random.key(0)
    mu, l, k, x = variables_generator(
        N=N,
        n_realizations=n_realizations,
        var_names=['x'],
        var_domains=[x_domain],
        hypervar_domains=[mu_domain, l_domain, k_domain],
        hypervar_names=['mu', 'l', 'k'],
        key=key
    ).values()
    y = f_to_learn(mu, l, k, x)
    X = jnp.stack([mu, l, k, x], axis=1)

    split_idx1, split_idx2 = int(N*n_realizations * 0.8), int(N*n_realizations * 0.9)
    X_train, X_val, X_test = X[:split_idx1], X[split_idx1:split_idx2], X[split_idx2:]
    y_train, y_val, y_test = y[:split_idx1], y[split_idx1:split_idx2], y[split_idx2:]

    dataset_train = Dataset(vars=X_train[:, 3], hypervars=X_train[:, :3], labels=y_train)
    dataset_val = Dataset(vars=X_val[:, 3], hypervars=X_val[:, :3], labels=y_val)
    dataset_test = Dataset(vars=X_test[:, 3], hypervars=X_test[:, :3], labels=y_test)

    train_dataloader = JaxDataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    val_dataloader = JaxDataLoader(dataset_val, batch_size=batch_size, shuffle=False)
    test_dataloader = JaxDataLoader(dataset_test, batch_size=batch_size, shuffle=False)

    # -------------------------------
    # Model initialization
    # -------------------------------
    tn_class = model_class_map[cfg['targetnetwork']['model']]
    tn_num_neurons = cfg['targetnetwork']['num_neurons']
    tn_kwargs = copy.deepcopy(cfg['targetnetwork']['kwargs'])
    if 'activation_functions' in tn_kwargs.keys():
        tn_kwargs['activation_functions'] = [activation_map[act] for act in tn_kwargs['activation_functions']]
    targetnetwork = tn_class(
        num_neurons = tn_num_neurons,
        rngs=nnx.Rngs(0),
        **tn_kwargs
    )
    
    num_params = targetnetwork.num_parameters()
    hn_class = model_class_map[cfg['hypernetwork']['model']]
    hn_num_neurons = cfg['hypernetwork']['num_neurons']
    hn_num_neurons[-1] = num_params  # Ensure output layer matches number of target network parameters
    hn_kwargs = copy.deepcopy(cfg['hypernetwork']['kwargs'])
    if 'activation_functions' in hn_kwargs.keys():
        hn_kwargs['activation_functions'] = [activation_map[act] for act in hn_kwargs['activation_functions']]
    hypernetwork = hn_class(
        num_neurons = hn_num_neurons,
        rngs=nnx.Rngs(0),
        **hn_kwargs
    )

    # -------------------------------
    # Optimizer setup
    # -------------------------------
    schedule_optimizer = nnx.Optimizer(
        hypernetwork,
        optax.adam(
            learning_rate=optax.schedules.cosine_decay_schedule(**cosine_decay_parameters)
        )
    )
    plateau_optimizer = nnx.Optimizer(
        hypernetwork,
        optax.chain(
            optax.adam(learning_rate=plateau_scheduler_parameters['init_value']),
            optax.contrib.reduce_on_plateau(
                factor=plateau_scheduler_parameters['factor'],
                patience=plateau_scheduler_parameters['patience'],
                accumulation_size=len(val_dataloader)
            )
        )
    )
    optimizer = schedule_optimizer if scheduler_type == 'cosine_decay' else plateau_optimizer

    # -------------------------------
    # Training
    # -------------------------------
    history = train_model(
        hypernetwork=hypernetwork,
        targetnetwork=targetnetwork,
        train_loader=train_dataloader,
        val_loader=val_dataloader,
        optimizer=optimizer,
        num_epochs=epochs,
        criterion=criterion,
        metrics=metrics,
        early_stopping_metric='RRMSE',
        early_stopping=early_stopping
    )

    # -------------------------------
    # Testing
    # -------------------------------
    test_metrics = test_model(
        hypernetwork=hypernetwork,
        targetnetwork=targetnetwork,
        loader=test_dataloader,
        metrics={'RRMSE': RRMSE(), 'RMSE': RMSE(), 'MSE': MSE(), 'MAE': MAE()},
    )

    print(f"Test Metrics: {test_metrics}")

    # -------------------------------
    # Save plots
    # -------------------------------
    training_loss_history = [m['loss'] for m in history['train_results']]
    val_loss_history = [m['loss'] for m in history['val_results']]

    plt.figure()
    plt.loglog(range(len(training_loss_history)), training_loss_history, label='Training Loss')
    plt.loglog(range(len(val_loss_history)), val_loss_history, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.legend()
    loss_plot_path = os.path.join(run_name, 'loss_plot.png')
    plt.savefig(loss_plot_path)
    plt.close()

    # Example predictions
    x_vector = jnp.linspace(-1, 1, 101)[:, None]
    N_examples = cfg['inference']['N_examples']
    mu_example, l_example, k_example = variables_generator(
        N=N_examples,
        n_realizations=1,
        var_names=['mu', 'l', 'k'],
        var_domains=[mu_domain, l_domain, k_domain],
        key=random.key(1)
    ).values()
    example_hypervars = jnp.stack([mu_example, l_example, k_example], axis=1)

    prediction_paths = []
    for i, hypervars in enumerate(example_hypervars):
        w = hypernetwork(hypervars)
        targetnetwork = assign_parameters(targetnetwork, w)
        mu, l, k = hypervars
        y_pred = targetnetwork(x_vector)
        v_f_to_learn = nnx.vmap(lambda x: f_to_learn(mu, l, k, x))
        y_vector = v_f_to_learn(x_vector)

        plt.figure()
        plt.plot(x_vector, y_pred, '--b')
        plt.plot(x_vector, y_vector)
        plt.legend(['Predicted', 'True'], loc='upper left')
        plt.title(f'l = {l:.2f}, k = {k:.2f}, mu = {mu:.2f}')
        plot_path = os.path.join(run_name, f'prediction_{i}.png')
        plt.savefig(plot_path)
        plt.close()
        prediction_paths.append(plot_path)

    # -------------------------------
    # Save model parameters
    # -------------------------------

    hypernetwork_path = os.path.join(run_name, 'hypernetwork_params')
    # TODO: FIX
    # _, state = nnx.split(hypernetwork)
    # checkpointer = ocp.StandardCheckpointer()
    # checkpointer.save(os.path.abspath(hypernetwork_path), state, asynchronous=False)

    # -------------------------------
    # Save JSON with all info
    # -------------------------------
    run_data = {
        'run_name': run_name,
        'datetime': datetime.now().isoformat(),
        'config': cfg,
        'test_metrics': test_metrics,
        'model': {
            'num_params_targetnetwork': targetnetwork.num_parameters(),
            'num_params_hypernetwork': hypernetwork.num_parameters(),
            'hypernetwork_path': hypernetwork_path,
        },
        'plots': {
            'loss_plot': loss_plot_path,
            'predictions': prediction_paths
        },
        'training_history': {
            'train_results': history['train_results'],
            'val_results': history['val_results']
        }
    }

    # Convert JAX arrays to list for JSON
    def convert(obj):
        if isinstance(obj, jnp.ndarray):
            return obj.tolist()
        if isinstance(obj, (list, tuple)):
            return [convert(o) for o in obj]
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        return obj

    json_path = os.path.join(run_name, 'run_data.json')
    with open(json_path, 'w') as f:
        json.dump(convert(run_data), f, indent=4)

if __name__ == "__main__":
    main()