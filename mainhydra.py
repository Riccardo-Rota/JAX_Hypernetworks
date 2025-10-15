import os

# CPU fallback
if 'JAX_PLATFORMS' not in os.environ:
    try:
        from jax import devices
        if not any(d.platform == 'gpu' for d in devices()): os.environ['JAX_PLATFORMS'] = 'cpu'
    except Exception:
        os.environ['JAX_PLATFORMS'] = 'cpu'

import hydra
from omegaconf import DictConfig, OmegaConf
import jax.numpy as jnp
import jax.random as random
import matplotlib.pyplot as plt
from data import Dataset, JaxDataLoader
from training import train_model, assign_parameters, EarlyStopping
from inference import test_model
from utils import variables_generator
from flax import nnx
import optax
from losses import CustomLoss, RRMSE, MAE, l2_loss
import datetime
from utils import save_model

@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    print("Configuration being used:")
    print(OmegaConf.to_yaml(cfg))

    run_path = os.getcwd()
    print(f"Results will be saved in: {run_path}")

    # Dataset Generation (TODO: instantiate from config)
    key = random.key(cfg.seed)
    f_to_learn = lambda mu, l, k, x: jnp.exp(-l*x) + mu*x + k*x**2
    data_cfg = cfg.data
    mu_domain, l_domain, k_domain, x_domain = map(tuple, [data_cfg.mu_domain, data_cfg.l_domain, data_cfg.k_domain, data_cfg.x_domain])

    variables = variables_generator(
        N=data_cfg.N, n_realizations=data_cfg.n_realizations,
        var_names=['x'], var_domains=[x_domain],
        hypervar_domains=[mu_domain, l_domain, k_domain],
        hypervar_names=['mu', 'l', 'k'], key=key
    )
    mu, l, k, x = variables.values()
    y = f_to_learn(mu, l, k, x)
    X = jnp.stack([mu, l, k, x], axis=1)

    split_idx1 = int(X.shape[0] * 0.8)
    split_idx2 = int(X.shape[0] * 0.9)
    
    dataset_train = Dataset(X[:split_idx1, 3], X[:split_idx1, :3], y[:split_idx1])
    dataset_val = Dataset(X[split_idx1:split_idx2, 3], X[split_idx1:split_idx2, :3], y[split_idx1:split_idx2])
    dataset_test = Dataset(X[split_idx2:, 3], X[split_idx2:, :3], y[split_idx2:])

    train_loader = JaxDataLoader(dataset_train, batch_size=cfg.training.batch_size, shuffle=True)
    val_loader = JaxDataLoader(dataset_val, batch_size=cfg.training.batch_size, shuffle=False)
    test_loader = JaxDataLoader(dataset_test, batch_size=cfg.training.batch_size, shuffle=False)

    # Instantiate Models using Hydra
    targetnetwork = hydra.utils.instantiate(cfg.targetnetwork)
    num_params = targetnetwork.num_parameters()
    print(f"Target network '{type(targetnetwork).__name__}' instantiated with {num_params} parameters.")

    OmegaConf.update(cfg.hypernetwork, "config.num_neurons.-1", num_params, merge=False) # Set output layer size to match target network parameters
    hypernetwork = hydra.utils.instantiate(cfg.hypernetwork)
    print(f"Hypernetwork '{type(hypernetwork).__name__}' instantiated.")
    
    # Instantiate Training Components
    criterion = hydra.utils.instantiate(cfg.training.criterion)
    metrics = {name: hydra.utils.instantiate(metric_cfg) for name, metric_cfg in cfg.training.metrics.items()}
    early_stopping = hydra.utils.instantiate(cfg.training.early_stopping)

    # Instantiate optimizer.
    if cfg.optimizer.tx._target_ == 'optax.chain': # For ReduceLROnPlateau, we need to pass accumulation_size
         OmegaConf.update(cfg.optimizer, "tx.transforms.1.accumulation_size", len(val_loader), merge=False)

    optimizer = hydra.utils.instantiate(cfg.optimizer, model=hypernetwork)

    # Run Training
    history = train_model(
        hypernetwork=hypernetwork,
        targetnetwork=targetnetwork,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        num_epochs=cfg.training.epochs,
        criterion=criterion,
        metrics=metrics,
        early_stopping=early_stopping
    )

    # Run Testing
    test_metrics = test_model(
        hypernetwork=hypernetwork,
        targetnetwork=targetnetwork,
        loader=test_loader,
        metrics=tuple(metrics.values()),
    )
    print(f"Test Metrics: {test_metrics}")

    # Save plots
    training_loss_history = [m['loss'] for m in history['train_results']]
    val_loss_history = [m['loss'] for m in history['val_results']]

    plt.figure()
    plt.loglog(range(len(training_loss_history)), training_loss_history, label='Training Loss')
    plt.loglog(range(len(val_loss_history)), val_loss_history, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.legend()
    loss_plot_path = 'loss_plot.png'
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
        plot_path = f'prediction_{i}.png'
        plt.savefig(plot_path)
        plt.close()
        prediction_paths.append(plot_path)

    # Save model parameters
    hypernetwork_path = 'hypernetwork_params'
    save_model(hypernetwork, hypernetwork_path)

    # Save JSON with all info
    run_data = {
        'run_name': cfg.run_name,
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

    json_path = 'run_data.json'
    with open(json_path, 'w') as f:
        json.dump(convert(run_data), f, indent=4)

if __name__ == "__main__":
    main()