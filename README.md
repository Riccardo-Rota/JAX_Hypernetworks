# JAX Hypernetworks

This is a library built on **JAX** and **Flax NNX**, implemented to create a new general framework for **hypernetworks**. Instead of training one neural network to approximate a single function, we train a network that *generates the weights* of another network on the fly, conditioned on a set of parameters, called *hypervariables*. This is particularly powerful for scientific computing, where one is often interested not in a single solution, but in how a solution evolves as physical parameters change.

The library is validated on two problems of increasing complexity:

- **`toy`** — regression of analytic, parameter-dependent functions. Computationally soft to run on a laptop CPU, ideal to understand the method.
- **`turbulence`** — reconstruction of 2D fields from the *turbulent radiative layer* dataset (astrophysical turbulence), conditioned on time. A realistic, GPU-friendly workload.

Both experiments run with the same underlying code. This flexibility is mainly provided by the fact that everything (model topology, data source, optimizer, losses, metrics, post-processing) is declared in **Hydra** configuration files, so experiments are reproducible and can be reshaped from the command line without editing the source code.

---

## Table of Contents

1. [General Idea](#1-general-idea)
2. [Repository structure](#2-repository-structure)
3. [Hydra](#3-hydra)
4. [Reproducibility: How to run code and tests](#4-reproducibility-how-to-run-code-and-tests)
5. [The guided tour: `make run-test*`](#5-the-guided-tour-make-run-test)
6. [General usage](#6-general-usage)
7. [Datasets](#7-datasets)
8. [Weights & Biases](#8-weights--biases)
9. [Running on an HPC cluster](#9-running-on-an-hpc-cluster)
10. [Troubleshooting & notes](#10-troubleshooting--notes)

---

## 1. General Idea

A **hypernetwork** is a neural network whose output is the set of weights of a second network, the
**target network**. In this library the data flow is always the same:

```
hypervariables (θ)  ──►  Hypernetwork  ──►  latent features
                                                 │
                                                 ▼
                                          Projection head(s)  ──►  generated weights
                                                                         │
   variables (x)  ───────────────────────────────────────────►  Target network  ──►  output
```

- **Hypervariables (`θ`)** are the parameters given as inputs to the *hypernetwork*.
- **Variables (`x`)** are the parameters given as inputs to the *target network*.
- General pipeline:
    1. The **Hypernetwork** maps `θ` to a latent code
    2. The **projection heads** expand that code into the actual kernels and biases of the target network
    3. The **target network** consumes `x` with those generated weights and produces the prediction.

The whole assembly is described as a small **directed acyclic graph (DAG)** of *blocks* and orchestrated
by the `HypernetworkManager`, which infers the execution order automatically. We provide two different families of neural netwoek architectures: plain **MLPs** and a **SIRENs** (sinusoidal representation network, useful when dealing with high-frequency signals).

---

## 2. Repository structure

| Folder | Purpose |
| --- | --- |
| **`models/`** | The neural building blocks. Defines `Hypernetwork`, `ProjectionHead`/`SirenHead` (the heads that turn latent codes into weights), `TargetNetwork` (the network whose weights are *injected* rather than trained directly), and the `HypernetworkManager` that connect together all the blocks. Target architectures `MLP` and `Siren` are also implemented here. |
| **`training/`** | The training engine, structured hierarchically like a Matryoshka doll. `train_model` drives the full execution, nesting `train_epoch` for individual epochs, which further nests `train_step` for single-batch execution (which is JIT-compiled). `hypernet_utils.py` manages the weight-injection mechanism (`build_state_from_parameters`, `assign_parameters`, `apply`). |
| **`inference/`** | `test_model` — a single evaluation pass over a test set returning the required metrics. |
| **`losses/`** | Composable loss functions: `L2Loss` (MSE), `LpLoss` (generic L-p norm) and `CombinedLoss` (a weighted sum of any of the above). |
| **`metrics/`** | Flax-based streaming metrics: `MSE`, `RMSE`, `RRMSE` (relative RMSE) and `MAE`. For our results, we use RRMSE as most valuable performance metric. |
| **`data_processing/`** | Dataset loading and construction. `ToyDataSource` *generates* samples by evaluating a parametric function over its domains; `InMemoryHDF5Source` loads a preprocessed HDF5 dataset into RAM. `build_dataset` wraps them in a Grain pipeline (shuffle/batch/iterate). For the turbulence problem, `download_data.py` fetches the dataset from Hugging Face, while `preprocessing.py` performs the temporal train/val/test split and normalization. |
| **`postprocessing/`** | Plotting utilities: loss curves (`plot_loss_curves`), 1D/2D prediction plots, and `plot_2d_hdf5_comparison` (prediction vs. ground truth vs. error for the turbulence fields). |
| **`utils/`** | General helpers. Most importantly the **custom Hydra resolvers** (`hydra_resolvers.py`), the safe math-expression parser `get_function_from_string` (used to define toy functions from strings), Orbax-based `save_model`/`load_model`, and learning-rate introspection. |
| **`config/`** | The full Hydra configuration tree (see [§3](#3-hydra)). This is where experiment parameters and configurations are defined. |
| **`scripts/`** | Thin shell wrappers invoked by the `Makefile`. They allow the code to run with venv environment, Docker Images or on a OpenPBS based cluster. |
| **`checkpoints/`** | Pre-trained model weights (downloaded from the GitHub release) used by the demonstration tests. |
| **`results/`** | Default output directory. Each run writes here its logs, `run_data.json`, figures and a snapshot of the resolved config under `.hydra/`. |
| **`main.py`** | The single entry point. Reads the composed config, instantiates everything and runs the requested phases (train / test / inference plots). |

---

## 3. Hydra

Hydra is the tool allowing flexibility in this project. The entire experiment setup is assembled from small, composable YAML files in `config/` and **any value can be overridden from the command line**. Nothing in `main.py` is hard-coded.

### 3.1 Config groups

```
config/
├── config.yaml              # root: defaults, seed, flags (train/test/plot), W&B settings
├── problem/                 # toy or turbulence problem
├── model/                   # architecture choice
├── data_source/             # dataset definitions
├── toy_function/            # allows to construct different parametric functions
├── training/                # epochs, batch size, metrics, early stopping
├── optimizer/               # adam/adamw + schedulers
├── loss/                    # l2.yaml | combined.yaml
├── preprocessing/           # turbulence dataset split & normalization
└── postprocessing/          # which plots to generate
```

The `problem` group is the *controller*: selecting `problem=toy` or `problem=turbulence` pulls in a set of `data_source`, `training`, `model` and `postprocessing` defaults consistent with the selected problem, so that the two workflows never get mixed.


## 4. Reproducibility: How to run code and tests

To use our library and reproduce the results we provide two alternatives:
1. a local **`uv`** environment
2. a pre-built **Docker** image (Option B)

Both are driven by the same `Makefile`, the only difference is the value of the `ENGINE` variable (`venv` or `docker`) and that:

* **Docker Image:** Recommended for immediate reproduction. It arrives fully self-contained with all required datasets and model checkpoints pre-installed. No external setup or downloads are required.
* **Local Setup (`uv`):** Requires you to manually load and place the datasets and checkpoints before running execution commands, as the local clone only contains the source code. This happens because hosting large binaries in Git violates repository best practices

### CPU / GPU flag

The execution device is selected with the `USE_GPU` flag of the `Makefile` (`true` / `false`, default
`false`). JAX is installed with CUDA 12 support.

- If you set `USE_GPU=true` but no NVIDIA driver/hardware is detected, `run_experiment.sh` prints a
  hardware warning and **safely reverts to CPU**.
- Independently, `main.py` sets `JAX_PLATFORMS=cpu` whenever no GPU device is visible, so JAX itself
  never errors out.

The toy problem is small and is best run on CPU, while the turbulence problem benefits from a GPU.

### Option A: run locally using a `uv` environment

> **NOTE:** we use `uv`, but a plain `python -m venv` works just as well.

1. Clone the repository:
   ```bash
   git clone git@github.com:Riccardo-Rota/JAX_Hypernetworks.git
   ```

2. Move inside the repository root folder:
   ```bash
   cd JAX_Hypernetworks
   ```

3. Create the virtual environment with `uv` (Python 3.12):
   ```bash
   uv venv jax_hypernetworks --python 3.12.3
   ```

4. Activate it:
   ```bash
   source jax_hypernetworks/bin/activate
   ```

5. Install the packages:
   ```bash
   uv pip install -r requirements.txt
   ```

6. *(Turbulence only)* Load the astrophysical turbulence dataset from Hugging Face and run the provided preprocessing scripts:
   ```bash
   make load-data
   ```
   The toy problem needs no download: its data is generated on the fly.

7. *(Optional)* Download the pre-trained checkpoints used by the demonstration tests, from our
   `v1.0.0` GitHub release:
   ```bash
   wget https://github.com/Riccardo-Rota/JAX_Hypernetworks/releases/download/v1.0.0/checkpoints.zip
   unzip checkpoints.zip
   rm checkpoints.zip
   ```
   **NOTE:** this step is mandatory if you want to run provided tests.

8. Set the `ENGINE` environment variable to `venv` once for all:
   ```bash
   export ENGINE="venv"
   ```

9. **Run the code.** You can now run experiments and the provided tests through the `Makefile`:
   ```bash
   make run-local
   ```
   See [§5](#5-the-guided-tour-make-run-test) for the test suite and [§6](#6-general-usage) for general
   usage.

### Option B: run with the Docker image

We also provide a ready-to-use Docker container, so you do not need to install any Python dependency
yourself.

1. Clone the repository:
   ```bash
   git clone git@github.com:Riccardo-Rota/JAX_Hypernetworks.git
   ```

2. Move inside the repository root folder:
   ```bash
   cd JAX_Hypernetworks
   ```

3. Pull the Docker image (install Docker and `docker login` first if needed):
   ```bash
   docker pull leonardobocchieri/jax-hypernetworks:latest
   ```
   > **NOTE:** the image is approximately 10 GB, so this step can take a few minutes.

4. Set the `ENGINE` environment variable to `docker` (once and for all):
   ```bash
   export ENGINE="docker"
   ```

5. **Run the code** exactly as in Option A, through the `Makefile`:
   ```bash
   make run-local
   ```

> **How the Docker engine works: what is baked in vs. mounted.** The image *contains* the
> **library** code that needs to be kept unchanged (together with all dependencies).
> It is important to notice that `run_experiment.sh` **bind-mounts** the parts you actually edit between experiments:
> `config/`, `main.py` and `results/`. They are therefore taken from your local clone into the container. 
> The `Makefile` and the `scripts/` themselves run on the *host* and merely orchestrate the `docker run` call.
> The practical consequence: modifying a configuration file or `main.py` locally takes effect immediately inside the container. However, modifying local library files (such as the `models/` directory) will have **no effect** on execution, because the container strictly executes the library copies baked into the Docker image.

> **Ownership note.** Files created from inside the container belong to `root`. If you later run locally
> and hit a permission error, reclaim ownership with `sudo chown -R $(whoami) results`.


---

## 5. The guided tour: `make run-test*`

The `Makefile` ships a sequence of **self-documenting demonstrations** that walk through the library's
capabilities, from inspecting a pre-trained model to comparing JIT compilation speed-ups. Run them with
your chosen engine already exported (`export ENGINE=venv` or `docker`). Each writes its output under a
dedicated `results/...` directory.

| Target | What it demonstrates |
| --- | --- |
| **`make run-test1`** | **Inference from a good checkpoint.** Loads a pre-trained toy MLP (`train_model=false`) and produces prediction and metric reports — the model fits the target function well. |
| **`make run-test2`** | **Training from scratch: the SIREN initialization matters.** Trains a *naive* SIREN (poor initialization → poor fit), then a properly-initialized SIREN — illustrating why SIREN's weight scheme is essential for high-frequency targets. |
| **`make run-test3`** | **Fine-tuning a checkpoint.** First loads a deliberately under-trained checkpoint (poor results), then resumes training (`train_model=true`) from those same weights to show the model recovering. |
| **`make run-test4`** | **A challenging target.** Trains a SIREN on a high-frequency sine wave (`toy_function=highfreq_sine`), stress-testing the method on a hard signal. |
| **`make run-test5`** | **JIT vs. no-JIT.** Runs the same short training twice — with and without JAX JIT compilation (`JAX_DISABLE_JIT=1`) — on CPU and prints the measured speed-up. |
| **`make run-test6`** | **Turbulence inference (MLP).** Loads a pre-trained turbulence MLP and reconstructs the velocity fields, comparing prediction against ground truth. *Requires the turbulence dataset (`make load-data`).* |
| **`make run-test7`** | **Turbulence inference (SIREN).** Same as test 6 but with a SIREN target on the *density* field. *Requires the turbulence dataset.* |

These targets double as **usage recipes**: each one is simply a `make run-local` call with a specific
`OVERRIDES` string, so reading the `Makefile` shows exactly how to compose your own experiments.

---

## 6. General usage

Every experiment is launched through `make run-local`, which accepts three knobs:

```bash
make run-local USE_GPU=<true|false> ENGINE=<venv|docker> OVERRIDES="<hydra overrides>"
```

- **`ENGINE`** — `venv` (local interpreter) or `docker`. Defaults to `venv`.
- **`USE_GPU`** — `true` or `false` (default `false`). See the [CPU/GPU flag](#cpu--gpu-flag) note above.
- **`OVERRIDES`** — any space-separated list of Hydra overrides. Alternatively, edit the files in
  `config/` directly.

Full example:

```bash
make run-local USE_GPU=true OVERRIDES="problem=toy model=toy_siren training.epochs=1000 use_wandb=true"
```

### Phases: what actually runs

`main.py` executes up to three phases, each controlled by a boolean flag (overridable):

- `train_model=true` — train and checkpoint the model.
- `test_model=true` — evaluate metrics on the test set.
- `plot_inference=true` — generate the post-processing figures.

To merely inspect a checkpoint, set `train_model=false checkpoint='checkpoints/<name>'`, exactly as the
demonstration tests do.

On GPU you may see one-off **XLA autotuning warnings** during compilation — these are harmless (see
[§10](#10-troubleshooting--notes)).

---

## 7. Datasets

### Toy

No download is required. `ToyDataSource` generates samples on demand by evaluating a parametric function
(parsed safely from a string via `get_function_from_string`) over its hyper- and variable domains. The
function and domains are defined in `config/toy_function/` — `default.yaml` for the standard target and
`highfreq_sine.yaml` for the hard, high-frequency case.

### Turbulence

The turbulence workload uses the **turbulent radiative layer 2D** dataset
([`polymathic-ai/turbulent_radiative_layer_2D`](https://huggingface.co/datasets/polymathic-ai/turbulent_radiative_layer_2D)
on Hugging Face). `make load-data` runs two steps:

1. `download_data.py` — fetches the raw HDF5 file for the selected cooling time `tcool` (configurable in
   `config/preprocessing/turbulence.yaml`).
2. `preprocessing.py` — performs a **temporal** train/val/test split (consecutive-timestep chunks are
   allocated to each split), computes normalization statistics **from the training split only**, and
   writes `turbulence_dataset_{train,val,test}.hdf5` into `datasets/`.

At run time `InMemoryHDF5Source` loads the relevant split fully into RAM. The fields, coordinates and
prediction targets (`density`, `pressure`, `velocity_x`, `velocity_y`, …) are declared in
`config/data_source/turbulence.yaml` and can be remapped from the command line — e.g.
`data_source.base_dataset.target_keys=['density']`.

---

## 8. Weights & Biases

The project integrates with [Weights & Biases](https://wandb.ai) for live monitoring and logging, but
**W&B is entirely optional and disabled by default** (`use_wandb=false`). To turn it on, add
`use_wandb=true` to your overrides — and make sure your project/entity are set:

```bash
make run-local OVERRIDES="problem=toy use_wandb=true wandb_settings.project=my_project wandb_settings.entity=my_team"
```

When enabled, the run uploads its metrics, the fully-resolved config and every generated figure, and
names the run after its problem/date/time.

### Authentication

Generate an API key from your W&B account and store it securely in a `~/.netrc` file in your home
directory (the same file works on both your local terminal and a cluster login node):

```bash
touch ~/.netrc
chmod 600 ~/.netrc
```

Then open the file in a text editor (e.g. `nano ~/.netrc`) and paste the following block:

```text
machine api.wandb.ai
  login user
  password <your_API_key>
```

*(Replace `<your_API_key>` with your actual 40-character Weights & Biases token. Do not change the word
`user`.)*

`run_experiment.sh` parses this file automatically: if a key is found it logs you in; if not, it forces
**offline** mode so a run never blocks on a login prompt.

### Syncing offline runs

Runs produced offline (typically on a cluster) can be uploaded afterwards:

```bash
RESULTS_DIR=results/runs_turbulence make sync-cluster
```

This walks the results tree for `offline-run-*` directories and `wandb sync`s them in parallel through
the apptainer image.

---

## 9. Running on an HPC cluster

For long or GPU-heavy jobs, the library runs on an HPC cluster via **apptainer** (the HPC-friendly
counterpart of Docker) and the **PBS** scheduler.

1. Clone the repository on the cluster (preferably under your `work` space).
2. Convert the Docker image to an apptainer `.sif`:
   ```bash
   apptainer pull jax-hypernetworks.sif docker://leonardobocchieri/jax-hypernetworks:latest
   ```
3. Submit a job with `make submit-cluster`. It accepts two extra arguments:
   - **`USE_GPU`** (`true`/`false`) — decides which queue the job is sent to (GPU or CPU).
   - **`OVERRIDES`** — the Hydra overrides string (or edit the files in `config/` directly).
   ```bash
   make submit-cluster USE_GPU=true OVERRIDES="problem=turbulence use_wandb=true training.epochs=500"
   ```

The PBS script (`scripts/submission.pbs`) stages the code onto fast local scratch, mirrors results back
to your submission directory every minute (so you can `tail -f` the live log), and rescues all output on
exit. A complete walk-through — VPN, SSH, queue monitoring, retrieving results — is in
[CLUSTER_TUTORIAL.md](CLUSTER_TUTORIAL.md).

### Watching logs live on the cluster

To follow a run's progress, find the path of the log being written and tail it:

```bash
ls results/runs_*/*/*/default.log        # Hydra/run log
ls results/runs_*/*/*/training_log.txt   # per-epoch training log

tail -F <path_to_log>
```

---

## 10. Troubleshooting & notes

### XLA autotuning warnings (GPU)

When running on GPU you may see repeated lines such as:

```
W external/xla/.../dot_search_space.cc:200] All configs were filtered out because none of them
sufficiently match the hints. ... Working around this by using the full hints set instead.
```

**These are warnings, not errors** — the run is fine. The warning concerns only how XLA chose the kernel
and it happens once, during compilation. It depends on your GPU and JAX version, so it may not show up on
every machine. To silence them, set `TF_CPP_MIN_LOG_LEVEL=2` before launching.

### Permissions after a Docker run

Files written from inside the container belong to `root`. Reclaim them with:

```bash
sudo chown -R $(whoami) results
```

### Further reading

- [DOCKER_TUTORIAL.md](DOCKER_TUTORIAL.md) — building, testing and publishing the Docker image (incl. WSL 2).
- [CLUSTER_TUTORIAL.md](CLUSTER_TUTORIAL.md) — end-to-end guide to the POLIMI HPC cluster.

---

## License

See [LICENSE](LICENSE).
