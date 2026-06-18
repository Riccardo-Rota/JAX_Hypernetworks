








# How to run code and tests

To use our library and test the results, we provide two alternatives: one using local uv environment, one using a Docker Image.

explain CPU/GPU flag

## Option A: run locally using uv environment

NOTE: we use uv but you can use also venv

1. Clone the repository:
```bash
    git clone git@github.com:Riccardo-Rota/JAX_Hypernetworks.git
```

2. Move inside repository root folder:
```bash
    cd JAX_Hypernetworks
```

3. Create virtual with uv:
```bash
    uv venv jax_hypernetworks --python 3.12.3
```

2. Activate it:
```bash
    source jax_hypernetworks/bin/activate
```

3. Install packages:
```bash
    uv pip install -r requirements.txt
```

4. Load astrophysical turbulence datasets from Hugging Face and perform preprocessing using provided preprocessing scripts:
```bash
    make load-data
```

5. Download pre-trained model checkpoints directly from our v1.0.0 release on GitHub:
```bash
    wget https://github.com/Riccardo-Rota/JAX_Hypernetworks/releases/download/v1.0.0/checkpoints.zip
    unzip checkpoints.zip
    rm checkpoints.zip
```

6. Set ENGINE environmental variable to venv:
```bash
    export ENGINE="venv"
```

6.  **Run the Code**:
You can now run code and provided tests using the `Makefile`.

**TODO:** write what each test means
**TODO:** explain how to use for general purposes (make run-local)

```bash
    make run-local
```

## Option B: run with Docker Image

We also provide the possibility to use Docker container

1. Clone the repository:
```bash
    git clone git@github.com:Riccardo-Rota/JAX_Hypernetworks.git
```

2. Move inside repository root folder:
```bash
    cd JAX_Hypernetworks
```

3. Pull Docker Image (if not done, install Docker and login):
```bash
    docker pull leonardobocchieri/jax-hypernetworks:latest
```
**NOTE**: this operation can require some minutes

4. Set ENGINE environmental variable to docker:
```bash
    export ENGINE="docker"
```

5. **Run the Code**:
You can now run code and provided tests using the `Makefile`.


**BE AWARE**: User is cloning entire repository: they see all library folders in their local machine. However, bash script files use folders inside Docker container, which relies on its own baked-in copies of those library folders.
**TODO:** explain that only config, main, scripts and makefile are not in docker (so local version is used through mount binding).



## Running on cluster with apptainer
To run on cluster `main.py` with apptainer:
- clone the repository in the cluster 
- pull docker image with apptainer by executing `apptainer pull jax-hypernetworks.sif docker://leonardobocchieri/jax-hypernetworks:latest`
- execute `make submit-cluster`. You can pass two extra arguments:
    - USE_GPU (true or false, default true, decides which queue to put our job in)
    - OVERRIDES (string containing all overrides to be added to hydra config. Alternatively, just modify config files in `config/`)
  Example: `make submit-cluster USE_GPU=true OVERRIDES="problem=toy use_wandb=true training.epochs=100"`

## Saving logs and checkpoints to Weights & Biases
To connect the run to wandb for interactive live monitoring and logging, login to your wandb account online and generate an API key.
Then, securely store it in a `~/.netrc` file in your home directory.

To create and secure the file (the commands are identical for both your local terminal and the cluster login node), execute:
```bash
touch ~/.netrc
chmod 600 ~/.netrc
```

Then, open the file in a text editor (e.g., `nano ~/.netrc`) and paste the following structure:
```text
machine api.wandb.ai
  login user
  password <your_API_key>
```
*(Note: Replace `<your_API_key>` with your actual 40-character Weights & Biases token. Do not change the word `user`).*

## Checking live the logs on cluster

To check live the logs, find the path to the log being written by the run, by running
`ls runs/runs_*/*/*/default.log`
or the one of the training function, by running
`ls runs/runs_*/*/*/training.txt`

Then to visualize the log run 
`tail -F <path_to_log>`

**TODO:** talk about W&B if you want to put it on true



## NOTE: XLA autotuning warnings

When running on GPU you may see repeated lines like:
```
W external/xla/.../dot_search_space.cc:200] All configs were filtered out because none of them
sufficiently match the hints. ... Working around this by using the full hints set instead.
```
**These are warnings, not errors** — the run is fine. The warning is only about how XLA chose the kernel and it happens only once, during compilation. It depends on your GPU and JAX version, so it may not show up on every machine.

To silence them, set `TF_CPP_MIN_LOG_LEVEL=2` before launching.