# PACSproject TODO for replication

1. Install Docker Desktop and open the Docker Desktop. Check if Docker is ready with:
```
    docker --version
```
2. Clone repository
3. Create dataset/ folder (TODO: need to check if strictly necessary)
4. Make .sh file executable:
```
    chmod +x run_toy.sh scripts/run_turbulence.sh
```
5. Run the code, either toy or turbulence with:
```
    ./scripts/run_toy.sh
```
or
```
    ./scripts/run_turbulence.sh
```

**BE AWARE**: User is cloning entire repository: they see all library folders in their local machine. However, bash script files use folders inside Docker container, which relies on its own baked-in copies of those library folders.

## Docker image installation: WSL (version 2 required)
1. Setup Docker Desktop:
    - install Docker Desktop from the [official website](https://www.docker.com/products/docker-desktop/). During installation, ensure the box that says "Use the WSL 2 based engine" is checked.
    - open Docker Desktop and navigate to Settings > Resources > WSL Integration. Enable integration with default WSL distro or any additional distro you use (e.g. Ubuntu). Click Apply & restart 
2. Download Docker image:
    - open WSL terminal
    - Execute `docker pull leonardobocchieri/jax-hypernetworks:latest` (~4 GB download, it may take a few minutes)

## Running locally with docker
To run locally `main.py` with docker, execute `make run-local`. You can pass two extra arguments:
- USE_GPU (true or false, default true, it falls back to CPU if GPU not available)
- OVERRIDES (string containing all overrides to be added to hydra config. Alternatively, just modify config files in `config/`)
Example: `make run-local USE_GPU=true OVERRIDES="problem=toy use_wandb=true training.epochs=100"`

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