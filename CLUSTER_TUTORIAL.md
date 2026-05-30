# HOW TO USE POLIMI CLUSTER

1. Download VPN provider and connect (https://www.ict.polimi.it/network/vpn/)
2. Connect to cluser and initialize. Open Ubuntu terminal and do:
```
    ssh username@10.78.18.100
```
**NOTE**: username= u<codice persona>

Enable the PBS commands (required on your first login or in a new shell):
```
    . /etc/profile.d/pbs.sh
```

3. Clone repository inside work folder (not in home since we have more space in work)
```
    cd /work/$(whoami)
    mkdir jax_project
    cd jax_project
    git clone ...
```
**NOTE**: you need to generate new ssh key for cluster

4. Use Appainter to pull Docker image hosted on Docker Hub
Inside repository in cluster, run:
```
    apptainer pull jax-hypernetworks.sif docker://leonardobocchieri/jax-hypernetworks:latest
```
In this way Appainter creates a jax-hypernetworks.sif (converted HPC-compatible Docker Image)

5. Create .pbs file:

To run on CPU:
```
#!/bin/bash
#PBS -N jax_training_cpu
#PBS -q cpu
#PBS -l select=1:ncpus=16
#PBS -l walltime=12:00:00
#PBS -joe

# Define paths
WORK_DIR="/work/$USER/jax_project"
SCRATCH_DIR="/scratch_local/$USER/$PBS_JOBID"

# Setup fast local scratch space on the allocated CPU node
mkdir -p "$SCRATCH_DIR"
cd "$SCRATCH_DIR"

# Copy your configuration, dataset, and entrypoint to the fast drive
cp -r "$WORK_DIR/config" .
cp -r "$WORK_DIR/dataset" .
cp "$WORK_DIR/main.py" .
mkdir results

# Execute the container without GPU support (--nv is removed)
apptainer exec \
  --bind $(pwd)/config:/app/config \
  --bind $(pwd)/dataset:/app/dataset \
  --bind $(pwd)/results:/app/results \
  --bind $(pwd)/main.py:/app/main.py \
  "$WORK_DIR/jax-hypernetworks.sif" bash -c "cd /app && python main.py"

# Copy the generated results back to persistent storage before the node cleans itself
cp -r results "$WORK_DIR/"
```

To run on GPU:
```
#!/bin/bash
#PBS -N jax_training
#PBS -q gpu
#PBS -l select=1:ncpus=4:ngpus=1
#PBS -l walltime=12:00:00
#PBS -joe

# Define paths
WORK_DIR="/work/$USER/jax_project"
SCRATCH_DIR="/scratch_local/$USER/$PBS_JOBID"

# Setup fast local scratch space on the allocated GPU node
mkdir -p "$SCRATCH_DIR"
cd "$SCRATCH_DIR"

# Copy your configuration, dataset, and entrypoint to the fast drive
cp -r "$WORK_DIR/config" .
cp -r "$WORK_DIR/dataset" .
cp "$WORK_DIR/main.py" .
mkdir results

# Execute the container
# --nv enables GPU support for NVIDIA hardware.
# --bind acts exactly like Docker's -v flag to map the wormholes.
apptainer exec --nv \
  --bind $(pwd)/config:/app/config \
  --bind $(pwd)/dataset:/app/dataset \
  --bind $(pwd)/results:/app/results \
  --bind $(pwd)/main.py:/app/main.py \
  "$WORK_DIR/jax-hypernetworks.sif" bash -c "cd /app && python main.py"

# Copy the generated results back to persistent storage before the node cleans itself
cp -r results "$WORK_DIR/"
```

6. Submit script to cluster queue:
```
    qsub train.pbs
```

The cluster will return a job ID. You can monitor the status of your execution with:
```
    qstat -u $(whoami)
```

We can inspect in real time any output using
```
    tail -f jax_training.o*
```

7. Retrieve results on local machine:
**NOTE**: Run outside cluster, in Ubuntu terminal
```
    scp -r username@10.78.18.100:/work/username/jax_project/results ~/JAX_Hypernetworks/
```