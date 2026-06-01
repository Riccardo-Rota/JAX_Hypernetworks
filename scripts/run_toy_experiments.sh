#!/bin/bash

# Ensure the script exits if any command fails
set -e

# Run the Hydra application with the multirun flag (-m)
# This will execute a grid search across all models, N values, and realizations.
python main.py -m \
    problem=toy \
    model=toy_mlp,toy_siren_naive \
    data_source.train.N=10000,100000,1000000 \
    data_source.train.n_realizations=1,10,100

# NOTE: interrotto dopo prime 4 run (con N=100000 ci ha messo troppo tempo))