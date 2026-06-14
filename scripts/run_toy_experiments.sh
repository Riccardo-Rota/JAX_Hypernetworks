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

# Direttamente sul cluster ho runnato


###########################
make submit-cluster OVERRIDES="problem=toy data_source.train.N=100 data_source.train.n_realizations=1 model=toy_mlp"
make submit-cluster OVERRIDES="problem=toy data_source.train.N=1000 data_source.train.n_realizations=1 model=toy_mlp"
make submit-cluster OVERRIDES="problem=toy data_source.train.N=10000 data_source.train.n_realizations=1 model=toy_mlp"

make submit-cluster OVERRIDES="problem=toy data_source.train.N=100 data_source.train.n_realizations=10 model=toy_mlp"
make submit-cluster OVERRIDES="problem=toy data_source.train.N=1000 data_source.train.n_realizations=10 model=toy_mlp"
make submit-cluster OVERRIDES="problem=toy data_source.train.N=10000 data_source.train.n_realizations=10 model=toy_mlp"

make submit-cluster OVERRIDES="problem=toy data_source.train.N=100 data_source.train.n_realizations=100 model=toy_mlp"
make submit-cluster OVERRIDES="problem=toy data_source.train.N=1000 data_source.train.n_realizations=100 model=toy_mlp"
make submit-cluster OVERRIDES="problem=toy data_source.train.N=10000 data_source.train.n_realizations=100 model=toy_mlp"



make submit-cluster OVERRIDES="problem=toy data_source.train.N=100 data_source.train.n_realizations=1 model=toy_siren"
make submit-cluster OVERRIDES="problem=toy data_source.train.N=1000 data_source.train.n_realizations=1 model=toy_siren"
make submit-cluster OVERRIDES="problem=toy data_source.train.N=10000 data_source.train.n_realizations=1 model=toy_siren"

make submit-cluster OVERRIDES="problem=toy data_source.train.N=100 data_source.train.n_realizations=10 model=toy_siren"
make submit-cluster OVERRIDES="problem=toy data_source.train.N=1000 data_source.train.n_realizations=10 model=toy_siren"
make submit-cluster OVERRIDES="problem=toy data_source.train.N=10000 data_source.train.n_realizations=10 model=toy_siren"

make submit-cluster OVERRIDES="problem=toy data_source.train.N=100 data_source.train.n_realizations=100 model=toy_siren"
make submit-cluster OVERRIDES="problem=toy data_source.train.N=1000 data_source.train.n_realizations=100 model=toy_siren"
make submit-cluster OVERRIDES="problem=toy data_source.train.N=10000 data_source.train.n_realizations=100 model=toy_siren"