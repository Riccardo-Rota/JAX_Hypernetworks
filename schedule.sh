#!/bin/bash

PARAMS="--multirun data.N=100,1000,10000 data.n_realizations=1,10,100 targetnetwork.num_neurons=[1,16,16,1],[1,32,32,1],[1,64,64,1]"

FUNC_1="'lambda mu, l, k, x: jnp.exp(-l*x) + mu*x + jnp.sin(k*jnp.pi*x)'"
FUNC_2="'lambda mu, l, k, x: jnp.exp(-l*x) + mu*x + k*x**2'"

# --- RUN 1: MLP + Function 1 ---
echo "Starting Run 1/4: MLP with Sinusoidal Function"
python mainhydra.py $PARAMS targetnetwork=mlp data.f_to_learn="$FUNC_1"

echo "Run 1 complete. Cooling down for 30 minutes..."
sleep 1800

# --- RUN 2: MLP + Function 2 ---
echo "Starting Run 2/4: MLP with Polynomial Function"
python mainhydra.py $PARAMS targetnetwork=mlp data.f_to_learn="$FUNC_2"

echo "Run 2 complete. Cooling down for 30 minutes..."
sleep 1800

# --- RUN 3: SIREN + Function 1 ---
echo "Starting Run 3/4: SIREN with Sinusoidal Function"
python mainhydra.py $PARAMS targetnetwork=siren data.f_to_learn="$FUNC_1"

echo "Run 3 complete. Cooling down for 30 minutes..."
sleep 1800

# --- RUN 4: SIREN + Function 2 ---
echo "Starting Run 4/4: SIREN with Polynomial Function"
python mainhydra.py $PARAMS targetnetwork=siren data.f_to_learn="$FUNC_2"

echo "All runs completed successfully."