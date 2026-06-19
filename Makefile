# Default Variables
ENGINE ?= venv
USE_GPU ?= false
OVERRIDES ?= 

.PHONY: load-data run-local submit-cluster sync-cluster run-test1 run-test2 run-test3 run-test4 run-test5 run-test6 run-test7

load-data:
	@echo "Downloading data..."
	python data_processing/download_data.py
	python data_processing/preprocessing.py

run-local:
	@echo "Running locally with $(ENGINE) (GPU: $(USE_GPU))..."
	ENGINE=$(ENGINE) USE_GPU=$(USE_GPU) bash scripts/run_experiment.sh $(OVERRIDES)

submit-cluster:
	@if [ "$(USE_GPU)" = "true" ]; then \
		echo "Submitting to HPC GPU queue..."; \
		qsub -q gpu -l select=1:ncpus=4:ngpus=1 -v USE_GPU=$(USE_GPU),OVERRIDES="$(OVERRIDES)" scripts/submission.pbs; \
	else \
		echo "Submitting to HPC CPU queue..."; \
		qsub -q cpu -l select=1:ncpus=16 -v USE_GPU=$(USE_GPU),OVERRIDES="$(OVERRIDES)" scripts/submission.pbs; \
	fi

# Upload logs and results to W&B server (for offline runs)
# example usage:
#   RESULTS_DIR=results/runs_turbulence make sync-cluster
sync-cluster:
	@echo "Syncing offline W&B runs via apptainer..."
	bash scripts/sync_wandb.sh


## TESTS

run-test1:
	@echo "=== Inspect the results of a pre-trained model which performs well ==="
	$(MAKE) run-local OVERRIDES="problem=toy model=toy_mlp train_model=false checkpoint='checkpoints/toy_mlp' hydra.run.dir='results/test1'"

run-test2:
	@echo "=== Train a naive SIREN from scratch (performs poorly), inspect results ==="
	$(MAKE) run-local OVERRIDES="problem=toy model=toy_siren_naive training.epochs=200 hydra.run.dir='results/test2/naive'"
	@echo "=== Train a SIREN from scratch (performs better), inspect results ==="
	$(MAKE) run-local OVERRIDES="problem=toy model=toy_siren training.epochs=200 hydra.run.dir='results/test2/siren'"

run-test3:
	@echo "=== Load a pre-trained model which performs poorly, inspect results ==="
	$(MAKE) run-local OVERRIDES="problem=toy model=toy_mlp train_model=false hydra.run.dir='results/test3/pretrained' checkpoint='checkpoints/toy_mlp_partial'"
	@echo "=== Fine-tune the model ==="
	$(MAKE) run-local OVERRIDES="problem=toy model=toy_mlp train_model=true training.epochs=200 hydra.run.dir='results/test3/finetuned' checkpoint='checkpoints/toy_mlp_partial'"

run-test4:
	@echo "=== Run with a challenging function: high-frequency sine wave ==="
	$(MAKE) run-local OVERRIDES="problem=toy model=toy_siren training.epochs=200 toy_function=highfreq_sine hydra.run.dir='results/test4'"

run-test5:
	@echo "=== Compare training speed with and without jit compilation ==="
	ENGINE=$(ENGINE) RUN_DIR=results/test5 bash scripts/jit_comparison.sh

run-test6:
	@echo "=== Load a pre-trained model for turbulence and inspect results ==="
	$(MAKE) run-local OVERRIDES="problem=turbulence model=turbulence_mlp train_model=false checkpoint='checkpoints/velocity_mlp' hydra.run.dir='results/test6'"

run-test7:
	@echo "=== Load a pre-trained model for turbulence and inspect results ==="
	$(MAKE) run-local OVERRIDES="problem=turbulence model=turbulence_siren train_model=false checkpoint='checkpoints/density_siren' data_source.base_dataset.target_keys=['density'] hydra.run.dir='results/test7'"