# Default Variables
ENGINE ?= venv
USE_GPU ?= true
OVERRIDES ?= 

.PHONY: run-local submit-cluster sync-cluster run-test1 run-test2

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


run-test1:	OVERRIDES= training.epochs=10
run-test1:	run-local

run-test2:	OVERRIDES= problem=toy use_wandb=false
run-test2:	run-local