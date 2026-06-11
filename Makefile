# --- Default Variables ---
USE_GPU ?= true
OVERRIDES ?= 

.PHONY: run-local submit-cluster sync-cluster

run-local:
	@echo "Running locally on Docker (GPU: $(USE_GPU))..."
	ENGINE=docker USE_GPU=$(USE_GPU) bash scripts/run_experiment.sh $(OVERRIDES)

submit-cluster:
	@if [ "$(USE_GPU)" = "true" ]; then \
		echo "Submitting to HPC GPU queue..."; \
		qsub -q gpu -l select=1:ncpus=4:ngpus=1 -v USE_GPU=$(USE_GPU),OVERRIDES="$(OVERRIDES)" scripts/submission.pbs; \
	else \
		echo "Submitting to HPC CPU queue..."; \
		qsub -q cpu -l select=1:ncpus=16 -v USE_GPU=$(USE_GPU),OVERRIDES="$(OVERRIDES)" scripts/submission.pbs; \
	fi

# Upload offline W&B runs to the cloud. Run on the LOGIN NODE (needs internet).
# Uses the wandb library inside the apptainer image -- no host install required.
# example usage:
#   RESULTS_DIR=results/runs_turbulence make sync-cluster
sync-cluster:
	@echo "Syncing offline W&B runs via apptainer..."
	bash scripts/sync_wandb.sh