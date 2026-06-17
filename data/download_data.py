from pathlib import Path
import sys
from omegaconf import DictConfig
from hydra import initialize_config_dir, compose
from huggingface_hub import hf_hub_download

REPO_ID = "polymathic-ai/turbulent_radiative_layer_2D"

# Find project root and datasets directory.
# Project root is file specific(download_data.py is two levels deep in the directory structure)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = PROJECT_ROOT / "datasets"

def download(tcool: str) -> None:
    """
    Download the HDF5 file corresponding to the given tcool value from the Hugging Face dataset.
    NOTE: We always pull from the 'train' folder because it contains the most trajectories for any given tcool value (8 out of 10).
    We do NOT use the HF train/valid/test split, because we will re-split ourselves along the time axis in preprocessing.py.
    Args:
        tcool (str): The tcool value to identify the specific HDF5 file to download.
    """

    # Safely format the float to 2 decimal places to ensure "0.1" becomes "0.10"
    tcool_str = f"{float(tcool):.2f}"
    
    # File name inside the Hugging Face dataset
    filename = f"data/train/turbulent_radiative_layer_tcool_{tcool_str}.hdf5"
    
    # Download using Hugging Face Hub API
    path = hf_hub_download(
        repo_id=REPO_ID,
        filename=filename,
        repo_type="dataset",
        local_dir=str(DATASETS_DIR),
    )

def main():
    # Hydra's Composition API doesn't automatically parse sys.argv, so we pass them manually.
    overrides = sys.argv[1:]
    config_dir = str(PROJECT_ROOT / "config")

    # Use the Composition API to load config without creating an output directory.
    with initialize_config_dir(version_base="1.3", config_dir=config_dir):
        cfg = compose(config_name="config", overrides=overrides)
        tcool = cfg.preprocessing.data.tcool
        download(tcool)

if __name__ == "__main__":
    main()