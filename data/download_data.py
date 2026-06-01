"""
download_data.py
----------------
Download a single HDF5 file from The Well's turbulent_radiative_layer_2D
Hugging Face dataset. We always pull from the 'train' folder because it
contains the most trajectories for any given tcool value (8 out of 10).
We do NOT use the HF train/valid/test split — we'll re-split ourselves
along the time axis in preprocessing.py.
"""
from pathlib import Path
import hydra
from omegaconf import DictConfig
from huggingface_hub import hf_hub_download

REPO_ID = "polymathic-ai/turbulent_radiative_layer_2D"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = PROJECT_ROOT / "datasets"

def download(tcool: str) -> str:
    # Safely format the float to 2 decimal places to ensure "0.1" becomes "0.10"
    tcool_str = f"{float(tcool):.2f}"
    
    filename = f"data/train/turbulent_radiative_layer_tcool_{tcool_str}.hdf5"
    
    path = hf_hub_download(
        repo_id=REPO_ID,
        filename=filename,
        repo_type="dataset",
        local_dir=str(DATASETS_DIR),
    )
    print(f"Downloaded: {path}")

@hydra.main(version_base="1.3", config_path="../config", config_name="config")
def main(cfg: DictConfig):
    tcool = cfg.preprocessing.data.tcool
    print(f"Targeting tcool: {tcool}")
    download(tcool)

if __name__ == "__main__":
    main()