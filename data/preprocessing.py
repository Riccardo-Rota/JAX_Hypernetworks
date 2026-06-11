"""
preprocessing.py
----------------
Reads a master HDF5 file from The Well, extracts ONE trajectory, splits the
TIME axis into train/val/test using a chunked random allocation, flattens
each split over (x, y), and saves each split as a separate HDF5 file.

Chunking strategy
-----------------
The temporal domain (T timesteps) is divided into consecutive chunks of
`chunk_size` timesteps. Within each chunk, timesteps are randomly assigned
to train/val/test according to the given ratios. This way every macroscopic
phase of the turbulent evolution appears in all three splits.

Each output row is float32: [time, x, y, density, pressure, vel_x, vel_y]
"""

from pathlib import Path

import h5py
import numpy as np
from omegaconf import DictConfig
import shutil
import sys
from hydra import initialize, compose

# Anchor everything to the project root; identical behaviour on any OS,
# regardless of the user's current working directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = PROJECT_ROOT / "datasets"


def temporal_chunk_split(
    n_timesteps: int,
    chunk_size: int,
    train_ratio: float,
    val_ratio: float,
    seed: int,
):
    """
    Return three sorted arrays of time indices (train, val, test).

    Within each chunk of `chunk_size` consecutive timesteps:
      - int(chunk_size * train_ratio) timesteps go to train
      - int(chunk_size * val_ratio)   timesteps go to val
      - the remainder                  go to test
    Allocation within the chunk is uniformly random.
    """
    rng = np.random.default_rng(seed)
    train_idx, val_idx, test_idx = [], [], []

    for chunk_start in range(0, n_timesteps, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n_timesteps)
        size = chunk_end - chunk_start

        n_tr = int(size * train_ratio)
        n_va = int(size * val_ratio)
        # remainder goes to test (preserves the ratio's intent on short chunks)

        chunk = rng.permutation(np.arange(chunk_start, chunk_end))
        train_idx.extend(chunk[:n_tr].tolist())
        val_idx.extend(chunk[n_tr:n_tr + n_va].tolist())
        test_idx.extend(chunk[n_tr + n_va:].tolist())

    return (
        np.sort(np.array(train_idx, dtype=int)),
        np.sort(np.array(val_idx,   dtype=int)),
        np.sort(np.array(test_idx,  dtype=int)),
    )


def _check_non_empty_splits(train_t, val_t, test_t, chunk_size,
                            train_ratio, val_ratio, test_ratio):
    """Raise an error if any split is empty."""
    for name, arr, ratio in [
        ("train", train_t, train_ratio),
        ("val",   val_t,   val_ratio),
        ("test",  test_t,  test_ratio),
    ]:
        if len(arr) == 0:
            per_chunk = chunk_size * ratio
            raise ValueError(
                f"'{name}' split is empty\n"
                f"Fix: increase chunk_size and/or {name}_ratio so that "
                f"chunk_size x {name}_ratio ≥ 1. "
            )


def prepare_datasets(
    master_hdf5_path: str,
    base_name: str,
    trajectory_index: int,
    chunk_size: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
):
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError(
            f"Split ratios must sum to 1. Got {train_ratio + val_ratio + test_ratio}"
        )

    # Load one trajectory
    with h5py.File(master_hdf5_path, "r") as f:
        n_traj = f["t0_fields"]["density"].shape[0]
        if trajectory_index >= n_traj:
            raise IndexError(
                f"trajectory_index {trajectory_index} out of range "
                f"(file has {n_traj} trajectories)"
            )

        time_points = f["dimensions"]["time"][:]
        x_points    = f["dimensions"]["x"][:]
        y_points    = f["dimensions"]["y"][:]

        density  = f["t0_fields"]["density"][trajectory_index]    # (T, X, Y)
        pressure = f["t0_fields"]["pressure"][trajectory_index]   # (T, X, Y)
        velocity = f["t1_fields"]["velocity"][trajectory_index]   # (T, X, Y, 2)

    n_timesteps = len(time_points)

    # Split in train, val, test along the time axis
    train_t, val_t, test_t = temporal_chunk_split(
        n_timesteps, chunk_size, train_ratio, val_ratio, seed
    )
    _check_non_empty_splits(
        train_t, val_t, test_t, chunk_size, train_ratio, val_ratio, test_ratio
    )

    print(f"==> {n_timesteps} timesteps -> chunks of {chunk_size}")
    print(f"    train: {len(train_t)} | val: {len(val_t)} | test: {len(test_t)} timesteps")

    splits_t = {"train": train_t, "val": val_t, "test": test_t}

    # Compute normalization stats from the TRAIN split only (in order to have no leakage)
    tgt_tr = np.concatenate(
    [density[train_t][..., None], pressure[train_t][..., None], velocity[train_t]],
    axis=-1).reshape(-1, 4)
    tgt_mean, tgt_std = tgt_tr.mean(0), tgt_tr.std(0) + 1e-8
    time_mean, time_std = time_points[train_t].mean(), time_points[train_t].std() + 1e-8

    # For each split: select timesteps, flatten over (x, y), shuffle, save
    rng = np.random.default_rng(seed)

    for split_name, t_idx in splits_t.items():
        density_s  = density[t_idx]
        pressure_s = pressure[t_idx]
        velocity_s = velocity[t_idx]
        time_s     = time_points[t_idx]

        labels = np.concatenate(
            [density_s[..., np.newaxis], pressure_s[..., np.newaxis], velocity_s],
            axis=-1,
        )
        labels_flat = labels.reshape(-1, 4)

        # Standardize labels and time using TRAIN split stats (no leakage)
        labels_flat = (labels_flat - tgt_mean) / tgt_std
        time_s = (time_s - time_mean) / time_std

        T_g, X_g, Y_g = np.meshgrid(time_s, x_points, y_points, indexing="ij")

        rows = np.concatenate(
            [
                T_g.ravel()[..., np.newaxis],
                X_g.ravel()[..., np.newaxis],
                Y_g.ravel()[..., np.newaxis],
                labels_flat,
            ],
            axis=-1,
        )
        rows = rows[rng.permutation(len(rows))]

        out_path = DATASETS_DIR / f"{base_name}_{split_name}.hdf5"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(out_path, "w") as f:
            f.create_dataset("data", data=rows.astype(np.float32))
            f.attrs["tgt_mean"], f.attrs["tgt_std"] = tgt_mean, tgt_std
            f.attrs["time_mean"], f.attrs["time_std"] = time_mean, time_std

        print(f"==> Saved {split_name:>5}: {len(rows):>10,} rows -> {out_path}")

def cleanup_raw_download():
    """Delete the raw HF download and HF cache, keeping only the split files."""
    raw_dir   = DATASETS_DIR / "data"        # raw HF files (data/train/*.hdf5)
    cache_dir = DATASETS_DIR / ".cache"      # huggingface_hub cache

    for path in (raw_dir, cache_dir):
        if path.exists():
            shutil.rmtree(path)
            print(f"==> Removed {path}")


def main():
    overrides = sys.argv[1:]
    with initialize(version_base="1.3", config_path="../config"):
        cfg = compose(config_name="config", overrides=overrides)

        tcool = cfg.preprocessing.data.tcool
        master_hdf5_path = (
            DATASETS_DIR
            / f"data/train/turbulent_radiative_layer_tcool_{float(tcool):.2f}.hdf5"
        )

        if not master_hdf5_path.exists():
            raise FileNotFoundError(
                f"Master HDF5 file not found: {master_hdf5_path}\n"
                f"  Run `python data/download_data.py` first "
                f"(or with `data.tcool={tcool}` to download the right file)."
            )

        prepare_datasets(
            master_hdf5_path=str(master_hdf5_path),
            base_name=cfg.preprocessing.data.base_name,
            trajectory_index=cfg.preprocessing.data.trajectory,
            chunk_size=cfg.preprocessing.data.chunk_size,
            train_ratio=cfg.preprocessing.data.train_ratio,
            val_ratio=cfg.preprocessing.data.val_ratio,
            test_ratio=cfg.preprocessing.data.test_ratio,
            seed=cfg.preprocessing.data.seed,
        )

        cleanup_raw_download()


if __name__ == "__main__":
    main()