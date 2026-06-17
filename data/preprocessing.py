from pathlib import Path

import h5py
import numpy as np
from omegaconf import DictConfig
import shutil
import sys
from hydra import initialize_config_dir, compose

# Find project root and datasets directory.
# Project root is file specific(download_data.py is two levels deep in the directory structure)
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
    Splits a temporal domain into train, validation, and test sets using a chunked random allocation strategy.
    
    Within each chunk of `chunk_size` consecutive timesteps, indices are uniformly and randomly allocated
    to train, validation, and test sets based on the provided ratios. The remainder goes to the test set.

    Args:
        n_timesteps (int): The total number of timesteps in the temporal domain.
        chunk_size (int): The number of consecutive timesteps in each chunk.
        train_ratio (float): The proportion of timesteps in each chunk to allocate to the training set.
        val_ratio (float): The proportion of timesteps in each chunk to allocate to the validation set.
        seed (int): The random seed for the numpy random number generator to ensure reproducibility.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing three sorted 1D numpy arrays of integers 
        representing the time indices for the train, validation, and test splits, respectively.
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
    
    """
    Validates that none of the generated temporal data splits (train, val, test) are empty.

    Args:
        train_t (np.ndarray): Array of time indices allocated to the training split.
        val_t (np.ndarray): Array of time indices allocated to the validation split.
        test_t (np.ndarray): Array of time indices allocated to the test split.
        chunk_size (int): The size of the temporal chunks used during splitting.
        train_ratio (float): The ratio of data allocated to the training split.
        val_ratio (float): The ratio of data allocated to the validation split.
        test_ratio (float): The ratio of data allocated to the test split.
    """
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
    hdf5_path: str,
    base_name: str,
    trajectory_index: int,
    chunk_size: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
):
    
    """
    Reads a HDF5 file, extracts a single trajectory, splits it temporally, normalizes the data,
    flattens the spatial dimensions, and saves the resulting splits into separate HDF5 files.

    Normalization statistics (mean and standard deviation) are computed strictly from the training 
    split to avoid data leakage and then applied to all splits. The output data is shuffled.

    Args:
        hdf5_path (str): The file path to the downloaded HDF5 dataset.
        base_name (str): The base prefix for the output HDF5 files.
        trajectory_index (int): The index of the specific trajectory to extract from the master file.
        chunk_size (int): The number of consecutive timesteps in each chunk for splitting.
        train_ratio (float): The proportion of data to allocate to the training set.
        val_ratio (float): The proportion of data to allocate to the validation set.
        test_ratio (float): The proportion of data to allocate to the test set.
        seed (int): The random seed used for splitting and shuffling to ensure reproducibility.
    """

    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError(
            f"Split ratios must sum to 1. Got {train_ratio + val_ratio + test_ratio}"
        )

    # Load one trajectory
    with h5py.File(hdf5_path, "r") as f:
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

    splits_t = {"train": train_t, "val": val_t, "test": test_t}

    # Compute normalization stats from the TRAIN split only (in order to have no leakage)
    norm_keys = ["density", "pressure", "velocity_x", "velocity_y"]
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
            f.attrs["norm_keys"] = norm_keys


def cleanup_raw_download(master_hdf5_path: str | Path) -> None:
    """
    Safely deletes the specific raw master HDF5 file to free disk space.

    Args:
        master_hdf5_path (str | Path): The precise path to the raw master file.
    """
    target_path = Path(master_hdf5_path)

    # Strict validation: Ensure it exists and is specifically a file, not a directory.
    if target_path.exists() and target_path.is_file():
        try:
            # Precise deletion: unlink() deletes only the specific file.
            target_path.unlink()
        except PermissionError:
            print(f"==> Error: Permission denied when attempting to remove {target_path}")
    else:
        print(f"==> Warning: Target raw file not found or is not a file: {target_path}")
    
    # Attempt to clean up empty parent directories
    # Hugging Face structure is typically: datasets/data/train/file.hdf5
    train_dir = target_path.parent
    data_dir = train_dir.parent

    for directory in [train_dir, data_dir]:
        if directory.exists() and directory.is_dir():
            try:
                # rmdir() acts as a strict safety gate: it fails if the folder is not empty
                directory.rmdir()
            except OSError:
                # OSError (specifically ENOTEMPTY) means other files exist.
                # We stop the loop here to avoid touching higher-level directories.
                break


def main():
    overrides = sys.argv[1:]
    config_dir = str(PROJECT_ROOT / "config")

    with initialize_config_dir(version_base="1.3", config_dir=config_dir):
        cfg = compose(config_name="config", overrides=overrides)

        tcool = cfg.preprocessing.data.tcool
        hdf5_path = (
            DATASETS_DIR
            / f"data/train/turbulent_radiative_layer_tcool_{float(tcool):.2f}.hdf5"
        )

        if not hdf5_path.exists():
            raise FileNotFoundError(
                f"Master HDF5 file not found: {hdf5_path}\n"
                f"  Run `python data/download_data.py` first "
                f"(or with `data.tcool={tcool}` to download the right file)."
            )

        prepare_datasets(
            hdf5_path=str(hdf5_path),
            base_name=cfg.preprocessing.data.base_name,
            trajectory_index=cfg.preprocessing.data.trajectory,
            chunk_size=cfg.preprocessing.data.chunk_size,
            train_ratio=cfg.preprocessing.data.train_ratio,
            val_ratio=cfg.preprocessing.data.val_ratio,
            test_ratio=cfg.preprocessing.data.test_ratio,
            seed=cfg.preprocessing.data.seed,
        )

        cleanup_raw_download(hdf5_path)


if __name__ == "__main__":
    main()