"""Utility functions for io."""

from typing import Mapping, Any

import os
from pathlib import Path

import pickle
import json
import nibabel as nib
import numpy as np
import pandas as pd


def load_data(
    path: os.PathLike,
) -> np.ndarray | Mapping[str, Any] | pd.DataFrame | None:
    """Load data from a file."""
    path = Path(path)

    if path.suffix in [".npy", ".npz"]:
        return np.load(path)
    elif path.suffix == ".json":
        return json.load(path.open())
    elif path.suffix in [".nii", ".nii.gz"]:
        return np.asarray(nib.load(path).dataobj)
    elif path.suffix in [".csv", ".tsv"]:
        return pd.read_csv(path)
    elif path.suffix in [".parquet"]:
        return pd.read_parquet(path)
    elif path.suffix in [".pkl"]:
        return pickle.load(path.open("rb"))
    else:
        raise ValueError(f"Unknown file type: {path.suffix}")


def save_data(data: Any, path: os.PathLike) -> Path:  # noqa: ANN401
    """Save data to a file."""
    path = Path(path)

    if path.suffix in [".npy", ".npz"]:
        np.save(path, data)
    elif path.suffix == ".json":
        json.dump(data, path.open("w"))
    elif path.suffix in [".nii", ".nii.gz"]:
        if isinstance(data, (tuple, list)):
            nib.save(nib.Nifti1Image(data[0], data[1]), path)
        else:
            nib.save(nib.Nifti1Image(data, np.eye(4)), path)
    elif path.suffix in [".csv", ".tsv"]:
        data.to_csv(path)
    elif path.suffix in [".parquet"]:
        data.to_parquet(path)
    elif path.suffix in [".pkl"]:
        pickle.dump(data, path.open("wb"))
    else:
        raise ValueError(f"Unknown file type: {path.suffix}")
    return path
