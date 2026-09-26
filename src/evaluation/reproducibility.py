"""Reproducibility helpers: seeds and experiment metadata."""

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
import torch

from src.data.io_utils import ensure_parent, write_json


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collect_environment_metadata(config: dict[str, Any] | None = None) -> dict[str, Any]:
    import rdkit

    metadata = {
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "sklearn_version": sklearn.__version__,
        "torch_version": torch.__version__,
        "torch_geometric_version": __import__("torch_geometric").__version__,
        "rdkit_version": rdkit.__version__,
        "device": str(get_device()),
        "config": config or {},
    }
    return metadata


def save_experiment_metadata(path: Path, config: dict[str, Any]) -> Path:
    payload = collect_environment_metadata(config)
    return write_json(path, payload)
