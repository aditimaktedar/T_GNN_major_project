"""PyTorch Geometric Dataset/DataLoader for the ML dataset from build_dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader

from src.data.config import ML_DATASET_PATH
from src.data.exceptions import MissingInputError, SchemaError
from src.data.io_utils import load_table_chunked
from src.data.splits import validate_patient_splits

REQUIRED_COLUMNS = (
    "patient_id",
    "drug_a",
    "drug_b",
    "drug_a_fingerprint",
    "drug_b_fingerprint",
    "label",
    "split",
)

LABEL_TO_INT = {"positive": 1, "negative": 0}
INT_TO_LABEL = {1: "positive", 0: "negative"}


def _fp_to_tensor(value: object) -> torch.Tensor:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        raise ValueError("Missing fingerprint.")
    arr = np.asarray(value, dtype=np.float32)
    return torch.from_numpy(arr)


def load_processed_dataset(path: Path | str = ML_DATASET_PATH) -> pd.DataFrame:
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise MissingInputError(
            f"Processed ML dataset not found at {dataset_path}. "
            "Run: python -m src.data.pipeline build"
        )
    frame = load_table_chunked(dataset_path)
    validate_ml_dataset_frame(frame)
    return frame


def validate_ml_dataset_frame(frame: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in frame.columns]
    if missing:
        raise SchemaError(
            f"ML dataset missing required columns: {missing}. "
            f"Available: {list(frame.columns)}"
        )
    splits = frame[["patient_id", "split"]].drop_duplicates()
    validate_patient_splits(splits)
    for col in ("drug_a_fingerprint", "drug_b_fingerprint"):
        sample = frame[col].dropna().iloc[0] if frame[col].notna().any() else None
        if sample is None:
            raise SchemaError(f"Column {col} has no valid fingerprints.")
        dim = len(np.asarray(sample))
        for value in frame[col].dropna():
            if len(np.asarray(value)) != dim:
                raise SchemaError(f"Inconsistent fingerprint dimension in {col}.")


def filter_trainable_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep rows with binary labels and valid fingerprints."""
    mask = frame["label"].isin(LABEL_TO_INT)
    filtered = frame.loc[mask].copy()
    valid = []
    for idx, row in filtered.iterrows():
        try:
            _fp_to_tensor(row["drug_a_fingerprint"])
            _fp_to_tensor(row["drug_b_fingerprint"])
            valid.append(idx)
        except ValueError:
            continue
    return filtered.loc[valid].reset_index(drop=True)


def row_to_pyg_data(row: pd.Series) -> Data:
    x_a = _fp_to_tensor(row["drug_a_fingerprint"])
    x_b = _fp_to_tensor(row["drug_b_fingerprint"])
    x = torch.stack([x_a, x_b], dim=0)
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    y = torch.tensor([LABEL_TO_INT[str(row["label"])]], dtype=torch.float)
    return Data(
        x=x,
        edge_index=edge_index,
        y=y,
        patient_id=str(row["patient_id"]),
        drug_a=str(row["drug_a"]),
        drug_b=str(row["drug_b"]),
        split=str(row["split"]),
    )


class DrugPairGraphDataset(Dataset):
    """PyG dataset where each item is a 2-node static graph for one drug pair."""

    def __init__(self, frame: pd.DataFrame):
        self.frame = filter_trainable_rows(frame)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> Data:
        return row_to_pyg_data(self.frame.iloc[index])


def split_frame(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    validate_ml_dataset_frame(frame)
    parts = {}
    for split_name in ("train", "val", "test"):
        parts[split_name] = frame.loc[frame["split"] == split_name].reset_index(drop=True)
    return parts


def assert_no_patient_leakage(frame: pd.DataFrame) -> None:
    patient_splits = frame[["patient_id", "split"]].drop_duplicates()
    validate_patient_splits(patient_splits)


def create_dataloaders(
    frame: pd.DataFrame,
    batch_size: int = 32,
    num_workers: int = 0,
) -> dict[str, DataLoader]:
    assert_no_patient_leakage(frame)
    loaders = {}
    for split_name, part in split_frame(frame).items():
        dataset = DrugPairGraphDataset(part)
        loaders[split_name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split_name == "train"),
            num_workers=num_workers,
        )
    return loaders


def fingerprint_dim_from_frame(frame: pd.DataFrame) -> int:
    validate_ml_dataset_frame(frame)
    sample = frame["drug_a_fingerprint"].dropna().iloc[0]
    return int(len(np.asarray(sample)))
