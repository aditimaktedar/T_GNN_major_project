"""Dataset adapter for ``data/processed/final_mimic_twosides_ml.csv``.

Reads SMILES-based multiclass interaction-type data. Does not modify source files.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
try:
    from torch_geometric.data import Batch, Data
    HAS_PYG = True
except ImportError:
    # Fallback dummy classes for environments without torch_geometric
    class Batch:  # pragma: no cover
        pass
    class Data:  # pragma: no cover
        pass
    HAS_PYG = False

from src.data.config import (
    DEFAULT_FINGERPRINT_N_BITS,
    DEFAULT_FINGERPRINT_RADIUS,
    FINAL_MIMIC_TWOSIDES_ML_PATH,
)
from src.data.exceptions import ConfigurationError, MissingInputError, SchemaError
from src.data.prepare_selected_ml_dataset import validate_pair_key_splits
from src.features.rdkit_features import morgan_fingerprint, parse_smiles

REQUIRED_COLUMNS = (
    "subject_id",
    "hadm_id",
    "drug_a",
    "drug_b",
    "smiles_a",
    "smiles_b",
    "pair_key",
    "type",
    "split",
)


class FingerprintCache:
    """Cache Morgan fingerprints keyed by SMILES string."""

    def __init__(
        self,
        radius: int = DEFAULT_FINGERPRINT_RADIUS,
        n_bits: int = DEFAULT_FINGERPRINT_N_BITS,
    ):
        self.radius = radius
        self.n_bits = n_bits
        self._cache: dict[str, list[int] | None] = {}

    def get(self, smiles: object) -> list[int] | None:
        key = str(smiles).strip()
        if key in self._cache:
            return self._cache[key]
        mol = parse_smiles(key)
        if mol is None:
            self._cache[key] = None
            return None
        fp = morgan_fingerprint(mol, radius=self.radius, n_bits=self.n_bits)
        self._cache[key] = fp
        return fp

    @property
    def fingerprint_dim(self) -> int:
        return self.n_bits


class TypeLabelEncoder:
    """Map TWOSIDES ``type`` integers to contiguous class indices (fit on train only)."""

    def __init__(self) -> None:
        self.type_to_index: dict[int, int] = {}
        self.index_to_type: dict[int, int] = {}

    def fit(self, types: pd.Series | np.ndarray | list[int]) -> "TypeLabelEncoder":
        unique_types = sorted({int(value) for value in types})
        self.type_to_index = {type_id: idx for idx, type_id in enumerate(unique_types)}
        self.index_to_type = {idx: type_id for type_id, idx in self.type_to_index.items()}
        return self

    @property
    def n_classes(self) -> int:
        return len(self.type_to_index)

    @property
    def classes(self) -> list[int]:
        return [self.index_to_type[idx] for idx in range(self.n_classes)]

    def transform(self, types: pd.Series | np.ndarray | list[int], unknown: int = -1) -> np.ndarray:
        encoded = []
        for value in types:
            encoded.append(self.type_to_index.get(int(value), unknown))
        return np.asarray(encoded, dtype=int)

    def fit_transform(self, types: pd.Series | np.ndarray | list[int]) -> np.ndarray:
        return self.fit(types).transform(types)


def load_final_ml_dataset(path: Path | str | None = None) -> pd.DataFrame:
    dataset_path = Path(path) if path is not None else FINAL_MIMIC_TWOSIDES_ML_PATH
    if not dataset_path.exists():
        raise MissingInputError(
            f"Processed ML dataset not found at {dataset_path}. "
            "Run prepare_final_mimic_twosides_ml first."
        )
    frame = pd.read_csv(dataset_path)
    validate_final_ml_frame(frame)
    return frame


def validate_final_ml_frame(frame: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in frame.columns]
    if missing:
        raise SchemaError(
            f"Final ML dataset missing columns: {missing}. Available: {list(frame.columns)}"
        )
    if frame["split"].isna().any():
        raise SchemaError("Split column contains missing values.")
    invalid_splits = set(frame["split"].astype(str)) - {"train", "val", "test"}
    if invalid_splits:
        raise SchemaError(f"Unexpected split values: {sorted(invalid_splits)}")
    pair_splits = frame.groupby("pair_key")["split"].nunique()
    if int((pair_splits > 1).sum()):
        raise ConfigurationError("Pair leakage: pair_key appears in more than one split.")


def assert_no_pair_leakage(frame: pd.DataFrame) -> None:
    validate_final_ml_frame(frame)
    pair_table = frame[["pair_key", "split"]].drop_duplicates()
    validate_pair_key_splits(pair_table.rename(columns={"pair_key": "pair_key"}))


def split_frame(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    validate_final_ml_frame(frame)
    parts: dict[str, pd.DataFrame] = {}
    for split_name in ("train", "val", "test"):
        parts[split_name] = frame.loc[frame["split"] == split_name].reset_index(drop=True)
    return parts


def attach_fingerprints(
    frame: pd.DataFrame,
    cache: FingerprintCache | None = None,
) -> pd.DataFrame:
    """Return a copy with Morgan fingerprint columns derived from SMILES."""
    validate_final_ml_frame(frame)
    fp_cache = cache or FingerprintCache()
    out = frame.copy()
    fps_a = []
    fps_b = []
    valid = []
    for _, row in out.iterrows():
        fp_a = fp_cache.get(row["smiles_a"])
        fp_b = fp_cache.get(row["smiles_b"])
        fps_a.append(fp_a)
        fps_b.append(fp_b)
        valid.append(fp_a is not None and fp_b is not None)
    out["drug_a_fingerprint"] = fps_a
    out["drug_b_fingerprint"] = fps_b
    out["fingerprint_valid"] = valid
    return out


def filter_trainable_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep rows with valid fingerprints and known encoded labels when present."""
    working = frame.copy()
    if "fingerprint_valid" not in working.columns:
        working = attach_fingerprints(working)
    mask = working["fingerprint_valid"].astype(bool)
    if "label_index" in working.columns:
        mask &= working["label_index"] >= 0
    return working.loc[mask].reset_index(drop=True)


def prepare_multiclass_frame(
    frame: pd.DataFrame,
    label_encoder: TypeLabelEncoder | None = None,
    fit_encoder_on: pd.DataFrame | None = None,
    cache: FingerprintCache | None = None,
) -> tuple[pd.DataFrame, TypeLabelEncoder, FingerprintCache]:
    """Attach fingerprints and class indices without modifying the source CSV."""
    fp_cache = cache or FingerprintCache()
    enriched = attach_fingerprints(frame, cache=fp_cache)
    encoder = label_encoder or TypeLabelEncoder()
    if fit_encoder_on is not None:
        encoder.fit(fit_encoder_on["type"])
    elif label_encoder is None:
        encoder.fit(enriched.loc[enriched["split"] == "train", "type"])
    enriched["label_index"] = encoder.transform(enriched["type"], unknown=-1)
    return enriched, encoder, fp_cache


def _fp_to_tensor(value: object) -> torch.Tensor:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        raise ValueError("Missing fingerprint.")
    return torch.from_numpy(np.asarray(value, dtype=np.float32))


def row_to_pyg_data(row: pd.Series) -> Data:
    x_a = _fp_to_tensor(row["drug_a_fingerprint"])
    x_b = _fp_to_tensor(row["drug_b_fingerprint"])
    x = torch.stack([x_a, x_b], dim=0)
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    y = torch.tensor([int(row["label_index"])], dtype=torch.long)
    return Data(
        x=x,
        edge_index=edge_index,
        y=y,
        type_id=int(row["type"]),
        pair_key=str(row["pair_key"]),
        subject_id=str(row["subject_id"]),
        split=str(row["split"]),
    )


class MulticlassDrugPairGraphDataset(Dataset):
    """PyG dataset: 2-node fingerprint graph per row, multiclass ``type`` label."""

    def __init__(self, frame: pd.DataFrame):
        self.frame = filter_trainable_rows(frame)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> Data:
        return row_to_pyg_data(self.frame.iloc[index])


def fingerprint_dim_from_frame(frame: pd.DataFrame) -> int:
    trainable = filter_trainable_rows(frame)
    if trainable.empty:
        raise SchemaError("No valid fingerprint rows available.")
    sample = trainable["drug_a_fingerprint"].iloc[0]
    return int(len(np.asarray(sample)))


def dataset_sanity_report(frame: pd.DataFrame) -> dict:
    """Non-training checks for the processed multiclass dataset."""
    validate_final_ml_frame(frame)
    splits = split_frame(frame)
    train_types = set(splits["train"]["type"].astype(int))
    val_types = set(splits["val"]["type"].astype(int))
    test_types = set(splits["test"]["type"].astype(int))
    enriched, encoder, cache = prepare_multiclass_frame(frame)
    trainable = filter_trainable_rows(enriched)
    return {
        "n_rows": int(len(frame)),
        "n_columns": int(len(frame.columns)),
        "n_train_rows": int(len(splits["train"])),
        "n_val_rows": int(len(splits["val"])),
        "n_test_rows": int(len(splits["test"])),
        "unique_pair_key": int(frame["pair_key"].nunique()),
        "unique_types_total": int(frame["type"].nunique()),
        "unique_types_train": int(len(train_types)),
        "unique_types_val": int(len(val_types)),
        "unique_types_test": int(len(test_types)),
        "val_types_not_in_train": sorted(val_types - train_types),
        "test_types_not_in_train": sorted(test_types - train_types),
        "n_classes_train_encoder": encoder.n_classes,
        "fingerprint_dim": cache.fingerprint_dim,
        "unique_smiles_a": int(frame["smiles_a"].nunique()),
        "unique_smiles_b": int(frame["smiles_b"].nunique()),
        "trainable_rows": int(len(trainable)),
        "invalid_fingerprint_rows": int(len(enriched) - len(trainable)),
        "duplicate_pair_key_type_rows": int(enriched.duplicated(subset=["pair_key", "type"]).sum()),
        "pair_split_leakage_pairs": int(
            frame.groupby("pair_key")["split"].nunique().gt(1).sum()
        ),
    }
