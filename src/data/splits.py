"""Patient-level train/validation/test splits with leakage checks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.config import (
    DEFAULT_SPLIT_SEED,
    DEFAULT_TEST_RATIO,
    DEFAULT_TRAIN_RATIO,
    DEFAULT_VAL_RATIO,
    METRICS_DIR,
    PATIENT_SPLITS_PATH,
)
from src.data.exceptions import ConfigurationError, MissingInputError
from src.data.io_utils import load_table_chunked, write_json, write_table


def validate_patient_splits(splits: pd.DataFrame) -> None:
    if not {"patient_id", "split"}.issubset(splits.columns):
        raise ConfigurationError("Split table must contain patient_id and split.")
    duplicated_patients = splits["patient_id"].duplicated().sum()
    if duplicated_patients:
        raise ConfigurationError(f"Patient leakage: {duplicated_patients} patient_id values appear more than once.")
    by_split = {
        name: set(group["patient_id"].astype(str))
        for name, group in splits.groupby("split")
    }
    overlap_tv = by_split.get("train", set()) & by_split.get("val", set())
    overlap_tt = by_split.get("train", set()) & by_split.get("test", set())
    overlap_vt = by_split.get("val", set()) & by_split.get("test", set())
    if overlap_tv or overlap_tt or overlap_vt:
        raise ConfigurationError(
            f"Patient leakage detected. train∩val={len(overlap_tv)}, "
            f"train∩test={len(overlap_tt)}, val∩test={len(overlap_vt)}"
        )


def split_patients(
    table: pd.DataFrame | Path,
    patient_column: str = "patient_id",
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    val_ratio: float = DEFAULT_VAL_RATIO,
    test_ratio: float = DEFAULT_TEST_RATIO,
    seed: int = DEFAULT_SPLIT_SEED,
    output_path: Path | None = None,
    stats_path: Path | None = None,
) -> tuple[pd.DataFrame, dict]:
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-8:
        raise ConfigurationError(f"Split ratios must sum to 1. Got {train_ratio}+{val_ratio}+{test_ratio}={total}")
    if isinstance(table, Path):
        if not table.exists():
            raise MissingInputError(f"Split step needs a table with patients at {table}.")
        frame = load_table_chunked(table)
    else:
        frame = table
    if patient_column not in frame.columns:
        raise MissingInputError(f"Missing {patient_column}. Columns: {list(frame.columns)}")

    patients = np.array(sorted({str(value) for value in frame[patient_column].dropna().astype(str)}))
    if len(patients) == 0:
        raise ConfigurationError("No patients available to split.")
    rng = np.random.default_rng(seed)
    shuffled = patients.copy()
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    # Remainder goes to test so every patient is assigned once.
    train_ids = shuffled[:n_train]
    val_ids = shuffled[n_train : n_train + n_val]
    test_ids = shuffled[n_train + n_val :]

    rows = (
        [{"patient_id": pid, "split": "train"} for pid in train_ids]
        + [{"patient_id": pid, "split": "val"} for pid in val_ids]
        + [{"patient_id": pid, "split": "test"} for pid in test_ids]
    )
    splits = pd.DataFrame(rows)
    validate_patient_splits(splits)
    if int(len(splits)) != int(len(patients)):
        raise ConfigurationError("Not all eligible patients were assigned exactly once.")

    stats = {
        "n_patients": int(len(patients)),
        "n_train": int((splits["split"] == "train").sum()),
        "n_val": int((splits["split"] == "val").sum()),
        "n_test": int((splits["split"] == "test").sum()),
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "seed": seed,
        "note": (
            "Default ratios are 0.70/0.15/0.15 because the project did not specify another split. "
            "The same patient never appears in more than one split."
        ),
    }
    out = output_path if output_path is not None else PATIENT_SPLITS_PATH
    write_table(splits, out)
    metrics_out = stats_path if stats_path is not None else METRICS_DIR / "split_statistics.json"
    write_json(metrics_out, stats)
    stats["output_path"] = str(out)
    return splits, stats
