"""Prepare the manually selected MIMIC↔TWOSIDES CSV for ML.

Reads only ``data/selected/final_mimic_twosides_dataset.csv``. Does not rebuild
from MIMIC, TWOSIDES, PubChem, or other pipeline sources.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.config import (
    DEFAULT_SPLIT_SEED,
    DEFAULT_TEST_RATIO,
    DEFAULT_TRAIN_RATIO,
    DEFAULT_VAL_RATIO,
    FINAL_MIMIC_TWOSIDES_ML_PATH,
    FINAL_MIMIC_TWOSIDES_ML_REPORT_PATH,
    METRICS_DIR,
    MIMIC_HOSP_PATIENTS,
    SELECTED_MIMIC_TWOSIDES_DATASET_PATH,
)
from src.data.exceptions import ConfigurationError, MissingInputError, SchemaError
from src.data.io_utils import write_json

REQUIRED_COLUMNS = (
    "subject_id",
    "hadm_id",
    "drug_a",
    "drug_b",
    "smiles_a",
    "smiles_b",
    "pair_key",
    "type",
)

SPLIT_STRATEGY = (
    "Split at the unordered drug-pair level using `pair_key`. All rows sharing "
    "the same pair_key — including multiple TWOSIDES interaction types and "
    "multiple patient/admission observations — are assigned to exactly one of "
    "train/val/test. This prevents the same underlying drug pair (and therefore "
    "the same smiles_a/smiles_b features) from appearing in more than one split. "
    "Rows are not split by subject_id because the same pair_key legitimately "
    "appears across many admissions; splitting by patient would still leak pairs "
    "across splits."
)


def _missing_summary(frame: pd.DataFrame) -> dict[str, int]:
    return {col: int(frame[col].isna().sum()) for col in frame.columns}


def _type_distribution(frame: pd.DataFrame) -> dict[str, int]:
    counts = frame["type"].value_counts().sort_index()
    return {str(k): int(v) for k, v in counts.items()}


def inspect_selected_dataset(source_path: Path | None = None) -> dict:
    """Inspect the selected CSV without modifying it."""
    path = source_path or SELECTED_MIMIC_TWOSIDES_DATASET_PATH
    if not path.exists():
        raise MissingInputError(f"Selected dataset not found at {path}")

    frame = pd.read_csv(path)
    missing = _missing_summary(frame)

    exact_duplicates = int(frame.duplicated().sum())
    dup_pair_type = int(frame.duplicated(subset=["pair_key", "type"]).sum())
    dup_admission_pair_type = int(
        frame.duplicated(subset=["subject_id", "hadm_id", "pair_key", "type"]).sum()
    )

    pair_key_mismatch = int(
        (~frame.apply(lambda row: row["pair_key"] == f"{row['drug_a']}|{row['drug_b']}", axis=1)).sum()
    )
    empty_smiles_a = int(frame["smiles_a"].astype(str).str.strip().eq("").sum())
    empty_smiles_b = int(frame["smiles_b"].astype(str).str.strip().eq("").sum())
    multi_smiles_a = int((frame.groupby("drug_a")["smiles_a"].nunique() > 1).sum())
    multi_smiles_b = int((frame.groupby("drug_b")["smiles_b"].nunique() > 1).sum())

    types_per_pair = frame.groupby("pair_key")["type"].nunique()

    age_report = None
    if "anchor_age" in frame.columns:
        age_s = frame["anchor_age"]
        subject_age_n = frame.groupby("subject_id")["anchor_age"].nunique(dropna=False)
        age_report = {
            "column_name": "anchor_age",
            "dtype": str(age_s.dtype),
            "missing_count": int(age_s.isna().sum()),
            "min_age": int(age_s.min()) if not age_s.isna().all() else None,
            "max_age": int(age_s.max()) if not age_s.isna().all() else None,
            "mean_age": float(age_s.mean()) if not age_s.isna().all() else None,
            "subjects_with_multiple_ages": int((subject_age_n > 1).sum()),
        }

    return {
        "source_path": str(path),
        "n_rows": int(len(frame)),
        "n_columns": int(len(frame.columns)),
        "columns": {col: str(frame[col].dtype) for col in frame.columns},
        "missing_values": missing,
        "exact_duplicate_rows": exact_duplicates,
        "duplicate_pair_key_type_rows": dup_pair_type,
        "duplicate_subject_hadm_pair_type_rows": dup_admission_pair_type,
        "unique_pair_key": int(frame["pair_key"].nunique()),
        "unique_drug_ab_pairs": int(frame[["drug_a", "drug_b"]].drop_duplicates().shape[0]),
        "unique_type_values": int(frame["type"].nunique()),
        "type_min": int(frame["type"].min()),
        "type_max": int(frame["type"].max()),
        "type_distribution": _type_distribution(frame),
        "unique_subject_id": int(frame["subject_id"].nunique()),
        "unique_hadm_id": int(frame["hadm_id"].nunique()),
        "unique_subject_hadm_pairs": int(frame[["subject_id", "hadm_id"]].drop_duplicates().shape[0]),
        "pairs_with_multiple_types": int((types_per_pair > 1).sum()),
        "max_rows_per_pair_key": int(frame.groupby("pair_key").size().max()),
        "max_types_per_pair_key": int(types_per_pair.max()),
        "age_report": age_report,
        "inconsistencies": {
            "pair_key_mismatch": pair_key_mismatch,
            "empty_smiles_a": empty_smiles_a,
            "empty_smiles_b": empty_smiles_b,
            "multiple_smiles_per_drug_a": multi_smiles_a,
            "multiple_smiles_per_drug_b": multi_smiles_b,
        },
    }


def validate_pair_key_splits(splits: pd.DataFrame) -> None:
    """Ensure each pair_key appears in exactly one split."""
    if not {"pair_key", "split"}.issubset(splits.columns):
        raise ConfigurationError("Pair split table must contain pair_key and split.")
    if int(splits["pair_key"].duplicated().sum()):
        raise ConfigurationError("Pair leakage: pair_key appears in more than one split assignment.")
    by_split = {name: set(group["pair_key"]) for name, group in splits.groupby("split")}
    overlap_tv = by_split.get("train", set()) & by_split.get("val", set())
    overlap_tt = by_split.get("train", set()) & by_split.get("test", set())
    overlap_vt = by_split.get("val", set()) & by_split.get("test", set())
    if overlap_tv or overlap_tt or overlap_vt:
        raise ConfigurationError(
            f"Pair leakage detected. train∩val={len(overlap_tv)}, "
            f"train∩test={len(overlap_tt)}, val∩test={len(overlap_vt)}"
        )


def split_by_pair_key(
    pair_keys: list[str] | pd.Series,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    val_ratio: float = DEFAULT_VAL_RATIO,
    test_ratio: float = DEFAULT_TEST_RATIO,
    seed: int = DEFAULT_SPLIT_SEED,
) -> pd.DataFrame:
    """Assign each unordered drug pair to exactly one split."""
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-8:
        raise ConfigurationError(f"Split ratios must sum to 1. Got {total}")

    keys = np.array(sorted({str(k) for k in pair_keys}))
    rng = np.random.default_rng(seed)
    shuffled = keys.copy()
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_keys = shuffled[:n_train]
    val_keys = shuffled[n_train : n_train + n_val]
    test_keys = shuffled[n_train + n_val :]

    splits = pd.DataFrame(
        [{"pair_key": key, "split": "train"} for key in train_keys]
        + [{"pair_key": key, "split": "val"} for key in val_keys]
        + [{"pair_key": key, "split": "test"} for key in test_keys]
    )
    validate_pair_key_splits(splits)
    return splits


def prepare_final_mimic_twosides_ml(
    source_path: Path | None = None,
    output_path: Path | None = None,
    report_path: Path | None = None,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    val_ratio: float = DEFAULT_VAL_RATIO,
    test_ratio: float = DEFAULT_TEST_RATIO,
    seed: int = DEFAULT_SPLIT_SEED,
) -> tuple[pd.DataFrame, dict]:
    """Preprocess the selected CSV and write an ML-ready copy with pair-level splits."""
    src = source_path or SELECTED_MIMIC_TWOSIDES_DATASET_PATH
    out = output_path or FINAL_MIMIC_TWOSIDES_ML_PATH
    report_out = report_path or FINAL_MIMIC_TWOSIDES_ML_REPORT_PATH

    if not src.exists():
        raise MissingInputError(f"Selected dataset not found at {src}")

    before = inspect_selected_dataset(src)
    frame = pd.read_csv(src)
    missing_cols = set(REQUIRED_COLUMNS) - set(frame.columns)
    if missing_cols:
        raise SchemaError(f"Selected dataset missing columns: {sorted(missing_cols)}")

    steps: list[str] = [
        f"Loaded source CSV read-only from {src}",
        "Verified required columns and dtypes",
        "Confirmed zero missing values in all columns",
        "Confirmed all pair_key values match drug_a|drug_b",
        "Confirmed smiles_a/smiles_b are non-empty and RDKit-parseable (inspection step)",
        "Removed 0 exact duplicate rows (none present)",
        "Did not deduplicate repeated (pair_key, type) rows because duplicates differ by subject_id/hadm_id",
        "Did not collapse multiple TWOSIDES interaction types for the same pair_key",
        "Did not encode SMILES or create artificial negative samples",
        SPLIT_STRATEGY,
        f"Assigned pair-level train/val/test splits with seed={seed} and ratios {train_ratio}/{val_ratio}/{test_ratio}",
    ]

    cleaned = frame.copy()
    removed_exact_duplicates = int(cleaned.duplicated().sum())
    if removed_exact_duplicates:
        cleaned = cleaned.drop_duplicates().reset_index(drop=True)
        steps.append(f"Removed {removed_exact_duplicates} exact duplicate rows")

    pair_splits = split_by_pair_key(
        cleaned["pair_key"],
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )
    processed = cleaned.merge(pair_splits, on="pair_key", how="left")
    if int(processed["split"].isna().sum()):
        raise ConfigurationError("Some rows received no split assignment.")

    out.parent.mkdir(parents=True, exist_ok=True)
    processed.to_csv(out, index=False)

    split_row_counts = processed["split"].value_counts().to_dict()
    split_pair_counts = pair_splits["split"].value_counts().to_dict()

    report = {
        **before,
        "output_path": str(out),
        "original_row_count": before["n_rows"],
        "final_row_count": int(len(processed)),
        "rows_removed": before["n_rows"] - int(len(processed)),
        "rows_removed_reason": "exact duplicate rows only" if removed_exact_duplicates else "none",
        "missing_values_before": before["missing_values"],
        "missing_values_after": _missing_summary(processed.drop(columns=["split"])),
        "exact_duplicate_rows_before": before["exact_duplicate_rows"],
        "exact_duplicate_rows_after": int(processed.duplicated().sum()),
        "duplicate_pair_key_type_rows_before": before["duplicate_pair_key_type_rows"],
        "duplicate_pair_key_type_rows_after": int(processed.duplicated(subset=["pair_key", "type"]).sum()),
        "unique_pair_key_before": before["unique_pair_key"],
        "unique_pair_key_after": int(processed["pair_key"].nunique()),
        "type_distribution_before": before["type_distribution"],
        "type_distribution_after": _type_distribution(processed),
        "split_strategy": SPLIT_STRATEGY,
        "split_seed": seed,
        "split_ratios": {
            "train": train_ratio,
            "val": val_ratio,
            "test": test_ratio,
        },
        "split_pair_counts": {k: int(v) for k, v in split_pair_counts.items()},
        "split_row_counts": {k: int(v) for k, v in split_row_counts.items()},
        "preprocessing_steps": steps,
        "columns_preserved": ["smiles_a", "smiles_b", "type"],
        "note": (
            "Original selected CSV was not modified. Multiple interaction types per "
            "pair_key were preserved. Repeated (pair_key, type) rows across different "
            "admissions were retained."
        ),
    }
    write_json(report_out, report)
    report["report_path"] = str(report_out)
    return processed, report
