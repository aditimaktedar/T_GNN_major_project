"""Frequency-filtered (363-class) ML dataset adapter.

Derives eligible TWOSIDES types from **train-split row counts only** (>= threshold).
Does not modify ``final_mimic_twosides_ml.csv``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.data.config import (
    FINAL_MIMIC_TWOSIDES_ML_FREQUENT363_PATH,
    FINAL_MIMIC_TWOSIDES_ML_PATH,
    FREQUENT363_MIN_TRAIN_ROWS,
)
from src.data.exceptions import ConfigurationError, MissingInputError, SchemaError
from src.data.final_ml_dataset import (
    FingerprintCache,
    TypeLabelEncoder,
    assert_no_pair_leakage,
    attach_fingerprints,
    filter_trainable_rows,
    prepare_multiclass_frame,
    split_frame,
    validate_final_ml_frame,
)

FORMULATION_NAME = "frequent363"
FORMULATION_DESCRIPTION = (
    "Frequency-filtered multiclass TWOSIDES interaction-type prediction. "
    "Includes only types with >= min_train_rows admission rows in the training split. "
    "No 'other' class; rare types are excluded."
)


def eligible_types_from_train(
    frame: pd.DataFrame,
    min_train_rows: int = FREQUENT363_MIN_TRAIN_ROWS,
) -> set[int]:
    """Return TWOSIDES type IDs with at least ``min_train_rows`` training rows."""
    validate_final_ml_frame(frame)
    train = frame.loc[frame["split"] == "train"]
    counts = train.groupby("type").size()
    return {int(type_id) for type_id, count in counts.items() if int(count) >= min_train_rows}


def filter_to_eligible_types(
    frame: pd.DataFrame,
    eligible_types: set[int],
) -> pd.DataFrame:
    """Keep rows whose ``type`` is in the eligible set; preserve columns and splits."""
    validate_final_ml_frame(frame)
    filtered = frame.loc[frame["type"].astype(int).isin(eligible_types)].copy()
    return filtered.reset_index(drop=True)


def build_frequent363_frame(
    source: pd.DataFrame,
    min_train_rows: int = FREQUENT363_MIN_TRAIN_ROWS,
) -> tuple[pd.DataFrame, set[int]]:
    """Build the in-memory frequent363 representation from the full ML dataset."""
    validate_final_ml_frame(source)
    eligible = eligible_types_from_train(source, min_train_rows=min_train_rows)
    if not eligible:
        raise SchemaError("No eligible types found for frequent363 formulation.")
    filtered = filter_to_eligible_types(source, eligible)
    assert_no_pair_leakage(filtered)
    return filtered, eligible


def load_frequent363_ml_dataset(path: Path | str | None = None) -> pd.DataFrame:
    dataset_path = Path(path) if path is not None else FINAL_MIMIC_TWOSIDES_ML_FREQUENT363_PATH
    if not dataset_path.exists():
        raise MissingInputError(
            f"Frequent363 dataset not found at {dataset_path}. "
            "Run: python -m src.data.prepare_frequent363_ml_dataset"
        )
    frame = pd.read_csv(dataset_path)
    validate_final_ml_frame(frame)
    assert_no_pair_leakage(frame)
    return frame


def load_frequent363_source_or_derived(
    source_path: Path | str | None = None,
    derived_path: Path | str | None = None,
) -> pd.DataFrame:
    """Load derived CSV if present; otherwise filter from source in memory."""
    derived = Path(derived_path) if derived_path is not None else FINAL_MIMIC_TWOSIDES_ML_FREQUENT363_PATH
    if derived.exists():
        return load_frequent363_ml_dataset(derived)
    source = pd.read_csv(source_path or FINAL_MIMIC_TWOSIDES_ML_PATH)
    frame, _eligible = build_frequent363_frame(source)
    return frame


def prepare_frequent363_multiclass_frame(
    frame: pd.DataFrame,
    label_encoder: TypeLabelEncoder | None = None,
    cache: FingerprintCache | None = None,
    min_train_rows: int = FREQUENT363_MIN_TRAIN_ROWS,
) -> tuple[pd.DataFrame, TypeLabelEncoder, FingerprintCache]:
    """Attach fingerprints and train-only label indices for the 363-class formulation."""
    validate_final_ml_frame(frame)
    eligible = eligible_types_from_train(frame, min_train_rows=min_train_rows)
    non_eligible = set(frame["type"].astype(int)) - eligible
    if non_eligible:
        raise SchemaError(f"Frame contains non-eligible types: {sorted(non_eligible)[:10]}")
    train_types = set(frame.loc[frame["split"] == "train", "type"].astype(int))
    if train_types != eligible:
        raise ConfigurationError(
            "Frequent363 frame train types do not match eligible-type definition."
        )

    enriched, encoder, fp_cache = prepare_multiclass_frame(
        frame,
        label_encoder=label_encoder,
        fit_encoder_on=frame.loc[frame["split"] == "train"],
        cache=cache,
    )
    if (enriched["label_index"] < 0).any():
        raise SchemaError("Frequent363 frame contains unencoded labels.")
    return enriched, encoder, fp_cache


def _pair_type_counts(frame: pd.DataFrame) -> dict[str, int]:
    dedup = frame.drop_duplicates(subset=["pair_key", "type"])
    return {
        "n_pair_types_dedup": int(len(dedup)),
        "n_pair_types_dedup_train": int(len(dedup.loc[dedup["split"] == "train"])),
        "n_pair_types_dedup_val": int(len(dedup.loc[dedup["split"] == "val"])),
        "n_pair_types_dedup_test": int(len(dedup.loc[dedup["split"] == "test"])),
    }


def frequent363_report(
    source: pd.DataFrame | None = None,
    filtered: pd.DataFrame | None = None,
    min_train_rows: int = FREQUENT363_MIN_TRAIN_ROWS,
) -> dict[str, Any]:
    """Reporting stats for the frequent363 formulation."""
    if source is None:
        source = pd.read_csv(FINAL_MIMIC_TWOSIDES_ML_PATH)
    validate_final_ml_frame(source)
    if filtered is None:
        filtered, eligible = build_frequent363_frame(source, min_train_rows=min_train_rows)
    else:
        eligible = eligible_types_from_train(source, min_train_rows=min_train_rows)

    source_train = source.loc[source["split"] == "train"]
    all_train_types = set(source_train["type"].astype(int))
    excluded_types = sorted(all_train_types - eligible)
    never_in_train = sorted(set(source["type"].astype(int)) - all_train_types)

    splits_source = split_frame(source)
    splits_filtered = split_frame(filtered)
    enriched, encoder, cache = prepare_frequent363_multiclass_frame(filtered)
    trainable = filter_trainable_rows(enriched)

    train_type_counts = (
        filtered.loc[filtered["split"] == "train"]
        .groupby("type")
        .size()
        .sort_values(ascending=False)
    )

    val_types = set(filtered.loc[filtered["split"] == "val", "type"].astype(int))
    test_types = set(filtered.loc[filtered["split"] == "test", "type"].astype(int))
    encoder_types = set(encoder.classes)

    pair_type_leakage = int(
        filtered.groupby(["pair_key", "type"])["split"].nunique().gt(1).sum()
    )
    pair_leakage = int(filtered.groupby("pair_key")["split"].nunique().gt(1).sum())

    return {
        "formulation": FORMULATION_NAME,
        "description": FORMULATION_DESCRIPTION,
        "source_dataset": str(FINAL_MIMIC_TWOSIDES_ML_PATH),
        "min_train_rows": min_train_rows,
        "n_eligible_types": len(eligible),
        "eligible_types": sorted(eligible),
        "n_excluded_train_types": len(excluded_types),
        "excluded_train_types": excluded_types,
        "types_never_in_source_train": never_in_train,
        "source_rows": {
            "train": int(len(splits_source["train"])),
            "val": int(len(splits_source["val"])),
            "test": int(len(splits_source["test"])),
            "total": int(len(source)),
        },
        "filtered_rows": {
            "train": int(len(splits_filtered["train"])),
            "val": int(len(splits_filtered["val"])),
            "test": int(len(splits_filtered["test"])),
            "total": int(len(filtered)),
        },
        "rows_removed": {
            "train": int(len(splits_source["train"]) - len(splits_filtered["train"])),
            "val": int(len(splits_source["val"]) - len(splits_filtered["val"])),
            "test": int(len(splits_source["test"]) - len(splits_filtered["test"])),
        },
        "pair_type_counts": _pair_type_counts(filtered),
        "trainable_rows_after_fingerprints": {
            "train": int(len(trainable.loc[trainable["split"] == "train"])),
            "val": int(len(trainable.loc[trainable["split"] == "val"])),
            "test": int(len(trainable.loc[trainable["split"] == "test"])),
            "total": int(len(trainable)),
        },
        "invalid_fingerprint_rows_removed": int(len(enriched) - len(trainable)),
        "encoder_n_classes": encoder.n_classes,
        "unique_types_in_filtered": {
            "train": int(filtered.loc[filtered["split"] == "train", "type"].nunique()),
            "val": len(val_types),
            "test": len(test_types),
        },
        "val_types_not_in_encoder": sorted(val_types - encoder_types),
        "test_types_not_in_encoder": sorted(test_types - encoder_types),
        "pair_split_leakage_pairs": pair_leakage,
        "pair_type_split_leakage": pair_type_leakage,
        "train_class_frequency": {
            "min": int(train_type_counts.min()),
            "max": int(train_type_counts.max()),
            "median": float(train_type_counts.median()),
            "mean": float(train_type_counts.mean()),
        },
        "fingerprint_dim": cache.fingerprint_dim,
    }
