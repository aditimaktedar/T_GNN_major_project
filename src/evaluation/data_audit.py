"""Audit processed datasets for counts, labels, and patient leakage."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.config import METRICS_DIR, ML_DATASET_PATH
from src.data.io_utils import write_json
from src.data.pyg_dataset import assert_no_patient_leakage, filter_trainable_rows, load_processed_dataset
from src.data.splits import validate_patient_splits


def audit_dataset(frame: pd.DataFrame | Path | None = None) -> dict:
    """Audit a processed ML dataset. Returns REAL DATA NOT AVAILABLE if none supplied."""
    if frame is None:
        if not ML_DATASET_PATH.exists():
            return {
                "status": "real_data_not_available",
                "message": "REAL DATA NOT AVAILABLE. No processed dataset at data/processed/ml_dataset.parquet.",
            }
        frame = load_processed_dataset(ML_DATASET_PATH)
    elif isinstance(frame, Path):
        if not frame.exists():
            return {
                "status": "real_data_not_available",
                "message": f"REAL DATA NOT AVAILABLE. File not found: {frame}",
            }
        frame = load_processed_dataset(frame)

    assert isinstance(frame, pd.DataFrame)
    synthetic_demo = bool(getattr(frame, "attrs", {}).get("synthetic_demo", False))

    leakage_detected = False
    leakage_message = None
    try:
        if "split" in frame.columns:
            patient_splits = frame[["patient_id", "split"]].drop_duplicates()
            validate_patient_splits(patient_splits)
            assert_no_patient_leakage(frame)
    except Exception as exc:
        leakage_detected = True
        leakage_message = str(exc)

    trainable = filter_trainable_rows(frame) if "label" in frame.columns else frame
    label_counts = frame["label"].value_counts().astype(int).to_dict() if "label" in frame.columns else {}
    split_counts = frame["split"].value_counts().astype(int).to_dict() if "split" in frame.columns else {}

    missing_fp_a = int(frame["drug_a_fingerprint"].isna().sum()) if "drug_a_fingerprint" in frame.columns else None
    missing_fp_b = int(frame["drug_b_fingerprint"].isna().sum()) if "drug_b_fingerprint" in frame.columns else None

    audit = {
        "status": "ok",
        "synthetic_demo": synthetic_demo,
        "warning": "SYNTHETIC SOFTWARE VALIDATION ONLY" if synthetic_demo else None,
        "n_patients": int(frame["patient_id"].nunique()) if "patient_id" in frame.columns else None,
        "n_drugs": int(len(set(frame["drug_a"]).union(set(frame["drug_b"]))))
        if {"drug_a", "drug_b"}.issubset(frame.columns)
        else None,
        "n_drug_pairs": int(frame["pair_key"].nunique()) if "pair_key" in frame.columns else int(len(frame)),
        "n_examples": int(len(frame)),
        "label_counts": label_counts,
        "positive_labels": int(label_counts.get("positive", 0)),
        "negative_labels": int(label_counts.get("negative", 0)),
        "unknown_unmatched_labels": int(label_counts.get("unknown", 0)),
        "trainable_examples": int(len(trainable)),
        "missing_drug_a_fingerprints": missing_fp_a,
        "missing_drug_b_fingerprints": missing_fp_b,
        "split_counts": split_counts,
        "patient_leakage_detected": leakage_detected,
        "patient_leakage_message": leakage_message,
    }
    return audit


def save_data_audit(audit: dict, output_path: Path | None = None) -> Path:
    path = output_path or METRICS_DIR / "data_audit.json"
    write_json(path, audit)
    return path
