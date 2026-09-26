"""Assemble an ML-ready dataset from pipeline intermediates."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.config import (
    LABELED_PAIRS_PATH,
    METRICS_DIR,
    ML_DATASET_PATH,
    MOLECULAR_FEATURES_PATH,
    PATIENT_SPLITS_PATH,
    PIPELINE_METADATA_PATH,
)
from src.data.exceptions import MissingInputError
from src.data.io_utils import load_table_chunked, write_json, write_table
from src.data.splits import validate_patient_splits


def _as_frame(obj: pd.DataFrame | Path, label: str, required: bool = True) -> tuple[pd.DataFrame | None, str | None]:
    if isinstance(obj, pd.DataFrame):
        return obj.copy(), "in-memory"
    if isinstance(obj, Path):
        if not obj.exists():
            if required:
                raise MissingInputError(
                    f"Dataset assembly is missing {label} at {obj}. Run the corresponding pipeline stage first."
                )
            return None, str(obj)
        return load_table_chunked(obj), str(obj)
    if obj is None:
        if required:
            raise MissingInputError(f"Dataset assembly is missing {label}.")
        return None, None
    raise MissingInputError(f"Unsupported {label} input: {type(obj)}")


def build_ml_dataset(
    labeled_pairs: pd.DataFrame | Path = LABELED_PAIRS_PATH,
    splits: pd.DataFrame | Path = PATIENT_SPLITS_PATH,
    molecular_features: pd.DataFrame | Path | None = MOLECULAR_FEATURES_PATH,
    output_path: Path | None = None,
    metadata_path: Path | None = None,
) -> tuple[pd.DataFrame, dict]:
    pairs, labeled_path = _as_frame(labeled_pairs, "labeled pairs", required=True)
    split_table, splits_path = _as_frame(splits, "patient splits", required=True)
    assert pairs is not None and split_table is not None
    validate_patient_splits(split_table)

    dataset = pairs.merge(split_table, on="patient_id", how="left")
    missing_split = int(dataset["split"].isna().sum())
    if missing_split:
        raise MissingInputError(
            f"{missing_split} pair rows have no patient split. All pair patients must appear in patient_splits."
        )

    features_frame, feature_path = _as_frame(
        molecular_features, "molecular features", required=False
    )
    feature_meta: dict
    if features_frame is not None and "drug_id" in features_frame.columns:
        a_feat = features_frame.rename(
            columns={
                "drug_id": "drug_a",
                "smiles": "drug_a_smiles",
                "smiles_valid": "drug_a_smiles_valid",
                "fingerprint": "drug_a_fingerprint",
                "skip_reason": "drug_a_feature_skip_reason",
            }
        )
        b_feat = features_frame.rename(
            columns={
                "drug_id": "drug_b",
                "smiles": "drug_b_smiles",
                "smiles_valid": "drug_b_smiles_valid",
                "fingerprint": "drug_b_fingerprint",
                "skip_reason": "drug_b_feature_skip_reason",
            }
        )
        keep_a = [col for col in a_feat.columns if col.startswith("drug_a")]
        keep_b = [col for col in b_feat.columns if col.startswith("drug_b")]
        dataset = dataset.merge(a_feat[keep_a], on="drug_a", how="left")
        dataset = dataset.merge(b_feat[keep_b], on="drug_b", how="left")
        feature_meta = {"attached": True, "feature_path": feature_path}
    else:
        feature_meta = {"attached": False, "reason": "molecular features were not available"}

    metadata = {
        "n_rows": int(len(dataset)),
        "n_patients": int(dataset["patient_id"].nunique()) if len(dataset) else 0,
        "label_counts": dataset["label"].value_counts().astype(int).to_dict() if "label" in dataset.columns else {},
        "split_counts": dataset["split"].value_counts().astype(int).to_dict() if "split" in dataset.columns else {},
        "labeled_pairs_path": labeled_path,
        "splits_path": splits_path,
        "molecular_features": feature_meta,
        "note": "Intermediate files were not deleted. This table joins pairs, labels, splits, and optional fingerprints.",
    }
    out = output_path if output_path is not None else ML_DATASET_PATH
    write_table(dataset, out)
    meta_out = metadata_path if metadata_path is not None else PIPELINE_METADATA_PATH
    write_json(meta_out, metadata)
    write_json(METRICS_DIR / "dataset_assembly_statistics.json", metadata)
    metadata["output_path"] = str(out)
    return dataset, metadata
