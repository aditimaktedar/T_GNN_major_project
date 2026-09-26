"""Multi-label DDI dataset construction and updateability pipeline.

Reads raw source CSV, aggregates unique drug pairs, builds multi-hot target
vectors, and saves the derived dataset and label mapping artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.config import (
    MULTILABEL_DATASET_AUDIT_PATH,
    MULTILABEL_FREQUENT363_DATASET_CSV,
    MULTILABEL_FREQUENT363_DATASET_PATH,
    MULTILABEL_FREQUENT363_LABEL_MAPPING_PATH,
    MULTILABEL_PAIR_SPLITS_PATH,
    SELECTED_MIMIC_TWOSIDES_DATASET_PATH,
)
from src.data.multilabel_audit import generate_multilabel_dataset_audit
from src.data.multilabel_splits import generate_pair_level_splits
from src.data.multilabel_validation import validate_multilabel_target


def load_versioned_label_mapping(
    path: Path | None = None,
    source_df: pd.DataFrame | None = None,
) -> dict[int, int]:
    """Load or generate the explicit versioned label mapping.

    If mapping JSON exists at `path`, load it.
    Otherwise, if source_df is provided, derive eligible FREQUENT363 types.
    """
    mapping_path = Path(path) if path is not None else MULTILABEL_FREQUENT363_LABEL_MAPPING_PATH
    if mapping_path.exists():
        payload = json.loads(mapping_path.read_text(encoding="utf-8"))
        # json keys are strings, parse to int -> int index
        mapping = {int(k): int(v) for k, v in payload["type_to_index"].items()}
        return mapping

    if source_df is not None:
        # Determine unique types sorted
        unique_types = sorted(source_df["type"].astype(int).unique())
        # For FREQUENT363 default subset (363 types), if 363 types exist in historical set:
        # Filter to 363 types if full source has 962
        type_counts = source_df["type"].astype(int).value_counts()
        top_types = sorted(type_counts.head(363).index.tolist())
        mapping = {int(type_id): idx for idx, type_id in enumerate(top_types)}
        save_versioned_label_mapping(mapping, mapping_path)
        return mapping

    raise FileNotFoundError(f"Label mapping not found at {mapping_path} and no source_df provided.")


def save_versioned_label_mapping(
    mapping: dict[int, int],
    path: Path | None = None,
) -> Path:
    target_path = Path(path) if path is not None else MULTILABEL_FREQUENT363_LABEL_MAPPING_PATH
    target_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": "FREQUENT363_v1",
        "n_classes": len(mapping),
        "type_to_index": {str(k): int(v) for k, v in sorted(mapping.items(), key=lambda x: x[1])},
        "index_to_type": {str(v): int(k) for k, v in sorted(mapping.items(), key=lambda x: x[1])},
    }
    target_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return target_path


def build_multilabel_dataset(
    source_path: Path | None = None,
    label_mapping_path: Path | None = None,
    output_parquet: Path | None = None,
    output_csv: Path | None = None,
    audit_path: Path | None = None,
    splits_path: Path | None = None,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build the pair-level multi-label dataset dynamically from source dataset.

    Returns
    -------
    tuple[pd.DataFrame, dict]
        The derived pair-level multi-label dataset and audit report.
    """
    src_path = Path(source_path) if source_path is not None else SELECTED_MIMIC_TWOSIDES_DATASET_PATH
    if not src_path.exists():
        raise FileNotFoundError(f"Source dataset not found at {src_path}")

    source_df = pd.read_csv(src_path)

    # 1. Load or initialize versioned label mapping
    # If historical frequent363 dataset exists, extract exact 363 label set to match label space
    mapping_path = Path(label_mapping_path) if label_mapping_path is not None else MULTILABEL_FREQUENT363_LABEL_MAPPING_PATH
    if not mapping_path.exists():
        freq363_csv = Path("data/processed/final_mimic_twosides_ml_frequent363.csv")
        if freq363_csv.exists():
            f363_df = pd.read_csv(freq363_csv)
            f363_types = sorted(f363_df["type"].astype(int).unique().tolist())
            label_mapping = {int(lbl): idx for idx, lbl in enumerate(f363_types)}
        else:
            type_counts = source_df["type"].astype(int).value_counts()
            top_types = sorted(type_counts.head(363).index.tolist())
            label_mapping = {int(lbl): idx for idx, lbl in enumerate(top_types)}
        save_versioned_label_mapping(label_mapping, mapping_path)
    else:
        label_mapping = load_versioned_label_mapping(mapping_path)

    n_classes = len(label_mapping)

    # 2. Detect unique drug pairs and aggregate
    # Filter source to valid label space
    valid_mask = source_df["type"].astype(int).isin(label_mapping.keys())
    valid_source = source_df.loc[valid_mask].copy()

    records = []
    grouped = valid_source.groupby("pair_key")
    for pair_key, group in grouped:
        first = group.iloc[0]
        drug_a = str(first["drug_a"])
        drug_b = str(first["drug_b"])
        smiles_a = str(first["smiles_a"])
        smiles_b = str(first["smiles_b"])
        mean_age = float(group["anchor_age"].mean())

        # Collect unique interaction types
        types_set = set(group["type"].astype(int).unique())
        labels_list = sorted([int(x) for x in types_set])

        # Build 363-dim multi-hot vector
        multi_hot = np.zeros(n_classes, dtype=int)
        for type_id in types_set:
            col_idx = label_mapping[type_id]
            multi_hot[col_idx] = 1

        records.append({
            "pair_key": str(pair_key),
            "drug_a": drug_a,
            "drug_b": drug_b,
            "smiles_a": smiles_a,
            "smiles_b": smiles_b,
            "anchor_age": mean_age,
            "labels": json.dumps(labels_list),
            "target": json.dumps([int(x) for x in multi_hot]),
        })

    derived_df = pd.DataFrame(records)

    # 3. Validate target construction assertions
    validate_multilabel_target(valid_source, derived_df, label_mapping)

    # 4. Generate pair-level splits
    out_splits_path = Path(splits_path) if splits_path is not None else MULTILABEL_PAIR_SPLITS_PATH
    split_map = generate_pair_level_splits(
        derived_df["pair_key"],
        seed=seed,
        output_path=out_splits_path,
    )
    derived_df["split"] = derived_df["pair_key"].map(split_map)

    # 5. Generate Audit
    audit = generate_multilabel_dataset_audit(source_df, derived_df, label_mapping)

    # 6. Save derived outputs
    out_csv = Path(output_csv) if output_csv is not None else MULTILABEL_FREQUENT363_DATASET_CSV
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    derived_df.to_csv(out_csv, index=False)

    out_parquet = Path(output_parquet) if output_parquet is not None else MULTILABEL_FREQUENT363_DATASET_PATH
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    try:
        derived_df.to_parquet(out_parquet, index=False)
    except (ImportError, Exception):
        pass  # CSV is primary saved artifact

    out_audit = Path(audit_path) if audit_path is not None else MULTILABEL_DATASET_AUDIT_PATH
    out_audit.parent.mkdir(parents=True, exist_ok=True)
    out_audit.write_text(json.dumps(audit, indent=2), encoding="utf-8")

    return derived_df, audit
