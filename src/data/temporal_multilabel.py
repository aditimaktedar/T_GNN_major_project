"""Temporal Multi-Label DDI Dataset Construction and Validation.

Joins admission-level temporal pair features with the canonical 363-dimensional
multi-hot targets and molecular SMILES representations from multilabel_frequent363_dataset.csv.
Performs rigorous target consistency, label index mapping, and split leakage validation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.config import (
    MULTILABEL_FREQUENT363_DATASET_CSV,
    MULTILABEL_FREQUENT363_LABEL_MAPPING_PATH,
    TEMPORAL_MULTILABEL_AUDIT_JSON,
    TEMPORAL_MULTILABEL_FREQUENT363_DATASET_CSV,
    TEMPORAL_MULTILABEL_FREQUENT363_DATASET_PATH,
    TEMPORAL_MULTILABEL_REPORT_MD,
    TEMPORAL_PAIR_FEATURES_CSV,
)
from src.data.exceptions import DatasetSchemaError, SplitLeakageError

TEMPORAL_MULTILABEL_COLUMNS = [
    "subject_id",
    "hadm_id",
    "pair_key",
    "drug_a",
    "drug_b",
    "smiles_a",
    "smiles_b",
    "anchor_age",
    "num_a_events",
    "num_b_events",
    "num_total_pair_events",
    "first_a_time",
    "first_b_time",
    "min_delta_hours",
    "median_delta_hours",
    "a_before_b",
    "b_before_a",
    "same_timestamp",
    "labels",
    "target",
    "split",
]


def join_temporal_multilabel_dataset(
    temporal_features_df: pd.DataFrame,
    multilabel_df: pd.DataFrame,
) -> pd.DataFrame:
    """Join temporal pair features with multi-label targets and molecular structures.

    Parameters
    ----------
    temporal_features_df : pd.DataFrame
        Temporal pair features with admission temporal stats and split.
    multilabel_df : pd.DataFrame
        Canonical frequent363 multi-label dataset with SMILES, labels, and targets.

    Returns
    -------
    pd.DataFrame
        Joined model-ready temporal multi-label dataset.
    """
    req_temporal = {
        "subject_id", "hadm_id", "pair_key", "drug_a", "drug_b", "anchor_age",
        "num_a_events", "num_b_events", "num_total_pair_events",
        "first_a_time", "first_b_time", "min_delta_hours", "median_delta_hours",
        "a_before_b", "b_before_a", "same_timestamp", "split",
    }
    missing_temporal = req_temporal - set(temporal_features_df.columns)
    if missing_temporal:
        raise DatasetSchemaError(f"temporal_features_df missing required columns: {sorted(missing_temporal)}")

    req_ml = {"pair_key", "smiles_a", "smiles_b", "labels", "target", "split"}
    missing_ml = req_ml - set(multilabel_df.columns)
    if missing_ml:
        raise DatasetSchemaError(f"multilabel_df missing required columns: {sorted(missing_ml)}")

    # Prepare multi-label subset for merge
    ml_subset = multilabel_df[["pair_key", "smiles_a", "smiles_b", "labels", "target", "split"]].copy()

    # Join strictly on pair_key and split to verify split congruence
    merged = temporal_features_df.merge(
        ml_subset,
        on=["pair_key", "split"],
        how="inner",
    )

    if len(merged) != len(temporal_features_df):
        missing_count = len(temporal_features_df) - len(merged)
        raise SplitLeakageError(
            f"Join mismatch: {missing_count} temporal observations could not be matched to multi-label target with matching split."
        )

    out_df = merged[TEMPORAL_MULTILABEL_COLUMNS].copy()
    return out_df


def validate_temporal_multilabel_dataset(
    df: pd.DataFrame,
    label_mapping: dict[int, int],
    source_split_map: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Validate target consistency, label space completeness, and split integrity.

    Parameters
    ----------
    df : pd.DataFrame
        Joined temporal multi-label dataframe.
    label_mapping : dict[int, int]
        Mapping from adverse interaction type ID to index (0..362).
    source_split_map : dict[str, str] | None, optional
        Source pair split mapping for verification.

    Returns
    -------
    dict[str, Any]
        Audit summary dictionary.
    """
    # 1. Schema check
    missing_cols = set(TEMPORAL_MULTILABEL_COLUMNS) - set(df.columns)
    if missing_cols:
        raise DatasetSchemaError(f"Missing required columns: {sorted(missing_cols)}")

    if len(df) == 0:
        raise ValueError("Dataset is empty.")

    # 2. Missing value check
    null_counts = df.isnull().sum().to_dict()
    total_nulls = sum(null_counts.values())
    if total_nulls > 0:
        raise ValueError(f"Found {total_nulls} missing values across dataset: {null_counts}")

    # 3. Target vector consistency checks
    n_classes = len(label_mapping)
    total_labels_count = 0
    active_labels_per_obs: list[int] = []

    for idx, row in df.iterrows():
        try:
            labels_list = json.loads(row["labels"])
            target_vec = json.loads(row["target"])
        except Exception as e:
            raise ValueError(f"Failed to parse JSON labels/target at row {idx}: {e}") from e

        if len(target_vec) != n_classes:
            raise ValueError(
                f"Row {idx} has target vector length {len(target_vec)}, expected {n_classes}."
            )

        if not all(v in (0, 1) for v in target_vec):
            raise ValueError(f"Row {idx} target vector contains non-binary values.")

        if sum(target_vec) != len(labels_list):
            raise ValueError(
                f"Row {idx} target active count ({sum(target_vec)}) does not match labels count ({len(labels_list)})."
            )

        for type_id in labels_list:
            if type_id not in label_mapping:
                raise ValueError(f"Row {idx} contains label {type_id} not in label_mapping.")
            target_idx = label_mapping[type_id]
            if target_vec[target_idx] != 1:
                raise ValueError(
                    f"Row {idx} label {type_id} maps to index {target_idx}, but target[{target_idx}] is {target_vec[target_idx]}."
                )

        total_labels_count += len(labels_list)
        active_labels_per_obs.append(len(labels_list))

    # 4. Split Leakage & Disjointness Check
    pair_splits = df.groupby("pair_key")["split"].nunique()
    multi_split_pairs = pair_splits[pair_splits > 1]
    if len(multi_split_pairs) > 0:
        raise SplitLeakageError(
            f"Split leakage detected! Pairs occurring across multiple splits: {multi_split_pairs.index.tolist()}"
        )

    if source_split_map is not None:
        for pk, assigned_split in df[["pair_key", "split"]].drop_duplicates().values:
            if pk in source_split_map and source_split_map[pk] != assigned_split:
                raise SplitLeakageError(
                    f"Pair {pk!r} assigned split {assigned_split!r} does not match source split {source_split_map[pk]!r}."
                )

    # 5. Order flag and event checks
    order_sums = df["a_before_b"] + df["b_before_a"] + df["same_timestamp"]
    if not (order_sums == 1).all():
        raise ValueError("Order indicators are not mutually exclusive and exhaustive.")

    if not (df["num_total_pair_events"] == (df["num_a_events"] + df["num_b_events"])).all():
        raise ValueError("num_total_pair_events does not match num_a_events + num_b_events.")

    # 6. Audit statistics
    total_observations = int(len(df))
    unique_pairs = int(df["pair_key"].nunique())
    unique_subjects = int(df["subject_id"].nunique())
    unique_admissions = int(df["hadm_id"].nunique())

    split_counts_obs = {str(k): int(v) for k, v in df["split"].value_counts().to_dict().items()}
    split_counts_pairs = {
        str(k): int(v) for k, v in df.groupby("split")["pair_key"].nunique().to_dict().items()
    }

    obs_per_pair = df["pair_key"].value_counts()

    audit_report = {
        "total_temporal_observations": total_observations,
        "unique_temporal_pairs": unique_pairs,
        "unique_subjects": unique_subjects,
        "unique_admissions": unique_admissions,
        "target_dimension": n_classes,
        "total_active_label_associations": total_labels_count,
        "labels_per_observation": {
            "min": int(min(active_labels_per_obs)),
            "median": float(np.median(active_labels_per_obs)),
            "mean": round(float(np.mean(active_labels_per_obs)), 4),
            "max": int(max(active_labels_per_obs)),
        },
        "split_counts_observations": split_counts_obs,
        "split_counts_unique_pairs": split_counts_pairs,
        "missing_values": {k: int(v) for k, v in null_counts.items()},
        "total_missing_values": total_nulls,
        "delta_hours_statistics": {
            "min_delta_hours": {
                "min": float(df["min_delta_hours"].min()),
                "median": float(df["min_delta_hours"].median()),
                "mean": float(df["min_delta_hours"].mean()),
                "max": float(df["min_delta_hours"].max()),
            },
            "median_delta_hours": {
                "min": float(df["median_delta_hours"].min()),
                "median": float(df["median_delta_hours"].median()),
                "mean": float(df["median_delta_hours"].mean()),
                "max": float(df["median_delta_hours"].max()),
            },
        },
        "repeated_observations_per_pair": {
            "min": int(obs_per_pair.min()),
            "median": float(obs_per_pair.median()),
            "mean": round(float(obs_per_pair.mean()), 4),
            "max": int(obs_per_pair.max()),
            "single_observation_pairs": int((obs_per_pair == 1).sum()),
            "repeated_observation_pairs": int((obs_per_pair > 1).sum()),
            "value_counts": {str(k): int(v) for k, v in obs_per_pair.value_counts().sort_index().to_dict().items()},
        },
        "initial_administration_order": {
            "a_before_b": int(df["a_before_b"].sum()),
            "b_before_a": int(df["b_before_a"].sum()),
            "same_timestamp": int(df["same_timestamp"].sum()),
        },
        "target_consistency_verified": True,
        "split_leakage_detected": False,
        "split_integrity_verified": True,
    }

    return audit_report


def generate_temporal_multilabel_markdown(audit: dict[str, Any]) -> str:
    """Generate Markdown report for temporal multi-label dataset validation."""
    lines = [
        "# Model-Ready Temporal Multi-Label Dataset Validation Report",
        "",
        "> [!IMPORTANT]",
        "> **Dataset Generation & Validation Verification:**",
        f"> - **Total Observations (Pair + Admission):** `{audit['total_temporal_observations']}`",
        f"> - **Target Dimension:** `{audit['target_dimension']}` multi-label classes",
        f"> - **Unique Temporal DDI Pairs:** `{audit['unique_temporal_pairs']}`",
        f"> - **Unique Patients (`subject_id`):** `{audit['unique_subjects']}`",
        f"> - **Unique Admissions (`hadm_id`):** `{audit['unique_admissions']}`",
        f"> - **Target Consistency:** `100% verified (Zero encoding or dimension errors)`",
        f"> - **Split Leakage:** `0 pairs across multiple splits (Verified Disjoint)`",
        f"> - **Total Missing Values:** `{audit['total_missing_values']}`",
        "",
        "---",
        "",
        "## 1. Dataset Overview",
        "",
        "| Metric | Value |",
        "| :--- | :---: |",
        f"| Total Temporal Observations | {audit['total_temporal_observations']} |",
        f"| Target Dimension ($C$) | {audit['target_dimension']} |",
        f"| Total Multi-Label Associations | {audit['total_active_label_associations']} |",
        f"| Unique Temporal Pairs | {audit['unique_temporal_pairs']} |",
        f"| Unique Subjects | {audit['unique_subjects']} |",
        f"| Unique Hospital Admissions | {audit['unique_admissions']} |",
        f"| Missing Values Across All Columns | {audit['total_missing_values']} |",
        "",
        "---",
        "",
        "## 2. Multi-Label Target Distribution per Observation",
        "",
        f"- **Min Active Labels / Observation:** `{audit['labels_per_observation']['min']}`",
        f"- **Median Active Labels / Observation:** `{audit['labels_per_observation']['median']}`",
        f"- **Mean Active Labels / Observation:** `{audit['labels_per_observation']['mean']}`",
        f"- **Max Active Labels / Observation:** `{audit['labels_per_observation']['max']}`",
        "",
        "---",
        "",
        "## 3. Preserved Split Distribution",
        "",
        "| Split | Observations Count | Observations % | Unique Pairs | Unique Pairs % |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ]

    total_obs = audit["total_temporal_observations"]
    total_pairs = audit["unique_temporal_pairs"]
    for s in ["train", "val", "test"]:
        obs_c = audit["split_counts_observations"].get(s, 0)
        pair_c = audit["split_counts_unique_pairs"].get(s, 0)
        obs_pct = (obs_c / total_obs * 100.0) if total_obs > 0 else 0.0
        pair_pct = (pair_c / total_pairs * 100.0) if total_pairs > 0 else 0.0
        lines.append(f"| `{s}` | {obs_c} | {obs_pct:.2f}% | {pair_c} | {pair_pct:.2f}% |")

    lines.extend([
        "",
        "> [!NOTE]",
        "> **Split Preservation:** Pair-level partition strictly preserved from `multilabel_frequent363_dataset.csv`. Zero overlap between train, val, and test splits.",
        "",
        "---",
        "",
        r"## 4. Temporal Interval ($\Delta t$) Distribution",
        "",
        "| Metric | Min (Hours) | Median (Hours) | Mean (Hours) | Max (Hours) |",
        "| :--- | :---: | :---: | :---: | :---: |",
        f"| **Minimum $\\Delta t$ (`min_delta_hours`)** | {audit['delta_hours_statistics']['min_delta_hours']['min']:.4f} | {audit['delta_hours_statistics']['min_delta_hours']['median']:.4f} | {audit['delta_hours_statistics']['min_delta_hours']['mean']:.4f} | {audit['delta_hours_statistics']['min_delta_hours']['max']:.4f} |",
        f"| **Median $\\Delta t$ (`median_delta_hours`)** | {audit['delta_hours_statistics']['median_delta_hours']['min']:.4f} | {audit['delta_hours_statistics']['median_delta_hours']['median']:.4f} | {audit['delta_hours_statistics']['median_delta_hours']['mean']:.4f} | {audit['delta_hours_statistics']['median_delta_hours']['max']:.4f} |",
        "",
        "---",
        "",
        "## 5. Initial Administration Order Distribution",
        "",
        "| Initial Administration Event | Count | Percentage |",
        "| :--- | :---: | :---: |",
    ])

    order = audit["initial_administration_order"]
    for k, name in [
        ("a_before_b", "Drug A Administered First (`a_before_b`)"),
        ("b_before_a", "Drug B Administered First (`b_before_a`)"),
        ("same_timestamp", "Simultaneous Initial Administration (`same_timestamp`)"),
    ]:
        cnt = order.get(k, 0)
        pct = (cnt / total_obs * 100.0) if total_obs > 0 else 0.0
        lines.append(f"| {name} | {cnt} | {pct:.2f}% |")

    lines.extend([
        "",
        "---",
        "",
        "## 6. Missingness Audit",
        "",
        "| Column Name | Missing Count | Status |",
        "| :--- | :---: | :--- |",
    ])

    for col, count in audit["missing_values"].items():
        status = "Clean (0 missing)" if count == 0 else f"FAIL ({count} missing)"
        lines.append(f"| `{col}` | {count} | {status} |")

    lines.append("")
    return "\n".join(lines)


def build_temporal_multilabel_dataset(
    temporal_features_path: Path | str = TEMPORAL_PAIR_FEATURES_CSV,
    multilabel_path: Path | str = MULTILABEL_FREQUENT363_DATASET_CSV,
    label_mapping_path: Path | str = MULTILABEL_FREQUENT363_LABEL_MAPPING_PATH,
    output_csv_path: Path | str = TEMPORAL_MULTILABEL_FREQUENT363_DATASET_CSV,
    output_parquet_path: Path | str = TEMPORAL_MULTILABEL_FREQUENT363_DATASET_PATH,
    report_md_path: Path | str = TEMPORAL_MULTILABEL_REPORT_MD,
    audit_json_path: Path | str = TEMPORAL_MULTILABEL_AUDIT_JSON,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Execute temporal multi-label dataset construction and validation pipeline."""
    temporal_path = Path(temporal_features_path)
    ml_path = Path(multilabel_path)
    lbl_path = Path(label_mapping_path)
    out_csv = Path(output_csv_path)
    out_pq = Path(output_parquet_path)
    rep_md = Path(report_md_path)
    aud_json = Path(audit_json_path)

    if not temporal_path.exists():
        raise FileNotFoundError(f"Temporal pair features file not found: {temporal_path}")
    if not ml_path.exists():
        raise FileNotFoundError(f"Multi-label dataset not found: {ml_path}")
    if not lbl_path.exists():
        raise FileNotFoundError(f"Label mapping file not found: {lbl_path}")

    temporal_df = pd.read_csv(temporal_path)
    ml_df = pd.read_csv(ml_path)
    mapping_payload = json.loads(lbl_path.read_text(encoding="utf-8"))
    label_mapping = {int(k): int(v) for k, v in mapping_payload["type_to_index"].items()}

    # Join
    joined_df = join_temporal_multilabel_dataset(temporal_df, ml_df)

    # Validate
    source_split_map = dict(zip(ml_df["pair_key"], ml_df["split"]))
    audit = validate_temporal_multilabel_dataset(
        joined_df,
        label_mapping=label_mapping,
        source_split_map=source_split_map,
    )

    # Save outputs
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_pq.parent.mkdir(parents=True, exist_ok=True)
    rep_md.parent.mkdir(parents=True, exist_ok=True)
    aud_json.parent.mkdir(parents=True, exist_ok=True)

    joined_df.to_csv(out_csv, index=False)
    try:
        joined_df.to_parquet(out_pq, index=False)
    except Exception:
        pass

    with open(aud_json, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)

    report_md_content = generate_temporal_multilabel_markdown(audit)
    with open(rep_md, "w", encoding="utf-8") as f:
        f.write(report_md_content)

    return joined_df, audit
