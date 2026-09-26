"""Temporal Pair Feature Extraction and Validation for DDI.

Extracts temporal pair-admission features from eMAR administration records,
joins with observed DDI pairs and admission metadata, preserves existing
pair-level train/val/test splits, and validates against data leakage and missingness.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.config import (
    MULTILABEL_FREQUENT363_DATASET_CSV,
    SELECTED_MIMIC_TWOSIDES_DATASET_PATH,
    TEMPORAL_EMAR_EVENTS_PATH,
    TEMPORAL_PAIR_FEATURES_AUDIT_JSON,
    TEMPORAL_PAIR_FEATURES_CSV,
    TEMPORAL_PAIR_FEATURES_REPORT_MD,
)
from src.data.exceptions import DatasetSchemaError, SplitLeakageError

TEMPORAL_PAIR_FEATURES_COLUMNS = [
    "subject_id",
    "hadm_id",
    "pair_key",
    "drug_a",
    "drug_b",
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
    "split",
]


def extract_temporal_pair_features(
    emar_df: pd.DataFrame,
    final_mimic_df: pd.DataFrame,
    pair_splits: dict[str, str] | pd.DataFrame,
) -> pd.DataFrame:
    """Extract temporal pair-level features for each observed DDI pair + admission.

    Parameters
    ----------
    emar_df : pd.DataFrame
        eMAR events table with columns ['subject_id', 'hadm_id', 'drug_cid', 'event_time', ...]
    final_mimic_df : pd.DataFrame
        Final selected MIMIC-TWOSIDES dataset with columns ['subject_id', 'hadm_id', 'drug_a', 'drug_b', 'pair_key', 'anchor_age', ...]
    pair_splits : dict[str, str] | pd.DataFrame
        Mapping or DataFrame with 'pair_key' -> 'split' ('train', 'val', 'test').

    Returns
    -------
    pd.DataFrame
        Extracted temporal pair features matching TEMPORAL_PAIR_FEATURES_COLUMNS.
    """
    # Validate required columns
    required_emar_cols = {"subject_id", "hadm_id", "drug_cid", "event_time"}
    missing_emar = required_emar_cols - set(emar_df.columns)
    if missing_emar:
        raise DatasetSchemaError(f"emar_df missing required columns: {sorted(missing_emar)}")

    required_fm_cols = {"subject_id", "hadm_id", "drug_a", "drug_b", "pair_key", "anchor_age"}
    missing_fm = required_fm_cols - set(final_mimic_df.columns)
    if missing_fm:
        raise DatasetSchemaError(f"final_mimic_df missing required columns: {sorted(missing_fm)}")

    # Resolve pair split mapping
    if isinstance(pair_splits, pd.DataFrame):
        if "pair_key" not in pair_splits.columns or "split" not in pair_splits.columns:
            raise DatasetSchemaError("pair_splits DataFrame must contain 'pair_key' and 'split' columns.")
        split_map = dict(zip(pair_splits["pair_key"], pair_splits["split"]))
    elif isinstance(pair_splits, dict):
        split_map = dict(pair_splits)
    else:
        raise TypeError("pair_splits must be a dict or pd.DataFrame.")

    # Standardize eMAR timestamps
    emar_clean = emar_df[["subject_id", "hadm_id", "drug_cid", "event_time"]].copy()
    emar_clean["event_time"] = pd.to_datetime(emar_clean["event_time"])

    # Group eMAR event times by (hadm_id, drug_cid)
    # Pre-aggregate event times as sorted lists for fast lookup
    emar_grouped: dict[tuple[int, str], list[pd.Timestamp]] = {}
    for (hadm, drug), grp in emar_clean.groupby(["hadm_id", "drug_cid"]):
        emar_grouped[(int(hadm), str(drug))] = sorted(grp["event_time"].tolist())

    # Extract distinct DDI pair admissions from final_mimic_df
    adm_pairs = (
        final_mimic_df[["subject_id", "hadm_id", "pair_key", "drug_a", "drug_b", "anchor_age"]]
        .drop_duplicates()
        .sort_values(["subject_id", "hadm_id", "pair_key"])
    )

    records: list[dict[str, Any]] = []

    for _, row in adm_pairs.iterrows():
        sub_id = int(row["subject_id"])
        hadm_id = int(row["hadm_id"])
        da = str(row["drug_a"])
        db = str(row["drug_b"])
        pk = str(row["pair_key"])
        anchor_age = int(row["anchor_age"])

        if pk not in split_map:
            raise SplitLeakageError(f"Pair {pk!r} not found in provided split mapping.")

        split_label = split_map[pk]

        # Check eMAR events for both drugs
        times_a = emar_grouped.get((hadm_id, da), [])
        times_b = emar_grouped.get((hadm_id, db), [])

        # Only retain observations where both drugs were administered in this admission
        if not times_a or not times_b:
            continue

        first_a = times_a[0]
        first_b = times_b[0]

        # Calculate all pairwise administration delta hours
        deltas = [
            abs((ta - tb).total_seconds()) / 3600.0
            for ta in times_a
            for tb in times_b
        ]
        min_delta_hours = min(deltas)
        median_delta_hours = float(np.median(deltas))

        # Order indicators based on initial administration
        a_before_b = 1 if first_a < first_b else 0
        b_before_a = 1 if first_b < first_a else 0
        same_timestamp = 1 if first_a == first_b else 0

        records.append({
            "subject_id": sub_id,
            "hadm_id": hadm_id,
            "pair_key": pk,
            "drug_a": da,
            "drug_b": db,
            "anchor_age": anchor_age,
            "num_a_events": len(times_a),
            "num_b_events": len(times_b),
            "num_total_pair_events": len(times_a) + len(times_b),
            "first_a_time": str(first_a),
            "first_b_time": str(first_b),
            "min_delta_hours": round(min_delta_hours, 4),
            "median_delta_hours": round(median_delta_hours, 4),
            "a_before_b": a_before_b,
            "b_before_a": b_before_a,
            "same_timestamp": same_timestamp,
            "split": split_label,
        })

    out_df = pd.DataFrame(records, columns=TEMPORAL_PAIR_FEATURES_COLUMNS)
    return out_df


def validate_temporal_pair_features(
    df: pd.DataFrame,
    source_split_map: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Validate temporal pair feature dataset and return structured audit metrics.

    Parameters
    ----------
    df : pd.DataFrame
        Temporal pair features dataframe.
    source_split_map : dict[str, str] | None, optional
        Source pair-to-split mapping to verify exact consistency.

    Returns
    -------
    dict[str, Any]
        Audit and validation summary dictionary.
    """
    # 1. Column structure validation
    missing_cols = set(TEMPORAL_PAIR_FEATURES_COLUMNS) - set(df.columns)
    if missing_cols:
        raise DatasetSchemaError(f"Missing required columns: {sorted(missing_cols)}")

    if len(df) == 0:
        raise ValueError("Temporal pair features dataframe is empty.")

    # 2. Missing value check
    null_counts = df.isnull().sum().to_dict()
    total_nulls = sum(null_counts.values())
    if total_nulls > 0:
        raise ValueError(f"Found {total_nulls} missing values across temporal pair dataset: {null_counts}")

    # 3. Order indicator exclusivity check
    order_sums = df["a_before_b"] + df["b_before_a"] + df["same_timestamp"]
    if not (order_sums == 1).all():
        invalid_rows = df[order_sums != 1]
        raise ValueError(f"Order indicator exclusivity failed for {len(invalid_rows)} rows.")

    # 4. Total event count consistency
    event_sum_match = df["num_total_pair_events"] == (df["num_a_events"] + df["num_b_events"])
    if not event_sum_match.all():
        raise ValueError("num_total_pair_events does not match num_a_events + num_b_events.")

    # 5. Delta range validity
    if (df["min_delta_hours"] < 0).any():
        raise ValueError("min_delta_hours contains negative values.")
    if (df["median_delta_hours"] < df["min_delta_hours"] - 1e-6).any():
        raise ValueError("median_delta_hours is less than min_delta_hours for some rows.")

    # 6. Pair-level split leakage check
    pair_splits = df.groupby("pair_key")["split"].nunique()
    multi_split_pairs = pair_splits[pair_splits > 1]
    if len(multi_split_pairs) > 0:
        raise SplitLeakageError(
            f"Split leakage detected! Pairs occurring across multiple splits: {multi_split_pairs.index.tolist()}"
        )

    # 7. Check consistency with source splits if provided
    if source_split_map is not None:
        for pk, assigned_split in df[["pair_key", "split"]].drop_duplicates().values:
            if pk in source_split_map and source_split_map[pk] != assigned_split:
                raise SplitLeakageError(
                    f"Pair {pk!r} assigned split {assigned_split!r} does not match source split {source_split_map[pk]!r}."
                )

    # Compute audit statistics
    total_observations = int(len(df))
    unique_pairs = int(df["pair_key"].nunique())
    unique_subjects = int(df["subject_id"].nunique())
    unique_admissions = int(df["hadm_id"].nunique())

    split_counts = {str(k): int(v) for k, v in df["split"].value_counts().to_dict().items()}
    split_pair_counts = {
        str(k): int(v)
        for k, v in df.groupby("split")["pair_key"].nunique().to_dict().items()
    }

    # Delta t statistics
    min_delta_stats = {
        "min": float(df["min_delta_hours"].min()),
        "median": float(df["min_delta_hours"].median()),
        "mean": float(df["min_delta_hours"].mean()),
        "max": float(df["min_delta_hours"].max()),
    }
    median_delta_stats = {
        "min": float(df["median_delta_hours"].min()),
        "median": float(df["median_delta_hours"].median()),
        "mean": float(df["median_delta_hours"].mean()),
        "max": float(df["median_delta_hours"].max()),
    }

    # Repeated observations per pair
    obs_per_pair = df["pair_key"].value_counts()
    repeated_obs_distribution = {
        "min": int(obs_per_pair.min()),
        "median": float(obs_per_pair.median()),
        "mean": round(float(obs_per_pair.mean()), 4),
        "max": int(obs_per_pair.max()),
        "single_observation_pairs": int((obs_per_pair == 1).sum()),
        "repeated_observation_pairs": int((obs_per_pair > 1).sum()),
        "value_counts": {str(k): int(v) for k, v in obs_per_pair.value_counts().sort_index().to_dict().items()},
    }

    # Order indicator totals
    order_totals = {
        "a_before_b": int(df["a_before_b"].sum()),
        "b_before_a": int(df["b_before_a"].sum()),
        "same_timestamp": int(df["same_timestamp"].sum()),
    }

    audit_report = {
        "total_temporal_observations": total_observations,
        "unique_temporal_pairs": unique_pairs,
        "unique_subjects": unique_subjects,
        "unique_admissions": unique_admissions,
        "split_counts_observations": split_counts,
        "split_counts_unique_pairs": split_pair_counts,
        "missing_values": {k: int(v) for k, v in null_counts.items()},
        "total_missing_values": total_nulls,
        "delta_hours_statistics": {
            "min_delta_hours": min_delta_stats,
            "median_delta_hours": median_delta_stats,
        },
        "repeated_observations_per_pair": repeated_obs_distribution,
        "initial_administration_order": order_totals,
        "split_leakage_detected": False,
        "split_integrity_verified": True,
    }

    return audit_report


def generate_validation_markdown(audit: dict[str, Any]) -> str:
    """Generate Markdown report for temporal pair feature validation."""
    lines = [
        "# Temporal Pair Feature Dataset Validation Report",
        "",
        "> [!IMPORTANT]",
        "> **Dataset Generation & Validation Verification:**",
        f"> - **Total Temporal Pair/Admission Observations:** `{audit['total_temporal_observations']}`",
        f"> - **Unique Temporal DDI Pairs:** `{audit['unique_temporal_pairs']}`",
        f"> - **Unique Patients (`subject_id`):** `{audit['unique_subjects']}`",
        f"> - **Unique Admissions (`hadm_id`):** `{audit['unique_admissions']}`",
        f"> - **Split Leakage:** `0 pairs across multiple splits (Verified Disjoint)`",
        f"> - **Total Missing Values:** `{audit['total_missing_values']}`",
        "",
        "---",
        "",
        "## 1. Summary Overview",
        "",
        "| Metric | Count |",
        "| :--- | :---: |",
        f"| Total Temporal Observations (Pair + Admission) | {audit['total_temporal_observations']} |",
        f"| Unique Temporal Pairs | {audit['unique_temporal_pairs']} |",
        f"| Unique Subjects | {audit['unique_subjects']} |",
        f"| Unique Hospital Admissions | {audit['unique_admissions']} |",
        f"| Missing Values Across All Columns | {audit['total_missing_values']} |",
        "",
        "---",
        "",
        "## 2. Train / Validation / Test Split Distribution",
        "",
        "Preserved strictly from the existing pair-level split (`multilabel_frequent363_dataset.csv` / `multilabel_pair_splits.json`).",
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
        "> **Split Integrity:** 100% of temporal pairs belong strictly to their predetermined split partition. Zero leakage between train, validation, and test sets.",
        "",
        "---",
        "",
        r"## 3. Temporal Interval ($\Delta t$) Distribution",
        "",
        "| Metric | Min (Hours) | Median (Hours) | Mean (Hours) | Max (Hours) |",
        "| :--- | :---: | :---: | :---: | :---: |",
        f"| **Minimum $\\Delta t$ (`min_delta_hours`)** | {audit['delta_hours_statistics']['min_delta_hours']['min']:.4f} | {audit['delta_hours_statistics']['min_delta_hours']['median']:.4f} | {audit['delta_hours_statistics']['min_delta_hours']['mean']:.4f} | {audit['delta_hours_statistics']['min_delta_hours']['max']:.4f} |",
        f"| **Median $\\Delta t$ (`median_delta_hours`)** | {audit['delta_hours_statistics']['median_delta_hours']['min']:.4f} | {audit['delta_hours_statistics']['median_delta_hours']['median']:.4f} | {audit['delta_hours_statistics']['median_delta_hours']['mean']:.4f} | {audit['delta_hours_statistics']['median_delta_hours']['max']:.4f} |",

        "",
        "---",
        "",
        "## 4. Initial Administration Order Distribution",
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
        "## 5. Repeated Observations per Pair",
        "",
        f"- **Min Observations / Pair:** `{audit['repeated_observations_per_pair']['min']}`",
        f"- **Median Observations / Pair:** `{audit['repeated_observations_per_pair']['median']}`",
        f"- **Mean Observations / Pair:** `{audit['repeated_observations_per_pair']['mean']}`",
        f"- **Max Observations / Pair:** `{audit['repeated_observations_per_pair']['max']}`",
        f"- **Single-Observation Pairs:** `{audit['repeated_observations_per_pair']['single_observation_pairs']}` ({audit['repeated_observations_per_pair']['single_observation_pairs'] / total_pairs * 100:.2f}%)",
        f"- **Repeated-Observation Pairs (>1 admission):** `{audit['repeated_observations_per_pair']['repeated_observation_pairs']}` ({audit['repeated_observations_per_pair']['repeated_observation_pairs'] / total_pairs * 100:.2f}%)",
        "",
        "### Frequency Breakdown",
        "",
        "| Observations per Pair ($k$) | Number of Pairs with $k$ Observations |",
        "| :---: | :---: |",
    ])

    for k, v in audit["repeated_observations_per_pair"]["value_counts"].items():
        lines.append(f"| {k} | {v} |")

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


def build_temporal_pair_features(
    emar_path: Path | str = TEMPORAL_EMAR_EVENTS_PATH,
    final_mimic_path: Path | str = SELECTED_MIMIC_TWOSIDES_DATASET_PATH,
    splits_source_path: Path | str = MULTILABEL_FREQUENT363_DATASET_CSV,
    output_csv_path: Path | str = TEMPORAL_PAIR_FEATURES_CSV,
    report_md_path: Path | str = TEMPORAL_PAIR_FEATURES_REPORT_MD,
    audit_json_path: Path | str = TEMPORAL_PAIR_FEATURES_AUDIT_JSON,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Execute end-to-end temporal pair features pipeline.

    Loads input datasets, extracts features, runs validations, writes outputs,
    and returns the dataframe and audit summary.
    """
    emar_path = Path(emar_path)
    final_mimic_path = Path(final_mimic_path)
    splits_source_path = Path(splits_source_path)
    output_csv_path = Path(output_csv_path)
    report_md_path = Path(report_md_path)
    audit_json_path = Path(audit_json_path)

    if not emar_path.exists():
        raise FileNotFoundError(f"eMAR events file not found: {emar_path}")
    if not final_mimic_path.exists():
        raise FileNotFoundError(f"Final MIMIC-TWOSIDES dataset not found: {final_mimic_path}")
    if not splits_source_path.exists():
        raise FileNotFoundError(f"Splits source dataset not found: {splits_source_path}")

    # Load source files
    emar_df = pd.read_csv(emar_path)
    final_mimic_df = pd.read_csv(final_mimic_path)
    splits_df = pd.read_csv(splits_source_path)

    # Extract features
    pair_features_df = extract_temporal_pair_features(
        emar_df=emar_df,
        final_mimic_df=final_mimic_df,
        pair_splits=splits_df,
    )

    # Validate
    source_split_map = dict(zip(splits_df["pair_key"], splits_df["split"]))
    audit = validate_temporal_pair_features(pair_features_df, source_split_map=source_split_map)

    # Ensure parent directories exist
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    report_md_path.parent.mkdir(parents=True, exist_ok=True)
    audit_json_path.parent.mkdir(parents=True, exist_ok=True)

    # Save outputs
    pair_features_df.to_csv(output_csv_path, index=False)
    with open(audit_json_path, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)

    report_md = generate_validation_markdown(audit)
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write(report_md)

    return pair_features_df, audit
