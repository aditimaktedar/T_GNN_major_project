"""Unit and integration tests for temporal pair feature extraction and validation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.config import (
    MULTILABEL_FREQUENT363_DATASET_CSV,
    SELECTED_MIMIC_TWOSIDES_DATASET_PATH,
    TEMPORAL_EMAR_EVENTS_PATH,
)
from src.data.exceptions import DatasetSchemaError, SplitLeakageError
from src.data.temporal_pair_features import (
    TEMPORAL_PAIR_FEATURES_COLUMNS,
    build_temporal_pair_features,
    extract_temporal_pair_features,
    generate_validation_markdown,
    validate_temporal_pair_features,
)


@pytest.fixture
def sample_splits() -> dict[str, str]:
    return {
        "CID000001|CID000002": "train",
        "CID000003|CID000004": "val",
        "CID000005|CID000006": "test",
    }


@pytest.fixture
def sample_final_mimic() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "subject_id": 101,
            "hadm_id": 201,
            "drug_a": "CID000001",
            "drug_b": "CID000002",
            "pair_key": "CID000001|CID000002",
            "type": 1,
            "anchor_age": 65,
        },
        {
            "subject_id": 101,
            "hadm_id": 201,
            "drug_a": "CID000001",
            "drug_b": "CID000002",
            "pair_key": "CID000001|CID000002",
            "type": 2,
            "anchor_age": 65,
        },
        {
            "subject_id": 102,
            "hadm_id": 202,
            "drug_a": "CID000003",
            "drug_b": "CID000004",
            "pair_key": "CID000003|CID000004",
            "type": 5,
            "anchor_age": 45,
        },
        {
            "subject_id": 103,
            "hadm_id": 203,
            "drug_a": "CID000005",
            "drug_b": "CID000006",
            "pair_key": "CID000005|CID000006",
            "type": 8,
            "anchor_age": 72,
        },
    ])


@pytest.fixture
def sample_emar() -> pd.DataFrame:
    return pd.DataFrame([
        # Admission 201: A before B, multiple events
        {"subject_id": 101, "hadm_id": 201, "drug_cid": "CID000001", "event_time": "2180-01-01 08:00:00"},
        {"subject_id": 101, "hadm_id": 201, "drug_cid": "CID000001", "event_time": "2180-01-01 20:00:00"},
        {"subject_id": 101, "hadm_id": 201, "drug_cid": "CID000002", "event_time": "2180-01-01 12:00:00"},
        # Admission 202: B before A
        {"subject_id": 102, "hadm_id": 202, "drug_cid": "CID000004", "event_time": "2180-02-01 06:00:00"},
        {"subject_id": 102, "hadm_id": 202, "drug_cid": "CID000003", "event_time": "2180-02-01 10:00:00"},
        # Admission 203: Same timestamp
        {"subject_id": 103, "hadm_id": 203, "drug_cid": "CID000005", "event_time": "2180-03-01 09:00:00"},
        {"subject_id": 103, "hadm_id": 203, "drug_cid": "CID000006", "event_time": "2180-03-01 09:00:00"},
    ])


def test_extract_temporal_pair_features_synthetic(
    sample_emar: pd.DataFrame,
    sample_final_mimic: pd.DataFrame,
    sample_splits: dict[str, str],
) -> None:
    features = extract_temporal_pair_features(sample_emar, sample_final_mimic, sample_splits)

    assert len(features) == 3
    assert list(features.columns) == TEMPORAL_PAIR_FEATURES_COLUMNS

    # Row 1: A before B
    r1 = features.iloc[0]
    assert r1["subject_id"] == 101
    assert r1["hadm_id"] == 201
    assert r1["pair_key"] == "CID000001|CID000002"
    assert r1["num_a_events"] == 2
    assert r1["num_b_events"] == 1
    assert r1["num_total_pair_events"] == 3
    assert r1["first_a_time"] == "2180-01-01 08:00:00"
    assert r1["first_b_time"] == "2180-01-01 12:00:00"
    # Deltas: |08:00-12:00| = 4h, |20:00-12:00| = 8h -> min=4.0, median=6.0
    assert r1["min_delta_hours"] == 4.0
    assert r1["median_delta_hours"] == 6.0
    assert r1["a_before_b"] == 1
    assert r1["b_before_a"] == 0
    assert r1["same_timestamp"] == 0
    assert r1["split"] == "train"

    # Row 2: B before A
    r2 = features.iloc[1]
    assert r2["subject_id"] == 102
    assert r2["hadm_id"] == 202
    assert r2["first_a_time"] == "2180-02-01 10:00:00"
    assert r2["first_b_time"] == "2180-02-01 06:00:00"
    assert r2["min_delta_hours"] == 4.0
    assert r2["median_delta_hours"] == 4.0
    assert r2["a_before_b"] == 0
    assert r2["b_before_a"] == 1
    assert r2["same_timestamp"] == 0
    assert r2["split"] == "val"

    # Row 3: Same timestamp
    r3 = features.iloc[2]
    assert r3["subject_id"] == 103
    assert r3["hadm_id"] == 203
    assert r3["first_a_time"] == "2180-03-01 09:00:00"
    assert r3["first_b_time"] == "2180-03-01 09:00:00"
    assert r3["min_delta_hours"] == 0.0
    assert r3["median_delta_hours"] == 0.0
    assert r3["a_before_b"] == 0
    assert r3["b_before_a"] == 0
    assert r3["same_timestamp"] == 1
    assert r3["split"] == "test"


def test_missing_drug_skipped(
    sample_final_mimic: pd.DataFrame,
    sample_splits: dict[str, str],
) -> None:
    # Only drug A is in eMAR, drug B is missing
    partial_emar = pd.DataFrame([
        {"subject_id": 101, "hadm_id": 201, "drug_cid": "CID000001", "event_time": "2180-01-01 08:00:00"},
    ])
    features = extract_temporal_pair_features(partial_emar, sample_final_mimic, sample_splits)
    assert len(features) == 0


def test_schema_error_on_missing_columns() -> None:
    invalid_emar = pd.DataFrame({"subject_id": [1], "event_time": ["2180-01-01"]})
    invalid_fm = pd.DataFrame({"subject_id": [1]})
    with pytest.raises(DatasetSchemaError):
        extract_temporal_pair_features(invalid_emar, invalid_fm, {})


def test_split_leakage_detection(
    sample_emar: pd.DataFrame,
    sample_final_mimic: pd.DataFrame,
) -> None:
    df = extract_temporal_pair_features(
        sample_emar,
        sample_final_mimic,
        {"CID000001|CID000002": "train", "CID000003|CID000004": "train", "CID000005|CID000006": "train"},
    )
    # Artificially modify one row to induce split leakage
    df_leak = df.copy()
    extra_row = df.iloc[0:1].copy()
    extra_row["split"] = "test"
    df_leak = pd.concat([df_leak, extra_row], ignore_index=True)

    with pytest.raises(SplitLeakageError, match="Split leakage detected"):
        validate_temporal_pair_features(df_leak)


def test_validation_and_markdown(
    sample_emar: pd.DataFrame,
    sample_final_mimic: pd.DataFrame,
    sample_splits: dict[str, str],
) -> None:
    df = extract_temporal_pair_features(sample_emar, sample_final_mimic, sample_splits)
    audit = validate_temporal_pair_features(df, source_split_map=sample_splits)

    assert audit["total_temporal_observations"] == 3
    assert audit["unique_temporal_pairs"] == 3
    assert audit["unique_subjects"] == 3
    assert audit["unique_admissions"] == 3
    assert audit["total_missing_values"] == 0
    assert audit["split_leakage_detected"] is False

    md = generate_validation_markdown(audit)
    assert "# Temporal Pair Feature Dataset Validation Report" in md
    assert "train" in md
    assert "val" in md
    assert "test" in md


def test_end_to_end_pipeline(tmp_path: Path) -> None:
    out_csv = tmp_path / "temporal_pair_features.csv"
    report_md = tmp_path / "validation.md"
    audit_json = tmp_path / "audit.json"

    df, audit = build_temporal_pair_features(
        emar_path=TEMPORAL_EMAR_EVENTS_PATH,
        final_mimic_path=SELECTED_MIMIC_TWOSIDES_DATASET_PATH,
        splits_source_path=MULTILABEL_FREQUENT363_DATASET_CSV,
        output_csv_path=out_csv,
        report_md_path=report_md,
        audit_json_path=audit_json,
    )

    assert out_csv.exists()
    assert report_md.exists()
    assert audit_json.exists()

    assert len(df) == 84
    assert df["pair_key"].nunique() == 62
    assert df["subject_id"].nunique() == 49
    assert df["hadm_id"].nunique() == 57
    assert list(df.columns) == TEMPORAL_PAIR_FEATURES_COLUMNS
    assert df.isnull().sum().sum() == 0

    assert audit["total_temporal_observations"] == 84
    assert audit["split_counts_observations"] == {"train": 58, "val": 16, "test": 10}
    assert audit["split_leakage_detected"] is False
