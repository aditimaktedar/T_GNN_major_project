"""Unit and integration tests for temporal multi-label dataset preparation and validation."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.data.config import (
    MULTILABEL_FREQUENT363_DATASET_CSV,
    MULTILABEL_FREQUENT363_LABEL_MAPPING_PATH,
    TEMPORAL_PAIR_FEATURES_CSV,
)
from src.data.exceptions import DatasetSchemaError, SplitLeakageError
from src.data.temporal_multilabel import (
    TEMPORAL_MULTILABEL_COLUMNS,
    build_temporal_multilabel_dataset,
    generate_temporal_multilabel_markdown,
    join_temporal_multilabel_dataset,
    validate_temporal_multilabel_dataset,
)


@pytest.fixture
def sample_temporal_features() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "subject_id": 101,
            "hadm_id": 201,
            "pair_key": "CID000001|CID000002",
            "drug_a": "CID000001",
            "drug_b": "CID000002",
            "anchor_age": 55,
            "num_a_events": 2,
            "num_b_events": 1,
            "num_total_pair_events": 3,
            "first_a_time": "2180-01-01 08:00:00",
            "first_b_time": "2180-01-01 12:00:00",
            "min_delta_hours": 4.0,
            "median_delta_hours": 6.0,
            "a_before_b": 1,
            "b_before_a": 0,
            "same_timestamp": 0,
            "split": "train",
        },
        {
            "subject_id": 102,
            "hadm_id": 202,
            "pair_key": "CID000003|CID000004",
            "drug_a": "CID000003",
            "drug_b": "CID000004",
            "anchor_age": 60,
            "num_a_events": 1,
            "num_b_events": 1,
            "num_total_pair_events": 2,
            "first_a_time": "2180-02-01 10:00:00",
            "first_b_time": "2180-02-01 06:00:00",
            "min_delta_hours": 4.0,
            "median_delta_hours": 4.0,
            "a_before_b": 0,
            "b_before_a": 1,
            "same_timestamp": 0,
            "split": "val",
        },
    ])


@pytest.fixture
def sample_multilabel_df() -> pd.DataFrame:
    # 3-class target space: types 10, 20, 30
    return pd.DataFrame([
        {
            "pair_key": "CID000001|CID000002",
            "drug_a": "CID000001",
            "drug_b": "CID000002",
            "smiles_a": "CCO",
            "smiles_b": "CCN",
            "anchor_age": 55.0,
            "labels": json.dumps([10, 30]),
            "target": json.dumps([1, 0, 1]),
            "split": "train",
        },
        {
            "pair_key": "CID000003|CID000004",
            "drug_a": "CID000003",
            "drug_b": "CID000004",
            "smiles_a": "CCC",
            "smiles_b": "CCCl",
            "anchor_age": 60.0,
            "labels": json.dumps([20]),
            "target": json.dumps([0, 1, 0]),
            "split": "val",
        },
    ])


@pytest.fixture
def sample_label_mapping() -> dict[int, int]:
    return {10: 0, 20: 1, 30: 2}


def test_join_temporal_multilabel_dataset(
    sample_temporal_features: pd.DataFrame,
    sample_multilabel_df: pd.DataFrame,
    sample_label_mapping: dict[int, int],
) -> None:
    df = join_temporal_multilabel_dataset(sample_temporal_features, sample_multilabel_df)
    assert len(df) == 2
    assert list(df.columns) == TEMPORAL_MULTILABEL_COLUMNS
    assert df["smiles_a"].tolist() == ["CCO", "CCC"]
    assert df["target"].iloc[0] == json.dumps([1, 0, 1])

    audit = validate_temporal_multilabel_dataset(df, sample_label_mapping)
    assert audit["total_temporal_observations"] == 2
    assert audit["target_dimension"] == 3
    assert audit["total_missing_values"] == 0
    assert audit["target_consistency_verified"] is True
    assert audit["split_leakage_detected"] is False


def test_join_schema_error_missing_column(sample_temporal_features: pd.DataFrame) -> None:
    bad_ml = pd.DataFrame({"pair_key": ["A|B"], "split": ["train"]})
    with pytest.raises(DatasetSchemaError):
        join_temporal_multilabel_dataset(sample_temporal_features, bad_ml)


def test_target_consistency_validation_errors(
    sample_temporal_features: pd.DataFrame,
    sample_multilabel_df: pd.DataFrame,
    sample_label_mapping: dict[int, int],
) -> None:
    df = join_temporal_multilabel_dataset(sample_temporal_features, sample_multilabel_df)

    # 1. Target dimension mismatch
    bad_dim_df = df.copy()
    bad_dim_df.at[0, "target"] = json.dumps([1, 0])
    with pytest.raises(ValueError, match="target vector length"):
        validate_temporal_multilabel_dataset(bad_dim_df, sample_label_mapping)

    # 2. Target sum mismatch with labels
    bad_sum_df = df.copy()
    bad_sum_df.at[0, "target"] = json.dumps([1, 1, 1])
    with pytest.raises(ValueError, match="target active count"):
        validate_temporal_multilabel_dataset(bad_sum_df, sample_label_mapping)

    # 3. Label not in mapping
    bad_label_df = df.copy()
    bad_label_df.at[0, "labels"] = json.dumps([999, 30])
    bad_label_df.at[0, "target"] = json.dumps([1, 0, 1])
    with pytest.raises(ValueError, match="not in label_mapping"):
        validate_temporal_multilabel_dataset(bad_label_df, sample_label_mapping)


def test_split_leakage_in_validation(
    sample_temporal_features: pd.DataFrame,
    sample_multilabel_df: pd.DataFrame,
    sample_label_mapping: dict[int, int],
) -> None:
    df = join_temporal_multilabel_dataset(sample_temporal_features, sample_multilabel_df)
    # Inject leakage row
    leak_row = df.iloc[0:1].copy()
    leak_row["split"] = "test"
    df_leak = pd.concat([df, leak_row], ignore_index=True)

    with pytest.raises(SplitLeakageError, match="Split leakage detected"):
        validate_temporal_multilabel_dataset(df_leak, sample_label_mapping)


def test_end_to_end_temporal_multilabel_pipeline(tmp_path: Path) -> None:
    out_csv = tmp_path / "temporal_multilabel.csv"
    out_pq = tmp_path / "temporal_multilabel.parquet"
    rep_md = tmp_path / "validation.md"
    aud_json = tmp_path / "audit.json"

    df, audit = build_temporal_multilabel_dataset(
        temporal_features_path=TEMPORAL_PAIR_FEATURES_CSV,
        multilabel_path=MULTILABEL_FREQUENT363_DATASET_CSV,
        label_mapping_path=MULTILABEL_FREQUENT363_LABEL_MAPPING_PATH,
        output_csv_path=out_csv,
        output_parquet_path=out_pq,
        report_md_path=rep_md,
        audit_json_path=aud_json,
    )

    assert out_csv.exists()
    assert rep_md.exists()
    assert aud_json.exists()

    assert len(df) == 84
    assert df["pair_key"].nunique() == 62
    assert df["subject_id"].nunique() == 49
    assert df["hadm_id"].nunique() == 57
    assert list(df.columns) == TEMPORAL_MULTILABEL_COLUMNS
    assert df.isnull().sum().sum() == 0

    assert audit["total_temporal_observations"] == 84
    assert audit["target_dimension"] == 363
    assert audit["split_counts_observations"] == {"train": 58, "val": 16, "test": 10}
    assert audit["target_consistency_verified"] is True
    assert audit["split_leakage_detected"] is False
