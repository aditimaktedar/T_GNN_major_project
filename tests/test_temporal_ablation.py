"""Unit and integration tests for temporal feature ablation study."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.config import TEMPORAL_MULTILABEL_FREQUENT363_DATASET_CSV
from src.models.temporal_ablation import (
    ABLATION_CONFIGS,
    execute_temporal_ablation_pipeline,
    generate_temporal_ablation_markdown,
    prepare_ablation_features,
    run_temporal_feature_ablation,
)


@pytest.fixture
def dummy_multilabel_dataset() -> pd.DataFrame:
    records = []
    # 10 train, 4 val, 2 test
    splits = ["train"] * 10 + ["val"] * 4 + ["test"] * 2
    for i, s in enumerate(splits):
        rec = {
            "subject_id": 100 + i,
            "hadm_id": 200 + i,
            "pair_key": f"CID001|CID00{i+2}",
            "drug_a": "CID001",
            "drug_b": f"CID00{i+2}",
            "anchor_age": 40 + i * 2,
            "num_a_events": (i % 3) + 1,
            "num_b_events": (i % 2) + 1,
            "num_total_pair_events": ((i % 3) + 1) + ((i % 2) + 1),
            "min_delta_hours": float(i * 0.5),
            "median_delta_hours": float(i * 1.5),
            "a_before_b": 1 if i % 3 == 0 else 0,
            "b_before_a": 1 if i % 3 == 1 else 0,
            "same_timestamp": 1 if i % 3 == 2 else 0,
            "labels": json.dumps([1, 2] if i % 2 == 0 else [2, 3]),
            "target": json.dumps([1, 1, 0] if i % 2 == 0 else [0, 1, 1]),
            "split": s,
        }
        records.append(rec)
    return pd.DataFrame(records)


def test_prepare_ablation_features_dimensions(dummy_multilabel_dataset: pd.DataFrame) -> None:
    train_df = dummy_multilabel_dataset[dummy_multilabel_dataset["split"] == "train"]
    val_df = dummy_multilabel_dataset[dummy_multilabel_dataset["split"] == "val"]
    test_df = dummy_multilabel_dataset[dummy_multilabel_dataset["split"] == "test"]

    # 1. Age-only (1 continuous, 0 binary)
    X_tr_age, X_val_age, X_te_age, _ = prepare_ablation_features(
        train_df, val_df, test_df, cont_cols=["anchor_age"], bin_cols=[]
    )
    assert X_tr_age.shape == (10, 1)
    assert X_val_age.shape == (4, 1)
    assert X_te_age.shape == (2, 1)
    assert np.isclose(X_tr_age.mean(), 0.0, atol=1e-5)

    # 2. Temporal-only (5 continuous, 3 binary)
    cont_cols = ["num_a_events", "num_b_events", "num_total_pair_events", "min_delta_hours", "median_delta_hours"]
    bin_cols = ["a_before_b", "b_before_a", "same_timestamp"]
    X_tr_temp, X_val_temp, X_te_temp, _ = prepare_ablation_features(
        train_df, val_df, test_df, cont_cols=cont_cols, bin_cols=bin_cols
    )
    assert X_tr_temp.shape == (10, 8)
    assert X_val_temp.shape == (4, 8)
    assert X_te_temp.shape == (2, 8)

    # 3. Age + Temporal (6 continuous, 3 binary)
    X_tr_full, X_val_full, X_te_full, _ = prepare_ablation_features(
        train_df, val_df, test_df, cont_cols=["anchor_age"] + cont_cols, bin_cols=bin_cols
    )
    assert X_tr_full.shape == (10, 9)
    assert X_val_full.shape == (4, 9)
    assert X_te_full.shape == (2, 9)


def test_ablation_execution_synthetic(dummy_multilabel_dataset: pd.DataFrame) -> None:
    results = run_temporal_feature_ablation(
        df=dummy_multilabel_dataset,
        epochs=10,
        lr=0.05,
    )

    assert results["study_name"] == "temporal_feature_ablation_study"
    assert len(results["configurations"]) == 4
    for key in ["prior_baseline", "age_only", "temporal_only", "age_temporal"]:
        assert key in results["configurations"]
        cfg_res = results["configurations"][key]
        assert "test_metrics_optimized_threshold" in cfg_res
        assert "test_metrics_default_threshold" in cfg_res
        assert "val_threshold_optimization" in cfg_res


def test_ablation_markdown_generation(dummy_multilabel_dataset: pd.DataFrame) -> None:
    results = run_temporal_feature_ablation(df=dummy_multilabel_dataset, epochs=5)
    md = generate_temporal_ablation_markdown(results)

    assert "# Temporal Feature Ablation Study Report" in md
    assert "Training-Prior Baseline" in md
    assert "Age-Only" in md
    assert "Temporal-Only" in md
    assert "Age + Temporal" in md
    assert "only 10 observations" in md.lower() or "10 observations" in md


def test_end_to_end_ablation_pipeline(tmp_path: Path) -> None:
    json_path = tmp_path / "ablation_results.json"
    md_path = tmp_path / "ablation_report.md"

    results, md = execute_temporal_ablation_pipeline(
        dataset_path=TEMPORAL_MULTILABEL_FREQUENT363_DATASET_CSV,
        results_json_path=json_path,
        report_md_path=md_path,
        epochs=50,
        lr=0.05,
    )

    assert json_path.exists()
    assert md_path.exists()

    assert results["dataset_split_counts"]["test"] == 10
    assert results["target_dimension"] == 363
    assert len(results["configurations"]) == 4
