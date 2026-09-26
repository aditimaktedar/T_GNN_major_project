"""Unit and integration tests for temporal multi-label baseline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.data.config import (
    MULTILABEL_FREQUENT363_LABEL_MAPPING_PATH,
    TEMPORAL_MULTILABEL_FREQUENT363_DATASET_CSV,
)
from src.models.temporal_multilabel_baseline import (
    CONTINUOUS_FEATURE_COLUMNS,
    TEMPORAL_FEATURE_COLUMNS,
    TemporalFeatureStandardizer,
    TemporalMultilabelLogisticRegression,
    evaluate_temporal_experiment,
    generate_baseline_markdown_report,
    train_temporal_baseline,
)


@pytest.fixture
def dummy_train_val_test_data() -> pd.DataFrame:
    records = []
    # Create 10 train, 4 val, 2 test records
    splits = ["train"] * 10 + ["val"] * 4 + ["test"] * 2
    for i, s in enumerate(splits):
        rec = {
            "subject_id": 100 + i,
            "hadm_id": 200 + i,
            "pair_key": f"CID001|CID00{i+2}",
            "drug_a": "CID001",
            "drug_b": f"CID00{i+2}",
            "anchor_age": 50 + i,
            "num_a_events": (i % 3) + 1,
            "num_b_events": (i % 2) + 1,
            "num_total_pair_events": ((i % 3) + 1) + ((i % 2) + 1),
            "min_delta_hours": float(i * 1.5),
            "median_delta_hours": float(i * 3.0),
            "a_before_b": 1 if i % 3 == 0 else 0,
            "b_before_a": 1 if i % 3 == 1 else 0,
            "same_timestamp": 1 if i % 3 == 2 else 0,
            "labels": json.dumps([1, 2] if i % 2 == 0 else [2, 3]),
            "target": json.dumps([1, 1, 0] if i % 2 == 0 else [0, 1, 1]),
            "split": s,
        }
        records.append(rec)
    return pd.DataFrame(records)


def test_standardizer(dummy_train_val_test_data: pd.DataFrame) -> None:
    train_df = dummy_train_val_test_data[dummy_train_val_test_data["split"] == "train"]
    val_df = dummy_train_val_test_data[dummy_train_val_test_data["split"] == "val"]

    std = TemporalFeatureStandardizer()
    X_train = std.fit_transform(train_df)
    X_val = std.transform(val_df)

    assert X_train.shape == (10, len(TEMPORAL_FEATURE_COLUMNS))
    assert X_val.shape == (4, len(TEMPORAL_FEATURE_COLUMNS))

    # Mean of standardized continuous training columns should be ~0
    n_cont = len(CONTINUOUS_FEATURE_COLUMNS)
    assert np.allclose(X_train[:, :n_cont].mean(axis=0), 0.0, atol=1e-5)


def test_model_forward_and_predict_proba() -> None:
    model = TemporalMultilabelLogisticRegression(input_dim=9, n_classes=3)
    x = torch.randn(5, 9)
    logits = model(x)
    assert logits.shape == (5, 3)

    probs = model.predict_proba(x)
    assert probs.shape == (5, 3)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()


def test_training_loop() -> None:
    X_tr = np.random.randn(20, 9).astype(np.float32)
    y_tr = np.random.randint(0, 2, size=(20, 5)).astype(np.float32)
    X_v = np.random.randn(5, 9).astype(np.float32)
    y_v = np.random.randint(0, 2, size=(5, 5)).astype(np.float32)

    model = TemporalMultilabelLogisticRegression(input_dim=9, n_classes=5)
    model, summary = train_temporal_baseline(
        model=model,
        X_train=X_tr,
        y_train=y_tr,
        X_val=X_v,
        y_val=y_v,
        epochs=10,
        lr=0.01,
    )

    assert summary["best_epoch"] >= 1
    assert "best_val_loss" in summary


def test_evaluate_temporal_experiment_full_dataset() -> None:
    df = pd.read_csv(TEMPORAL_MULTILABEL_FREQUENT363_DATASET_CSV)
    mapping_payload = json.loads(Path(MULTILABEL_FREQUENT363_LABEL_MAPPING_PATH).read_text(encoding="utf-8"))
    label_mapping = {int(k): int(v) for k, v in mapping_payload["type_to_index"].items()}

    results = evaluate_temporal_experiment(
        df=df,
        label_mapping=label_mapping,
        epochs=50,
        lr=0.05,
    )

    assert results["target_dimension"] == 363
    assert "test_metrics_optimized_threshold" in results["temporal_model_evaluation"]
    assert "test_metrics_optimized_threshold" in results["prior_baseline_evaluation"]

    md = generate_baseline_markdown_report(results)
    assert "# Temporal Multi-Label Baseline Evaluation Report" in md
    assert "Micro-F1" in md
    assert "Training Prior Baseline" in md
