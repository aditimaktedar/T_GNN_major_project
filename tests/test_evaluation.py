"""Tests for evaluation utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.evaluation.ablation import compare_predictions
from src.evaluation.classification import compute_classification_metrics
from src.evaluation.drugbank_sanity import check_predictions_against_drugbank
from src.evaluation.pharmacogenetics import attach_pharmacogenetic_evidence
from src.evaluation.reproducibility import collect_environment_metadata, set_global_seed
from src.evaluation.xai_metrics import evaluate_xai, fidelity_score, sparsity_score, stability_score


def test_classification_metrics_all_present() -> None:
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])
    y_prob = np.array([0.1, 0.9, 0.4, 0.2])
    metrics = compute_classification_metrics(y_true, y_pred, y_prob)
    for key in ("accuracy", "precision", "recall", "f1", "auroc", "pr_auc", "mcc"):
        assert key in metrics


def test_single_class_edge_case() -> None:
    y_true = np.array([1, 1, 1])
    y_pred = np.array([1, 1, 1])
    y_prob = np.array([0.9, 0.8, 0.7])
    metrics = compute_classification_metrics(y_true, y_pred, y_prob)
    assert metrics["auroc"] is None


def test_xai_metrics() -> None:
    assert fidelity_score(0.9, 0.85) == pytest.approx(0.95)
    assert sparsity_score(np.array([1, 0, 0, 0])) == pytest.approx(0.75)
    assert stability_score(np.array([1, 0, 1]), np.array([1, 0, 1])) == pytest.approx(1.0)
    result = evaluate_xai(
        np.array([0.9, 0.2]),
        np.array([0.85, 0.25]),
        np.array([[1, 0, 0], [0, 1, 0]]),
        np.array([[1, 0, 0], [0, 0, 1]]),
    )
    assert "fidelity_mean" in result
    assert "sparsity_mean" in result
    assert result["stability_mean"] is not None


def test_missing_drugbank_and_pharmgkb(tmp_path: Path) -> None:
    preds = pd.DataFrame({"drug_a": ["A"], "drug_b": ["B"]})
    db = check_predictions_against_drugbank(preds, drugbank_dir=tmp_path / "missing")
    assert db["status"] == "drugbank_data_not_available"
    pg = attach_pharmacogenetic_evidence(preds, pharmgkb_dir=tmp_path / "missing")
    assert pg["status"] == "pharmgkb_data_not_available"


def test_ablation_compare() -> None:
    y_true = np.array([0, 1, 1, 0])
    baseline_pred = np.array([0, 1, 0, 0])
    rag_pred = np.array([0, 1, 1, 0])
    baseline_prob = np.array([0.2, 0.8, 0.4, 0.3])
    rag_prob = np.array([0.1, 0.9, 0.7, 0.2])
    result = compare_predictions(y_true, baseline_pred, baseline_prob, rag_pred, rag_prob)
    assert "delta_rag_minus_baseline" in result
    assert "f1" in result["baseline"]


def test_reproducibility_metadata() -> None:
    set_global_seed(42)
    meta = collect_environment_metadata({"seed": 42})
    assert "torch_version" in meta
    assert "device" in meta
