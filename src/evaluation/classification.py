"""Classification metrics for Member B baselines."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _safe_metric(name: str, fn, default: float | None = None) -> float | None:
    try:
        value = float(fn())
        if np.isnan(value):
            return default
        return value
    except ValueError:
        return default


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
) -> dict[str, Any]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_prob is not None:
        y_prob = np.asarray(y_prob, dtype=float)
        metrics["auroc"] = _safe_metric("auroc", lambda: roc_auc_score(y_true, y_prob))
        metrics["pr_auc"] = _safe_metric("pr_auc", lambda: average_precision_score(y_true, y_prob))
    else:
        metrics["auroc"] = None
        metrics["pr_auc"] = None
    metrics["mcc"] = _safe_metric("mcc", lambda: matthews_corrcoef(y_true, y_pred), default=0.0)
    metrics["n_samples"] = int(len(y_true))
    metrics["n_positive"] = int((y_true == 1).sum())
    metrics["n_negative"] = int((y_true == 0).sum())
    return metrics
