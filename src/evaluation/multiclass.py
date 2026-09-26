"""Multiclass classification metrics for interaction-type prediction."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    top_k_accuracy_score,
)


def compute_multiclass_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
    labels: list[int] | None = None,
    top_k: tuple[int, ...] = (3, 5),
) -> dict[str, Any]:
    """Compute multiclass metrics for encoded class indices."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    seen_mask = y_true >= 0
    metrics: dict[str, Any] = {
        "n_samples": int(len(y_true)),
        "n_seen_label_samples": int(seen_mask.sum()),
        "n_unseen_label_samples": int((~seen_mask).sum()),
    }
    if not seen_mask.any():
        metrics.update(
            {
                "accuracy": None,
                "macro_f1": None,
                "micro_f1": None,
                "balanced_accuracy": None,
            }
        )
        return metrics

    yt = y_true[seen_mask]
    yp = y_pred[seen_mask]
    label_values = labels if labels is not None else sorted(set(yt.tolist()))
    metrics["accuracy"] = float(accuracy_score(yt, yp))
    metrics["macro_f1"] = float(f1_score(yt, yp, average="macro", zero_division=0, labels=label_values))
    metrics["micro_f1"] = float(f1_score(yt, yp, average="micro", zero_division=0, labels=label_values))
    metrics["balanced_accuracy"] = float(balanced_accuracy_score(yt, yp))

    if y_prob is not None:
        prob = np.asarray(y_prob, dtype=float)
        prob = prob[seen_mask]
        if prob.ndim == 2 and prob.shape[1] > 0 and np.all(yt < prob.shape[1]):
            label_axis = list(range(prob.shape[1]))
            for k in top_k:
                if prob.shape[1] >= k:
                    metrics[f"top_{k}_accuracy"] = float(
                        top_k_accuracy_score(yt, prob, k=k, labels=label_axis)
                    )
    return metrics
