"""Multi-label evaluation metrics module.

Computes 13 standard multi-label evaluation metrics from ground-truth binary matrix
and predicted probability matrix.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    hamming_loss,
    jaccard_score,
)


def _precision_recall_at_k(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    k: int,
) -> tuple[float, float]:
    """Calculate Precision@k and Recall@k per sample, averaged over samples."""
    n_samples = y_true.shape[0]
    if n_samples == 0:
        return 0.0, 0.0

    precisions = []
    recalls = []

    for i in range(n_samples):
        true_pos_indices = set(np.where(y_true[i] == 1)[0])
        n_true = len(true_pos_indices)
        if n_true == 0:
            continue

        # Top-k predicted indices
        top_k_indices = set(np.argsort(y_prob[i])[-k:])
        hits = len(top_k_indices & true_pos_indices)

        precisions.append(hits / float(k))
        recalls.append(hits / float(n_true))

    mean_precision = float(np.mean(precisions)) if precisions else 0.0
    mean_recall = float(np.mean(recalls)) if recalls else 0.0
    return mean_precision, mean_recall


def compute_multilabel_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Calculate 13 multi-label classification metrics.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth binary matrix of shape (n_samples, n_classes).
    y_prob : np.ndarray
        Predicted probabilities of shape (n_samples, n_classes).
    threshold : float, default=0.5
        Decision threshold for binarizing probabilities.

    Returns
    -------
    dict[str, Any]
        Dictionary containing all 13 multi-label metrics.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    if y_true.shape != y_prob.shape:
        raise ValueError(
            f"Shape mismatch: y_true shape {y_true.shape} vs y_prob shape {y_prob.shape}"
        )

    # Binarize predictions using threshold
    y_pred = (y_prob >= threshold).astype(int)

    # 1. Micro F1
    micro_f1 = float(f1_score(y_true, y_pred, average="micro", zero_division=0))

    # 2. Macro F1
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    # 3. Hamming Loss
    h_loss = float(hamming_loss(y_true, y_pred))

    # 4. Jaccard Score (Micro)
    jaccard = float(jaccard_score(y_true, y_pred, average="micro", zero_division=0))

    # 5. Mean Average Precision (mAP / Micro AP)
    try:
        mAP = float(average_precision_score(y_true, y_prob, average="micro"))
    except Exception:
        mAP = 0.0

    # 6-13: Precision@k and Recall@k for k in [1, 3, 5, 10]
    p1, r1 = _precision_recall_at_k(y_true, y_prob, k=1)
    p3, r3 = _precision_recall_at_k(y_true, y_prob, k=3)
    p5, r5 = _precision_recall_at_k(y_true, y_prob, k=5)
    p10, r10 = _precision_recall_at_k(y_true, y_prob, k=10)

    return {
        "micro_f1": round(micro_f1, 6),
        "macro_f1": round(macro_f1, 6),
        "hamming_loss": round(h_loss, 6),
        "jaccard_score": round(jaccard, 6),
        "mAP": round(mAP, 6),
        "precision_at_1": round(p1, 6),
        "recall_at_1": round(r1, 6),
        "precision_at_3": round(p3, 6),
        "recall_at_3": round(r3, 6),
        "precision_at_5": round(p5, 6),
        "recall_at_5": round(r5, 6),
        "precision_at_10": round(p10, 6),
        "recall_at_10": round(r10, 6),
        "threshold": threshold,
        "n_samples": int(y_true.shape[0]),
        "n_classes": int(y_true.shape[1]),
    }
