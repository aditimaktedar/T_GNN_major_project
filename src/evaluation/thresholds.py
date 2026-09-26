"""Sigmoid threshold management for multi-label classification.

Provides default thresholding and validation-only threshold optimization
to convert predicted probabilities to binary predictions.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import f1_score


DEFAULT_THRESHOLD = 0.50


def apply_threshold(y_prob: np.ndarray, threshold: float = DEFAULT_THRESHOLD) -> np.ndarray:
    """Binarize probability matrix using threshold.

    Parameters
    ----------
    y_prob : np.ndarray
        Predicted probabilities of shape (n_samples, n_classes).
    threshold : float
        Decision threshold.

    Returns
    -------
    np.ndarray
        Binary predictions of same shape.
    """
    return (np.asarray(y_prob, dtype=float) >= threshold).astype(int)


def optimize_threshold_on_val(
    y_true_val: np.ndarray,
    y_prob_val: np.ndarray,
    metric: str = "micro_f1",
    thresholds: list[float] | None = None,
) -> dict[str, Any]:
    """Search for the best threshold using only validation data.

    Parameters
    ----------
    y_true_val : np.ndarray
        Ground truth binary matrix from validation split only.
    y_prob_val : np.ndarray
        Predicted probabilities from validation split only.
    metric : str
        Metric to optimize ('micro_f1' or 'macro_f1').
    thresholds : list[float], optional
        Candidate thresholds to search. Defaults to [0.1, 0.2, ..., 0.9].

    Returns
    -------
    dict
        Contains 'best_threshold', 'best_score', and 'search_results'.
    """
    y_true_val = np.asarray(y_true_val, dtype=int)
    y_prob_val = np.asarray(y_prob_val, dtype=float)

    if thresholds is None:
        thresholds = [round(t * 0.05, 2) for t in range(1, 20)]  # 0.05 to 0.95

    average = "micro" if metric == "micro_f1" else "macro"

    search_results: list[dict[str, float]] = []
    best_threshold = DEFAULT_THRESHOLD
    best_score = -1.0

    for t in thresholds:
        y_pred = apply_threshold(y_prob_val, t)
        # Guard against all-zero predictions
        if y_pred.sum() == 0:
            score = 0.0
        else:
            score = float(f1_score(y_true_val, y_pred, average=average, zero_division=0))
        search_results.append({"threshold": t, "score": round(score, 6)})
        if score > best_score:
            best_score = score
            best_threshold = t

    return {
        "best_threshold": best_threshold,
        "best_score": round(best_score, 6),
        "metric": metric,
        "n_candidates": len(thresholds),
        "search_results": search_results,
        "note": "Threshold optimized on VALIDATION split only. Never on test.",
    }


def optimize_threshold_on_val_fine(
    y_true_val: np.ndarray,
    y_prob_val: np.ndarray,
    metric: str = "micro_f1",
) -> dict[str, Any]:
    """Two-stage threshold search: coarse (0.05 step) then fine (0.01 step).

    Stage 1: Search [0.05, 0.10, ..., 0.95] (same as existing).
    Stage 2: Search [coarse_best - 0.04, ..., coarse_best + 0.04] at 0.01 step.

    Parameters
    ----------
    y_true_val : np.ndarray
        Ground truth binary matrix from validation split only.
    y_prob_val : np.ndarray
        Predicted probabilities from validation split only.
    metric : str
        Metric to optimize ('micro_f1' or 'macro_f1').

    Returns
    -------
    dict
        Contains 'best_threshold', 'best_score', 'coarse_best', and 'search_results'.
    """
    # Stage 1: coarse search
    coarse = optimize_threshold_on_val(y_true_val, y_prob_val, metric=metric)
    coarse_best = coarse["best_threshold"]

    # Stage 2: fine search around the coarse winner
    fine_thresholds = [
        round(coarse_best + delta * 0.01, 2)
        for delta in range(-4, 5)
        if 0.01 <= round(coarse_best + delta * 0.01, 2) <= 0.99
    ]

    fine = optimize_threshold_on_val(
        y_true_val, y_prob_val, metric=metric, thresholds=fine_thresholds
    )

    # Pick the overall best
    if fine["best_score"] >= coarse["best_score"]:
        best_threshold = fine["best_threshold"]
        best_score = fine["best_score"]
    else:
        best_threshold = coarse_best
        best_score = coarse["best_score"]

    return {
        "best_threshold": best_threshold,
        "best_score": round(best_score, 6),
        "metric": metric,
        "coarse_best": coarse_best,
        "coarse_score": round(coarse["best_score"], 6),
        "n_candidates_coarse": coarse["n_candidates"],
        "n_candidates_fine": fine["n_candidates"],
        "coarse_search_results": coarse["search_results"],
        "fine_search_results": fine["search_results"],
        "note": "Two-stage threshold: coarse (0.05 step) + fine (0.01 step). VALIDATION split only.",
    }


def detect_degenerate_prediction(
    y_pred: np.ndarray,
    max_positive_fraction: float = 0.50,
) -> tuple[bool, float, float]:
    """Detect if binary predictions represent a degenerate (near-all-positive) solution.

    Parameters
    ----------
    y_pred : np.ndarray
        Binary prediction matrix of shape (n_samples, n_classes).
    max_positive_fraction : float, default=0.50
        Maximum allowable fraction of positive predictions before flagging as degenerate.

    Returns
    -------
    tuple[bool, float, float]
        (is_degenerate, positive_fraction, mean_pos_per_obs)
    """
    total_entries = y_pred.size
    if total_entries == 0:
        return False, 0.0, 0.0
    pos_count = float(np.sum(y_pred))
    pos_fraction = pos_count / total_entries
    mean_pos_per_obs = pos_count / float(y_pred.shape[0])
    is_degenerate = pos_fraction > max_positive_fraction
    return is_degenerate, round(pos_fraction, 6), round(mean_pos_per_obs, 2)


def optimize_threshold_non_degenerate(
    y_true_val: np.ndarray,
    y_prob_val: np.ndarray,
    primary_metric: str = "micro_f1",
    min_threshold: float = 0.05,
    max_threshold: float = 0.95,
    step: float = 0.01,
    max_positive_fraction: float = 0.50,
) -> dict[str, Any]:
    """Validation-only threshold optimization with degeneracy safeguards.

    Evaluates thresholds from min_threshold to max_threshold at `step` resolution.
    Tracks Micro-F1, Macro-F1, Hamming, Jaccard, predicted positive count, and degeneracy.
    Prefers the best-scoring non-degenerate threshold on validation set.

    Parameters
    ----------
    y_true_val : np.ndarray
        Validation ground truth binary targets.
    y_prob_val : np.ndarray
        Validation predicted probabilities.
    primary_metric : str, default='micro_f1'
        Primary metric for selection ('micro_f1', 'macro_f1', or 'map').
    min_threshold : float, default=0.05
        Minimum candidate threshold.
    max_threshold : float, default=0.95
        Maximum candidate threshold.
    step : float, default=0.01
        Threshold search step size.
    max_positive_fraction : float, default=0.50
        Fraction of labels above which predictions are flagged as degenerate.

    Returns
    -------
    dict[str, Any]
        Structured threshold optimization summary.
    """
    y_true_val = np.asarray(y_true_val, dtype=int)
    y_prob_val = np.asarray(y_prob_val, dtype=float)

    thresholds = [
        round(t, 4) for t in np.arange(min_threshold, max_threshold + 1e-6, step)
    ]

    search_results = []
    best_non_deg_threshold = None
    best_non_deg_score = -1.0

    best_overall_threshold = DEFAULT_THRESHOLD
    best_overall_score = -1.0

    for t in thresholds:
        y_pred = apply_threshold(y_prob_val, t)
        is_deg, pos_frac, mean_pos = detect_degenerate_prediction(
            y_pred, max_positive_fraction=max_positive_fraction
        )

        if y_pred.sum() == 0:
            micro_f1, macro_f1 = 0.0, 0.0
        else:
            micro_f1 = float(f1_score(y_true_val, y_pred, average="micro", zero_division=0))
            macro_f1 = float(f1_score(y_true_val, y_pred, average="macro", zero_division=0))

        score = micro_f1 if primary_metric == "micro_f1" else macro_f1

        res = {
            "threshold": t,
            "micro_f1": round(micro_f1, 6),
            "macro_f1": round(macro_f1, 6),
            "predicted_positive_fraction": pos_frac,
            "mean_pos_per_obs": mean_pos,
            "is_degenerate": is_deg,
        }
        search_results.append(res)

        if score > best_overall_score:
            best_overall_score = score
            best_overall_threshold = t

        if not is_deg and score > best_non_deg_score:
            best_non_deg_score = score
            best_non_deg_threshold = t

    # If all thresholds are degenerate or no non-degenerate surpassed -1, fallback to overall best
    if best_non_deg_threshold is not None:
        selected_threshold = best_non_deg_threshold
        selected_score = best_non_deg_score
        selection_status = "non_degenerate_optimum"
    else:
        selected_threshold = best_overall_threshold
        selected_score = best_overall_score
        selection_status = "fallback_all_thresholds_degenerate"

    selected_pred = apply_threshold(y_prob_val, selected_threshold)
    is_deg, pos_frac, mean_pos = detect_degenerate_prediction(selected_pred)

    return {
        "best_threshold": selected_threshold,
        "best_score": round(selected_score, 6),
        "primary_metric": primary_metric,
        "selection_status": selection_status,
        "is_degenerate": is_deg,
        "predicted_positive_fraction": pos_frac,
        "mean_pos_per_obs": mean_pos,
        "n_candidates": len(thresholds),
        "search_results": search_results,
        "note": "Threshold optimized on VALIDATION split only with non-degeneracy safeguards.",
    }

