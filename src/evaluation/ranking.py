"""Top-K interaction ranking utility for multi-label DDI predictions.

Provides ranked outputs with pair_key, interaction_type, probability, and
true/false positive indicator.
"""

from __future__ import annotations

from typing import Any

import json

import numpy as np
import pandas as pd


def rank_predictions_for_pair(
    pair_key: str,
    y_prob: np.ndarray,
    y_true: np.ndarray | None,
    label_mapping: dict[int, int],
    k: int = 5,
) -> list[dict[str, Any]]:
    """Rank predicted interactions for a single drug pair.

    Parameters
    ----------
    pair_key : str
        Identifier for the drug pair.
    y_prob : np.ndarray
        Predicted probabilities of shape (n_classes,).
    y_true : np.ndarray or None
        Ground truth binary vector of shape (n_classes,), or None if unknown.
    label_mapping : dict[int, int]
        Mapping of type_id -> column_index in the multi-hot vector.
    k : int
        Number of top predictions to return.

    Returns
    -------
    list[dict]
        Top-k ranked interactions with type, probability, and correctness.
    """
    y_prob = np.asarray(y_prob, dtype=float)
    idx_to_type = {int(v): int(k_) for k_, v in label_mapping.items()}

    top_k_indices = np.argsort(y_prob)[::-1][:k]

    ranked = []
    for rank_pos, col_idx in enumerate(top_k_indices):
        type_id = idx_to_type.get(int(col_idx), int(col_idx))
        entry = {
            "rank": rank_pos + 1,
            "pair_key": pair_key,
            "interaction_type": type_id,
            "column_index": int(col_idx),
            "probability": round(float(y_prob[col_idx]), 6),
        }
        if y_true is not None:
            y_true_arr = np.asarray(y_true, dtype=int)
            entry["is_true_positive"] = bool(y_true_arr[col_idx] == 1)
        ranked.append(entry)

    return ranked


def rank_predictions_batch(
    pair_keys: list[str],
    y_prob: np.ndarray,
    y_true: np.ndarray | None,
    label_mapping: dict[int, int],
    k: int = 5,
) -> list[dict[str, Any]]:
    """Rank predictions for a batch of drug pairs.

    Parameters
    ----------
    pair_keys : list[str]
        List of pair identifiers.
    y_prob : np.ndarray
        Predicted probabilities of shape (n_samples, n_classes).
    y_true : np.ndarray or None
        Ground truth binary matrix of shape (n_samples, n_classes).
    label_mapping : dict[int, int]
        Mapping of type_id -> column_index.
    k : int
        Number of top predictions per pair.

    Returns
    -------
    list[dict]
        Flat list of all ranked predictions across all pairs.
    """
    all_ranked = []
    for i, pair_key in enumerate(pair_keys):
        y_true_row = y_true[i] if y_true is not None else None
        ranked = rank_predictions_for_pair(
            pair_key=pair_key,
            y_prob=y_prob[i],
            y_true=y_true_row,
            label_mapping=label_mapping,
            k=k,
        )
        all_ranked.extend(ranked)
    return all_ranked


def format_rankings_as_dataframe(rankings: list[dict[str, Any]]) -> pd.DataFrame:
    """Convert ranked predictions list to a clean DataFrame.

    Returns
    -------
    pd.DataFrame
        Columns: rank, pair_key, interaction_type, probability, is_true_positive.
    """
    if not rankings:
        return pd.DataFrame(
            columns=["rank", "pair_key", "interaction_type", "probability", "is_true_positive"]
        )
    return pd.DataFrame(rankings)


def compute_topk_hit_rates(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    k_values: list[int] | None = None,
) -> dict[str, float]:
    """Compute hit-rate@k: fraction of samples where at least one true positive is in top-k.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth binary matrix (n_samples, n_classes).
    y_prob : np.ndarray
        Predicted probabilities (n_samples, n_classes).
    k_values : list[int], optional
        K values to compute. Defaults to [1, 3, 5, 10].

    Returns
    -------
    dict[str, float]
        hit_rate_at_1, hit_rate_at_3, hit_rate_at_5, hit_rate_at_10.
    """
    if k_values is None:
        k_values = [1, 3, 5, 10]

    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    n_samples = y_true.shape[0]

    results = {}
    for k in k_values:
        hits = 0
        for i in range(n_samples):
            true_pos = set(np.where(y_true[i] == 1)[0])
            if not true_pos:
                continue
            top_k = set(np.argsort(y_prob[i])[-k:])
            if top_k & true_pos:
                hits += 1
        total = sum(1 for i in range(n_samples) if y_true[i].sum() > 0)
        results[f"hit_rate_at_{k}"] = round(hits / max(total, 1), 6)

    return results
