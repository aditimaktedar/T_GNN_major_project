"""Dataset audit module for multi-label DDI dataset.

Computes comprehensive statistics on pair aggregation, label co-occurrence,
and distribution without hardcoded expectations.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd


def generate_multilabel_dataset_audit(
    source_frame: pd.DataFrame,
    derived_frame: pd.DataFrame,
    label_mapping: dict[int, int] | list[int] | None = None,
) -> dict[str, Any]:
    """Generate detailed audit statistics for the multi-label dataset.

    Parameters
    ----------
    source_frame : pd.DataFrame
        Raw or preprocessed source frame.
    derived_frame : pd.DataFrame
        Derived pair-level multi-label frame.
    label_mapping : dict or list, optional
        Active label mapping.

    Returns
    -------
    dict
        Structured audit report.
    """
    n_source_rows = int(len(source_frame))
    n_unique_pairs = int(derived_frame["pair_key"].nunique())
    source_types = set(source_frame["type"].astype(int))
    n_unique_source_labels = int(len(source_types))

    # Pair-label associations in source dataset
    source_pair_labels = source_frame[["pair_key", "type"]].drop_duplicates()
    n_pair_label_associations = int(len(source_pair_labels))

    # Calculate labels per pair from derived target
    labels_per_pair_counts = []
    for target in derived_frame["target"]:
        if isinstance(target, str):
            target = json.loads(target)
        arr = np.asarray(target, dtype=int)
        labels_per_pair_counts.append(int(arr.sum()))

    counts_arr = np.asarray(labels_per_pair_counts, dtype=int)
    min_l = int(np.min(counts_arr)) if len(counts_arr) > 0 else 0
    max_l = int(np.max(counts_arr)) if len(counts_arr) > 0 else 0
    mean_l = float(np.mean(counts_arr)) if len(counts_arr) > 0 else 0.0
    median_l = float(np.median(counts_arr)) if len(counts_arr) > 0 else 0.0

    n_1_label = int((counts_arr == 1).sum())
    n_2_labels = int((counts_arr == 2).sum())
    n_3_labels = int((counts_arr == 3).sum())
    n_gt3_labels = int((counts_arr > 3).sum())

    # Label frequency distribution across unique pairs
    label_frequencies = {}
    if label_mapping is not None:
        if isinstance(label_mapping, dict):
            idx_to_type = {int(v): int(k) for k, v in label_mapping.items()}
        else:
            idx_to_type = {idx: int(lbl) for idx, lbl in enumerate(label_mapping)}

        target_matrix = np.vstack([
            json.loads(t) if isinstance(t, str) else np.asarray(t, dtype=int)
            for t in derived_frame["target"]
        ])
        pos_per_col = target_matrix.sum(axis=0)
        for col_idx, count in enumerate(pos_per_col):
            type_id = idx_to_type.get(col_idx, col_idx)
            label_frequencies[str(type_id)] = int(count)

    return {
        "source_rows": n_source_rows,
        "unique_drug_pairs": n_unique_pairs,
        "unique_interaction_labels": n_unique_source_labels,
        "pair_label_associations": n_pair_label_associations,
        "labels_per_pair": {
            "min": min_l,
            "max": max_l,
            "mean": round(mean_l, 4),
            "median": median_l,
        },
        "pairs_by_label_count": {
            "exactly_1_label": n_1_label,
            "exactly_2_labels": n_2_labels,
            "exactly_3_labels": n_3_labels,
            "more_than_3_labels": n_gt3_labels,
        },
        "label_frequency_distribution": label_frequencies,
    }
