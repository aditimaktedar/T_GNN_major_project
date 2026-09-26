"""Target validation module for multi-label DDI target construction.

Verifies that target matrix generation and pair aggregation are strictly
correct and free of data loss or schema errors.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd


def validate_multilabel_target(
    source_frame: pd.DataFrame,
    derived_frame: pd.DataFrame,
    label_mapping: dict[int, int] | Sequence[int],
) -> dict[str, Any]:
    """Validate multi-label target construction and pair aggregation.

    Parameters
    ----------
    source_frame : pd.DataFrame
        Raw or preprocessed source dataset containing `pair_key` and `type`.
    derived_frame : pd.DataFrame
        Aggregated derived dataset containing `pair_key` and `target` multi-hot vectors.
    label_mapping : dict or list
        Mapping or list of valid label IDs.

    Raises
    ------
    AssertionError
        If any validation rule fails.
    """
    if isinstance(label_mapping, dict):
        expected_n_classes = len(label_mapping)
        label_set = set(label_mapping.keys())
    else:
        expected_n_classes = len(label_mapping)
        label_set = set(int(lbl) for lbl in label_mapping)

    # Rule 1: Every source pair is represented
    source_pairs = set(source_frame["pair_key"].astype(str))
    derived_pairs = set(derived_frame["pair_key"].astype(str))
    missing_pairs = source_pairs - derived_pairs
    assert not missing_pairs, f"Target Validation Error: {len(missing_pairs)} source pairs missing from derived dataset."

    # Rule 2: No duplicate pair keys exist in derived dataset
    assert len(derived_frame) == len(derived_pairs), (
        f"Target Validation Error: Duplicate pair_key entries found in derived dataset. "
        f"Total rows: {len(derived_frame)}, unique pairs: {len(derived_pairs)}"
    )

    # Rule 3: Validate multi-hot vectors
    for idx, row in derived_frame.iterrows():
        pair_key = row["pair_key"]
        target = row["target"]
        if isinstance(target, str):
            import json
            target = json.loads(target)
        target_arr = np.asarray(target, dtype=int)

        # Vector length matches label mapping
        assert len(target_arr) == expected_n_classes, (
            f"Target Validation Error: Pair {pair_key} multi-hot length {len(target_arr)} "
            f"does not match expected length {expected_n_classes}."
        )

        # Values are strictly 0 or 1
        unique_vals = set(np.unique(target_arr))
        assert unique_vals.issubset({0, 1}), (
            f"Target Validation Error: Pair {pair_key} contains non-binary values: {unique_vals}"
        )

        # Each pair has at least one positive label
        assert target_arr.sum() > 0, (
            f"Target Validation Error: Pair {pair_key} has zero positive labels in target vector."
        )

    # Rule 4: No in-vocabulary pair-label association is lost
    valid_source = source_frame.loc[source_frame["type"].astype(int).isin(label_set)]
    pair_to_labels = valid_source.groupby("pair_key")["type"].apply(lambda s: set(s.astype(int))).to_dict()

    if isinstance(label_mapping, dict):
        type_to_idx = {int(k): int(v) for k, v in label_mapping.items()}
    else:
        type_to_idx = {int(lbl): idx for idx, lbl in enumerate(label_mapping)}

    for row in derived_frame.itertuples():
        pair_key = str(row.pair_key)
        target = getattr(row, "target")
        if isinstance(target, str):
            import json
            target = json.loads(target)
        target_arr = np.asarray(target, dtype=int)

        expected_types = pair_to_labels.get(pair_key, set())
        for type_id in expected_types:
            col_idx = type_to_idx[type_id]
            assert target_arr[col_idx] == 1, (
                f"Target Validation Error: Pair {pair_key} missing association for type {type_id} at index {col_idx}."
            )

    return {
        "status": "PASSED",
        "n_unique_pairs": len(derived_pairs),
        "n_classes": expected_n_classes,
        "note": "All target validation rules passed cleanly.",
    }
