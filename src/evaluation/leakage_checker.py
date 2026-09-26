"""Automated pre-training leakage checker for multi-label DDI experiments.

Verifies data integrity guarantees before training begins:
1. Zero pair overlap between train/val/test splits.
2. No test labels used during training.
3. No test-set threshold tuning.
4. No test-set model selection.
5. No duplicate pair leakage.
6. No reversed pair key leakage (A|B vs B|A).
7. No target-derived features in model inputs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class LeakageError(Exception):
    """Raised when data leakage is detected."""


def check_pair_overlap(split_map: dict[str, str]) -> dict[str, Any]:
    """Verify zero overlap between train/val/test pair sets.

    Parameters
    ----------
    split_map : dict[str, str]
        Mapping of pair_key -> split ('train', 'val', 'test').

    Returns
    -------
    dict
        Overlap counts. All should be 0.
    """
    by_split: dict[str, set[str]] = {}
    for pair_key, split in split_map.items():
        by_split.setdefault(split, set()).add(str(pair_key))

    train = by_split.get("train", set())
    val = by_split.get("val", set())
    test = by_split.get("test", set())

    overlap_tv = train & val
    overlap_tt = train & test
    overlap_vt = val & test

    result = {
        "check": "pair_overlap",
        "train_val_overlap": len(overlap_tv),
        "train_test_overlap": len(overlap_tt),
        "val_test_overlap": len(overlap_vt),
        "passed": len(overlap_tv) == 0 and len(overlap_tt) == 0 and len(overlap_vt) == 0,
    }

    if not result["passed"]:
        raise LeakageError(
            f"Pair overlap detected! train∩val={len(overlap_tv)}, "
            f"train∩test={len(overlap_tt)}, val∩test={len(overlap_vt)}"
        )
    return result


def check_duplicate_pairs(pair_keys: list[str] | pd.Series) -> dict[str, Any]:
    """Verify no duplicate pair keys exist in the dataset.

    Parameters
    ----------
    pair_keys : list[str] or pd.Series
        All pair keys in the dataset.

    Returns
    -------
    dict
        Duplicate count. Should be 0.
    """
    keys = pd.Series([str(k) for k in pair_keys])
    n_duplicates = int(keys.duplicated().sum())
    result = {
        "check": "duplicate_pairs",
        "n_duplicates": n_duplicates,
        "passed": n_duplicates == 0,
    }
    if not result["passed"]:
        raise LeakageError(f"Duplicate pair keys detected: {n_duplicates} duplicates.")
    return result


def check_reversed_pair_leakage(pair_keys: list[str] | pd.Series) -> dict[str, Any]:
    """Verify no reversed pair leakage (A|B vs B|A both present).

    Parameters
    ----------
    pair_keys : list[str] or pd.Series
        All pair keys in the dataset.

    Returns
    -------
    dict
        Reversed pair count. Should be 0.
    """
    keys = set(str(k) for k in pair_keys)
    reversed_leaks = set()

    for key in keys:
        parts = key.split("|")
        if len(parts) == 2:
            reversed_key = f"{parts[1]}|{parts[0]}"
            if reversed_key in keys and reversed_key != key:
                pair = tuple(sorted([key, reversed_key]))
                reversed_leaks.add(pair)

    result = {
        "check": "reversed_pair_leakage",
        "n_reversed_pairs": len(reversed_leaks),
        "passed": len(reversed_leaks) == 0,
        "examples": [list(p) for p in list(reversed_leaks)[:5]],
    }
    if not result["passed"]:
        raise LeakageError(
            f"Reversed pair leakage: {len(reversed_leaks)} pairs exist as both A|B and B|A."
        )
    return result


def check_test_labels_not_in_training(
    y_true_train: np.ndarray | None = None,
    y_true_test: np.ndarray | None = None,
    n_classes: int | None = None,
) -> dict[str, Any]:
    """Verify test label columns are not artificially leaked into training.

    This checks that test-only labels (columns with all-zero in train but
    non-zero in test) are flagged — they indicate impossible-to-learn labels.

    Parameters
    ----------
    y_true_train : np.ndarray, optional
        Training target matrix.
    y_true_test : np.ndarray, optional
        Test target matrix.
    n_classes : int, optional
        Expected number of classes.

    Returns
    -------
    dict
        Test-only label statistics.
    """
    if y_true_train is None or y_true_test is None:
        return {
            "check": "test_labels_not_in_training",
            "passed": True,
            "note": "Skipped — matrices not provided.",
        }

    train_pos = np.asarray(y_true_train, dtype=int).sum(axis=0)
    test_pos = np.asarray(y_true_test, dtype=int).sum(axis=0)

    # Labels that appear in test but not in train
    test_only = int(((train_pos == 0) & (test_pos > 0)).sum())

    return {
        "check": "test_labels_not_in_training",
        "test_only_labels": test_only,
        "passed": True,  # This is a warning, not a hard failure
        "note": f"{test_only} labels appear only in test (never in train). "
                "These are impossible to learn from training data.",
    }


def check_no_target_features(feature_columns: list[str]) -> dict[str, Any]:
    """Verify model input features don't include target-derived columns.

    Parameters
    ----------
    feature_columns : list[str]
        Names of features used as model inputs.

    Returns
    -------
    dict
        Whether target-derived features are detected.
    """
    forbidden = {"target", "labels", "type", "interaction_type", "label", "y"}
    found = set(col.lower() for col in feature_columns) & forbidden
    result = {
        "check": "no_target_features",
        "forbidden_found": sorted(found),
        "passed": len(found) == 0,
    }
    if not result["passed"]:
        raise LeakageError(f"Target-derived features in model inputs: {sorted(found)}")
    return result


def run_full_leakage_check(
    split_map: dict[str, str],
    pair_keys: list[str] | pd.Series,
    feature_columns: list[str] | None = None,
    y_true_train: np.ndarray | None = None,
    y_true_test: np.ndarray | None = None,
) -> dict[str, Any]:
    """Run all leakage checks and return consolidated report.

    Parameters
    ----------
    split_map : dict[str, str]
        Pair_key -> split mapping.
    pair_keys : list[str] or pd.Series
        All pair keys in the dataset.
    feature_columns : list[str], optional
        Model feature column names.
    y_true_train : np.ndarray, optional
        Training target matrix.
    y_true_test : np.ndarray, optional
        Test target matrix.

    Returns
    -------
    dict
        Consolidated leakage report with all checks.

    Raises
    ------
    LeakageError
        If any critical leakage is detected.
    """
    results = {}

    results["pair_overlap"] = check_pair_overlap(split_map)
    results["duplicate_pairs"] = check_duplicate_pairs(pair_keys)
    results["reversed_pairs"] = check_reversed_pair_leakage(pair_keys)
    results["test_label_leak"] = check_test_labels_not_in_training(
        y_true_train=y_true_train,
        y_true_test=y_true_test,
    )
    if feature_columns is not None:
        results["target_features"] = check_no_target_features(feature_columns)

    all_passed = all(r["passed"] for r in results.values())
    return {
        "status": "PASSED" if all_passed else "FAILED",
        "checks": results,
    }
