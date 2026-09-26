"""Tests for target formulation analysis utilities."""

from __future__ import annotations

import pandas as pd

from src.evaluation.target_formulation_analysis import analyze_target_formulations


def _tiny_frame() -> pd.DataFrame:
    rows = []
    for pair, type_id, split in [
        ("A|B", 0, "train"),
        ("A|B", 1, "train"),
        ("A|B", 0, "train"),
        ("C|D", 2, "train"),
        ("E|F", 0, "val"),
        ("G|H", 1, "test"),
    ]:
        rows.append(
            {
                "subject_id": 1,
                "hadm_id": 1,
                "drug_a": "x",
                "drug_b": "y",
                "smiles_a": "CCO",
                "smiles_b": "CCC",
                "pair_key": pair,
                "type": type_id,
                "split": split,
            }
        )
    return pd.DataFrame(rows)


def test_analyze_target_formulations_structure() -> None:
    report = analyze_target_formulations(_tiny_frame(), threshold=1)
    assert report["option1_frequency_filtered_multiclass"]["pair_type_split_leakage"] == 0
    assert report["option2_pair_level_multilabel"]["n_unique_pairs"] == 4
    assert report["option2_pair_level_multilabel"]["labels_per_pair"]["min"] >= 1
