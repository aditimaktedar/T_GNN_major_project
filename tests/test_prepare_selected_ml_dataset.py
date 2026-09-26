"""Tests for selected manual MIMIC↔TWOSIDES ML dataset preparation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data.exceptions import ConfigurationError
from src.data.prepare_selected_ml_dataset import (
    inspect_selected_dataset,
    prepare_final_mimic_twosides_ml,
    split_by_pair_key,
    validate_pair_key_splits,
)


def _write_sample(path: Path) -> None:
    rows = []
    for pair_key, smiles_a, smiles_b in [
        ("CID1|CID2", "CC", "CCC"),
        ("CID3|CID4", "CCCC", "CCCCC"),
        ("CID5|CID6", "C", "CC"),
    ]:
        for subject_id, hadm_id, type_id in [
            (1, 10, 0),
            (2, 20, 1),
            (1, 11, 0),
        ]:
            drug_a, drug_b = pair_key.split("|")
            rows.append(
                {
                    "subject_id": subject_id,
                    "hadm_id": hadm_id,
                    "drug_a": drug_a,
                    "drug_b": drug_b,
                    "smiles_a": smiles_a,
                    "smiles_b": smiles_b,
                    "pair_key": pair_key,
                    "type": type_id,
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)


def test_pair_split_has_no_leakage() -> None:
    splits = split_by_pair_key(["a|b", "c|d", "e|f", "g|h"], seed=7)
    validate_pair_key_splits(splits)
    assert len(splits) == 4


def test_prepare_preserves_types_and_assigns_pair_splits(tmp_path: Path) -> None:
    src = tmp_path / "selected.csv"
    _write_sample(src)
    out = tmp_path / "processed.csv"
    report = tmp_path / "report.json"

    processed, stats = prepare_final_mimic_twosides_ml(
        source_path=src,
        output_path=out,
        report_path=report,
        seed=99,
    )

    assert len(processed) == 9
    assert stats["rows_removed"] == 0
    assert set(processed.columns) >= {"smiles_a", "smiles_b", "type", "split"}
    assert processed["type"].nunique() == 2
    for pair_key, group in processed.groupby("pair_key"):
        assert group["split"].nunique() == 1
        assert group["smiles_a"].nunique() == 1
        assert group["smiles_b"].nunique() == 1


def test_exact_duplicate_removed(tmp_path: Path) -> None:
    src = tmp_path / "selected.csv"
    frame = pd.DataFrame(
        [
            {
                "subject_id": 1,
                "hadm_id": 1,
                "drug_a": "CID1",
                "drug_b": "CID2",
                "smiles_a": "C",
                "smiles_b": "CC",
                "pair_key": "CID1|CID2",
                "type": 0,
            }
        ]
    )
    frame = pd.concat([frame, frame], ignore_index=True)
    frame.to_csv(src, index=False)

    processed, stats = prepare_final_mimic_twosides_ml(
        source_path=src,
        output_path=tmp_path / "out.csv",
        report_path=tmp_path / "report.json",
        seed=1,
    )
    assert len(processed) == 1
    assert stats["rows_removed"] == 1


def test_validate_pair_key_splits_rejects_overlap() -> None:
    splits = pd.DataFrame({"pair_key": ["a|b", "a|b"], "split": ["train", "val"]})
    with pytest.raises(ConfigurationError):
        validate_pair_key_splits(splits)
