"""Tests for final_mimic_twosides_ml dataset adapter."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from torch_geometric.data import Batch

from src.data.exceptions import SchemaError
from src.data.final_ml_dataset import (
    MulticlassDrugPairGraphDataset,
    TypeLabelEncoder,
    assert_no_pair_leakage,
    attach_fingerprints,
    dataset_sanity_report,
    load_final_ml_dataset,
    prepare_multiclass_frame,
    split_frame,
)


def _write_tiny_ml_csv(path: Path) -> None:
    rows = []
    pairs = [
        ("CID1|CID2", "CCO", "CC", 0),
        ("CID3|CID4", "CCC", "CCCC", 1),
        ("CID5|CID6", "CCO", "CCC", 2),
    ]
    for pair_key, smiles_a, smiles_b, type_id in pairs:
        drug_a, drug_b = pair_key.split("|")
        for hadm_id in (1, 2):
            rows.append(
                {
                    "subject_id": 100 + hadm_id,
                    "hadm_id": hadm_id,
                    "drug_a": drug_a,
                    "drug_b": drug_b,
                    "smiles_a": smiles_a,
                    "smiles_b": smiles_b,
                    "pair_key": pair_key,
                    "type": type_id,
                    "split": "train" if pair_key != "CID5|CID6" else "test",
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)


def test_type_label_encoder_fit_on_train_only() -> None:
    encoder = TypeLabelEncoder()
    encoder.fit([0, 2, 5])
    assert encoder.n_classes == 3
    assert encoder.transform([0, 2, 5]).tolist() == [0, 1, 2]
    assert encoder.transform([99], unknown=-1).tolist() == [-1]


def test_attach_fingerprints_and_pyg_dataset(tmp_path: Path) -> None:
    csv_path = tmp_path / "ml.csv"
    _write_tiny_ml_csv(csv_path)
    frame = load_final_ml_dataset(csv_path)
    enriched, encoder, _cache = prepare_multiclass_frame(frame)
    assert encoder.n_classes == 2
    assert enriched["fingerprint_valid"].all()
    ds = MulticlassDrugPairGraphDataset(enriched)
    trainable = enriched.loc[enriched["label_index"] >= 0]
    assert len(ds) == len(trainable)
    batch = Batch.from_data_list([ds[0], ds[1]])
    assert batch.x.shape[0] == 4
    assert batch.y.shape[0] == 2


def test_pair_split_no_leakage_and_sanity_report(tmp_path: Path) -> None:
    csv_path = tmp_path / "ml.csv"
    _write_tiny_ml_csv(csv_path)
    frame = load_final_ml_dataset(csv_path)
    assert_no_pair_leakage(frame)
    splits = split_frame(frame)
    assert len(splits["train"]) == 4
    assert len(splits["test"]) == 2
    report = dataset_sanity_report(frame)
    assert report["n_classes_train_encoder"] == 2
    assert report["pair_split_leakage_pairs"] == 0


def test_missing_columns_raise_schema_error() -> None:
    with pytest.raises(SchemaError):
        attach_fingerprints(pd.DataFrame({"smiles_a": ["CC"]}))
