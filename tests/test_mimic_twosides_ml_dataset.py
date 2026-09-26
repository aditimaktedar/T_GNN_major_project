"""Tests for streaming MIMIC ↔ TWOSIDES ML dataset preparation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest

from src.data.exceptions import ConfigurationError
from src.data.mimic_twosides_ml_dataset import (
    assemble_ml_dataset,
    build_patient_splits_table,
    canonicalize_mimic_pair,
    check_patient_leakage,
    collect_negative_candidate_pairs,
    inspect_labeled_pairs_columns,
    prepare_mimic_twosides_ml_dataset,
    sample_negative_pairs,
    stream_dedupe_positives,
    validate_real_data_inputs,
)
from src.data.splits import validate_patient_splits


def _write_mapping(path: Path) -> None:
    pd.DataFrame(
        {
            "mimic_drug_name_norm": ["drug_a", "drug_b", "drug_c", "drug_x"],
            "twosides_drug_id": [
                "CID000000001",
                "CID000000002",
                "CID000000003",
                "CID000000004",
            ],
            "mapping_status": ["matched"] * 4,
            "mapping_method": ["local_pubchem_cache"] * 4,
        }
    ).to_parquet(path, index=False)


def _write_twosides_unique_pairs(path: Path) -> None:
    pd.DataFrame(
        {
            "pair_key": ["CID000000001||CID000000002"],
            "drug_a": ["CID000000001"],
            "drug_b": ["CID000000002"],
        }
    ).to_parquet(path, index=False)


def _write_labeled(path: Path) -> None:
    rows = []
    for patient_id in (1, 1, 2):
        rows.append(
            {
                "patient_id": patient_id,
                "admission_id": 100 + patient_id,
                "mimic_drug_a": "drug_b",
                "mimic_drug_b": "drug_a",
                "mimic_pair_key": "drug_b||drug_a",
                "twosides_drug_a": "CID000000001",
                "twosides_drug_b": "CID000000002",
                "twosides_pair_key": "CID000000001||CID000000002",
                "interaction_type": 0,
                "neg_sample_drug": "CID000000003",
                "label": "positive",
                "label_source": "twosides",
                "pair_rule": "same_admission",
            }
        )
        rows.append(
            {
                "patient_id": patient_id,
                "admission_id": 100 + patient_id,
                "mimic_drug_a": "drug_a",
                "mimic_drug_b": "drug_b",
                "mimic_pair_key": "drug_a||drug_b",
                "twosides_drug_a": "CID000000001",
                "twosides_drug_b": "CID000000002",
                "twosides_pair_key": "CID000000001||CID000000002",
                "interaction_type": 1,
                "neg_sample_drug": "CID000000004",
                "label": "positive",
                "label_source": "twosides",
                "pair_rule": "same_admission",
            }
        )
    # Duplicate patient/pair/type row for dedup test.
    rows.append(dict(rows[0]))
    pd.DataFrame(rows).to_parquet(path, index=False)


def _write_drug_pairs(path: Path) -> None:
    pd.DataFrame(
        {
            "patient_id": [1, 2, 3, 4],
            "admission_id": [10, 20, 30, 40],
            "drug_a": ["drug_a", "drug_a", "drug_a", "drug_a"],
            "drug_b": ["drug_c", "drug_x", "drug_b", "drug_b"],
            "pair_key": ["drug_a||drug_c", "drug_a||drug_x", "drug_a||drug_b", "drug_b||drug_a"],
            "pair_rule": ["same_admission"] * 4,
        }
    ).to_parquet(path, index=False)


def test_canonicalize_mimic_pair() -> None:
    a, b, key = canonicalize_mimic_pair("zocor", "aspirin")
    assert a == "aspirin"
    assert b == "zocor"
    assert key == "aspirin||zocor"


def test_inspect_labeled_pairs_columns(tmp_path: Path) -> None:
    labeled = tmp_path / "labeled.parquet"
    _write_labeled(labeled)
    info = inspect_labeled_pairs_columns(labeled)
    assert info["num_rows"] == 7
    assert "interaction_type" in info["columns"]
    assert "patient_id" in info["field_definitions"]


def test_stream_dedupe_preserves_interaction_types(tmp_path: Path) -> None:
    labeled = tmp_path / "labeled.parquet"
    _write_labeled(labeled)
    out = tmp_path / "positives.parquet"
    _, stats = stream_dedupe_positives(
        labeled_path=labeled,
        output_path=out,
        batch_size=3,
        n_buckets=4,
        temp_dir=tmp_path / "buckets",
    )
    frame = pd.read_parquet(out)
    assert stats["duplicate_rows_removed"] >= 1
    assert stats["positive_rows_after_dedup"] == 4
    assert set(frame["interaction_type"].astype(int)) == {0, 1}
    assert (frame["drug_a"] <= frame["drug_b"]).all()
    assert frame["twosides_pair_key"].iloc[0] == "CID000000001||CID000000002"


def test_negative_pool_only_real_unmatched_pairs(tmp_path: Path) -> None:
    mapping = tmp_path / "mapping.parquet"
    pairs = tmp_path / "pairs.parquet"
    unique = tmp_path / "unique.parquet"
    _write_mapping(mapping)
    _write_drug_pairs(pairs)
    _write_twosides_unique_pairs(unique)

    pool, stats = collect_negative_candidate_pairs(
        mimic_pairs_path=pairs,
        mapping_path=mapping,
        unique_pairs_path=unique,
        batch_size=2,
    )
    assert stats["eligible_negative_pairs_after_patient_pair_dedup"] == 2
    assert set(pool["mimic_pair_key"]) == {"drug_a||drug_c", "drug_a||drug_x"}
    assert "drug_a||drug_b" not in set(pool["mimic_pair_key"])


def test_deterministic_negative_sampling(tmp_path: Path) -> None:
    pool = pd.DataFrame(
        {
            "patient_id": [1, 2, 3, 4, 5],
            "admission_id": [10, 20, 30, 40, 50],
            "drug_a": ["a"] * 5,
            "drug_b": ["b", "c", "d", "e", "f"],
            "mimic_pair_key": [f"a||{x}" for x in ["b", "c", "d", "e", "f"]],
            "twosides_drug_a": ["CID000000001"] * 5,
            "twosides_drug_b": ["CID000000003"] * 5,
            "twosides_pair_key": ["CID000000001||CID000000003"] * 5,
            "pair_rule": ["same_admission"] * 5,
        }
    )
    first, _ = sample_negative_pairs(pool, n_positives=4, negative_ratio=0.5, seed=7)
    second, _ = sample_negative_pairs(pool, n_positives=4, negative_ratio=0.5, seed=7)
    other_seed, _ = sample_negative_pairs(pool, n_positives=4, negative_ratio=0.5, seed=8)
    assert len(first) == 2
    assert first["mimic_pair_key"].tolist() == second["mimic_pair_key"].tolist()
    assert first["label"].eq("negative").all()
    assert first["interaction_type"].isna().all()
    assert first["label_source"].eq("mimic_unmatched_twosides_pair").all()
    assert first["mimic_pair_key"].tolist() != other_seed["mimic_pair_key"].tolist() or len(pool) <= 2


def test_patient_level_split_has_no_leakage() -> None:
    splits, stats = build_patient_splits_table(
        patient_ids=[str(i) for i in range(20)],
        seed=123,
    )
    validate_patient_splits(splits)
    assert stats["n_patients"] == 20
    assert stats["n_train_patients"] + stats["n_val_patients"] + stats["n_test_patients"] == 20


def test_assemble_ml_dataset_assigns_splits_by_patient(tmp_path: Path) -> None:
    positives = tmp_path / "pos.parquet"
    negatives = tmp_path / "neg.parquet"
    splits_path = tmp_path / "splits.parquet"
    out = tmp_path / "dataset.parquet"

    pd.DataFrame(
        {
            "patient_id": [1, 1, 2],
            "drug_a": ["a", "a", "c"],
            "drug_b": ["b", "b", "d"],
            "mimic_pair_key": ["a||b", "a||b", "c||d"],
            "twosides_drug_a": ["CID1", "CID1", "CID3"],
            "twosides_drug_b": ["CID2", "CID2", "CID4"],
            "twosides_pair_key": ["CID1||CID2", "CID1||CID2", "CID3||CID4"],
            "interaction_type": [0, 1, 0],
            "label": ["positive"] * 3,
            "label_source": ["twosides"] * 3,
            "pair_rule": ["same_admission"] * 3,
        }
    ).to_parquet(positives, index=False)
    pd.DataFrame(
        {
            "patient_id": [3],
            "admission_id": [30],
            "drug_a": ["e"],
            "drug_b": ["f"],
            "mimic_pair_key": ["e||f"],
            "twosides_drug_a": ["CID5"],
            "twosides_drug_b": ["CID6"],
            "twosides_pair_key": ["CID5||CID6"],
            "interaction_type": pd.Series([pd.NA], dtype="Int64"),
            "label": ["negative"],
            "label_source": ["mimic_unmatched_twosides_pair"],
            "pair_rule": ["same_admission"],
        }
    ).to_parquet(negatives, index=False)
    pd.DataFrame(
        {
            "patient_id": ["1", "2", "3"],
            "split": ["train", "val", "test"],
        }
    ).to_parquet(splits_path, index=False)

    _, stats = assemble_ml_dataset(
        positives_path=positives,
        negatives_path=negatives,
        splits_path=splits_path,
        output_path=out,
        batch_size=2,
    )
    dataset = pd.read_parquet(out)
    assert stats["positive_rows"] == 3
    assert stats["negative_rows"] == 1
    assert set(dataset.loc[dataset["patient_id"] == 1, "split"]) == {"train"}
    assert set(dataset.loc[dataset["patient_id"] == 2, "split"]) == {"val"}
    assert set(dataset.loc[dataset["patient_id"] == 3, "split"]) == {"test"}


def test_end_to_end_prepare_is_reproducible(tmp_path: Path) -> None:
    labeled = tmp_path / "labeled.parquet"
    pairs = tmp_path / "pairs.parquet"
    mapping = tmp_path / "mapping.parquet"
    unique = tmp_path / "unique.parquet"
    _write_labeled(labeled)
    _write_drug_pairs(pairs)
    _write_mapping(mapping)
    _write_twosides_unique_pairs(unique)

    common = dict(
        labeled_path=labeled,
        mimic_pairs_path=pairs,
        mapping_path=mapping,
        unique_pairs_path=unique,
        negatives_path=tmp_path / "neg.parquet",
        positives_path=tmp_path / "pos.parquet",
        splits_path=tmp_path / "splits.parquet",
        dataset_path=tmp_path / "dataset.parquet",
        stats_path=tmp_path / "stats1.json",
        negative_ratio=1.0,
        seed=99,
        batch_size=3,
        n_buckets=4,
    )
    stats1 = prepare_mimic_twosides_ml_dataset(**common)
    common["stats_path"] = tmp_path / "stats2.json"
    common["negatives_path"] = tmp_path / "neg2.parquet"
    common["positives_path"] = tmp_path / "pos2.parquet"
    common["splits_path"] = tmp_path / "splits2.parquet"
    common["dataset_path"] = tmp_path / "dataset2.parquet"
    stats2 = prepare_mimic_twosides_ml_dataset(**common)

    ds1 = pd.read_parquet(tmp_path / "dataset.parquet")
    ds2 = pd.read_parquet(tmp_path / "dataset2.parquet")
    assert stats1["negative_count"] == stats2["negative_count"]
    assert stats1["positive_count"] == stats2["positive_count"]
    pd.testing.assert_frame_equal(
        ds1.sort_values(["patient_id", "drug_a", "drug_b", "interaction_type"], na_position="last").reset_index(drop=True),
        ds2.sort_values(["patient_id", "drug_a", "drug_b", "interaction_type"], na_position="last").reset_index(drop=True),
    )


def test_validate_real_data_inputs_rejects_small_labeled(tmp_path: Path) -> None:
    labeled = tmp_path / "labeled.parquet"
    mapping = tmp_path / "mapping.parquet"
    unique = tmp_path / "unique.parquet"
    pd.DataFrame({"patient_id": [1], "label": ["positive"]}).to_parquet(labeled, index=False)
    pd.DataFrame(
        {
            "mapping_status": ["matched"] * 489,
            "mapping_method": ["local_pubchem_cache"] * 489,
            "twosides_drug_id": [f"CID{i:09d}" for i in range(489)],
        }
    ).to_parquet(mapping, index=False)
    pairs = pd.DataFrame({"pair_key": [f"CID{i:09d}||CID{(i+1):09d}" for i in range(70000)]})
    pairs.to_parquet(unique, index=False)

    with pytest.raises(ConfigurationError, match="Labeled pairs has"):
        validate_real_data_inputs(
            labeled_path=labeled,
            mapping_path=mapping,
            unique_pairs_path=unique,
            require_real_scale=True,
        )


def test_validate_real_data_inputs_rejects_clobbered_twosides(tmp_path: Path, monkeypatch) -> None:
    labeled = tmp_path / "labeled.parquet"
    mapping = tmp_path / "mapping.parquet"
    unique = tmp_path / "unique.parquet"
    pd.DataFrame({"patient_id": [1], "label": ["positive"]}).to_parquet(labeled, index=False)
    pd.DataFrame(
        {
            "mapping_status": ["matched"] * 489,
            "mapping_method": ["local_pubchem_cache"] * 489,
            "twosides_drug_id": [f"CID{i:09d}" for i in range(489)],
        }
    ).to_parquet(mapping, index=False)
    _write_twosides_unique_pairs(unique)

    class _Meta:
        num_rows = 900_000_000

    class _FakePf:
        metadata = _Meta()
        schema = type("S", (), {"names": ["patient_id", "label"]})()

    real_pf = pq.ParquetFile

    def _pf(path):
        resolved = Path(path)
        if resolved == labeled:
            return _FakePf()
        return real_pf(path)

    monkeypatch.setattr("src.data.mimic_twosides_ml_dataset.pq.ParquetFile", _pf)

    with pytest.raises(ConfigurationError, match="TWOSIDES unique pairs"):
        validate_real_data_inputs(
            labeled_path=labeled,
            mapping_path=mapping,
            unique_pairs_path=unique,
            require_real_scale=True,
        )


def test_assembled_dataset_has_no_patient_leakage(tmp_path: Path) -> None:
    splits = tmp_path / "splits.parquet"
    build_patient_splits_table(
        patient_ids=[str(i) for i in range(30)],
        seed=42,
        output_path=splits,
    )
    split_table = pd.read_parquet(splits)
    check_patient_leakage(split_table)


def test_streaming_uses_parquet_batches(tmp_path: Path) -> None:
    labeled = tmp_path / "labeled.parquet"
    rows = []
    for i in range(50):
        rows.append(
            {
                "patient_id": i % 5,
                "admission_id": i,
                "mimic_drug_a": "drug_a",
                "mimic_drug_b": "drug_b",
                "mimic_pair_key": "drug_a||drug_b",
                "twosides_drug_a": "CID000000001",
                "twosides_drug_b": "CID000000002",
                "twosides_pair_key": "CID000000001||CID000000002",
                "interaction_type": i % 3,
                "neg_sample_drug": "CID000000003",
                "label": "positive",
                "label_source": "twosides",
                "pair_rule": "same_admission",
            }
        )
    pd.DataFrame(rows).to_parquet(labeled, index=False)

    _, stats = stream_dedupe_positives(
        labeled_path=labeled,
        output_path=tmp_path / "pos.parquet",
        batch_size=10,
        n_buckets=8,
        temp_dir=tmp_path / "buckets",
    )
    assert stats["labeled_rows_streamed"] == 50
    assert stats["positive_rows_after_dedup"] <= 50
