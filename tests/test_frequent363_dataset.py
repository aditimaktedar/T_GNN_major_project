"""Tests for the frequency-filtered 363-class formulation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data.config import (
    FINAL_MIMIC_TWOSIDES_ML_FREQUENT363_PATH,
    FINAL_MIMIC_TWOSIDES_ML_PATH,
    FREQUENT363_MIN_TRAIN_ROWS,
)
from src.data.dataset_paths import (
    load_multiclass_dataset,
    normalize_formulation,
    prepare_multiclass_for_formulation,
    resolve_formulation_path,
)
from src.data.exceptions import ConfigurationError, SchemaError
from src.data.final_ml_dataset import assert_no_pair_leakage, load_final_ml_dataset
from src.data.frequent363_ml_dataset import (
    build_frequent363_frame,
    eligible_types_from_train,
    filter_to_eligible_types,
    frequent363_report,
    load_frequent363_ml_dataset,
    prepare_frequent363_multiclass_frame,
)
from src.data.molecular_pair_dataset import prepare_molecular_frequent363_frame
from src.data.prepare_frequent363_ml_dataset import prepare_frequent363_ml_dataset
from src.evaluation.frequent363_protocol import (
    SLICE_NAME,
    describe_frequent363_slices,
    evaluate_frequent363_protocol,
)


def _build_tiny_source_frame() -> pd.DataFrame:
    rows = []
    specs = [
        ("PAIR0", "CCO", "CCC", 0, "train"),
        ("PAIR0", "CCO", "CCC", 0, "train"),
        ("PAIR0", "CCO", "CCC", 0, "train"),
        ("PAIR1", "CCCC", "CC", 1, "train"),
        ("PAIR1", "CCCC", "CC", 1, "train"),
        ("PAIR2", "CCO", "CCCC", 2, "train"),
        ("PAIR3", "CCC", "CC", 99, "train"),
        ("PAIR4", "CC", "CCO", 0, "val"),
        ("PAIR4", "CC", "CCO", 1, "val"),
        ("PAIR4", "CC", "CCO", 99, "val"),
        ("PAIR5", "CCC", "CCCC", 2, "test"),
        ("PAIR5", "CCC", "CCCC", 100, "test"),
    ]
    for i, (pair_key, smiles_a, smiles_b, type_id, split) in enumerate(specs):
        rows.append(
            {
                "subject_id": 100 + i,
                "hadm_id": i,
                "drug_a": f"A{i}",
                "drug_b": f"B{i}",
                "smiles_a": smiles_a,
                "smiles_b": smiles_b,
                "pair_key": pair_key,
                "type": type_id,
                "split": split,
            }
        )
    return pd.DataFrame(rows)


def test_eligible_types_from_train_only() -> None:
    frame = _build_tiny_source_frame()
    eligible = eligible_types_from_train(frame, min_train_rows=2)
    assert eligible == {0, 1}
    assert 2 not in eligible
    assert 99 not in eligible


def test_filter_excludes_rare_types_no_other_class() -> None:
    frame = _build_tiny_source_frame()
    filtered, eligible = build_frequent363_frame(frame, min_train_rows=2)
    assert eligible == {0, 1}
    assert set(filtered["type"].astype(int)) <= eligible
    assert 99 not in set(filtered["type"].astype(int))
    assert 100 not in set(filtered["type"].astype(int))
    assert_no_pair_leakage(filtered)


def test_encoder_fit_on_train_only() -> None:
    frame = _build_tiny_source_frame()
    filtered, _ = build_frequent363_frame(frame, min_train_rows=2)
    enriched, encoder, _ = prepare_frequent363_multiclass_frame(filtered, min_train_rows=2)
    assert encoder.n_classes == 2
    val = enriched.loc[enriched["split"] == "val"]
    assert (val["label_index"] >= 0).all()
    test = enriched.loc[enriched["split"] == "test"]
    assert (test["label_index"] >= 0).all()


def test_non_eligible_frame_rejected() -> None:
    frame = _build_tiny_source_frame()
    with pytest.raises(SchemaError):
        prepare_frequent363_multiclass_frame(frame)


def test_molecular_frequent363_frame() -> None:
    frame = _build_tiny_source_frame()
    filtered, _ = build_frequent363_frame(frame, min_train_rows=2)
    enriched, encoder, _fp, graph_cache = prepare_molecular_frequent363_frame(
        filtered, min_train_rows=2
    )
    assert encoder.n_classes == 2
    assert "graph_valid" in enriched.columns
    assert graph_cache.n_cached >= 1


def test_dataset_paths_formulation() -> None:
    assert normalize_formulation("955") == "955"
    assert normalize_formulation("frequent363") == "frequent363"
    with pytest.raises(ConfigurationError):
        normalize_formulation("unknown")
    assert resolve_formulation_path("frequent363").name == "final_mimic_twosides_ml_frequent363.csv"


def test_frequent363_protocol_on_tiny_frame() -> None:
    frame = _build_tiny_source_frame()
    filtered, _ = build_frequent363_frame(frame, min_train_rows=2)
    enriched, encoder, _ = prepare_frequent363_multiclass_frame(filtered, min_train_rows=2)
    predictions = {}
    for split in ("val", "test"):
        subset = enriched.loc[enriched["split"] == split].reset_index(drop=True)
        payload = subset.copy()
        payload["y_pred"] = payload["label_index"].astype(int)
        predictions[split] = payload
    report = evaluate_frequent363_protocol(filtered, predictions, encoder)
    assert report["slice"] == SLICE_NAME
    assert "pair_type_dedup_level" in report["splits"]["val"]
    assert "top_1_accuracy" in report["splits"]["test"]["pair_type_dedup_level"]["metrics"]


@pytest.mark.skipif(
    not FINAL_MIMIC_TWOSIDES_ML_PATH.exists(),
    reason="Full ML dataset not available locally",
)
def test_real_frequent363_counts_match_analysis() -> None:
    source = load_final_ml_dataset()
    filtered, eligible = build_frequent363_frame(source, min_train_rows=FREQUENT363_MIN_TRAIN_ROWS)
    report = frequent363_report(source=source, filtered=filtered)

    assert len(eligible) == 363
    assert report["filtered_rows"]["train"] == 13458
    assert report["filtered_rows"]["val"] == 2841
    assert report["filtered_rows"]["test"] == 2584
    assert report["pair_type_counts"]["n_pair_types_dedup_test"] == 2074
    assert report["n_excluded_train_types"] == 592
    assert report["pair_split_leakage_pairs"] == 0
    assert report["val_types_not_in_encoder"] == []
    assert report["test_types_not_in_encoder"] == []
    assert report["encoder_n_classes"] == 363


@pytest.mark.skipif(
    not FINAL_MIMIC_TWOSIDES_ML_PATH.exists(),
    reason="Full ML dataset not available locally",
)
def test_prepare_and_load_derived_csv(tmp_path: Path) -> None:
    out = tmp_path / "frequent363.csv"
    rep = tmp_path / "report.json"
    filtered, report = prepare_frequent363_ml_dataset(
        source_path=FINAL_MIMIC_TWOSIDES_ML_PATH,
        output_path=out,
        report_path=rep,
    )
    assert out.exists()
    assert rep.exists()
    loaded = load_frequent363_ml_dataset(out)
    assert len(loaded) == len(filtered)
    assert len(report["eligible_types"]) == 363


@pytest.mark.skipif(
    not FINAL_MIMIC_TWOSIDES_ML_FREQUENT363_PATH.exists(),
    reason="Derived frequent363 CSV not built yet",
)
def test_load_prepared_frequent363_dataset() -> None:
    frame = load_multiclass_dataset("frequent363")
    enriched, encoder, _ = prepare_multiclass_for_formulation(frame, "frequent363")
    assert encoder.n_classes == 363
    assert_no_pair_leakage(frame)
    slices = describe_frequent363_slices(frame)
    assert slices["encoder_n_classes"] == 363
    assert slices["splits"]["test"]["pair_type_dedup_level"]["n_pair_types"] == 2074
