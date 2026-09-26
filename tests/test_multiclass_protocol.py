"""Tests for the multiclass evaluation protocol."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.final_ml_dataset import prepare_multiclass_frame
from src.evaluation.multiclass_protocol import (
    apply_slice_mask,
    deduplicate_pair_type,
    describe_evaluation_slices,
    evaluate_multiclass_protocol,
    frequent_train_types,
    random_baseline_predictions,
    train_type_counts,
)


def _build_protocol_frame() -> pd.DataFrame:
    rows = []
    specs = [
        ("PAIR0", "CCO", "CCC", 0, "train"),
        ("PAIR0", "CCO", "CCC", 0, "train"),
        ("PAIR1", "CCCC", "CC", 1, "train"),
        ("PAIR1", "CCCC", "CC", 1, "train"),
        ("PAIR2", "CCO", "CCCC", 2, "train"),
        ("PAIR2", "CCO", "CCCC", 2, "train"),
        ("PAIR2", "CCO", "CCCC", 2, "train"),
        ("PAIR3", "CCC", "CC", 0, "val"),
        ("PAIR3", "CCC", "CC", 1, "val"),
        ("PAIR4", "CC", "CCO", 99, "val"),
        ("PAIR5", "CCC", "CCCC", 1, "test"),
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


def test_train_type_counts_and_frequent_types() -> None:
    frame = _build_protocol_frame()
    counts = train_type_counts(frame)
    assert counts[0] == 2
    assert counts[2] == 3
    frequent = frequent_train_types(frame, min_examples=2)
    assert frequent == {0, 1, 2}


def test_deduplicate_pair_type() -> None:
    frame = _build_protocol_frame()
    enriched, encoder, _ = prepare_multiclass_frame(frame)
    train_rows = enriched.loc[enriched["split"] == "train"]
    y_true = train_rows["label_index"].astype(int).to_numpy()
    y_pred = y_true.copy()
    y_prob = None
    dedup_frame, dedup_true, dedup_pred, _ = deduplicate_pair_type(train_rows, y_true, y_pred, y_prob)
    assert len(dedup_frame) == 3
    assert len(dedup_true) == 3


def test_describe_evaluation_slices() -> None:
    frame = _build_protocol_frame()
    report = describe_evaluation_slices(frame, min_train_examples=2)
    assert report["encoder_n_classes"] == 3
    assert report["slices"]["val_seen_class"]["row_level"]["n_rows"] == 2
    assert report["slices"]["val_seen_class"]["pair_type_dedup_level"]["n_pair_types"] == 2
    assert report["slices"]["val_frequent_ge_20"]["row_level"]["n_rows"] == 2


def test_evaluate_multiclass_protocol_random_baseline() -> None:
    frame = _build_protocol_frame()
    enriched, encoder, _ = prepare_multiclass_frame(frame)
    predictions_by_split = {}
    for split in ("val", "test"):
        eval_frame = enriched.loc[enriched["split"] == split].reset_index(drop=True)
        evaluable = eval_frame.loc[eval_frame["label_index"] >= 0].reset_index(drop=True)
        y_pred, y_prob = random_baseline_predictions(evaluable, encoder.n_classes, seed=7)
        payload = evaluable.copy()
        payload["y_pred"] = y_pred
        payload["y_prob"] = list(y_prob)
        predictions_by_split[split] = payload

    results = evaluate_multiclass_protocol(
        frame,
        predictions_by_split,
        encoder,
        min_train_examples=2,
    )
    val_primary = results["splits"]["val"]["seen_class"]["pair_type_dedup_level"]["metrics"]
    assert val_primary["top_1_accuracy"] is not None
    assert val_primary["top_3_accuracy"] is not None
    if val_primary["top_5_accuracy"] is not None:
        assert val_primary["top_5_accuracy"] >= val_primary["top_1_accuracy"]
    assert "primary_reporting" in results


def test_unseen_type_excluded_from_seen_slice() -> None:
    frame = _build_protocol_frame()
    enriched, encoder, _ = prepare_multiclass_frame(frame)
    val = enriched.loc[enriched["split"] == "val"].reset_index(drop=True)
    counts = train_type_counts(enriched)
    mask = apply_slice_mask(val, "seen_class", train_counts=counts)
    assert mask.sum() == 2
    unseen_mask = val["label_index"] < 0
    assert unseen_mask.sum() == 1
