"""Tests for multiclass baseline models."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.baselines.multiclass_logistic_regression import MulticlassLogisticRegressionBaseline
from src.baselines.multiclass_static_gat import MulticlassStaticGATTrainer
from src.baselines.run_multiclass_baselines import run_sanity_checks
from src.data.final_ml_dataset import prepare_multiclass_frame


def _build_tiny_frame() -> pd.DataFrame:
    rows = []
    specs = [
        ("PAIR0", "CCO", "CCC", 0, "train"),
        ("PAIR0", "CCO", "CCC", 0, "train"),
        ("PAIR1", "CCCC", "CC", 1, "train"),
        ("PAIR1", "CCCC", "CC", 1, "train"),
        ("PAIR2", "CCO", "CCCC", 2, "train"),
        ("PAIR3", "CCC", "CC", 0, "val"),
        ("PAIR3", "CCC", "CC", 1, "val"),
    ]
    for i, (pair_key, smiles_a, smiles_b, type_id, split) in enumerate(specs):
        rows.append(
            {
                "subject_id": 100 + i,
                "hadm_id": i,
                "drug_a": pair_key.split("|")[0] if "|" in pair_key else f"A{i}",
                "drug_b": f"B{i}",
                "smiles_a": smiles_a,
                "smiles_b": smiles_b,
                "pair_key": pair_key,
                "type": type_id,
                "split": split,
            }
        )
    return pd.DataFrame(rows)


def test_multiclass_logistic_regression_roundtrip(tmp_path: Path) -> None:
    frame, encoder, _ = prepare_multiclass_frame(_build_tiny_frame())
    train = frame.loc[frame["split"] == "train"]
    val = frame.loc[frame["split"] == "val"]
    model = MulticlassLogisticRegressionBaseline(seed=42, max_iter=200)
    model.fit(train, encoder)
    preds = model.predict(val)
    assert len(preds) == len(val)
    ckpt = tmp_path / "mc_lr.joblib"
    model.save(ckpt)
    loaded = MulticlassLogisticRegressionBaseline.load(ckpt)
    assert len(loaded.predict(val)) == len(val)


def test_multiclass_static_gat_train(tmp_path: Path) -> None:
    frame, encoder, _ = prepare_multiclass_frame(_build_tiny_frame())
    train = frame.loc[frame["split"] == "train"]
    val = frame.loc[frame["split"] == "val"]
    trainer = MulticlassStaticGATTrainer(
        input_dim=2048,
        n_classes=encoder.n_classes,
        hidden_dim=16,
        heads=2,
        epochs=2,
        seed=42,
    )
    ckpt = tmp_path / "mc_gat.pt"
    history = trainer.fit(train, val, encoder, batch_size=2, checkpoint_path=ckpt)
    assert ckpt.exists()
    assert "best_val_macro_f1" in history
    assert len(trainer.predict(val, batch_size=2)) == len(val)


def test_run_sanity_checks_on_tiny_frame() -> None:
    config = {
        "seed": 42,
        "pair_feature_method": "concat",
        "logistic_regression": {"C": 1.0, "class_weight": "balanced"},
        "static_gat": {
            "hidden_dim": 16,
            "heads": 2,
            "dropout": 0.1,
            "learning_rate": 0.01,
            "class_weight": "balanced",
        },
        "sanity_check": {
            "max_train_rows": 6,
            "max_val_rows": 2,
            "logistic_regression_max_iter": 50,
            "static_gat_epochs": 1,
            "batch_size": 2,
        },
    }
    results = run_sanity_checks(_build_tiny_frame(), config, ["logistic_regression", "static_gat"])
    assert results["n_classes"] == 3
    assert "logistic_regression_val_metrics" in results
    assert "static_gat_val_metrics" in results
