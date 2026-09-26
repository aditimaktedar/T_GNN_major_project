"""Tests for baseline models using synthetic demo data only."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.baselines.logistic_regression import LogisticRegressionBaseline
from src.baselines.run_baselines import build_demo_ml_dataset, run_baselines
from src.baselines.static_gat import StaticGATTrainer
from src.data.pyg_dataset import split_frame


def test_logistic_regression_train_predict_save_load(tmp_path: Path) -> None:
    demo_frame = build_demo_ml_dataset()
    splits = split_frame(demo_frame)
    model = LogisticRegressionBaseline(seed=42)
    model.fit(splits["train"])
    probs = model.predict_proba(splits["test"])
    preds = model.predict(splits["test"])
    assert len(probs) == len(splits["test"])
    assert set(np.unique(preds)).issubset({0, 1})
    ckpt = tmp_path / "lr.joblib"
    model.save(ckpt)
    loaded = LogisticRegressionBaseline.load(ckpt)
    assert np.allclose(loaded.predict_proba(splits["test"]), probs)


def test_static_gat_train_and_checkpoint(tmp_path: Path) -> None:
    demo_frame = build_demo_ml_dataset()
    splits = split_frame(demo_frame)
    trainer = StaticGATTrainer(input_dim=16, hidden_dim=16, heads=2, epochs=3, seed=42)
    ckpt = tmp_path / "gat.pt"
    history = trainer.fit(splits["train"], splits["val"], batch_size=4, checkpoint_path=ckpt)
    assert ckpt.exists()
    assert "best_val_f1" in history
    probs = trainer.predict_proba(splits["test"], batch_size=4)
    assert len(probs) == len(splits["test"])
    loaded = StaticGATTrainer.load(ckpt)
    assert len(loaded.predict(splits["test"], batch_size=4)) == len(splits["test"])


def test_run_baselines_demo(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("src.baselines.run_baselines.BASELINES_DIR", tmp_path / "baselines")
    monkeypatch.setattr("src.baselines.run_baselines.METRICS_DIR", tmp_path / "metrics")
    frame = build_demo_ml_dataset()
    config = {
        "seed": 42,
        "pair_feature_method": "concat",
        "logistic_regression": {"C": 1.0, "class_weight": "balanced", "max_iter": 1000},
        "static_gat": {
            "hidden_dim": 16,
            "heads": 2,
            "dropout": 0.1,
            "learning_rate": 0.01,
            "epochs": 2,
            "batch_size": 4,
            "class_weight": "balanced",
        },
        "demo": {"fingerprint_bits": 16},
    }
    results = run_baselines(frame, config, ["logistic_regression", "static_gat"], "synthetic_demo")
    assert results["synthetic_demo"] is True
    assert "logistic_regression" in results
    assert "static_gat" in results
