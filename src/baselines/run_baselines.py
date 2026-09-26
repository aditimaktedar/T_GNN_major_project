"""Run Logistic Regression and Static GAT baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.baselines.logistic_regression import LogisticRegressionBaseline
from src.baselines.static_gat import StaticGATTrainer
from src.data.config import PROJECT_ROOT
from src.data.pyg_dataset import (
    assert_no_patient_leakage,
    filter_trainable_rows,
    fingerprint_dim_from_frame,
    load_processed_dataset,
    split_frame,
)
from src.evaluation.classification import compute_classification_metrics
from src.evaluation.report import save_metrics_report
from src.evaluation.reproducibility import collect_environment_metadata, save_experiment_metadata, set_global_seed

CONFIG_PATH = PROJECT_ROOT / "configs" / "baselines.json"
BASELINES_DIR = PROJECT_ROOT / "results" / "baselines"
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"


def load_config(path: Path | None = None) -> dict:
    config_path = path if path is not None else CONFIG_PATH
    return json.loads(config_path.read_text(encoding="utf-8"))


def build_demo_ml_dataset(seed: int = 42, n_bits: int = 16) -> pd.DataFrame:
    """Synthetic ML dataset for software validation only — NOT scientific results."""
    rng = np.random.default_rng(seed)
    rows = []
    patients = [
        ("demo_p1", "train"),
        ("demo_p2", "train"),
        ("demo_p3", "train"),
        ("demo_p4", "val"),
        ("demo_p5", "test"),
        ("demo_p6", "test"),
    ]
    drugs = {
        "SYN-A": rng.integers(0, 2, size=n_bits).astype(float).tolist(),
        "SYN-B": rng.integers(0, 2, size=n_bits).astype(float).tolist(),
        "SYN-C": rng.integers(0, 2, size=n_bits).astype(float).tolist(),
        "SYN-D": rng.integers(0, 2, size=n_bits).astype(float).tolist(),
    }
    pairs = [
        ("demo_p1", "SYN-A", "SYN-B", "positive", "train"),
        ("demo_p2", "SYN-A", "SYN-C", "negative", "train"),
        ("demo_p3", "SYN-B", "SYN-D", "positive", "train"),
        ("demo_p4", "SYN-C", "SYN-D", "negative", "val"),
        ("demo_p5", "SYN-A", "SYN-D", "positive", "test"),
        ("demo_p6", "SYN-B", "SYN-C", "negative", "test"),
        ("demo_p1", "SYN-A", "SYN-D", "negative", "train"),
        ("demo_p2", "SYN-B", "SYN-C", "positive", "train"),
    ]
    for patient_id, drug_a, drug_b, label, split in pairs:
        rows.append(
            {
                "patient_id": patient_id,
                "drug_a": drug_a,
                "drug_b": drug_b,
                "drug_a_fingerprint": drugs[drug_a],
                "drug_b_fingerprint": drugs[drug_b],
                "label": label,
                "split": split,
            }
        )
    frame = pd.DataFrame(rows)
    frame.attrs["synthetic_demo"] = True
    return frame


def _evaluate_split(model, frame: pd.DataFrame, model_type: str) -> dict:
    eval_frame = filter_trainable_rows(frame)
    y_true = eval_frame["label"].map({"positive": 1, "negative": 0}).astype(int).to_numpy()
    if model_type == "logistic_regression":
        y_pred = model.predict(eval_frame)
        y_prob = model.predict_proba(eval_frame)
    else:
        y_pred = model.predict(eval_frame)
        y_prob = model.predict_proba(eval_frame)
    return compute_classification_metrics(y_true, y_pred, y_prob)


def run_logistic_regression(frame: pd.DataFrame, config: dict, experiment_prefix: str) -> dict:
    lr_cfg = config["logistic_regression"]
    splits = split_frame(frame)
    model = LogisticRegressionBaseline(
        C=lr_cfg["C"],
        class_weight=lr_cfg.get("class_weight"),
        seed=config["seed"],
        pair_feature_method=config.get("pair_feature_method", "concat"),
        max_iter=lr_cfg.get("max_iter", 1000),
    )
    model.fit(splits["train"])
    ckpt = BASELINES_DIR / f"{experiment_prefix}_logistic_regression.joblib"
    model.save(ckpt)
    metrics = {}
    for split_name, split_frame_data in splits.items():
        if len(split_frame_data) == 0:
            continue
        metrics[split_name] = _evaluate_split(model, split_frame_data, "logistic_regression")
    metadata = collect_environment_metadata({**config, "model": "logistic_regression", "checkpoint": str(ckpt)})
    save_experiment_metadata(
        METRICS_DIR / f"{experiment_prefix}_logistic_regression_metadata.json",
        {**config, "model": "logistic_regression"},
    )
    test_metrics = metrics.get("test") or metrics.get("val") or {}
    save_metrics_report(f"{experiment_prefix}_logistic_regression", test_metrics, metadata)
    return {"checkpoint": str(ckpt), "metrics": metrics}


def run_static_gat(frame: pd.DataFrame, config: dict, experiment_prefix: str) -> dict:
    gat_cfg = config["static_gat"]
    splits = split_frame(frame)
    input_dim = fingerprint_dim_from_frame(frame)
    trainer = StaticGATTrainer(
        input_dim=input_dim,
        hidden_dim=gat_cfg["hidden_dim"],
        heads=gat_cfg["heads"],
        dropout=gat_cfg["dropout"],
        learning_rate=gat_cfg["learning_rate"],
        epochs=gat_cfg["epochs"],
        seed=config["seed"],
        class_weight=gat_cfg.get("class_weight"),
    )
    ckpt = BASELINES_DIR / f"{experiment_prefix}_static_gat.pt"
    history = trainer.fit(
        splits["train"],
        splits["val"],
        batch_size=gat_cfg.get("batch_size", 32),
        checkpoint_path=ckpt,
    )
    metrics = {}
    for split_name, split_frame_data in splits.items():
        if len(split_frame_data) == 0:
            continue
        metrics[split_name] = _evaluate_split(trainer, split_frame_data, "static_gat")
    metadata = collect_environment_metadata({**config, "model": "static_gat", "checkpoint": str(ckpt), "history": history})
    save_experiment_metadata(
        METRICS_DIR / f"{experiment_prefix}_static_gat_metadata.json",
        {**config, "model": "static_gat"},
    )
    test_metrics = metrics.get("test") or metrics.get("val") or {}
    save_metrics_report(f"{experiment_prefix}_static_gat", test_metrics, metadata)
    return {"checkpoint": str(ckpt), "metrics": metrics, "history": history}


def run_baselines(
    frame: pd.DataFrame,
    config: dict,
    models: list[str],
    experiment_prefix: str,
) -> dict:
    set_global_seed(config["seed"])
    assert_no_patient_leakage(frame)
    results = {"experiment_prefix": experiment_prefix, "synthetic_demo": bool(frame.attrs.get("synthetic_demo", False))}
    if "logistic_regression" in models:
        results["logistic_regression"] = run_logistic_regression(frame, config, experiment_prefix)
    if "static_gat" in models:
        results["static_gat"] = run_static_gat(frame, config, experiment_prefix)
    summary_path = METRICS_DIR / f"{experiment_prefix}_run_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(results, indent=2, default=str) + "\n", encoding="utf-8")
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Member B baselines.")
    parser.add_argument("--demo", action="store_true", help="Run synthetic software-validation demo only.")
    parser.add_argument("--data", type=str, help="Path to processed ml_dataset.parquet from build_dataset.")
    parser.add_argument("--config", type=str, default=str(CONFIG_PATH))
    parser.add_argument(
        "--models",
        nargs="+",
        default=["logistic_regression", "static_gat"],
        choices=["logistic_regression", "static_gat"],
    )
    parser.add_argument("--experiment-prefix", default="demo" )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(Path(args.config))
    if args.demo:
        frame = build_demo_ml_dataset(seed=config["seed"], n_bits=config["demo"]["fingerprint_bits"])
        prefix = "synthetic_demo"
        print("SYNTHETIC SOFTWARE TEST ONLY — NOT SCIENTIFIC RESULTS")
    elif args.data:
        frame = load_processed_dataset(args.data)
        prefix = args.experiment_prefix
        print("REAL DATA MODE — results depend on supplied processed dataset.")
    else:
        print("Provide --demo or --data PATH", file=__import__("sys").stderr)
        return 1
    BASELINES_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    results = run_baselines(frame, config, args.models, prefix)
    print(json.dumps(results, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
