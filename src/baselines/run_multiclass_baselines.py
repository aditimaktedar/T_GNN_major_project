"""Run multiclass baselines on final_mimic_twosides_ml.csv."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.baselines.multiclass_logistic_regression import MulticlassLogisticRegressionBaseline
from src.baselines.multiclass_static_gat import MulticlassStaticGATTrainer
from src.data.config import PROJECT_ROOT
from src.data.dataset_paths import (
    default_experiment_prefix,
    evaluate_protocol_for_formulation,
    load_multiclass_dataset,
    normalize_formulation,
    prepare_multiclass_for_formulation,
    resolve_formulation_path,
)
from src.data.final_ml_dataset import (
    assert_no_pair_leakage,
    dataset_sanity_report,
    filter_trainable_rows,
    fingerprint_dim_from_frame,
    split_frame,
)
from src.data.frequent363_ml_dataset import frequent363_report
from src.evaluation.master_results import record_training_result
from src.evaluation.multiclass import compute_multiclass_metrics
from src.evaluation.reproducibility import collect_environment_metadata, get_device, set_global_seed

CONFIG_PATH = PROJECT_ROOT / "configs" / "multiclass_baselines.json"
FREQUENT363_CONFIG_PATH = PROJECT_ROOT / "configs" / "frequent363_baselines.json"
BASELINES_DIR = PROJECT_ROOT / "results" / "baselines"
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"


def load_config(path: Path | None = None) -> dict:
    config_path = path if path is not None else CONFIG_PATH
    return json.loads(config_path.read_text(encoding="utf-8"))


def _evaluate_multiclass(model, frame: pd.DataFrame, n_classes: int) -> dict:
    eval_frame = filter_trainable_rows(frame)
    y_true = eval_frame["label_index"].astype(int).to_numpy()
    y_pred = model.predict(eval_frame)
    y_prob = model.predict_proba(eval_frame)
    return compute_multiclass_metrics(
        y_true,
        y_pred,
        y_prob,
        labels=list(range(n_classes)),
    )


def _predictions_payload(
    model,
    split_frame: pd.DataFrame,
    batch_size: int,
) -> pd.DataFrame:
    eval_frame = filter_trainable_rows(split_frame).reset_index(drop=True)
    y_pred = model.predict(eval_frame)
    y_prob = model.predict_proba(eval_frame)
    payload = eval_frame.copy()
    payload["y_pred"] = y_pred
    payload["y_prob"] = list(y_prob)
    return payload


def _evaluate_with_protocol(
    frame: pd.DataFrame,
    model,
    encoder,
    batch_size: int,
    formulation: str = "955",
    min_train_examples: int = 20,
) -> dict:
    predictions_by_split = {}
    for split_name in ("val", "test"):
        split_data = frame.loc[frame["split"] == split_name].reset_index(drop=True)
        predictions_by_split[split_name] = _predictions_payload(model, split_data, batch_size)
    return evaluate_protocol_for_formulation(
        formulation,
        frame,
        predictions_by_split,
        encoder,
        min_train_examples=min_train_examples,
    )


def run_sanity_checks(
    frame: pd.DataFrame,
    config: dict,
    models: list[str],
    formulation: str = "955",
) -> dict:
    """Fast validation before any long training run."""
    report = dataset_sanity_report(frame)
    assert_no_pair_leakage(frame)
    enriched, encoder, _cache = prepare_multiclass_for_formulation(frame, formulation)
    train = enriched.loc[enriched["split"] == "train"].head(config["sanity_check"]["max_train_rows"])
    val = enriched.loc[enriched["split"] == "val"].head(config["sanity_check"]["max_val_rows"])
    results: dict = {
        "formulation": normalize_formulation(formulation),
        "dataset_report": report,
        "sanity_train_rows": int(len(train)),
        "sanity_val_rows": int(len(val)),
        "n_classes": encoder.n_classes,
        "device": str(get_device()),
    }
    if normalize_formulation(formulation) == "frequent363":
        results["frequent363_report"] = frequent363_report(filtered=frame)

    if "logistic_regression" in models:
        lr = MulticlassLogisticRegressionBaseline(
            C=config["logistic_regression"]["C"],
            class_weight=config["logistic_regression"].get("class_weight"),
            seed=config["seed"],
            pair_feature_method=config.get("pair_feature_method", "concat"),
            max_iter=config["sanity_check"]["logistic_regression_max_iter"],
        )
        lr.fit(train, encoder)
        results["logistic_regression_val_metrics"] = _evaluate_multiclass(lr, val, encoder.n_classes)

    if "static_gat" in models:
        gat_cfg = config["static_gat"]
        trainer = MulticlassStaticGATTrainer(
            input_dim=fingerprint_dim_from_frame(enriched),
            n_classes=encoder.n_classes,
            hidden_dim=gat_cfg["hidden_dim"],
            heads=gat_cfg["heads"],
            dropout=gat_cfg["dropout"],
            learning_rate=gat_cfg["learning_rate"],
            epochs=config["sanity_check"]["static_gat_epochs"],
            seed=config["seed"],
            class_weight=gat_cfg.get("class_weight"),
        )
        history = trainer.fit(
            train,
            val,
            encoder,
            batch_size=config["sanity_check"]["batch_size"],
        )
        results["static_gat_val_metrics"] = _evaluate_multiclass(trainer, val, encoder.n_classes)
        results["static_gat_history"] = history
        results["static_gat_device"] = str(trainer.device)

    return results


def run_logistic_regression(
    enriched: pd.DataFrame,
    encoder,
    config: dict,
    experiment_prefix: str,
    formulation: str = "955",
) -> dict:
    lr_cfg = config["logistic_regression"]
    splits = split_frame(enriched)
    model = MulticlassLogisticRegressionBaseline(
        C=lr_cfg["C"],
        class_weight=lr_cfg.get("class_weight"),
        seed=config["seed"],
        pair_feature_method=config.get("pair_feature_method", "concat"),
        max_iter=lr_cfg.get("max_iter", 1000),
    )
    model.fit(splits["train"], encoder)
    ckpt = BASELINES_DIR / f"{experiment_prefix}_multiclass_logistic_regression.joblib"
    model.save(ckpt)
    metrics = {}
    for split_name, split_data in splits.items():
        if len(split_data):
            metrics[split_name] = _evaluate_multiclass(model, split_data, encoder.n_classes)
    batch_size = config.get("static_gat", {}).get("batch_size", 32)
    protocol = _evaluate_with_protocol(
        enriched,
        model,
        encoder,
        batch_size=batch_size,
        formulation=formulation,
    )
    return {
        "checkpoint": str(ckpt),
        "metrics": metrics,
        "evaluation_protocol": protocol,
    }


def run_static_gat(
    enriched: pd.DataFrame,
    encoder,
    config: dict,
    experiment_prefix: str,
    formulation: str = "955",
) -> dict:
    gat_cfg = config["static_gat"]
    splits = split_frame(enriched)
    batch_size = gat_cfg.get("batch_size", 32)
    trainer = MulticlassStaticGATTrainer(
        input_dim=fingerprint_dim_from_frame(enriched),
        n_classes=encoder.n_classes,
        hidden_dim=gat_cfg["hidden_dim"],
        heads=gat_cfg["heads"],
        dropout=gat_cfg["dropout"],
        learning_rate=gat_cfg["learning_rate"],
        epochs=gat_cfg["epochs"],
        seed=config["seed"],
        class_weight=gat_cfg.get("class_weight"),
    )
    ckpt = BASELINES_DIR / f"{experiment_prefix}_multiclass_static_gat.pt"
    history = trainer.fit(
        splits["train"],
        splits["val"],
        encoder,
        batch_size=batch_size,
        checkpoint_path=ckpt,
    )
    metrics = {}
    for split_name, split_data in splits.items():
        if len(split_data):
            metrics[split_name] = _evaluate_multiclass(trainer, split_data, encoder.n_classes)
    protocol = _evaluate_with_protocol(
        enriched,
        trainer,
        encoder,
        batch_size=batch_size,
        formulation=formulation,
    )
    return {
        "checkpoint": str(ckpt),
        "device": str(trainer.device),
        "metrics": metrics,
        "evaluation_protocol": protocol,
        "history": history,
        "final_train_loss": history["train_loss"][-1] if history.get("train_loss") else None,
        "final_val_loss": history["val_loss"][-1] if history.get("val_loss") else None,
    }


def run_multiclass_baselines(
    frame: pd.DataFrame,
    config: dict,
    models: list[str],
    experiment_prefix: str,
    formulation: str = "955",
) -> dict:
    set_global_seed(config["seed"])
    assert_no_pair_leakage(frame)
    enriched, encoder, _cache = prepare_multiclass_for_formulation(frame, formulation)
    results = {
        "formulation": normalize_formulation(formulation),
        "experiment_prefix": experiment_prefix,
        "dataset_path": str(resolve_formulation_path(formulation)),
        "n_classes": encoder.n_classes,
        "dataset_report": dataset_sanity_report(frame),
    }
    if normalize_formulation(formulation) == "frequent363":
        results["frequent363_report"] = frequent363_report(filtered=frame)
    if "logistic_regression" in models:
        results["logistic_regression"] = run_logistic_regression(
            enriched, encoder, config, experiment_prefix, formulation=formulation
        )
        record_training_result(formulation=formulation, model="logistic_regression", payload=results)
    if "static_gat" in models:
        results["static_gat"] = run_static_gat(
            enriched, encoder, config, experiment_prefix, formulation=formulation
        )
        record_training_result(formulation=formulation, model="static_gat", payload=results)
    results["master_results_path"] = str(METRICS_DIR / "MASTER_RESULTS.json")
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run multiclass baselines on ML dataset CSV")
    parser.add_argument(
        "--formulation",
        default="955",
        choices=["955", "frequent363"],
        help="Target formulation: 955-class (full) or frequency-filtered 363-class.",
    )
    parser.add_argument("--data", default=None, help="Override dataset path (default from --formulation).")
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["logistic_regression", "static_gat"],
        choices=["logistic_regression", "static_gat"],
    )
    parser.add_argument("--experiment-prefix", default=None)
    parser.add_argument(
        "--sanity-check",
        action="store_true",
        help="Run fast validation and tiny training loop before full training.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    formulation = normalize_formulation(args.formulation)
    config_path = Path(args.config) if args.config else (
        FREQUENT363_CONFIG_PATH if formulation == "frequent363" else CONFIG_PATH
    )
    config = load_config(config_path)
    data_path = args.data or str(resolve_formulation_path(formulation))
    experiment_prefix = args.experiment_prefix or default_experiment_prefix(formulation)
    frame = load_multiclass_dataset(formulation, data_path)
    BASELINES_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    if args.sanity_check:
        results = run_sanity_checks(frame, config, args.models, formulation=formulation)
        results["mode"] = "sanity_check"
        results["models"] = args.models
        suffix = "_".join(args.models) if args.models != ["logistic_regression", "static_gat"] else ""
        out_name = f"{experiment_prefix}_sanity_check{('_' + suffix) if suffix else ''}.json"
        out = METRICS_DIR / out_name
        out.write_text(json.dumps(results, indent=2, default=str) + "\n", encoding="utf-8")
        print(json.dumps(results, indent=2, default=str))
        return 0

    results = run_multiclass_baselines(
        frame, config, args.models, experiment_prefix, formulation=formulation
    )
    results["environment"] = collect_environment_metadata(config)
    print(json.dumps(results, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
