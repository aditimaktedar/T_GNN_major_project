"""Train and evaluate the dual molecular GNN baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.baselines.run_multiclass_baselines import _evaluate_with_protocol
from src.data.config import PROJECT_ROOT
from src.data.dataset_paths import (
    default_experiment_prefix,
    load_multiclass_dataset,
    normalize_formulation,
    resolve_formulation_path,
)
from src.data.final_ml_dataset import (
    assert_no_pair_leakage,
    dataset_sanity_report,
    split_frame,
)
from src.data.frequent363_ml_dataset import frequent363_report
from src.data.molecular_pair_dataset import (
    filter_molecular_trainable_rows,
    prepare_molecular_frequent363_frame,
    prepare_molecular_multiclass_frame,
)
from src.evaluation.master_results import record_training_result
from src.evaluation.multiclass import compute_multiclass_metrics
from src.evaluation.reproducibility import collect_environment_metadata, get_device, set_global_seed
from src.models.dual_molecular_gnn import MulticlassMolecularGNNTrainer

CONFIG_PATH = PROJECT_ROOT / "configs" / "molecular_gnn.json"
BASELINES_DIR = PROJECT_ROOT / "results" / "baselines"
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"


def load_config(path: Path | None = None) -> dict:
    config_path = path if path is not None else CONFIG_PATH
    return json.loads(config_path.read_text(encoding="utf-8"))


def _evaluate_split(trainer: MulticlassMolecularGNNTrainer, split_frame: pd.DataFrame, batch_size: int) -> dict:
    eval_frame = filter_molecular_trainable_rows(split_frame)
    y_true = eval_frame["label_index"].astype(int).to_numpy()
    y_pred = trainer.predict(eval_frame, batch_size=batch_size)
    y_prob = trainer.predict_proba(eval_frame, batch_size=batch_size)
    return compute_multiclass_metrics(
        y_true,
        y_pred,
        y_prob,
        labels=list(range(trainer.n_classes)),
    )


def _prepare_molecular_frame(frame: pd.DataFrame, formulation: str):
    if normalize_formulation(formulation) == "frequent363":
        return prepare_molecular_frequent363_frame(frame)
    return prepare_molecular_multiclass_frame(frame)


def run_sanity_check(frame: pd.DataFrame, config: dict, formulation: str = "955") -> dict:
    assert_no_pair_leakage(frame)
    enriched, encoder, _fp_cache, graph_cache = _prepare_molecular_frame(frame, formulation)
    sanity_cfg = config["sanity_check"]
    gnn_cfg = config["molecular_gnn"]
    train = enriched.loc[enriched["split"] == "train"].head(sanity_cfg["max_train_rows"])
    val = enriched.loc[enriched["split"] == "val"].head(sanity_cfg["max_val_rows"])

    trainer = MulticlassMolecularGNNTrainer(
        n_classes=encoder.n_classes,
        hidden_dim=gnn_cfg["hidden_dim"],
        num_layers=gnn_cfg["num_layers"],
        dropout=gnn_cfg["dropout"],
        fusion_hidden=gnn_cfg["fusion_hidden"],
        learning_rate=gnn_cfg["learning_rate"],
        epochs=sanity_cfg["epochs"],
        seed=config["seed"],
        class_weight=gnn_cfg.get("class_weight"),
        graph_cache=graph_cache,
    )
    history = trainer.fit(
        train,
        val,
        encoder,
        batch_size=sanity_cfg["batch_size"],
    )
    val_metrics = _evaluate_split(trainer, val, sanity_cfg["batch_size"])
    trainable = filter_molecular_trainable_rows(enriched)
    result = {
        "formulation": normalize_formulation(formulation),
        "mode": "sanity_check",
        "device": str(trainer.device),
        "n_classes": encoder.n_classes,
        "sanity_train_rows": int(len(filter_molecular_trainable_rows(train))),
        "sanity_val_rows": int(len(filter_molecular_trainable_rows(val))),
        "total_trainable_rows": int(len(trainable)),
        "unique_smiles_cached": graph_cache.n_cached,
        "valid_graphs_cached": graph_cache.n_valid,
        "history": history,
        "val_metrics": val_metrics,
        "checks_passed": bool(history["train_loss"]) and val_metrics.get("accuracy") is not None,
    }
    if normalize_formulation(formulation) == "frequent363":
        result["frequent363_report"] = frequent363_report(filtered=frame)
    return result


def run_molecular_gnn(
    frame: pd.DataFrame,
    config: dict,
    experiment_prefix: str,
    formulation: str = "955",
) -> dict:
    set_global_seed(config["seed"])
    assert_no_pair_leakage(frame)
    enriched, encoder, _fp_cache, graph_cache = _prepare_molecular_frame(frame, formulation)
    gnn_cfg = config["molecular_gnn"]
    batch_size = gnn_cfg.get("batch_size", 32)

    trainer = MulticlassMolecularGNNTrainer(
        n_classes=encoder.n_classes,
        hidden_dim=gnn_cfg["hidden_dim"],
        num_layers=gnn_cfg["num_layers"],
        dropout=gnn_cfg["dropout"],
        fusion_hidden=gnn_cfg["fusion_hidden"],
        learning_rate=gnn_cfg["learning_rate"],
        epochs=gnn_cfg["epochs"],
        seed=config["seed"],
        class_weight=gnn_cfg.get("class_weight"),
        graph_cache=graph_cache,
    )
    splits = split_frame(enriched)
    ckpt = BASELINES_DIR / f"{experiment_prefix}_multiclass_molecular_gnn.pt"
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
            metrics[split_name] = _evaluate_split(trainer, split_data, batch_size)

    protocol = _evaluate_with_protocol(
        enriched, trainer, encoder, batch_size=batch_size, formulation=formulation
    )

    result = {
        "formulation": normalize_formulation(formulation),
        "experiment_prefix": experiment_prefix,
        "checkpoint": str(ckpt),
        "device": str(trainer.device),
        "dataset_path": str(resolve_formulation_path(formulation)),
        "n_classes": encoder.n_classes,
        "dataset_report": dataset_sanity_report(frame),
        "graph_cache": {
            "unique_smiles_cached": graph_cache.n_cached,
            "valid_graphs_cached": graph_cache.n_valid,
        },
        "metrics": metrics,
        "evaluation_protocol": protocol,
        "history": history,
        "final_train_loss": history["train_loss"][-1] if history.get("train_loss") else None,
        "final_val_loss": history["val_loss"][-1] if history.get("val_loss") else None,
        "environment": collect_environment_metadata(config),
    }
    if normalize_formulation(formulation) == "frequent363":
        result["frequent363_report"] = frequent363_report(filtered=frame)
    record_training_result(formulation=formulation, model="molecular_gnn", payload=result)
    result["master_results_path"] = str(METRICS_DIR / "MASTER_RESULTS.json")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train dual molecular GNN on ML dataset CSV")
    parser.add_argument(
        "--formulation",
        default="955",
        choices=["955", "frequent363"],
        help="Target formulation: 955-class (full) or frequency-filtered 363-class.",
    )
    parser.add_argument("--data", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--experiment-prefix", default=None)
    parser.add_argument("--sanity-check", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    formulation = normalize_formulation(args.formulation)
    config_path = Path(args.config) if args.config else CONFIG_PATH
    config = load_config(config_path)
    data_path = args.data or str(resolve_formulation_path(formulation))
    experiment_prefix = args.experiment_prefix or default_experiment_prefix(formulation)
    frame = load_multiclass_dataset(formulation, data_path)
    BASELINES_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    if args.sanity_check:
        results = run_sanity_check(frame, config, formulation=formulation)
        out = METRICS_DIR / f"{experiment_prefix}_molecular_gnn_sanity_check.json"
        out.write_text(json.dumps(results, indent=2, default=str) + "\n", encoding="utf-8")
        print(json.dumps(results, indent=2, default=str))
        return 0 if results.get("checks_passed") else 1

    results = run_molecular_gnn(frame, config, experiment_prefix, formulation=formulation)
    print(json.dumps(results, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
