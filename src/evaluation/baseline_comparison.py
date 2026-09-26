"""Compare multiclass baseline models using the evaluation protocol (inference-only)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.baselines.multiclass_logistic_regression import MulticlassLogisticRegressionBaseline
from src.baselines.multiclass_static_gat import MulticlassStaticGATTrainer
from src.baselines.run_multiclass_baselines import _evaluate_with_protocol
from src.data.config import FINAL_MIMIC_TWOSIDES_ML_PATH, METRICS_DIR, PROJECT_ROOT
from src.data.final_ml_dataset import load_final_ml_dataset, prepare_multiclass_frame
from src.data.molecular_pair_dataset import prepare_molecular_multiclass_frame
from src.evaluation.master_results import load_protocol
from src.models.dual_molecular_gnn import MulticlassMolecularGNNTrainer

BASELINES_DIR = PROJECT_ROOT / "results" / "baselines"

MODEL_SPECS: dict[str, dict[str, Any]] = {
    "logistic_regression": {
        "display_name": "Multiclass Logistic Regression",
        "checkpoint": BASELINES_DIR / "final_mimic_twosides_multiclass_logistic_regression.joblib",
        "protocol_path": METRICS_DIR / "final_mimic_twosides_multiclass_logistic_regression_protocol.json",
        "summary_path": METRICS_DIR / "final_mimic_twosides_multiclass_logistic_regression_summary.json",
        "loader": "logistic_regression",
    },
    "static_gat": {
        "display_name": "Multiclass Static GAT",
        "checkpoint": BASELINES_DIR / "final_mimic_twosides_multiclass_static_gat.pt",
        "protocol_path": METRICS_DIR / "final_mimic_twosides_multiclass_static_gat_protocol.json",
        "summary_path": METRICS_DIR / "final_mimic_twosides_multiclass_static_gat_summary.json",
        "loader": "static_gat",
    },
    "molecular_gnn": {
        "display_name": "Atom-level Molecular GNN (GIN)",
        "checkpoint": BASELINES_DIR / "final_mimic_twosides_multiclass_molecular_gnn.pt",
        "protocol_path": METRICS_DIR / "final_mimic_twosides_multiclass_molecular_gnn_protocol.json",
        "summary_path": METRICS_DIR / "final_mimic_twosides_multiclass_molecular_gnn_summary.json",
        "loader": "molecular_gnn",
    },
}

METRIC_KEYS = (
    "top_1_accuracy",
    "top_3_accuracy",
    "top_5_accuracy",
    "macro_f1",
    "micro_f1",
    "balanced_accuracy",
)

COMPARISON_SLICES = (
    ("seen_class", "pair_type_dedup_level"),
    ("frequent_ge_20", "pair_type_dedup_level"),
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_slice_metrics(protocol: dict[str, Any], split: str, slice_name: str, level: str) -> dict[str, Any]:
    block = protocol["splits"][split][slice_name][level]
    sample_key = "n_samples" if level == "row_level" else "n_pair_types"
    return {
        "n_samples": block.get(sample_key),
        "n_classes": block.get("n_classes"),
        **block["metrics"],
    }


def _load_or_evaluate_protocol(model_key: str, frame, enriched, encoder, batch_size: int = 32) -> dict[str, Any]:
    spec = MODEL_SPECS[model_key]
    master_protocol = load_protocol("955", model_key)
    if master_protocol:
        return master_protocol

    protocol_path = spec["protocol_path"]
    if protocol_path.exists():
        return _load_json(protocol_path)

    checkpoint = spec["checkpoint"]
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing checkpoint for {model_key}: {checkpoint}")

    loader = spec["loader"]
    if loader == "logistic_regression":
        model = MulticlassLogisticRegressionBaseline.load(checkpoint)
        protocol = _evaluate_with_protocol(enriched, model, encoder, batch_size=batch_size)
    elif loader == "static_gat":
        trainer = MulticlassStaticGATTrainer.load(checkpoint)
        protocol = _evaluate_with_protocol(enriched, trainer, encoder, batch_size=batch_size)
    elif loader == "molecular_gnn":
        enriched_mol, encoder_mol, _fp, cache = prepare_molecular_multiclass_frame(frame)
        trainer = MulticlassMolecularGNNTrainer.load(checkpoint, graph_cache=cache)
        protocol = _evaluate_with_protocol(enriched_mol, trainer, encoder_mol, batch_size=batch_size)
    else:
        raise ValueError(f"Unknown loader: {loader}")

    protocol_path.parent.mkdir(parents=True, exist_ok=True)
    protocol_path.write_text(json.dumps(protocol, indent=2, default=str) + "\n", encoding="utf-8")
    return protocol


def _relative_delta(value: float | None, baseline: float | None) -> float | None:
    if value is None or baseline is None or baseline == 0:
        return None
    return (value - baseline) / baseline


def _best_model(values: dict[str, float | None]) -> list[str]:
    ranked = sorted(
        ((name, val) for name, val in values.items() if val is not None),
        key=lambda item: item[1],
        reverse=True,
    )
    if not ranked:
        return []
    best_val = ranked[0][1]
    return [name for name, val in ranked if val == best_val]


def build_baseline_comparison(
    data_path: Path | str | None = None,
    batch_size: int = 32,
) -> dict[str, Any]:
    frame = load_final_ml_dataset(data_path)
    enriched, encoder, _cache = prepare_multiclass_frame(frame)

    protocols: dict[str, dict[str, Any]] = {}
    provenance: dict[str, Any] = {}
    for model_key, spec in MODEL_SPECS.items():
        protocol_cached = spec["protocol_path"].exists()
        protocol = _load_or_evaluate_protocol(model_key, frame, enriched, encoder, batch_size=batch_size)
        protocols[model_key] = protocol
        summary = _load_json(spec["summary_path"]) if spec["summary_path"].exists() else {}
        provenance[model_key] = {
            "display_name": spec["display_name"],
            "checkpoint": str(spec["checkpoint"]),
            "protocol_path": str(spec["protocol_path"]),
            "protocol_from_cache": protocol_cached,
            "device": summary.get("device"),
        }

    comparison: dict[str, Any] = {
        "dataset_path": str(FINAL_MIMIC_TWOSIDES_ML_PATH),
        "encoder_n_classes": encoder.n_classes,
        "evaluation_protocol": "docs/EVALUATION_PROTOCOL.md",
        "models": provenance,
        "splits": {},
        "best_by_metric": {},
        "relative_to_logistic_regression": {},
    }

    for split in ("val", "test"):
        comparison["splits"][split] = {}
        for slice_name, level in COMPARISON_SLICES:
            slice_key = f"{slice_name}/{level.replace('_level', '')}"
            rows: dict[str, dict[str, Any]] = {}
            for model_key in MODEL_SPECS:
                rows[model_key] = _extract_slice_metrics(protocols[model_key], split, slice_name, level)
            comparison["splits"][split][slice_key] = rows

            for metric in METRIC_KEYS:
                metric_values = {model_key: rows[model_key].get(metric) for model_key in MODEL_SPECS}
                best_key = f"{split}/{slice_key}/{metric}"
                comparison["best_by_metric"][best_key] = {
                    "best_models": _best_model(metric_values),
                    "values": metric_values,
                }
                lr_val = metric_values.get("logistic_regression")
                comparison["relative_to_logistic_regression"][best_key] = {
                    model_key: _relative_delta(metric_values.get(model_key), lr_val)
                    for model_key in ("static_gat", "molecular_gnn")
                }

    comparison["experiment_parity"] = {
        "same_dataset": True,
        "same_split_column": True,
        "same_label_encoder_fit": "train split only, 955 classes",
        "same_evaluation_protocol": True,
        "same_primary_slices": [f"{s}/{l.replace('_level', '')}" for s, l in COMPARISON_SLICES],
    }
    return comparison


def format_pct(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{100 * value:.3f}%"


def format_relative(value: float | None) -> str:
    if value is None:
        return "—"
    sign = "+" if value >= 0 else ""
    return f"{sign}{100 * value:.1f}%"
