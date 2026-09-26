"""Ablation comparison framework for baseline vs RAG/evidence experiments."""

from __future__ import annotations

from pathlib import Path

from src.data.io_utils import load_table_chunked, write_json
from src.evaluation.classification import compute_classification_metrics


METRIC_KEYS = ("accuracy", "f1", "auroc", "pr_auc", "mcc")


def compare_experiments(
    baseline_metrics: dict,
    rag_metrics: dict,
    xai_metrics: dict | None = None,
) -> dict:
    comparison = {"baseline": {}, "rag": {}, "delta_rag_minus_baseline": {}}
    for key in METRIC_KEYS:
        base = baseline_metrics.get(key)
        rag = rag_metrics.get(key)
        comparison["baseline"][key] = base
        comparison["rag"][key] = rag
        if base is not None and rag is not None:
            comparison["delta_rag_minus_baseline"][key] = float(rag) - float(base)
        else:
            comparison["delta_rag_minus_baseline"][key] = None
    if xai_metrics:
        comparison["xai"] = xai_metrics
    return comparison


def compare_from_result_files(
    baseline_path: Path,
    rag_path: Path,
    output_path: Path | None = None,
) -> dict:
    """Compare metrics JSON files produced by two experiments."""
    import json

    baseline_payload = json.loads(Path(baseline_path).read_text(encoding="utf-8"))
    rag_payload = json.loads(Path(rag_path).read_text(encoding="utf-8"))
    baseline_metrics = baseline_payload.get("metrics", baseline_payload)
    rag_metrics = rag_payload.get("metrics", rag_payload)
    result = compare_experiments(baseline_metrics, rag_metrics)
    if output_path:
        write_json(output_path, result)
    return result


def compare_predictions(
    y_true,
    baseline_pred,
    baseline_prob,
    rag_pred,
    rag_prob,
) -> dict:
    baseline_metrics = compute_classification_metrics(y_true, baseline_pred, baseline_prob)
    rag_metrics = compute_classification_metrics(y_true, rag_pred, rag_prob)
    return compare_experiments(baseline_metrics, rag_metrics)
