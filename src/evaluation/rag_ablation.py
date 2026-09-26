"""RAG ablation evaluation — accepts result files from Member C.

Member C owns RAG implementation. This module never fabricates RAG outputs.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.data.config import METRICS_DIR
from src.data.io_utils import ensure_parent, write_json
from src.evaluation.ablation import compare_experiments

CLASSIFICATION_KEYS = ("accuracy", "f1", "auroc", "pr_auc", "mcc")
XAI_KEYS = ("fidelity_mean", "sparsity_mean", "stability_mean")


def _load_metrics_file(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Metrics file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("metrics", payload.get("aggregate", payload))


def compare_rag_ablation(
    without_rag_path: Path,
    with_rag_path: Path,
    without_rag_xai_path: Path | None = None,
    with_rag_xai_path: Path | None = None,
    synthetic_demo: bool = False,
) -> dict:
    """Compare WITHOUT RAG vs WITH RAG experiment outputs from Member C."""
    try:
        without_metrics = _load_metrics_file(without_rag_path)
        with_metrics = _load_metrics_file(with_rag_path)
    except FileNotFoundError as exc:
        return {
            "status": "rag_results_not_available",
            "message": (
                f"{exc}. Supply Member C result files with --without-rag and --with-rag."
            ),
        }

    comparison = compare_experiments(without_metrics, with_metrics)
    xai_comparison = None
    if without_rag_xai_path and with_rag_xai_path:
        if without_rag_xai_path.exists() and with_rag_xai_path.exists():
            without_xai = _load_metrics_file(without_rag_xai_path)
            with_xai = _load_metrics_file(with_rag_xai_path)
            xai_comparison = compare_experiments(without_xai, with_xai)
            comparison["xai"] = xai_comparison

    result = {
        "status": "ok",
        "synthetic_demo": synthetic_demo,
        "warning": "SYNTHETIC SOFTWARE VALIDATION ONLY" if synthetic_demo else None,
        "without_rag": without_metrics,
        "with_rag": with_metrics,
        "comparison": comparison,
    }
    return result


def save_rag_ablation_report(result: dict) -> dict[str, str]:
    if result.get("status") != "ok":
        return {"status": result.get("status"), "message": result.get("message")}

    rows = []
    for label, key in [("WITHOUT RAG", "without_rag"), ("WITH RAG", "with_rag")]:
        metrics = result[key]
        rows.append(
            {
                "Condition": label,
                "Accuracy": metrics.get("accuracy"),
                "F1": metrics.get("f1"),
                "AUROC": metrics.get("auroc"),
                "PR-AUC": metrics.get("pr_auc"),
                "MCC": metrics.get("mcc"),
            }
        )
    table = pd.DataFrame(rows)
    csv_path = METRICS_DIR / "rag_ablation.csv"
    json_path = METRICS_DIR / "rag_ablation.json"
    ensure_parent(csv_path)
    table.to_csv(csv_path, index=False)
    write_json(json_path, result)
    return {"csv": str(csv_path), "json": str(json_path)}
