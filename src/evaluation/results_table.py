"""Build model comparison tables from baseline and TGNN result files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.config import METRICS_DIR, PROJECT_ROOT, RESULTS_DIR
from src.data.io_utils import ensure_parent, write_json

TABLES_DIR = RESULTS_DIR / "tables"
METRIC_COLUMNS = ["Model", "Accuracy", "Precision", "Recall", "F1", "AUROC", "PR-AUC", "MCC"]


def _metrics_row(model_name: str, metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "Model": model_name,
        "Accuracy": metrics.get("accuracy"),
        "Precision": metrics.get("precision"),
        "Recall": metrics.get("recall"),
        "F1": metrics.get("f1"),
        "AUROC": metrics.get("auroc"),
        "PR-AUC": metrics.get("pr_auc"),
        "MCC": metrics.get("mcc"),
    }


def load_baseline_metrics(experiment_prefix: str, model_key: str) -> dict[str, Any]:
    path = METRICS_DIR / f"{experiment_prefix}_{model_key}_metrics.json"
    if not path.exists():
        raise FileNotFoundError(f"Baseline metrics not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("metrics", payload)


def load_tgnn_metrics(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("metrics", payload)


def build_comparison_table(
    rows: list[dict[str, Any]],
    synthetic_demo: bool = False,
) -> pd.DataFrame:
    table = pd.DataFrame(rows, columns=METRIC_COLUMNS)
    if synthetic_demo:
        table.attrs["warning"] = "SYNTHETIC SOFTWARE VALIDATION ONLY"
    return table


def save_comparison_table(
    table: pd.DataFrame,
    synthetic_demo: bool = False,
    output_csv: Path | None = None,
    output_json: Path | None = None,
    output_md: Path | None = None,
) -> dict[str, Path]:
    csv_path = output_csv or METRICS_DIR / "model_comparison.csv"
    json_path = output_json or METRICS_DIR / "model_comparison.json"
    md_path = output_md or TABLES_DIR / "baseline_comparison.csv"
    ensure_parent(csv_path)
    ensure_parent(json_path)
    ensure_parent(md_path)

    warning = "SYNTHETIC SOFTWARE VALIDATION ONLY" if synthetic_demo else None
    md_file = md_path.with_suffix(".md")
    md_lines = []
    if warning:
        md_lines.append(f"# {warning}\n")
    md_lines.append("| " + " | ".join(METRIC_COLUMNS) + " |")
    md_lines.append("| " + " | ".join(["---"] * len(METRIC_COLUMNS)) + " |")
    for _, row in table.iterrows():
        md_lines.append("| " + " | ".join(str(row[col]) for col in METRIC_COLUMNS) + " |")
    md_file.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    table.to_csv(csv_path, index=False)
    payload = {
        "warning": warning,
        "rows": table.to_dict(orient="records"),
    }
    write_json(json_path, payload)
    table.to_csv(md_path, index=False)
    return {"csv": csv_path, "json": json_path, "md": md_file, "tables_csv": md_path}


def build_from_baseline_run_summary(
    summary_path: Path,
    tgnn_metrics_path: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Path]]:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    synthetic_demo = bool(payload.get("synthetic_demo", False))
    prefix = payload.get("experiment_prefix", "synthetic_demo")
    rows = []
    name_map = {
        "logistic_regression": "Logistic Regression",
        "static_gat": "Static GAT",
    }
    for key, display in name_map.items():
        if key in payload:
            test_metrics = payload[key]["metrics"].get("test") or payload[key]["metrics"].get("val") or {}
            rows.append(_metrics_row(display, test_metrics))
    if tgnn_metrics_path and tgnn_metrics_path.exists():
        rows.append(_metrics_row("TemporalDDI-GNN", load_tgnn_metrics(tgnn_metrics_path)))
    table = build_comparison_table(rows, synthetic_demo=synthetic_demo)
    paths = save_comparison_table(table, synthetic_demo=synthetic_demo)
    return table, paths
