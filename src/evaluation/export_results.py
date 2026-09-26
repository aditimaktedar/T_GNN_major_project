"""Export consolidated paper-ready result artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.data.config import METRICS_DIR, RESULTS_DIR
from src.data.io_utils import ensure_parent


TABLES_DIR = RESULTS_DIR / "tables"


def export_all_tables() -> dict[str, str]:
    """Copy/summarize existing metric files into results/tables/."""
    ensure_parent(TABLES_DIR / ".gitkeep")
    exported = {}

    model_csv = METRICS_DIR / "model_comparison.csv"
    if model_csv.exists():
        dest = TABLES_DIR / "baseline_comparison.csv"
        dest.write_text(model_csv.read_text(encoding="utf-8"), encoding="utf-8")
        exported["baseline_comparison"] = str(dest)

    xai_json = METRICS_DIR / "xai_metrics.json"
    if xai_json.exists():
        payload = json.loads(xai_json.read_text(encoding="utf-8"))
        rows = [{"Metric": k, "Value": v} for k, v in payload.get("aggregate", {}).items()]
        if payload.get("warning"):
            rows.insert(0, {"Metric": "warning", "Value": payload["warning"]})
        xai_table = pd.DataFrame(rows)
        xai_path = TABLES_DIR / "xai_comparison.csv"
        xai_table.to_csv(xai_path, index=False)
        exported["xai_comparison"] = str(xai_path)

    for name in (
        "model_comparison.json",
        "xai_metrics.json",
        "drugbank_sanity.json",
        "rag_ablation.json",
        "data_audit.json",
        "reproducibility.json",
    ):
        src = METRICS_DIR / name
        if src.exists():
            exported[name.replace(".json", "")] = str(src)

    return exported
