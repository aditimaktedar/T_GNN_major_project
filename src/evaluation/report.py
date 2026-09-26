"""Write readable metric summaries and JSON reports."""

from __future__ import annotations

from pathlib import Path

from src.data.config import METRICS_DIR
from src.data.io_utils import ensure_parent, write_json


def format_metrics_summary(experiment_name: str, metrics: dict, metadata: dict | None = None) -> str:
    lines = [
        f"Experiment: {experiment_name}",
        "========================",
    ]
    for key in ("accuracy", "precision", "recall", "f1", "auroc", "pr_auc", "mcc"):
        if key in metrics:
            lines.append(f"{key}: {metrics[key]}")
    if metadata:
        lines.append("")
        lines.append(f"device: {metadata.get('device')}")
        lines.append(f"seed: {metadata.get('config', {}).get('seed')}")
    return "\n".join(lines) + "\n"


def save_metrics_report(
    experiment_name: str,
    metrics: dict,
    metadata: dict | None = None,
    output_dir: Path | None = None,
) -> tuple[Path, Path]:
    out_dir = output_dir if output_dir is not None else METRICS_DIR
    json_path = out_dir / f"{experiment_name}_metrics.json"
    summary_path = out_dir / f"{experiment_name}_summary.txt"
    payload = {"experiment": experiment_name, "metrics": metrics}
    if metadata:
        payload["metadata"] = metadata
    write_json(json_path, payload)
    ensure_parent(summary_path)
    summary_path.write_text(format_metrics_summary(experiment_name, metrics, metadata), encoding="utf-8")
    return json_path, summary_path
