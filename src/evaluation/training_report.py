"""Standardized terminal reporting and JSON serialization for DDI model training.

Provides consistent formatting across Pipeline 1 (Static) and Pipeline 2 (Temporal)
model training workflows.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def format_terminal_report(data: dict[str, Any]) -> str:
    """Format a clean, standardized training report for terminal display.

    Parameters
    ----------
    data : dict[str, Any]
        Dictionary containing pipeline, model, dataset, training, threshold,
        test_metrics, checkpoint_path, and results_path.

    Returns
    -------
    str
        Formatted terminal report block.
    """
    pipeline = data.get("pipeline", "Static")
    model = data.get("model", "Unknown Model")
    dataset = data.get("dataset", "Frequent363")
    test_obs = data.get("test_size", data.get("test_observations", "N/A"))
    test_pairs = data.get("unique_test_pairs", test_obs)
    n_labels = data.get("number_of_labels", data.get("n_classes", 363))

    training = data.get("training_summary", {})
    epochs = training.get("total_epochs", data.get("epochs", "N/A"))
    best_epoch = training.get("best_epoch", "N/A")
    best_val_loss = training.get("best_val_loss", data.get("val_loss", "N/A"))
    val_f1_sel = training.get("val_micro_f1_selected_threshold", training.get("best_val_micro_f1", "N/A"))
    val_f1_def = training.get("val_micro_f1_default_threshold")

    threshold_val = data.get("selected_threshold", 0.50)
    threshold_set = data.get("threshold_selection_set", "validation")

    tm = data.get("test_metrics", {})

    def _fmt_f(val: Any) -> str:
        if isinstance(val, (int, float)):
            return f"{float(val):.6f}"
        return str(val)

    best_val_loss_str = _fmt_f(best_val_loss) if isinstance(best_val_loss, (int, float)) else str(best_val_loss)
    best_val_f1_sel_str = _fmt_f(val_f1_sel) if isinstance(val_f1_sel, (int, float)) else str(val_f1_sel)

    training_lines = [
        "Training:",
        f"  Epochs: {epochs}",
        f"  Best epoch: {best_epoch}",
        f"  Best validation loss: {best_val_loss_str}",
        f"  Validation Micro-F1 (selected threshold {threshold_val}): {best_val_f1_sel_str}",
    ]
    if val_f1_def is not None:
        training_lines.append(f"  Validation Micro-F1 (default 0.50)  : {_fmt_f(val_f1_def)}")

    lines = [
        "=" * 60,
        "MODEL TRAINING COMPLETE",
        "=" * 60,
        "",
        f"Pipeline: {pipeline}",
        f"Model: {model}",
        f"Dataset: {dataset}",
        f"Test observations: {test_obs}",
        f"Test unique pairs: {test_pairs}",
        f"Number of labels: {n_labels}",
        "",
        *training_lines,
        "",
        "Threshold:",
        f"  Selected threshold: {threshold_val}",
        f"  Selection set: {threshold_set}",
        "",
        "TEST PERFORMANCE",
        "-" * 60,
        f"Micro-F1       : {_fmt_f(tm.get('micro_f1', 0.0))}",
        f"Macro-F1       : {_fmt_f(tm.get('macro_f1', 0.0))}",
        f"Hamming Loss   : {_fmt_f(tm.get('hamming_loss', 0.0))}",
        f"Jaccard        : {_fmt_f(tm.get('jaccard_score', 0.0))}",
        f"mAP            : {_fmt_f(tm.get('mAP', 0.0))}",
        "",
        f"Precision@1    : {_fmt_f(tm.get('precision_at_1', 0.0))}",
        f"Recall@1       : {_fmt_f(tm.get('recall_at_1', 0.0))}",
        "",
        f"Precision@5    : {_fmt_f(tm.get('precision_at_5', 0.0))}",
        f"Recall@5       : {_fmt_f(tm.get('recall_at_5', 0.0))}",
        "",
        f"Precision@10   : {_fmt_f(tm.get('precision_at_10', 0.0))}",
        f"Recall@10      : {_fmt_f(tm.get('recall_at_10', 0.0))}",
        "",
        "-" * 60,
        "CHECKPOINT",
        f"Path: {data.get('checkpoint_path', 'N/A')}",
        "",
        "RESULTS",
        f"Path: {data.get('results_path', 'N/A')}",
        "",
        "=" * 60,
    ]
    return "\n".join(lines)


def build_result_payload(
    pipeline: str,
    model: str,
    dataset: str,
    number_of_labels: int,
    train_size: int,
    val_size: int,
    test_size: int,
    unique_train_pairs: int,
    unique_val_pairs: int,
    unique_test_pairs: int,
    training_config: dict[str, Any],
    training_summary: dict[str, Any],
    selected_threshold: float,
    threshold_selection_method: str,
    test_metrics: dict[str, Any],
    checkpoint_path: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Construct a clean, standardized JSON results dictionary."""
    timestamp = datetime.now(timezone.utc).isoformat()
    payload = {
        "pipeline": pipeline,
        "model": model,
        "dataset": dataset,
        "number_of_labels": number_of_labels,
        "train_size": train_size,
        "val_size": val_size,
        "test_size": test_size,
        "unique_train_pairs": unique_train_pairs,
        "unique_val_pairs": unique_val_pairs,
        "unique_test_pairs": unique_test_pairs,
        "training_config": training_config,
        "best_epoch": training_summary.get("best_epoch"),
        "train_loss": training_summary.get("final_train_loss"),
        "val_loss": training_summary.get("best_val_loss"),
        "selected_threshold": selected_threshold,
        "threshold_selection_method": threshold_selection_method,
        "test_metrics": test_metrics,
        "checkpoint_path": checkpoint_path,
        "timestamp": timestamp,
        **(extra or {}),
    }
    return payload


def save_json_results(payload: dict[str, Any], filepath: Path | str) -> Path:
    """Save dictionary as formatted JSON."""
    out_path = Path(filepath)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return out_path
