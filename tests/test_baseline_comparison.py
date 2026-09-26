"""Tests for baseline comparison reporting."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.evaluation.baseline_comparison import (
    MODEL_SPECS,
    _best_model,
    _extract_slice_metrics,
    _relative_delta,
    build_baseline_comparison,
)
from src.evaluation.run_baseline_comparison import render_markdown


def test_relative_delta_and_best_model() -> None:
    assert _relative_delta(0.002, 0.001) == pytest.approx(1.0)
    assert _best_model({"a": 0.1, "b": 0.2, "c": 0.2}) == ["b", "c"]


def test_extract_slice_metrics_from_protocol_file() -> None:
    path = MODEL_SPECS["static_gat"]["protocol_path"]
    if not path.exists():
        pytest.skip("Static GAT protocol file not available.")
    protocol = json.loads(path.read_text(encoding="utf-8"))
    metrics = _extract_slice_metrics(protocol, "test", "seen_class", "pair_type_dedup_level")
    assert metrics["top_3_accuracy"] is not None
    assert metrics["n_samples"] == 3227


def test_build_baseline_comparison_produces_all_models() -> None:
    missing = [spec["checkpoint"] for spec in MODEL_SPECS.values() if not spec["checkpoint"].exists()]
    if missing:
        pytest.skip(f"Missing checkpoints: {missing}")
    comparison = build_baseline_comparison()
    assert set(comparison["models"]) == set(MODEL_SPECS)
    assert "test/seen_class/pair_type_dedup/top_3_accuracy" in comparison["best_by_metric"]


def test_render_markdown_contains_models() -> None:
    metric_row = {
        "n_samples": 1,
        "n_classes": 1,
        "top_1_accuracy": 0.01,
        "top_3_accuracy": 0.01,
        "top_5_accuracy": 0.01,
        "macro_f1": 0.001,
        "micro_f1": 0.01,
        "balanced_accuracy": 0.01,
    }
    comparison = {
        "models": {
            "logistic_regression": {
                "display_name": "Multiclass Logistic Regression",
                "checkpoint": "/tmp/lr.joblib",
                "protocol_from_cache": True,
            },
            "static_gat": {
                "display_name": "Multiclass Static GAT",
                "checkpoint": "/tmp/gat.pt",
                "protocol_from_cache": True,
            },
            "molecular_gnn": {
                "display_name": "Atom-level Molecular GNN (GIN)",
                "checkpoint": "/tmp/gnn.pt",
                "protocol_from_cache": True,
            },
        },
        "experiment_parity": {
            "same_dataset": True,
            "same_split_column": True,
            "same_evaluation_protocol": True,
        },
        "splits": {
            "val": {
                "seen_class/pair_type_dedup": {
                    "logistic_regression": metric_row,
                    "static_gat": metric_row,
                    "molecular_gnn": metric_row,
                },
                "frequent_ge_20/pair_type_dedup": {
                    "logistic_regression": metric_row,
                    "static_gat": metric_row,
                    "molecular_gnn": metric_row,
                },
            },
            "test": {
                "seen_class/pair_type_dedup": {
                    "logistic_regression": metric_row,
                    "static_gat": {**metric_row, "top_3_accuracy": 0.02},
                    "molecular_gnn": {**metric_row, "top_3_accuracy": 0.015},
                },
                "frequent_ge_20/pair_type_dedup": {
                    "logistic_regression": metric_row,
                    "static_gat": metric_row,
                    "molecular_gnn": {**metric_row, "top_3_accuracy": 0.02},
                },
            },
        },
        "best_by_metric": {
            f"test/seen_class/pair_type_dedup/{metric}": {
                "best_models": ["static_gat"],
                "values": {"logistic_regression": 0.01, "static_gat": 0.02, "molecular_gnn": 0.015},
            }
            for metric in (
                "top_1_accuracy",
                "top_3_accuracy",
                "top_5_accuracy",
                "macro_f1",
                "micro_f1",
                "balanced_accuracy",
            )
        }
        | {
            f"test/frequent_ge_20/pair_type_dedup/{metric}": {
                "best_models": ["molecular_gnn"],
                "values": {"logistic_regression": 0.01, "static_gat": 0.01, "molecular_gnn": 0.02},
            }
            for metric in (
                "top_1_accuracy",
                "top_3_accuracy",
                "top_5_accuracy",
                "macro_f1",
                "micro_f1",
                "balanced_accuracy",
            )
        },
        "relative_to_logistic_regression": {
            f"{split}/seen_class/pair_type_dedup/{metric}": {
                "static_gat": 1.0,
                "molecular_gnn": 0.5,
            }
            for split in ("val", "test")
            for metric in (
                "top_1_accuracy",
                "top_3_accuracy",
                "top_5_accuracy",
                "macro_f1",
                "micro_f1",
                "balanced_accuracy",
            )
        }
        | {
            f"{split}/frequent_ge_20/pair_type_dedup/{metric}": {
                "static_gat": 0.5,
                "molecular_gnn": 1.0,
            }
            for split in ("val", "test")
            for metric in (
                "top_1_accuracy",
                "top_3_accuracy",
                "top_5_accuracy",
                "macro_f1",
                "micro_f1",
                "balanced_accuracy",
            )
        },
    }
    comparison["best_by_metric"].update(
        {
            f"val/seen_class/pair_type_dedup/{metric}": {
                "best_models": ["logistic_regression"],
                "values": {"logistic_regression": 0.01, "static_gat": 0.01, "molecular_gnn": 0.01},
            }
            for metric in (
                "top_1_accuracy",
                "top_3_accuracy",
                "top_5_accuracy",
                "macro_f1",
                "micro_f1",
                "balanced_accuracy",
            )
        }
        | {
            f"val/frequent_ge_20/pair_type_dedup/{metric}": {
                "best_models": ["molecular_gnn"],
                "values": {"logistic_regression": 0.01, "static_gat": 0.01, "molecular_gnn": 0.02},
            }
            for metric in (
                "top_1_accuracy",
                "top_3_accuracy",
                "top_5_accuracy",
                "macro_f1",
                "micro_f1",
                "balanced_accuracy",
            )
        }
    )
    md = render_markdown(comparison)
    assert "Multiclass Logistic Regression" in md
    assert "Baseline Comparison" in md
