"""Tests for the manual training workflow, CLI entry-points, reporting, and JSON schema.

Verifies:
- All training CLI entry points import successfully.
- Argument parsers configure expected default paths, flags, and hyperparameters.
- Result JSON schemas contain all required metadata and metrics.
- Terminal performance report formatting matches specification.
- Threshold selection uses validation data only (no test data leakage).
- Existing datasets, splits, and 363-dimensional target formulations are preserved.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import torch

from src.data.config import (
    CHECKPOINTS_DIR,
    METRICS_DIR,
    MULTILABEL_FREQUENT363_DATASET_CSV,
    TEMPORAL_MULTILABEL_FREQUENT363_DATASET_CSV,
)
from src.evaluation.multilabel_metrics import compute_multilabel_metrics
from src.evaluation.thresholds import optimize_threshold_on_val
from src.evaluation.training_report import (
    build_result_payload,
    format_terminal_report,
    save_json_results,
)
from src.models.train_static_baseline import build_parser as build_static_baseline_parser
from src.models.train_static_gat import build_parser as build_static_gat_parser
from src.models.train_molecular_gnn import build_parser as build_molecular_gnn_parser
from src.models.train_temporal_baseline import build_parser as build_temporal_baseline_parser
from src.models.train_temporal_gnn import build_parser as build_temporal_gnn_parser


class TestCLIEntryPoints:
    """Verify CLI entry point parsers and argument defaults."""

    def test_static_baseline_parser(self):
        parser = build_static_baseline_parser()
        args = parser.parse_args([])
        assert args.dataset == str(MULTILABEL_FREQUENT363_DATASET_CSV)
        assert args.epochs == 200
        assert args.n_bits == 2048
        assert not args.no_age
        assert "static_logistic_regression" in args.results_json
        assert "static_logistic_regression" in args.checkpoint

    def test_static_gat_parser(self):
        parser = build_static_gat_parser()
        args = parser.parse_args([])
        assert args.dataset == str(MULTILABEL_FREQUENT363_DATASET_CSV)
        assert args.hidden_dim == 64
        assert args.heads == 4
        assert not args.no_age
        assert "static_gat" in args.results_json
        assert "static_gat" in args.checkpoint

    def test_molecular_gnn_parser(self):
        parser = build_molecular_gnn_parser()
        args = parser.parse_args([])
        assert args.dataset == str(MULTILABEL_FREQUENT363_DATASET_CSV)
        assert args.hidden_dim == 64
        assert args.num_layers == 3
        assert not args.no_age
        assert "static_molecular_gnn" in args.results_json
        assert "static_molecular_gnn" in args.checkpoint

    def test_temporal_baseline_parser(self):
        parser = build_temporal_baseline_parser()
        args = parser.parse_args([])
        assert args.dataset == str(TEMPORAL_MULTILABEL_FREQUENT363_DATASET_CSV)
        assert args.epochs == 200
        assert "temporal_baseline" in args.results_json
        assert "temporal_baseline" in args.checkpoint

    def test_temporal_gnn_parser(self):
        parser = build_temporal_gnn_parser()
        args = parser.parse_args([])
        assert args.dataset == str(TEMPORAL_MULTILABEL_FREQUENT363_DATASET_CSV)
        assert args.hidden_dim == 64


class TestReportingAndSchema:
    """Verify JSON schema and terminal report formatting."""

    def test_build_result_payload_schema(self, tmp_path):
        sample_metrics = {
            "micro_f1": 0.55,
            "macro_f1": 0.12,
            "hamming_loss": 0.04,
            "jaccard_score": 0.40,
            "mAP": 0.60,
            "precision_at_1": 0.80,
            "recall_at_1": 0.01,
            "precision_at_3": 0.70,
            "recall_at_3": 0.03,
            "precision_at_5": 0.65,
            "recall_at_5": 0.05,
            "precision_at_10": 0.50,
            "recall_at_10": 0.09,
            "threshold": 0.35,
            "n_samples": 23,
            "n_classes": 363,
        }
        summary = {
            "best_epoch": 42,
            "best_val_loss": 0.1234,
            "best_val_micro_f1": 0.6543,
            "final_train_loss": 0.0987,
        }

        payload = build_result_payload(
            pipeline="Static",
            model="Molecular GNN",
            dataset="Frequent363",
            number_of_labels=363,
            train_size=101,
            val_size=21,
            test_size=23,
            unique_train_pairs=101,
            unique_val_pairs=21,
            unique_test_pairs=23,
            training_config={"epochs": 200, "lr": 0.001},
            training_summary=summary,
            selected_threshold=0.35,
            threshold_selection_method="validation_micro_f1",
            test_metrics=sample_metrics,
            checkpoint_path="results/checkpoints/static_molecular_gnn.pt",
        )

        required_keys = [
            "pipeline",
            "model",
            "dataset",
            "number_of_labels",
            "train_size",
            "val_size",
            "test_size",
            "unique_train_pairs",
            "unique_val_pairs",
            "unique_test_pairs",
            "training_config",
            "best_epoch",
            "train_loss",
            "val_loss",
            "selected_threshold",
            "threshold_selection_method",
            "test_metrics",
            "checkpoint_path",
            "timestamp",
        ]
        for key in required_keys:
            assert key in payload, f"Missing required JSON key: {key}"

        json_file = tmp_path / "test_result.json"
        save_json_results(payload, json_file)
        assert json_file.exists()
        loaded = json.loads(json_file.read_text(encoding="utf-8"))
        assert loaded["number_of_labels"] == 363
        assert loaded["test_metrics"]["micro_f1"] == 0.55

    def test_format_terminal_report(self):
        data = {
            "pipeline": "Static",
            "model": "Molecular GNN",
            "dataset": "Frequent363",
            "test_size": 23,
            "unique_test_pairs": 23,
            "number_of_labels": 363,
            "training_summary": {
                "total_epochs": 200,
                "best_epoch": 42,
                "best_val_loss": 0.123456,
                "best_val_micro_f1": 0.654321,
            },
            "selected_threshold": 0.35,
            "threshold_selection_set": "validation",
            "test_metrics": {
                "micro_f1": 0.543210,
                "macro_f1": 0.123456,
                "hamming_loss": 0.054321,
                "jaccard_score": 0.432100,
                "mAP": 0.612345,
                "precision_at_1": 0.826087,
                "recall_at_1": 0.012345,
                "precision_at_5": 0.652174,
                "recall_at_5": 0.054321,
                "precision_at_10": 0.521739,
                "recall_at_10": 0.098765,
            },
            "checkpoint_path": "results/checkpoints/static_molecular_gnn.pt",
            "results_path": "results/metrics/static_molecular_gnn.json",
        }

        report = format_terminal_report(data)
        assert "MODEL TRAINING COMPLETE" in report
        assert "Pipeline: Static" in report
        assert "Model: Molecular GNN" in report
        assert "Number of labels: 363" in report
        assert "Test observations: 23" in report
        assert "Selected threshold: 0.35" in report
        assert "Selection set: validation" in report
        assert "Micro-F1       : 0.543210" in report
        assert "Precision@1    : 0.826087" in report
        assert "Path: results/checkpoints/static_molecular_gnn.pt" in report


class TestValidationThresholdIntegrity:
    """Ensure threshold is selected using validation data only, without test set leakage."""

    def test_threshold_selection_strictly_on_val(self):
        rng = np.random.default_rng(42)
        n_val = 21
        n_test = 23
        n_classes = 363

        y_val = rng.integers(0, 2, size=(n_val, n_classes))
        p_val = rng.random(size=(n_val, n_classes))

        y_test = rng.integers(0, 2, size=(n_test, n_classes))
        p_test = rng.random(size=(n_test, n_classes))

        # Optimize on validation only
        val_opt = optimize_threshold_on_val(y_val, p_val, metric="micro_f1")
        sel_thresh = val_opt["best_threshold"]

        # Ensure validation was used
        assert "Threshold optimized on VALIDATION split only" in val_opt["note"]
        assert 0.05 <= sel_thresh <= 0.95

        # Compute test metrics strictly using the chosen threshold
        test_m = compute_multilabel_metrics(y_test, p_test, threshold=sel_thresh)
        assert test_m["threshold"] == sel_thresh
        assert test_m["n_samples"] == n_test
        assert test_m["n_classes"] == n_classes


class TestDatasetIntegrity:
    """Verify static and temporal dataset split counts and multi-label formulation."""

    def test_static_dataset_splits(self):
        if not MULTILABEL_FREQUENT363_DATASET_CSV.exists():
            pytest.skip("Dataset file not present")

        df = pd.read_csv(MULTILABEL_FREQUENT363_DATASET_CSV)
        assert len(df) == 145
        assert df["split"].value_counts()["train"] == 101
        assert df["split"].value_counts()["val"] == 21
        assert df["split"].value_counts()["test"] == 23
        assert df[df["split"] == "test"]["pair_key"].nunique() == 23

        # Target dimension check
        first_target = json.loads(df["target"].iloc[0])
        assert len(first_target) == 363

    def test_temporal_dataset_splits(self):
        if not TEMPORAL_MULTILABEL_FREQUENT363_DATASET_CSV.exists():
            pytest.skip("Dataset file not present")

        df = pd.read_csv(TEMPORAL_MULTILABEL_FREQUENT363_DATASET_CSV)
        assert len(df) == 84
        assert df["split"].value_counts()["train"] == 58
        assert df["split"].value_counts()["val"] == 16
        assert df["split"].value_counts()["test"] == 10
        assert df[df["split"] == "test"]["pair_key"].nunique() == 10

        # Target dimension check
        first_target = json.loads(df["target"].iloc[0])
        assert len(first_target) == 363
