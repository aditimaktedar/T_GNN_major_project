"""CLI Entry-point for Pipeline 2: Temporal Multi-Label Baseline Training & Evaluation.

Command:
    python -m src.models.train_temporal_baseline

Trains a multi-label baseline on admission temporal features and evaluates
using canonical metrics (Micro-F1, Macro-F1, Hamming Loss, mAP, P@K, R@K).
Produces:
- results/metrics/temporal_baseline_results.json
- reports/temporal_baseline_evaluation.md
- results/checkpoints/temporal_baseline.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch

from src.data.config import (
    CHECKPOINTS_DIR,
    METRICS_DIR,
    MULTILABEL_FREQUENT363_LABEL_MAPPING_PATH,
    REPORTS_DIR,
    TEMPORAL_BASELINE_CHECKPOINT_PATH,
    TEMPORAL_BASELINE_REPORT_MD,
    TEMPORAL_BASELINE_RESULTS_JSON,
    TEMPORAL_MULTILABEL_FREQUENT363_DATASET_CSV,
)
from src.evaluation.training_report import (
    build_result_payload,
    format_terminal_report,
    save_json_results,
)
from src.models.temporal_multilabel_baseline import (
    evaluate_temporal_experiment,
    generate_baseline_markdown_report,
)


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Train and evaluate Pipeline 2: Temporal Multi-Label Baseline (Logistic Regression)."
    )
    parser.add_argument(
        "--dataset",
        default=str(TEMPORAL_MULTILABEL_FREQUENT363_DATASET_CSV),
        help="Path to temporal multi-label frequent363 dataset CSV.",
    )
    parser.add_argument(
        "--label-mapping",
        default=str(MULTILABEL_FREQUENT363_LABEL_MAPPING_PATH),
        help="Path to label mapping JSON.",
    )
    parser.add_argument(
        "--results-json",
        default=str(TEMPORAL_BASELINE_RESULTS_JSON),
        help="Output path for evaluation results JSON.",
    )
    parser.add_argument(
        "--report-md",
        default=str(TEMPORAL_BASELINE_REPORT_MD),
        help="Output path for evaluation report Markdown.",
    )
    parser.add_argument(
        "--checkpoint",
        default=str(TEMPORAL_BASELINE_CHECKPOINT_PATH),
        help="Output path for trained model checkpoint.",
    )
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs.")
    parser.add_argument("--lr", type=float, default=0.05, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-3, help="Weight decay.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI execution entry-point."""
    args = build_parser().parse_args(argv)

    dataset_path = Path(args.dataset)
    mapping_path = Path(args.label_mapping)
    results_json_path = Path(args.results_json)
    report_md_path = Path(args.report_md)
    checkpoint_path = Path(args.checkpoint)

    if not dataset_path.exists():
        print(f"[ERROR] Dataset not found at {dataset_path}", file=sys.stderr)
        return 1
    if not mapping_path.exists():
        print(f"[ERROR] Label mapping not found at {mapping_path}", file=sys.stderr)
        return 1

    df = pd.read_csv(dataset_path)
    mapping_payload = json.loads(mapping_path.read_text(encoding="utf-8"))
    label_mapping = {int(k): int(v) for k, v in mapping_payload["type_to_index"].items()}

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    results, model, standardizer = evaluate_temporal_experiment(
        df=df,
        label_mapping=label_mapping,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        return_model=True,
    )

    # Save checkpoint
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": model.input_dim,
            "n_classes": model.n_classes,
            "standardizer": standardizer.to_dict(),
            "selected_threshold": results["temporal_model_evaluation"]["val_threshold_optimization"]["best_threshold"],
            "train_summary": results["training_summary"],
        },
        checkpoint_path,
    )

    t_metrics = results["temporal_model_evaluation"]["test_metrics_optimized_threshold"]
    best_t = results["temporal_model_evaluation"]["val_threshold_optimization"]["best_threshold"]

    # Build standardized JSON payload
    payload = build_result_payload(
        pipeline="Temporal",
        model="Temporal Logistic Regression",
        dataset="Frequent363",
        number_of_labels=results["target_dimension"],
        train_size=len(train_df),
        val_size=len(val_df),
        test_size=len(test_df),
        unique_train_pairs=int(train_df["pair_key"].nunique()),
        unique_val_pairs=int(val_df["pair_key"].nunique()),
        unique_test_pairs=int(test_df["pair_key"].nunique()),
        training_config={
            "epochs": args.epochs,
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
            "features": results["temporal_features"],
        },
        training_summary=results["training_summary"],
        selected_threshold=best_t,
        threshold_selection_method="validation_micro_f1",
        test_metrics=t_metrics,
        checkpoint_path=str(checkpoint_path),
        extra={
            "full_evaluation": results,
        },
    )

    save_json_results(payload, results_json_path)

    # Save Markdown report
    report_md_path.parent.mkdir(parents=True, exist_ok=True)
    md_report = generate_baseline_markdown_report(results)
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write(md_report)

    # Print final terminal report
    report_data = {
        "pipeline": "Temporal",
        "model": "Temporal Logistic Regression",
        "dataset": "Frequent363",
        "test_size": len(test_df),
        "unique_test_pairs": int(test_df["pair_key"].nunique()),
        "number_of_labels": results["target_dimension"],
        "training_summary": results["training_summary"],
        "selected_threshold": best_t,
        "threshold_selection_set": "validation",
        "test_metrics": t_metrics,
        "checkpoint_path": str(checkpoint_path),
        "results_path": str(results_json_path),
    }

    print("\n" + format_terminal_report(report_data))
    return 0


if __name__ == "__main__":
    sys.exit(main())
