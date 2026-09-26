"""CLI Entry-point for Temporal Feature Ablation Study.

Command: python -m src.models.run_temporal_ablation

Evaluates 4 feature configurations on the preserved pair-level split:
1. Training-prior baseline
2. Age-only (anchor_age)
3. Temporal-only (num_a_events, num_b_events, num_total_pair_events,
                  min_delta_hours, median_delta_hours, a_before_b, b_before_a, same_timestamp)
4. Age + Temporal (all 9 admission features)

Produces:
- results/metrics/temporal_feature_ablation_results.json
- reports/temporal_feature_ablation_report.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.data.config import (
    TEMPORAL_ABLATION_REPORT_MD,
    TEMPORAL_ABLATION_RESULTS_JSON,
    TEMPORAL_MULTILABEL_FREQUENT363_DATASET_CSV,
)
from src.models.temporal_ablation import execute_temporal_ablation_pipeline


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Run temporal feature ablation study on multi-label DDI dataset."
    )
    parser.add_argument(
        "--dataset",
        default=str(TEMPORAL_MULTILABEL_FREQUENT363_DATASET_CSV),
        help="Path to temporal multi-label frequent363 dataset CSV.",
    )
    parser.add_argument(
        "--results-json",
        default=str(TEMPORAL_ABLATION_RESULTS_JSON),
        help="Output path for ablation results JSON.",
    )
    parser.add_argument(
        "--report-md",
        default=str(TEMPORAL_ABLATION_REPORT_MD),
        help="Output path for ablation report Markdown.",
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
    results_json_path = Path(args.results_json)
    report_md_path = Path(args.report_md)

    print("==================================================")
    print("      TEMPORAL FEATURE ABLATION STUDY             ")
    print("==================================================")
    print(f"Dataset Path:    {dataset_path}")
    print(f"Results JSON:    {results_json_path}")
    print(f"Report Markdown: {report_md_path}")
    print(f"Epochs:          {args.epochs}")
    print(f"Learning Rate:   {args.lr}")
    print(f"Weight Decay:    {args.weight_decay}")
    print(f"Seed:            {args.seed}")
    print("--------------------------------------------------")

    study_results, _md = execute_temporal_ablation_pipeline(
        dataset_path=dataset_path,
        results_json_path=results_json_path,
        report_md_path=report_md_path,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
    )

    print("\n[OK] Temporal feature ablation study complete.")
    print(f"[OK] Saved results JSON: {results_json_path}")
    print(f"[OK] Saved report MD:    {report_md_path}")
    print("\n--- ABLATION TEST EVALUATION SUMMARY (N=10 Test Observations) ---")
    cfgs = study_results["configurations"]
    for key, cfg in cfgs.items():
        opt_t = cfg["val_threshold_optimization"]["best_threshold"]
        m = cfg["test_metrics_optimized_threshold"]
        print(f"\nConfiguration: {cfg['config_name']} (Dim: {cfg['input_dim']}, Best Threshold = {opt_t:.2f}):")
        print(f"  Micro-F1:     {m['micro_f1']:.4f}")
        print(f"  Macro-F1:     {m['macro_f1']:.4f}")
        print(f"  Hamming Loss: {m['hamming_loss']:.4f}")
        print(f"  Jaccard:      {m['jaccard_score']:.4f}")
        print(f"  mAP:          {m['mAP']:.4f}")
        print(f"  P@5 / R@5:    {m['precision_at_5']:.4f} / {m['recall_at_5']:.4f}")

    print("\n==================================================")
    print("NOTE: Test split contains 10 observations. Results represent observational baseline signals without claims of statistical superiority.")
    print("SUCCESS: Temporal feature ablation complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
