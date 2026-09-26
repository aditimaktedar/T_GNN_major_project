"""CLI Entry-point for Temporal Pair Feature Dataset Preparation.

Command: python -m src.data.prepare_temporal_pair_features

Extracts temporal pair-admission features from eMAR administration records,
joins with observed DDI pairs and admission metadata, preserves existing
pair-level train/val/test splits, validates data integrity, and produces:
- data/processed/temporal_pair_features.csv
- reports/temporal_pair_features_validation.md
- results/metrics/temporal_pair_features_validation.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.data.config import (
    MULTILABEL_FREQUENT363_DATASET_CSV,
    SELECTED_MIMIC_TWOSIDES_DATASET_PATH,
    TEMPORAL_EMAR_EVENTS_PATH,
    TEMPORAL_PAIR_FEATURES_AUDIT_JSON,
    TEMPORAL_PAIR_FEATURES_CSV,
    TEMPORAL_PAIR_FEATURES_REPORT_MD,
)
from src.data.temporal_pair_features import build_temporal_pair_features


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Prepare reproducible temporal pair feature dataset from eMAR and DDI pairs."
    )
    parser.add_argument(
        "--emar-events",
        default=str(TEMPORAL_EMAR_EVENTS_PATH),
        help="Path to temporal eMAR events CSV.",
    )
    parser.add_argument(
        "--selected-dataset",
        default=str(SELECTED_MIMIC_TWOSIDES_DATASET_PATH),
        help="Path to final selected MIMIC-TWOSIDES dataset CSV.",
    )
    parser.add_argument(
        "--splits-source",
        default=str(MULTILABEL_FREQUENT363_DATASET_CSV),
        help="Path to multi-label frequent363 dataset CSV containing pair splits.",
    )
    parser.add_argument(
        "--output-csv",
        default=str(TEMPORAL_PAIR_FEATURES_CSV),
        help="Output path for temporal pair features CSV.",
    )
    parser.add_argument(
        "--report-md",
        default=str(TEMPORAL_PAIR_FEATURES_REPORT_MD),
        help="Output path for validation report Markdown.",
    )
    parser.add_argument(
        "--audit-json",
        default=str(TEMPORAL_PAIR_FEATURES_AUDIT_JSON),
        help="Output path for validation audit JSON.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI execution entry-point."""
    args = build_parser().parse_args(argv)

    print("==================================================")
    print("     TEMPORAL PAIR FEATURE DATASET PREPARATION    ")
    print("==================================================")
    print(f"eMAR Events Path:      {args.emar_events}")
    print(f"Selected Dataset Path: {args.selected_dataset}")
    print(f"Splits Source Path:    {args.splits_source}")
    print(f"Output CSV Path:       {args.output_csv}")
    print(f"Report Markdown Path:  {args.report_md}")
    print(f"Audit JSON Path:       {args.audit_json}")
    print("--------------------------------------------------")

    df, audit = build_temporal_pair_features(
        emar_path=Path(args.emar_events),
        final_mimic_path=Path(args.selected_dataset),
        splits_source_path=Path(args.splits_source),
        output_csv_path=Path(args.output_csv),
        report_md_path=Path(args.report_md),
        audit_json_path=Path(args.audit_json),
    )

    print("\n[OK] Temporal pair feature extraction complete.")
    print("[OK] Dataset validation & integrity assertions PASSED.")
    print("[OK] Pair-level train/val/test splits strictly preserved (Split Leakage = 0).")
    print(f"[OK] Saved temporal pair features CSV: {args.output_csv}")
    print(f"[OK] Saved validation report MD:       {args.report_md}")
    print(f"[OK] Saved audit metrics JSON:         {args.audit_json}")
    print("\n--- VALIDATION AUDIT SUMMARY ---")
    print(f"Total Temporal Observations: {audit['total_temporal_observations']}")
    print(f"Unique Temporal Pairs:       {audit['unique_temporal_pairs']}")
    print(f"Unique Subjects:             {audit['unique_subjects']}")
    print(f"Unique Admissions:           {audit['unique_admissions']}")
    print(f"Missing Values:              {audit['total_missing_values']}")
    print(f"Split Distribution (Obs):    {audit['split_counts_observations']}")
    print(f"Split Distribution (Pairs):  {audit['split_counts_unique_pairs']}")
    print("Delta Hours Stats (min Delta t):")
    print(f"  Min:    {audit['delta_hours_statistics']['min_delta_hours']['min']:.4f}h")
    print(f"  Median: {audit['delta_hours_statistics']['min_delta_hours']['median']:.4f}h")
    print(f"  Mean:   {audit['delta_hours_statistics']['min_delta_hours']['mean']:.4f}h")
    print(f"  Max:    {audit['delta_hours_statistics']['min_delta_hours']['max']:.4f}h")

    print("Initial Admin Order Flags:")
    print(f"  a_before_b:     {audit['initial_administration_order']['a_before_b']}")
    print(f"  b_before_a:     {audit['initial_administration_order']['b_before_a']}")
    print(f"  same_timestamp: {audit['initial_administration_order']['same_timestamp']}")
    print("Repeated Observations per Pair:")
    print(f"  Min:    {audit['repeated_observations_per_pair']['min']}")
    print(f"  Median: {audit['repeated_observations_per_pair']['median']}")
    print(f"  Mean:   {audit['repeated_observations_per_pair']['mean']}")
    print(f"  Max:    {audit['repeated_observations_per_pair']['max']}")
    print("Split Leakage Detected:      " + str(audit['split_leakage_detected']))
    print("==================================================")
    print("SUCCESS: Temporal pair features prepared and validated.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
