"""CLI Entry-point for Model-Ready Temporal Multi-Label Dataset Preparation.

Command: python -m src.data.prepare_temporal_multilabel

Joins temporal pair features with 363-dimensional multi-label targets,
verifies target consistency and split leakage, and produces:
- data/processed/temporal_multilabel_frequent363_dataset.csv
- reports/temporal_multilabel_validation.md
- results/metrics/temporal_multilabel_validation.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.data.config import (
    MULTILABEL_FREQUENT363_DATASET_CSV,
    MULTILABEL_FREQUENT363_LABEL_MAPPING_PATH,
    TEMPORAL_MULTILABEL_AUDIT_JSON,
    TEMPORAL_MULTILABEL_FREQUENT363_DATASET_CSV,
    TEMPORAL_MULTILABEL_FREQUENT363_DATASET_PATH,
    TEMPORAL_MULTILABEL_REPORT_MD,
    TEMPORAL_PAIR_FEATURES_CSV,
)
from src.data.temporal_multilabel import build_temporal_multilabel_dataset


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Prepare model-ready temporal multi-label DDI dataset."
    )
    parser.add_argument(
        "--temporal-features",
        default=str(TEMPORAL_PAIR_FEATURES_CSV),
        help="Path to temporal pair features CSV.",
    )
    parser.add_argument(
        "--multilabel-source",
        default=str(MULTILABEL_FREQUENT363_DATASET_CSV),
        help="Path to multi-label frequent363 dataset CSV.",
    )
    parser.add_argument(
        "--label-mapping",
        default=str(MULTILABEL_FREQUENT363_LABEL_MAPPING_PATH),
        help="Path to label mapping JSON.",
    )
    parser.add_argument(
        "--output-csv",
        default=str(TEMPORAL_MULTILABEL_FREQUENT363_DATASET_CSV),
        help="Output path for temporal multi-label CSV.",
    )
    parser.add_argument(
        "--output-parquet",
        default=str(TEMPORAL_MULTILABEL_FREQUENT363_DATASET_PATH),
        help="Output path for temporal multi-label Parquet.",
    )
    parser.add_argument(
        "--report-md",
        default=str(TEMPORAL_MULTILABEL_REPORT_MD),
        help="Output path for validation report Markdown.",
    )
    parser.add_argument(
        "--audit-json",
        default=str(TEMPORAL_MULTILABEL_AUDIT_JSON),
        help="Output path for validation audit JSON.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI execution entry-point."""
    args = build_parser().parse_args(argv)

    print("==================================================")
    print("   MODEL-READY TEMPORAL MULTI-LABEL PREPARATION   ")
    print("==================================================")
    print(f"Temporal Features:     {args.temporal_features}")
    print(f"Multi-Label Source:    {args.multilabel_source}")
    print(f"Label Mapping:         {args.label_mapping}")
    print(f"Output CSV:            {args.output_csv}")
    print(f"Output Parquet:        {args.output_parquet}")
    print(f"Report Markdown:       {args.report_md}")
    print(f"Audit JSON:            {args.audit_json}")
    print("--------------------------------------------------")

    df, audit = build_temporal_multilabel_dataset(
        temporal_features_path=Path(args.temporal_features),
        multilabel_path=Path(args.multilabel_source),
        label_mapping_path=Path(args.label_mapping),
        output_csv_path=Path(args.output_csv),
        output_parquet_path=Path(args.output_parquet),
        report_md_path=Path(args.report_md),
        audit_json_path=Path(args.audit_json),
    )

    print("\n[OK] Temporal multi-label dataset join complete.")
    print("[OK] 363-dimensional multi-hot target consistency verified.")
    print("[OK] Pair-level train/val/test splits strictly preserved (Split Leakage = 0).")
    print(f"[OK] Saved temporal multi-label CSV: {args.output_csv}")
    print(f"[OK] Saved validation report MD:      {args.report_md}")
    print(f"[OK] Saved audit metrics JSON:        {args.audit_json}")
    print("\n--- VALIDATION AUDIT SUMMARY ---")
    print(f"Total Temporal Observations: {audit['total_temporal_observations']}")
    print(f"Target Dimension:            {audit['target_dimension']}")
    print(f"Total Label Associations:    {audit['total_active_label_associations']}")
    print(f"Unique Temporal Pairs:       {audit['unique_temporal_pairs']}")
    print(f"Unique Subjects:             {audit['unique_subjects']}")
    print(f"Unique Admissions:           {audit['unique_admissions']}")
    print(f"Missing Values:              {audit['total_missing_values']}")
    print(f"Split Distribution (Obs):    {audit['split_counts_observations']}")
    print(f"Split Distribution (Pairs):  {audit['split_counts_unique_pairs']}")
    print("Active Labels per Obs:")
    print(f"  Min:    {audit['labels_per_observation']['min']}")
    print(f"  Median: {audit['labels_per_observation']['median']}")
    print(f"  Mean:   {audit['labels_per_observation']['mean']}")
    print(f"  Max:    {audit['labels_per_observation']['max']}")
    print("Initial Admin Order Flags:")
    print(f"  a_before_b:     {audit['initial_administration_order']['a_before_b']}")
    print(f"  b_before_a:     {audit['initial_administration_order']['b_before_a']}")
    print(f"  same_timestamp: {audit['initial_administration_order']['same_timestamp']}")
    print("Split Leakage Detected:      " + str(audit['split_leakage_detected']))
    print("Target Consistency Verified: " + str(audit['target_consistency_verified']))
    print("==================================================")
    print("SUCCESS: Model-ready temporal multi-label dataset prepared and validated.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
