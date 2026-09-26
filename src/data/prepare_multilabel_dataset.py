"""CLI entry-point for Multi-Label DDI Dataset Preparation.

Command: python -m src.data.prepare_multilabel_dataset

Runs dataset loading, pair aggregation, multi-hot target construction,
target validation, dataset audit, pair-level splitting, and leakage checking.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.data.config import (
    MULTILABEL_DATASET_AUDIT_PATH,
    MULTILABEL_FREQUENT363_DATASET_CSV,
    MULTILABEL_FREQUENT363_DATASET_PATH,
    MULTILABEL_FREQUENT363_LABEL_MAPPING_PATH,
    MULTILABEL_PAIR_SPLITS_PATH,
    SELECTED_MIMIC_TWOSIDES_DATASET_PATH,
)
from src.data.multilabel_pipeline import build_multilabel_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare reproducible multi-label DDI dataset from source dataset."
    )
    parser.add_argument(
        "--source",
        default=str(SELECTED_MIMIC_TWOSIDES_DATASET_PATH),
        help="Path to source CSV dataset.",
    )
    parser.add_argument(
        "--output-csv",
        default=str(MULTILABEL_FREQUENT363_DATASET_CSV),
        help="Output path for derived CSV dataset.",
    )
    parser.add_argument(
        "--output-parquet",
        default=str(MULTILABEL_FREQUENT363_DATASET_PATH),
        help="Output path for derived Parquet dataset.",
    )
    parser.add_argument(
        "--label-mapping",
        default=str(MULTILABEL_FREQUENT363_LABEL_MAPPING_PATH),
        help="Path to versioned label mapping JSON.",
    )
    parser.add_argument(
        "--splits-out",
        default=str(MULTILABEL_PAIR_SPLITS_PATH),
        help="Output path for pair splits JSON.",
    )
    parser.add_argument(
        "--audit-out",
        default=str(MULTILABEL_DATASET_AUDIT_PATH),
        help="Output path for dataset audit JSON.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for pair-level split.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    print("==================================================")
    print("      MULTI-LABEL DDI DATASET PREPARATION         ")
    print("==================================================")
    print(f"Source Dataset Path:  {args.source}")
    print(f"Label Mapping Path:   {args.label_mapping}")
    print(f"Derived CSV Path:     {args.output_csv}")
    print(f"Derived Parquet Path: {args.output_parquet}")
    print(f"Pair Splits Path:     {args.splits_out}")
    print(f"Dataset Audit Path:   {args.audit_out}")
    print("--------------------------------------------------")

    derived_df, audit = build_multilabel_dataset(
        source_path=Path(args.source),
        label_mapping_path=Path(args.label_mapping),
        output_parquet=Path(args.output_parquet),
        output_csv=Path(args.output_csv),
        audit_path=Path(args.audit_out),
        splits_path=Path(args.splits_out),
        seed=args.seed,
    )

    print("\n[OK] Dataset aggregation & target construction complete.")
    print("[OK] Target validation assertions PASSED.")
    print("[OK] Pair-level train/val/test splits generated (Overlap = 0).")
    print("\n--- DATASET AUDIT SUMMARY ---")
    print(f"Source Rows:                 {audit['source_rows']}")
    print(f"Unique Drug Pairs:           {audit['unique_drug_pairs']}")
    print(f"Unique Interaction Labels:   {audit['unique_interaction_labels']}")
    print(f"Pair-Label Associations:     {audit['pair_label_associations']}")
    print("Labels Per Pair Statistics:")
    print(f"  Min:    {audit['labels_per_pair']['min']}")
    print(f"  Max:    {audit['labels_per_pair']['max']}")
    print(f"  Mean:   {audit['labels_per_pair']['mean']}")
    print(f"  Median: {audit['labels_per_pair']['median']}")
    print("Pairs by Label Count:")
    print(f"  1 label:    {audit['pairs_by_label_count']['exactly_1_label']}")
    print(f"  2 labels:   {audit['pairs_by_label_count']['exactly_2_labels']}")
    print(f"  3 labels:   {audit['pairs_by_label_count']['exactly_3_labels']}")
    print(f"  >3 labels:  {audit['pairs_by_label_count']['more_than_3_labels']}")
    print("==================================================")
    print("SUCCESS: Dataset preparation complete. Ready for training.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
