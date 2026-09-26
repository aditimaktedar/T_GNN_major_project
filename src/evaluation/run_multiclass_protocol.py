"""CLI: describe evaluation slices and run protocol sanity checks (no training)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.data.config import FINAL_MIMIC_TWOSIDES_ML_PATH, METRICS_DIR, PROJECT_ROOT
from src.data.final_ml_dataset import load_final_ml_dataset, prepare_multiclass_frame
from src.evaluation.multiclass_protocol import (
    describe_evaluation_slices,
    evaluate_multiclass_protocol,
    random_baseline_predictions,
)

DEFAULT_OUTPUT = METRICS_DIR / "multiclass_evaluation_protocol_slices.json"
SANITY_OUTPUT = METRICS_DIR / "multiclass_evaluation_protocol_sanity.json"


def run_describe_slices(frame, min_train_examples: int) -> dict:
    return describe_evaluation_slices(frame, min_train_examples=min_train_examples)


def run_sanity_check(frame, min_train_examples: int, seed: int) -> dict:
    """Verify slice counts and metric computation using random baseline predictions."""
    enriched, encoder, _cache = prepare_multiclass_frame(frame)
    slice_report = describe_evaluation_slices(frame, min_train_examples=min_train_examples)

    predictions_by_split = {}
    for split in ("val", "test"):
        eval_frame = enriched.loc[enriched["split"] == split].reset_index(drop=True)
        evaluable = eval_frame.loc[eval_frame["label_index"] >= 0].reset_index(drop=True)
        y_pred, y_prob = random_baseline_predictions(evaluable, encoder.n_classes, seed=seed)
        payload = evaluable.copy()
        payload["y_pred"] = y_pred
        payload["y_prob"] = list(y_prob)
        predictions_by_split[split] = payload

    protocol_results = evaluate_multiclass_protocol(
        frame,
        predictions_by_split,
        encoder,
        min_train_examples=min_train_examples,
    )

    # Structural checks
    checks = []
    val_seen_dedup = slice_report["slices"]["val_seen_class"]["pair_type_dedup_level"]["n_pair_types"]
    test_seen_dedup = slice_report["slices"]["test_seen_class"]["pair_type_dedup_level"]["n_pair_types"]
    checks.append({"check": "val_seen_class_dedup_n_pair_types", "value": val_seen_dedup, "pass": val_seen_dedup > 0})
    checks.append({"check": "test_seen_class_dedup_n_pair_types", "value": test_seen_dedup, "pass": test_seen_dedup > 0})

    val_level = protocol_results["splits"]["val"]["seen_class"]["pair_type_dedup_level"]
    val_metrics = val_level["metrics"]
    for metric in ("top_1_accuracy", "top_3_accuracy", "top_5_accuracy", "macro_f1", "micro_f1"):
        value = val_metrics[metric]
        checks.append(
            {
                "check": f"val_seen_class_dedup_{metric}_computed",
                "value": value,
                "pass": value is not None and 0.0 <= float(value) <= 1.0,
            }
        )

    top5 = float(val_metrics["top_5_accuracy"])
    top1 = float(val_metrics["top_1_accuracy"])
    checks.append(
        {
            "check": "top_5_gte_top_1",
            "value": {"top_1": top1, "top_5": top5},
            "pass": top5 >= top1,
        }
    )

    all_pass = all(item["pass"] for item in checks)
    return {
        "mode": "sanity_check",
        "seed": seed,
        "min_train_examples_for_frequent_slice": min_train_examples,
        "slice_counts": slice_report,
        "random_baseline_protocol_results": protocol_results,
        "checks": checks,
        "all_checks_passed": all_pass,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Multiclass DDI evaluation protocol utilities")
    parser.add_argument("--data", default=str(FINAL_MIMIC_TWOSIDES_ML_PATH))
    parser.add_argument("--min-train-examples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--describe-slices",
        action="store_true",
        help="Report row/class counts per evaluation slice (no predictions).",
    )
    parser.add_argument(
        "--sanity-check",
        action="store_true",
        help="Verify slice counts and metric computation with random predictions.",
    )
    parser.add_argument("--output", default=None, help="Optional JSON output path.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    frame = load_final_ml_dataset(args.data)

    if args.sanity_check:
        results = run_sanity_check(frame, args.min_train_examples, args.seed)
        out = Path(args.output) if args.output else SANITY_OUTPUT
    elif args.describe_slices:
        results = run_describe_slices(frame, args.min_train_examples)
        out = Path(args.output) if args.output else DEFAULT_OUTPUT
    else:
        # Default: describe slices
        results = run_describe_slices(frame, args.min_train_examples)
        out = Path(args.output) if args.output else DEFAULT_OUTPUT

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2, default=str))
    if args.sanity_check and not results.get("all_checks_passed", True):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
