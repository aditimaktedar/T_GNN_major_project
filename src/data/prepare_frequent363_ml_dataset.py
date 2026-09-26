"""Build the derived frequent363 CSV from the full ML dataset (read-only source)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.data.config import (
    FINAL_MIMIC_TWOSIDES_ML_FREQUENT363_PATH,
    FINAL_MIMIC_TWOSIDES_ML_FREQUENT363_REPORT_PATH,
    FINAL_MIMIC_TWOSIDES_ML_PATH,
    FREQUENT363_MIN_TRAIN_ROWS,
)
from src.data.exceptions import MissingInputError
from src.data.final_ml_dataset import load_final_ml_dataset
from src.data.frequent363_ml_dataset import build_frequent363_frame, frequent363_report


def prepare_frequent363_ml_dataset(
    source_path: Path | None = None,
    output_path: Path | None = None,
    report_path: Path | None = None,
    min_train_rows: int = FREQUENT363_MIN_TRAIN_ROWS,
) -> tuple[pd.DataFrame, dict]:
    """Write derived frequent363 CSV and report without modifying the source file."""
    src = source_path or FINAL_MIMIC_TWOSIDES_ML_PATH
    if not Path(src).exists():
        raise MissingInputError(f"Source ML dataset not found at {src}")

    source = load_final_ml_dataset(src)
    filtered, _eligible = build_frequent363_frame(source, min_train_rows=min_train_rows)
    report = frequent363_report(source=source, filtered=filtered, min_train_rows=min_train_rows)

    out = output_path or FINAL_MIMIC_TWOSIDES_ML_FREQUENT363_PATH
    rep = report_path or FINAL_MIMIC_TWOSIDES_ML_FREQUENT363_REPORT_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    rep.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(out, index=False)
    rep.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    report["output_path"] = str(out)
    report["report_path"] = str(rep)
    return filtered, report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build frequent363 derived ML dataset")
    parser.add_argument("--source", default=str(FINAL_MIMIC_TWOSIDES_ML_PATH))
    parser.add_argument("--output", default=str(FINAL_MIMIC_TWOSIDES_ML_FREQUENT363_PATH))
    parser.add_argument("--report", default=str(FINAL_MIMIC_TWOSIDES_ML_FREQUENT363_REPORT_PATH))
    parser.add_argument("--min-train-rows", type=int, default=FREQUENT363_MIN_TRAIN_ROWS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _frame, report = prepare_frequent363_ml_dataset(
        source_path=Path(args.source),
        output_path=Path(args.output),
        report_path=Path(args.report),
        min_train_rows=args.min_train_rows,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
