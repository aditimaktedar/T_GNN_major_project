"""CLI: describe frequent363 evaluation slices (no training)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.data.config import FINAL_MIMIC_TWOSIDES_ML_FREQUENT363_PATH, METRICS_DIR
from src.data.frequent363_ml_dataset import load_frequent363_ml_dataset
from src.evaluation.frequent363_protocol import describe_frequent363_slices

DEFAULT_OUTPUT = METRICS_DIR / "frequent363_evaluation_protocol_slices.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Describe frequent363 evaluation slices")
    parser.add_argument("--data", default=str(FINAL_MIMIC_TWOSIDES_ML_FREQUENT363_PATH))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args(argv)

    frame = load_frequent363_ml_dataset(args.data)
    report = describe_frequent363_slices(frame)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
