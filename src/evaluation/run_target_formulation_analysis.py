"""CLI: regenerate target formulation analysis JSON (read-only on dataset)."""

from __future__ import annotations

import argparse
import json

from src.evaluation.target_formulation_analysis import write_target_formulation_analysis


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze alternative DDI target formulations")
    parser.add_argument("--threshold", type=int, default=20)
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)
    report = write_target_formulation_analysis(threshold=args.threshold)
    print(json.dumps(report["formulation_comparison"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
