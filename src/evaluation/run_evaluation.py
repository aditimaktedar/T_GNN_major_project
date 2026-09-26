"""End-to-end Member B evaluation runner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.baselines.run_baselines import (
    BASELINES_DIR,
    METRICS_DIR,
    build_demo_ml_dataset,
    load_config,
    run_baselines,
)
from src.data.config import PROJECT_ROOT
from src.evaluation.adapters import member_a_tgnn_status, member_a_xai_status
from src.evaluation.data_audit import audit_dataset, save_data_audit
from src.evaluation.drugbank_sanity import check_explanations_against_drugbank
from src.evaluation.export_results import export_all_tables
from src.evaluation.rag_ablation import compare_rag_ablation, save_rag_ablation_report
from src.evaluation.reproducibility_report import build_reproducibility_report, save_reproducibility_report
from src.evaluation.results_table import build_from_baseline_run_summary
from src.data.io_utils import write_json

CONFIG_PATH = PROJECT_ROOT / "configs" / "baselines.json"


def build_demo_explanations(frame: pd.DataFrame) -> pd.DataFrame:
    """Synthetic explanation objects for software testing only."""
    rows = []
    test_frame = frame[frame["split"] == "test"] if "split" in frame.columns else frame
    for idx, row in test_frame.iterrows():
        mask = [1.0, 0.0, 1.0, 0.0]
        prob = 0.8 if row["label"] == "positive" else 0.2
        masked = prob - 0.1 if row["label"] == "positive" else prob + 0.05
        rows.append(
            {
                "patient_id": row["patient_id"],
                "drug_a": row["drug_a"],
                "drug_b": row["drug_b"],
                "prediction": int(row["label"] == "positive"),
                "probability": prob,
                "original_probability": prob,
                "masked_probability": masked,
                "explanation_mask": mask,
                "important_features": mask,
                "synthetic_demo": True,
            }
        )
    rep_rows = []
    for item in rows:
        perturbed = list(item["explanation_mask"])
        if perturbed:
            perturbed[0] = 1.0 - perturbed[0]
        rep_rows.append({**item, "explanation_mask": perturbed})
    return pd.DataFrame(rows), pd.DataFrame(rep_rows)


def run_demo_evaluation(config: dict) -> dict:
    print("SYNTHETIC SOFTWARE VALIDATION ONLY — NOT SCIENTIFIC RESULTS")
    results = {"mode": "demo", "synthetic_demo": True, "warning": "SYNTHETIC SOFTWARE VALIDATION ONLY"}

    frame = build_demo_ml_dataset(seed=config["seed"], n_bits=config["demo"]["fingerprint_bits"])
    summary_path = METRICS_DIR / "synthetic_demo_run_summary.json"
    if not summary_path.exists():
        BASELINES_DIR.mkdir(parents=True, exist_ok=True)
        METRICS_DIR.mkdir(parents=True, exist_ok=True)
        run_baselines(frame, config, ["logistic_regression", "static_gat"], "synthetic_demo")

    table, table_paths = build_from_baseline_run_summary(summary_path)
    results["model_comparison"] = {"paths": {k: str(v) for k, v in table_paths.items()}, "rows": len(table)}

    explanations, repeated = build_demo_explanations(frame)
    from src.evaluation.run_xai_evaluation import evaluate_explanations

    aggregate, per_example = evaluate_explanations(explanations, repeated)
    xai_payload = {
        "status": "ok",
        "synthetic_demo": True,
        "warning": "SYNTHETIC SOFTWARE VALIDATION ONLY",
        "aggregate": aggregate,
        "n_examples": int(len(per_example)),
    }
    write_json(METRICS_DIR / "xai_metrics.json", xai_payload)
    per_example.to_csv(METRICS_DIR / "xai_per_example.csv", index=False)
    results["xai"] = xai_payload

    drugbank = check_explanations_against_drugbank(explanations)
    write_json(METRICS_DIR / "drugbank_sanity.json", drugbank)
    if drugbank.get("matches"):
        pd.DataFrame(drugbank["matches"]).to_csv(METRICS_DIR / "drugbank_sanity.csv", index=False)
    results["drugbank"] = drugbank

    rag = compare_rag_ablation(
        METRICS_DIR / "missing_without_rag.json",
        METRICS_DIR / "missing_with_rag.json",
        synthetic_demo=True,
    )
    rag_paths = save_rag_ablation_report(rag)
    if rag.get("status") != "ok":
        write_json(METRICS_DIR / "rag_ablation.json", rag)
    results["rag_ablation"] = {**rag, "paths": rag_paths}

    audit = audit_dataset(frame)
    save_data_audit(audit)
    results["data_audit"] = audit

    repro = build_reproducibility_report(
        config=config,
        dataset_paths={"demo_dataset": "in-memory synthetic"},
        dataset_sources=["synthetic_demo"],
        synthetic_demo=True,
    )
    save_reproducibility_report(repro)
    results["reproducibility"] = repro

    results["member_a"] = {
        "tgnn": member_a_tgnn_status(),
        "xai": member_a_xai_status(),
    }
    results["export"] = export_all_tables()
    summary_out = METRICS_DIR / "evaluation_demo_summary.json"
    write_json(summary_out, results)
    results["summary_path"] = str(summary_out)
    return results


def run_real_evaluation(
    config: dict,
    baseline_summary: Path | None = None,
    explanations_path: Path | None = None,
    tgnn_metrics_path: Path | None = None,
    drugbank_path: Path | None = None,
    without_rag_path: Path | None = None,
    with_rag_path: Path | None = None,
    dataset_path: Path | None = None,
) -> dict:
    results = {"mode": "real", "synthetic_demo": False}
    summary = baseline_summary or METRICS_DIR / "synthetic_demo_run_summary.json"
    if summary.exists():
        _, paths = build_from_baseline_run_summary(summary, tgnn_metrics_path=tgnn_metrics_path)
        results["model_comparison"] = {k: str(v) for k, v in paths.items()}
    else:
        results["model_comparison"] = {"status": "skipped", "message": "No baseline summary found."}

    if explanations_path and explanations_path.exists():
        results["xai"] = run_xai_evaluation(explanations_path=explanations_path)
    else:
        status = member_a_xai_status()
        results["xai"] = {"status": "skipped", "message": status.get("reason")}

    if explanations_path and explanations_path.exists():
        exp = pd.read_csv(explanations_path) if explanations_path.suffix == ".csv" else pd.read_parquet(explanations_path)
        drugbank = check_explanations_against_drugbank(exp, drugbank_path=drugbank_path)
        write_json(METRICS_DIR / "drugbank_sanity.json", drugbank)
        if drugbank.get("matches"):
            pd.DataFrame(drugbank["matches"]).to_csv(METRICS_DIR / "drugbank_sanity.csv", index=False)
        results["drugbank"] = drugbank
    else:
        results["drugbank"] = {"status": "skipped", "message": "No explanations supplied for DrugBank check."}

    if without_rag_path and with_rag_path:
        rag = compare_rag_ablation(without_rag_path, with_rag_path)
        results["rag_ablation"] = {**rag, "paths": save_rag_ablation_report(rag)}
    else:
        results["rag_ablation"] = {
            "status": "rag_results_not_available",
            "message": "Supply Member C result files with --without-rag and --with-rag.",
        }

    audit = audit_dataset(dataset_path)
    save_data_audit(audit)
    results["data_audit"] = audit

    repro = build_reproducibility_report(config=config, synthetic_demo=False)
    save_reproducibility_report(repro)
    results["reproducibility"] = repro
    results["export"] = export_all_tables()
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Member B evaluation runner.")
    parser.add_argument("--demo", action="store_true", help="Synthetic software-validation evaluation only.")
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--baseline-summary", type=str)
    parser.add_argument("--explanations", type=str, help="Explanation file from Member A.")
    parser.add_argument("--tgnn-metrics", type=str, help="TemporalDDI-GNN metrics JSON when available.")
    parser.add_argument("--drugbank-path", type=str, help="Local licensed DrugBank export path.")
    parser.add_argument("--without-rag", type=str, help="Member C metrics JSON without RAG.")
    parser.add_argument("--with-rag", type=str, help="Member C metrics JSON with RAG.")
    parser.add_argument("--dataset", type=str, help="Processed ml_dataset.parquet for audit.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(Path(args.config))
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR = PROJECT_ROOT / "results" / "tables"
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    if args.demo:
        results = run_demo_evaluation(config)
    else:
        results = run_real_evaluation(
            config,
            baseline_summary=Path(args.baseline_summary) if args.baseline_summary else None,
            explanations_path=Path(args.explanations) if args.explanations else None,
            tgnn_metrics_path=Path(args.tgnn_metrics) if args.tgnn_metrics else None,
            drugbank_path=Path(args.drugbank_path) if args.drugbank_path else None,
            without_rag_path=Path(args.without_rag) if args.without_rag else None,
            with_rag_path=Path(args.with_rag) if args.with_rag else None,
            dataset_path=Path(args.dataset) if args.dataset else None,
        )
    print(json.dumps({k: v for k, v in results.items() if k not in ("reproducibility",)}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
