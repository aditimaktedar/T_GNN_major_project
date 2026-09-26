"""Integration tests for Member B evaluation pipeline (synthetic fixtures only)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.baselines.run_baselines import build_demo_ml_dataset, run_baselines
from src.evaluation.adapters import member_a_xai_status
from src.evaluation.data_audit import audit_dataset
from src.evaluation.drugbank_sanity import check_explanations_against_drugbank, check_predictions_against_drugbank
from src.evaluation.rag_ablation import compare_rag_ablation, save_rag_ablation_report
from src.evaluation.reproducibility_report import build_reproducibility_report, save_reproducibility_report
from src.evaluation.results_table import build_comparison_table, save_comparison_table
from src.evaluation.run_evaluation import build_demo_explanations, run_demo_evaluation
from src.evaluation.run_xai_evaluation import evaluate_explanations


@pytest.fixture
def demo_config():
    return {
        "seed": 42,
        "pair_feature_method": "concat",
        "logistic_regression": {"C": 1.0, "class_weight": "balanced", "max_iter": 200},
        "static_gat": {
            "hidden_dim": 16,
            "heads": 2,
            "dropout": 0.1,
            "learning_rate": 0.01,
            "epochs": 2,
            "batch_size": 4,
            "class_weight": "balanced",
        },
        "demo": {"fingerprint_bits": 16},
    }


def test_model_comparison_table(demo_config, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("src.evaluation.results_table.METRICS_DIR", tmp_path / "metrics")
    monkeypatch.setattr("src.evaluation.results_table.TABLES_DIR", tmp_path / "results" / "tables")
    frame = build_demo_ml_dataset()
    monkeypatch.setattr("src.baselines.run_baselines.METRICS_DIR", tmp_path / "metrics")
    monkeypatch.setattr("src.baselines.run_baselines.BASELINES_DIR", tmp_path / "baselines")
    run_baselines(frame, demo_config, ["logistic_regression", "static_gat"], "synthetic_demo")
    summary = tmp_path / "metrics" / "synthetic_demo_run_summary.json"
    from src.evaluation.results_table import build_from_baseline_run_summary

    table, paths = build_from_baseline_run_summary(summary)
    assert len(table) == 2
    assert "Logistic Regression" in table["Model"].values
    assert "Static GAT" in table["Model"].values
    assert paths["csv"].exists()


def test_xai_evaluation_with_synthetic_explanations() -> None:
    frame = build_demo_ml_dataset()
    explanations, repeated = build_demo_explanations(frame)
    aggregate, per = evaluate_explanations(explanations, repeated)
    assert "fidelity_mean" in aggregate
    assert "sparsity_mean" in aggregate
    assert aggregate["stability_mean"] is not None
    assert len(per) > 0


def test_drugbank_missing(tmp_path: Path) -> None:
    preds = pd.DataFrame({"drug_a": ["A"], "drug_b": ["B"]})
    result = check_predictions_against_drugbank(preds, drugbank_dir=tmp_path / "missing")
    assert result["status"] == "drugbank_data_not_available"


def test_rag_missing(tmp_path: Path) -> None:
    result = compare_rag_ablation(
        tmp_path / "no_without.json",
        tmp_path / "no_with.json",
        synthetic_demo=True,
    )
    assert result["status"] == "rag_results_not_available"


def test_rag_comparison_with_files(tmp_path: Path) -> None:
    without = tmp_path / "without.json"
    with_rag = tmp_path / "with.json"
    without.write_text(json.dumps({"metrics": {"accuracy": 0.5, "f1": 0.4, "auroc": 0.6, "pr_auc": 0.55, "mcc": 0.1}}), encoding="utf-8")
    with_rag.write_text(json.dumps({"metrics": {"accuracy": 0.7, "f1": 0.6, "auroc": 0.8, "pr_auc": 0.75, "mcc": 0.3}}), encoding="utf-8")
    monkeypatch_metrics = tmp_path / "metrics"
    monkeypatch_metrics.mkdir()
    import src.evaluation.rag_ablation as rag_mod

    old = rag_mod.METRICS_DIR
    rag_mod.METRICS_DIR = monkeypatch_metrics
    try:
        result = compare_rag_ablation(without, with_rag, synthetic_demo=True)
        paths = save_rag_ablation_report(result)
        assert result["status"] == "ok"
        assert Path(paths["json"]).exists()
    finally:
        rag_mod.METRICS_DIR = old


def test_data_audit_demo() -> None:
    frame = build_demo_ml_dataset()
    audit = audit_dataset(frame)
    assert audit["status"] == "ok"
    assert audit["synthetic_demo"] is True
    assert audit["patient_leakage_detected"] is False


def test_data_audit_real_missing() -> None:
    audit = audit_dataset(None)
    if not Path("data/processed/ml_dataset.parquet").exists():
        assert audit["status"] == "real_data_not_available"


def test_reproducibility_report_no_credentials(tmp_path: Path) -> None:
    report = build_reproducibility_report(config={"seed": 42}, synthetic_demo=True)
    save_reproducibility_report(report, tmp_path / "reproducibility.json")
    text = (tmp_path / "reproducibility.json").read_text(encoding="utf-8")
    assert "credentials_included" in text
    assert "false" in text.lower() or '"credentials_included": false' in text.replace(" ", "")
    assert "password" not in text.lower()


def test_member_a_xai_stub() -> None:
    status = member_a_xai_status()
    assert status["available"] is False


def test_full_demo_evaluation(demo_config, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("src.evaluation.run_evaluation.METRICS_DIR", tmp_path / "metrics")
    monkeypatch.setattr("src.evaluation.run_evaluation.BASELINES_DIR", tmp_path / "baselines")
    monkeypatch.setattr("src.baselines.run_baselines.METRICS_DIR", tmp_path / "metrics")
    monkeypatch.setattr("src.baselines.run_baselines.BASELINES_DIR", tmp_path / "baselines")
    monkeypatch.setattr("src.evaluation.results_table.METRICS_DIR", tmp_path / "metrics")
    monkeypatch.setattr("src.evaluation.results_table.TABLES_DIR", tmp_path / "results" / "tables")
    monkeypatch.setattr("src.evaluation.data_audit.METRICS_DIR", tmp_path / "metrics")
    monkeypatch.setattr("src.evaluation.reproducibility_report.METRICS_DIR", tmp_path / "metrics")
    monkeypatch.setattr("src.evaluation.export_results.METRICS_DIR", tmp_path / "metrics")
    monkeypatch.setattr("src.evaluation.export_results.RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr("src.evaluation.export_results.TABLES_DIR", tmp_path / "results" / "tables")
    monkeypatch.setattr("src.evaluation.rag_ablation.METRICS_DIR", tmp_path / "metrics")
    monkeypatch.setattr("src.evaluation.run_evaluation.PROJECT_ROOT", tmp_path)
    (tmp_path / "results" / "tables").mkdir(parents=True)

    results = run_demo_evaluation(demo_config)
    assert results["synthetic_demo"] is True
    assert (tmp_path / "metrics" / "model_comparison.csv").exists()
    assert (tmp_path / "metrics" / "xai_metrics.json").exists()
    assert (tmp_path / "metrics" / "data_audit.json").exists()
    assert (tmp_path / "metrics" / "reproducibility.json").exists()
