"""Tests for the canonical MASTER_RESULTS.json store."""

from __future__ import annotations

import json
from pathlib import Path

from src.evaluation.master_results import (
    build_model_entry,
    ingest_legacy_sources,
    load_master,
    load_protocol,
    record_training_result,
    upsert_model_result,
)


def test_upsert_overwrites_same_formulation_and_model(tmp_path: Path) -> None:
    master_path = tmp_path / "MASTER_RESULTS.json"
    first = build_model_entry(
        formulation="frequent363",
        model="logistic_regression",
        n_classes=363,
        dataset_path="data/processed/final_mimic_twosides_ml_frequent363.csv",
        experiment_prefix="demo",
        metrics={"test": {"accuracy": 0.1, "n_samples": 10}},
    )
    upsert_model_result("frequent363", "logistic_regression", first, path=master_path)
    second = build_model_entry(
        formulation="frequent363",
        model="logistic_regression",
        n_classes=363,
        dataset_path="data/processed/final_mimic_twosides_ml_frequent363.csv",
        experiment_prefix="demo",
        metrics={"test": {"accuracy": 0.2, "n_samples": 10}},
    )
    upsert_model_result("frequent363", "logistic_regression", second, path=master_path)

    master = load_master(master_path)
    models = master["formulations"]["frequent363"]["models"]
    assert list(models) == ["logistic_regression"]
    entry = models["logistic_regression"]
    assert entry["metrics"]["test"]["accuracy"] == 0.2
    assert entry["supersedes_run_id"] == first["run_id"]
    assert entry["run_id"] != first["run_id"]


def test_record_training_result_keeps_separate_models(tmp_path: Path) -> None:
    master_path = tmp_path / "MASTER_RESULTS.json"
    payload = {
        "formulation": "955",
        "n_classes": 955,
        "dataset_path": "data/processed/final_mimic_twosides_ml.csv",
        "experiment_prefix": "final_mimic_twosides",
        "logistic_regression": {
            "checkpoint": "results/baselines/lr.joblib",
            "metrics": {"test": {"accuracy": 0.01, "n_samples": 5}},
            "evaluation_protocol": {
                "encoder_n_classes": 955,
                "splits": {
                    "test": {
                        "seen_class": {
                            "pair_type_dedup_level": {
                                "n_pair_types": 5,
                                "metrics": {"top_1_accuracy": 0.01},
                            }
                        }
                    }
                },
            },
        },
    }
    record_training_result(
        formulation="955",
        model="logistic_regression",
        payload=payload,
        path=master_path,
    )
    payload["static_gat"] = {
        "checkpoint": "results/baselines/gat.pt",
        "metrics": {"test": {"accuracy": 0.02, "n_samples": 5}},
        "evaluation_protocol": {"encoder_n_classes": 955, "splits": {}},
    }
    record_training_result(
        formulation="955",
        model="static_gat",
        payload=payload,
        path=master_path,
    )
    master = load_master(master_path)
    models = master["formulations"]["955"]["models"]
    assert set(models) == {"logistic_regression", "static_gat"}
    protocol = load_protocol("955", "logistic_regression", path=master_path)
    assert protocol["encoder_n_classes"] == 955


def test_ingest_legacy_sources_from_archive_files(tmp_path: Path, monkeypatch) -> None:
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir()
    protocol = {
        "encoder_n_classes": 955,
        "splits": {
            "test": {
                "seen_class": {
                    "pair_type_dedup_level": {
                        "n_pair_types": 3,
                        "metrics": {"top_1_accuracy": 0.001},
                    }
                }
            }
        },
    }
    protocol_path = metrics_dir / "lr_protocol.json"
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")

    summary = {
        "n_classes": 363,
        "dataset_path": "data/processed/final_mimic_twosides_ml_frequent363.csv",
        "experiment_prefix": "x",
        "checkpoint": "results/baselines/gnn.pt",
        "metrics": {"test": {"accuracy": 0.002, "n_samples": 8}},
        "evaluation_protocol": {
            "formulation": "frequent363",
            "encoder_n_classes": 363,
            "splits": {
                "test": {
                    "pair_type_dedup_level": {
                        "n_pair_types": 4,
                        "metrics": {"top_1_accuracy": 0.002},
                    }
                }
            },
        },
    }
    summary_path = metrics_dir / "gnn_summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    monkeypatch.setattr(
        "src.evaluation.master_results.LEGACY_SOURCES",
        [
            {
                "formulation": "955",
                "model": "logistic_regression",
                "protocol_path": protocol_path,
                "n_classes": 955,
                "dataset_path": "data/processed/final_mimic_twosides_ml.csv",
                "experiment_prefix": "final_mimic_twosides",
            },
            {
                "formulation": "frequent363",
                "model": "molecular_gnn",
                "summary_path": summary_path,
                "n_classes": 363,
                "dataset_path": "data/processed/final_mimic_twosides_ml_frequent363.csv",
                "experiment_prefix": "x",
            },
        ],
    )
    out = tmp_path / "MASTER_RESULTS.json"
    master = ingest_legacy_sources(out)
    assert "logistic_regression" in master["formulations"]["955"]["models"]
    assert "molecular_gnn" in master["formulations"]["frequent363"]["models"]
    gnn = master["formulations"]["frequent363"]["models"]["molecular_gnn"]
    assert gnn["metrics"]["test"]["accuracy"] == 0.002
    assert gnn["splits"]["test"]["pair_type_dedup_level"]["n_pair_types"] == 4


def test_canonical_master_results_file_structure() -> None:
    from src.data.config import MASTER_RESULTS_PATH

    if not MASTER_RESULTS_PATH.exists():
        pytest.skip("Canonical MASTER_RESULTS.json not yet generated on disk.")

    master = load_master(MASTER_RESULTS_PATH)
    assert master["schema_version"] == 1
    assert master["source_of_truth"] is True
    assert set(master["formulations"].keys()) == {"955", "frequent363"}

    for formulation, expected_classes in (("955", 955), ("frequent363", 363)):
        bucket = master["formulations"][formulation]
        assert bucket["n_classes"] == expected_classes
        assert set(bucket["models"].keys()) == {"logistic_regression", "static_gat", "molecular_gnn"}
        for model_name, model_data in bucket["models"].items():
            assert model_data["run_id"]
            assert model_data["checkpoint"]
            assert "test" in model_data["splits"]
            test_split = model_data["splits"]["test"]
            assert "row_level" in test_split or "slices" in test_split or "pair_type_dedup_level" in test_split

