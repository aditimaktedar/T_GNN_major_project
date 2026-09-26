"""Canonical experiment-result store.

``results/metrics/MASTER_RESULTS.json`` is the single source of truth for
multiclass model metrics. Training and evaluation upsert one
``formulations[formulation].models[model]`` entry per run instead of writing
a new summary file.
"""

from __future__ import annotations

import argparse
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.data.config import MASTER_RESULTS_PATH, METRICS_DIR, PROJECT_ROOT
SCHEMA_VERSION = 1

DISPLAY_NAMES = {
    "logistic_regression": "Multiclass Logistic Regression",
    "static_gat": "Multiclass Static GAT",
    "molecular_gnn": "Atom-level Molecular GNN (GIN)",
}

# Legacy per-run summaries. Kept on disk as archive; ingest once into MASTER_RESULTS.
LEGACY_SOURCES: list[dict[str, Any]] = [
    {
        "formulation": "955",
        "model": "logistic_regression",
        "protocol_path": METRICS_DIR / "final_mimic_twosides_multiclass_logistic_regression_protocol.json",
        "checkpoint": PROJECT_ROOT / "results" / "baselines" / "final_mimic_twosides_multiclass_logistic_regression.joblib",
        "n_classes": 955,
        "dataset_path": "data/processed/final_mimic_twosides_ml.csv",
        "experiment_prefix": "final_mimic_twosides",
    },
    {
        "formulation": "955",
        "model": "static_gat",
        "summary_path": METRICS_DIR / "final_mimic_twosides_multiclass_static_gat_summary.json",
        "protocol_path": METRICS_DIR / "final_mimic_twosides_multiclass_static_gat_protocol.json",
        "n_classes": 955,
        "dataset_path": "data/processed/final_mimic_twosides_ml.csv",
        "experiment_prefix": "final_mimic_twosides",
    },
    {
        "formulation": "955",
        "model": "molecular_gnn",
        "summary_path": METRICS_DIR / "final_mimic_twosides_multiclass_molecular_gnn_summary.json",
        "protocol_path": METRICS_DIR / "final_mimic_twosides_multiclass_molecular_gnn_protocol.json",
        "n_classes": 955,
        "dataset_path": "data/processed/final_mimic_twosides_ml.csv",
        "experiment_prefix": "final_mimic_twosides",
    },
    {
        "formulation": "frequent363",
        "model": "logistic_regression",
        "run_summary_path": METRICS_DIR / "final_mimic_twosides_frequent363_multiclass_run_summary.json",
        "run_summary_key": "logistic_regression",
        "n_classes": 363,
        "dataset_path": "data/processed/final_mimic_twosides_ml_frequent363.csv",
        "experiment_prefix": "final_mimic_twosides_frequent363",
    },
    {
        "formulation": "frequent363",
        "model": "static_gat",
        "summary_path": METRICS_DIR / "final_mimic_twosides_frequent363_multiclass_static_gat_summary.json",
        "protocol_path": METRICS_DIR / "final_mimic_twosides_frequent363_multiclass_static_gat_protocol.json",
        "run_summary_path": METRICS_DIR / "final_mimic_twosides_frequent363_multiclass_run_summary.json",
        "n_classes": 363,
        "dataset_path": "data/processed/final_mimic_twosides_ml_frequent363.csv",
        "experiment_prefix": "final_mimic_twosides_frequent363",
    },
    {
        "formulation": "frequent363",
        "model": "molecular_gnn",
        "summary_path": METRICS_DIR / "final_mimic_twosides_frequent363_multiclass_molecular_gnn_summary.json",
        "protocol_path": METRICS_DIR / "final_mimic_twosides_frequent363_multiclass_molecular_gnn_protocol.json",
        "n_classes": 363,
        "dataset_path": "data/processed/final_mimic_twosides_ml_frequent363.csv",
        "experiment_prefix": "final_mimic_twosides_frequent363",
    },
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _mtime_iso(path: Path) -> str | None:
    if not path.exists():
        return None
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).replace(microsecond=0).isoformat()


def _new_run_id(formulation: str, model: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{formulation}_{model}_{stamp}_{uuid.uuid4().hex[:8]}"


def project_relative(path: Path | str | None) -> str | None:
    if path is None:
        return None
    raw = Path(path)
    try:
        return str(raw.resolve().relative_to(PROJECT_ROOT.resolve()))
    except (ValueError, OSError):
        return str(path)


def empty_master() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "description": (
            "Canonical experiment results for TemporalDDI-GNN Member B multiclass models. "
            "One entry per formulation + model. Reruns overwrite that entry."
        ),
        "source_of_truth": True,
        "updated_at": None,
        "formulations": {},
    }


def load_master(path: Path | None = None) -> dict[str, Any]:
    target = path or MASTER_RESULTS_PATH
    if not target.exists():
        return empty_master()
    payload = json.loads(target.read_text(encoding="utf-8"))
    if "formulations" not in payload:
        raise ValueError(f"Invalid master results file (missing formulations): {target}")
    return payload


def save_master(master: dict[str, Any], path: Path | None = None) -> Path:
    target = path or MASTER_RESULTS_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    master["schema_version"] = SCHEMA_VERSION
    master["source_of_truth"] = True
    master["updated_at"] = _utc_now()
    target.write_text(json.dumps(master, indent=2, default=str) + "\n", encoding="utf-8")
    return target


def get_model_entry(
    master: dict[str, Any],
    formulation: str,
    model: str,
) -> dict[str, Any] | None:
    return master.get("formulations", {}).get(formulation, {}).get("models", {}).get(model)


def _row_level_from_metrics(metrics_block: dict[str, Any] | None) -> dict[str, Any] | None:
    if not metrics_block:
        return None
    formatted = {
        "top_1_accuracy": metrics_block.get("top_1_accuracy", metrics_block.get("accuracy")),
        "top_3_accuracy": metrics_block.get("top_3_accuracy"),
        "top_5_accuracy": metrics_block.get("top_5_accuracy"),
        "macro_f1": metrics_block.get("macro_f1"),
        "micro_f1": metrics_block.get("micro_f1"),
        "balanced_accuracy": metrics_block.get("balanced_accuracy"),
    }
    return {
        "level": "row",
        "n_samples": metrics_block.get("n_samples"),
        "n_seen_label_samples": metrics_block.get("n_seen_label_samples"),
        "n_unseen_label_samples": metrics_block.get("n_unseen_label_samples"),
        "metrics": formatted,
    }


def splits_view(metrics: dict[str, Any] | None, protocol: dict[str, Any] | None) -> dict[str, Any]:
    """Normalized split → level/slice view used for reporting."""
    splits: dict[str, Any] = {}
    for split_name, block in (metrics or {}).items():
        if isinstance(block, dict):
            row = _row_level_from_metrics(block)
            if row is not None:
                splits.setdefault(split_name, {})["row_level"] = row

    for split_name, block in (protocol or {}).get("splits", {}).items():
        if not isinstance(block, dict):
            continue
        dest = splits.setdefault(split_name, {})
        if "row_level" in block:
            dest["row_level"] = block["row_level"]
        if "pair_type_dedup_level" in block:
            dest["pair_type_dedup_level"] = block["pair_type_dedup_level"]
        slice_keys = [
            key
            for key in block
            if key not in {"row_level", "pair_type_dedup_level"}
            and isinstance(block[key], dict)
        ]
        if slice_keys:
            dest["slices"] = {key: block[key] for key in slice_keys}
    return splits


def compact_dataset_report(report: dict[str, Any] | None) -> dict[str, Any] | None:
    if not report:
        return None
    keep = (
        "n_rows",
        "n_train_rows",
        "n_val_rows",
        "n_test_rows",
        "unique_pair_key",
        "unique_types_total",
        "unique_types_train",
        "unique_types_val",
        "unique_types_test",
        "n_classes_train_encoder",
        "pair_split_leakage_pairs",
        "trainable_rows",
        "invalid_fingerprint_rows",
    )
    return {key: report[key] for key in keep if key in report}


def build_model_entry(
    *,
    formulation: str,
    model: str,
    n_classes: int | None,
    dataset_path: str | None,
    experiment_prefix: str | None,
    checkpoint: str | None = None,
    device: str | None = None,
    metrics: dict[str, Any] | None = None,
    evaluation_protocol: dict[str, Any] | None = None,
    dataset_report: dict[str, Any] | None = None,
    final_train_loss: float | None = None,
    final_val_loss: float | None = None,
    environment: dict[str, Any] | None = None,
    source_files: list[str] | None = None,
    extra: dict[str, Any] | None = None,
    updated_at: str | None = None,
) -> dict[str, Any]:
    protocol_n_classes = None
    if evaluation_protocol:
        protocol_n_classes = evaluation_protocol.get("encoder_n_classes")
    entry: dict[str, Any] = {
        "run_id": _new_run_id(formulation, model),
        "updated_at": updated_at or _utc_now(),
        "model": model,
        "display_name": DISPLAY_NAMES.get(model, model),
        "formulation": formulation,
        "n_classes": n_classes if n_classes is not None else protocol_n_classes,
        "dataset_path": dataset_path,
        "experiment_prefix": experiment_prefix,
        "checkpoint": project_relative(checkpoint) if checkpoint else None,
        "device": device,
        "metrics": metrics or {},
        "evaluation_protocol": evaluation_protocol,
        "splits": splits_view(metrics, evaluation_protocol),
        "dataset_report": compact_dataset_report(dataset_report),
        "final_train_loss": final_train_loss,
        "final_val_loss": final_val_loss,
        "source_files": source_files or [],
    }
    if environment:
        entry["environment"] = environment
    if extra:
        entry["extra"] = extra
    return entry


def upsert_model_result(
    formulation: str,
    model: str,
    entry: dict[str, Any],
    *,
    path: Path | None = None,
    keep_run_history: bool = False,
) -> dict[str, Any]:
    """Insert or overwrite ``formulations[formulation].models[model]``.

    Duplicate (formulation, model) keys are not created. The previous ``run_id``
    is recorded as ``supersedes_run_id``. Set ``keep_run_history=True`` only when
    a compact prior-run index is requested.
    """
    target = path or MASTER_RESULTS_PATH
    master = load_master(target)
    formulations = master.setdefault("formulations", {})
    bucket = formulations.setdefault(
        formulation,
        {"formulation": formulation, "models": {}},
    )
    models = bucket.setdefault("models", {})
    previous = models.get(model)
    if previous:
        entry["supersedes_run_id"] = previous.get("run_id")
        if keep_run_history:
            history = list(previous.get("run_history") or [])
            history.append(
                {
                    "run_id": previous.get("run_id"),
                    "updated_at": previous.get("updated_at"),
                    "checkpoint": previous.get("checkpoint"),
                }
            )
            entry["run_history"] = history
    models[model] = entry
    if entry.get("n_classes") is not None:
        bucket["n_classes"] = entry["n_classes"]
    if entry.get("dataset_path"):
        bucket["dataset_path"] = entry["dataset_path"]
    save_master(master, target)
    return master


def record_training_result(
    *,
    formulation: str,
    model: str,
    payload: dict[str, Any],
    path: Path | None = None,
) -> dict[str, Any]:
    """Write one trained model's metrics into MASTER_RESULTS.json."""
    nested = payload.get(model) if isinstance(payload.get(model), dict) else None
    block = nested if nested is not None else payload
    n_classes = payload.get("n_classes", block.get("n_classes"))
    if n_classes is None and block.get("evaluation_protocol"):
        n_classes = block["evaluation_protocol"].get("encoder_n_classes")
    entry = build_model_entry(
        formulation=formulation,
        model=model,
        n_classes=n_classes,
        dataset_path=project_relative(payload.get("dataset_path") or block.get("dataset_path")),
        experiment_prefix=payload.get("experiment_prefix") or block.get("experiment_prefix"),
        checkpoint=block.get("checkpoint"),
        device=block.get("device"),
        metrics=block.get("metrics"),
        evaluation_protocol=block.get("evaluation_protocol"),
        dataset_report=payload.get("dataset_report") or block.get("dataset_report"),
        final_train_loss=block.get("final_train_loss"),
        final_val_loss=block.get("final_val_loss"),
        environment=payload.get("environment") or block.get("environment"),
        source_files=["training_run"],
    )
    return upsert_model_result(formulation, model, entry, path=path)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def ingest_legacy_sources(path: Path | None = None) -> dict[str, Any]:
    """Merge archived per-model summary/protocol files into MASTER_RESULTS.json."""
    target = path or MASTER_RESULTS_PATH
    ingested = []
    for spec in LEGACY_SOURCES:
        formulation = spec["formulation"]
        model = spec["model"]
        source_files: list[str] = []
        summary: dict[str, Any] = {}
        protocol: dict[str, Any] | None = None
        timestamps: list[str] = []

        summary_path: Path | None = spec.get("summary_path")
        if summary_path is not None and summary_path.exists():
            summary = _load_json(summary_path)
            source_files.append(project_relative(summary_path) or str(summary_path))
            stamp = _mtime_iso(summary_path)
            if stamp:
                timestamps.append(stamp)

        run_summary_path: Path | None = spec.get("run_summary_path")
        run_key = spec.get("run_summary_key", model)
        if run_summary_path is not None and run_summary_path.exists():
            run_payload = _load_json(run_summary_path)
            source_files.append(project_relative(run_summary_path) or str(run_summary_path))
            stamp = _mtime_iso(run_summary_path)
            if stamp:
                timestamps.append(stamp)
            if not summary and isinstance(run_payload.get(run_key), dict):
                summary = dict(run_payload[run_key])
                summary.setdefault("dataset_path", run_payload.get("dataset_path"))
                summary.setdefault("n_classes", run_payload.get("n_classes"))
                summary.setdefault("experiment_prefix", run_payload.get("experiment_prefix"))
                summary.setdefault("dataset_report", run_payload.get("dataset_report"))

        protocol_path: Path | None = spec.get("protocol_path")
        if protocol_path is not None and protocol_path.exists():
            protocol = _load_json(protocol_path)
            source_files.append(project_relative(protocol_path) or str(protocol_path))
            stamp = _mtime_iso(protocol_path)
            if stamp:
                timestamps.append(stamp)
        elif summary.get("evaluation_protocol"):
            protocol = summary["evaluation_protocol"]
        elif model == "logistic_regression":
            candidate_ckpt = summary.get("checkpoint") or spec.get("checkpoint")
            if candidate_ckpt and Path(candidate_ckpt).exists():
                try:
                    from src.baselines.multiclass_logistic_regression import MulticlassLogisticRegressionBaseline
                    from src.baselines.run_multiclass_baselines import _evaluate_with_protocol
                    from src.data.dataset_paths import load_multiclass_dataset, prepare_multiclass_for_formulation

                    frame = load_multiclass_dataset(formulation)
                    enriched, encoder, _cache = prepare_multiclass_for_formulation(frame, formulation)
                    ckpt_model = MulticlassLogisticRegressionBaseline.load(candidate_ckpt)
                    protocol = _evaluate_with_protocol(
                        enriched, ckpt_model, encoder, batch_size=32, formulation=formulation
                    )
                except Exception:
                    protocol = None

        checkpoint = summary.get("checkpoint") or spec.get("checkpoint")
        if checkpoint is not None:
            checkpoint = str(checkpoint)

        if not summary and protocol is None:
            continue

        entry = build_model_entry(
            formulation=formulation,
            model=model,
            n_classes=summary.get("n_classes") or spec.get("n_classes"),
            dataset_path=project_relative(summary.get("dataset_path")) or spec.get("dataset_path"),
            experiment_prefix=summary.get("experiment_prefix") or spec.get("experiment_prefix"),
            checkpoint=checkpoint,
            device=summary.get("device"),
            metrics=summary.get("metrics"),
            evaluation_protocol=protocol,
            dataset_report=summary.get("dataset_report"),
            final_train_loss=summary.get("final_train_loss"),
            final_val_loss=summary.get("final_val_loss"),
            environment=summary.get("environment"),
            source_files=source_files,
            updated_at=max(timestamps) if timestamps else _utc_now(),
        )
        upsert_model_result(formulation, model, entry, path=target)
        ingested.append(f"{formulation}/{model}")

    master = load_master(target)
    master["legacy_ingest"] = {
        "ingested_entries": ingested,
        "note": (
            "Legacy per-run summary and protocol JSON files were copied into this "
            "file and left on disk as archive. New training writes only MASTER_RESULTS.json."
        ),
    }
    save_master(master, target)
    return master


def load_protocol(formulation: str, model: str, path: Path | None = None) -> dict[str, Any] | None:
    entry = get_model_entry(load_master(path), formulation, model)
    if not entry:
        return None
    return entry.get("evaluation_protocol")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Canonical MASTER_RESULTS.json helpers")
    parser.add_argument(
        "--ingest-existing",
        action="store_true",
        help="Merge archived summary/protocol files into MASTER_RESULTS.json.",
    )
    parser.add_argument("--output", default=str(MASTER_RESULTS_PATH))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out = Path(args.output)
    if args.ingest_existing:
        master = ingest_legacy_sources(out)
        counts = {
            formulation: sorted(bucket.get("models", {}).keys())
            for formulation, bucket in master.get("formulations", {}).items()
        }
        print(json.dumps({"path": str(out), "formulations": counts}, indent=2))
        return 0
    parser = build_parser()
    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
