"""Evaluation protocol for the frequency-filtered 363-class formulation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.data.frequent363_ml_dataset import (
    prepare_frequent363_multiclass_frame,
    split_frame,
)
from src.data.final_ml_dataset import TypeLabelEncoder
from src.evaluation.multiclass import compute_multiclass_metrics
from src.evaluation.multiclass_protocol import (
    LEVEL_DEFINITIONS,
    PRIMARY_METRICS,
    SUPPLEMENTARY_METRICS,
    deduplicate_pair_type,
)

SLICE_NAME = "full_363"
SLICE_DEFINITION = (
    "All rows in the frequent363 dataset (types with >= 20 train rows; no 'other' class)."
)

PRIMARY_LEVEL = "pair_type_dedup"
SUPPLEMENTARY_LEVEL = "row"

METRIC_KEYS = (
    "top_1_accuracy",
    "top_3_accuracy",
    "top_5_accuracy",
    "macro_f1",
    "micro_f1",
    "balanced_accuracy",
)


def _format_metrics(raw: dict[str, Any]) -> dict[str, Any]:
    return {
        "top_1_accuracy": raw.get("accuracy"),
        "top_3_accuracy": raw.get("top_3_accuracy"),
        "top_5_accuracy": raw.get("top_5_accuracy"),
        "macro_f1": raw.get("macro_f1"),
        "micro_f1": raw.get("micro_f1"),
        "balanced_accuracy": raw.get("balanced_accuracy"),
    }


def _evaluate_level(
    frame: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None,
    n_classes: int,
    level: str,
) -> dict[str, Any]:
    raw = compute_multiclass_metrics(
        y_true,
        y_pred,
        y_prob,
        labels=list(range(n_classes)),
        top_k=(3, 5),
    )
    metrics = _format_metrics(raw)
    sample_key = "n_samples" if level == "row" else "n_pair_types"
    return {
        "level": level,
        "level_definition": LEVEL_DEFINITIONS[level],  # type: ignore[arg-type]
        sample_key: int(raw.get("n_samples", len(frame))),
        "n_classes": int(frame["type"].nunique()) if len(frame) else 0,
        "metrics": metrics,
    }


def describe_frequent363_slices(frame: pd.DataFrame) -> dict[str, Any]:
    """Row and pair-type counts per split for the frequent363 dataset."""
    enriched, encoder, _cache = prepare_frequent363_multiclass_frame(frame)
    report: dict[str, Any] = {
        "formulation": "frequent363",
        "encoder_n_classes": encoder.n_classes,
        "slice": SLICE_NAME,
        "slice_definition": SLICE_DEFINITION,
        "splits": {},
    }
    for split in ("train", "val", "test"):
        subset = enriched.loc[enriched["split"] == split].reset_index(drop=True)
        dedup = subset.drop_duplicates(subset=["pair_key", "type"], keep="first")
        report["splits"][split] = {
            "row_level": {"n_rows": int(len(subset)), "n_classes": int(subset["type"].nunique())},
            "pair_type_dedup_level": {
                "n_pair_types": int(len(dedup)),
                "n_classes": int(dedup["type"].nunique()) if len(dedup) else 0,
            },
        }
    return report


def evaluate_frequent363_protocol(
    frame: pd.DataFrame,
    predictions_by_split: dict[str, pd.DataFrame],
    label_encoder: TypeLabelEncoder,
) -> dict[str, Any]:
    """Evaluate val/test predictions under the frequent363 protocol."""
    results: dict[str, Any] = {
        "formulation": "frequent363",
        "encoder_n_classes": label_encoder.n_classes,
        "slice": SLICE_NAME,
        "slice_definition": SLICE_DEFINITION,
        "primary_level": PRIMARY_LEVEL,
        "primary_metrics": list(PRIMARY_METRICS),
        "supplementary_metrics": list(SUPPLEMENTARY_METRICS),
        "splits": {},
    }
    required_cols = {"pair_key", "type", "split", "label_index", "y_pred"}

    for split in ("val", "test"):
        payload = predictions_by_split[split]
        eval_frame = payload.copy()
        missing = required_cols - set(eval_frame.columns)
        if missing:
            raise ValueError(f"Split {split} missing columns: {sorted(missing)}")

        y_pred = np.asarray(eval_frame["y_pred"], dtype=int)
        if "y_prob" in eval_frame.columns:
            first = eval_frame["y_prob"].iloc[0]
            if isinstance(first, (list, np.ndarray)):
                y_prob = np.vstack([np.asarray(row, dtype=float) for row in eval_frame["y_prob"]])
            else:
                y_prob = eval_frame["y_prob"].to_numpy(dtype=float)
        else:
            y_prob = None

        y_true = eval_frame["label_index"].astype(int).to_numpy()
        split_report = {
            "row_level": _evaluate_level(eval_frame, y_true, y_pred, y_prob, label_encoder.n_classes, "row"),
        }
        dedup_frame, dedup_true, dedup_pred, dedup_prob = deduplicate_pair_type(
            eval_frame, y_true, y_pred, y_prob
        )
        split_report["pair_type_dedup_level"] = _evaluate_level(
            dedup_frame,
            dedup_true,
            dedup_pred,
            dedup_prob,
            label_encoder.n_classes,
            "pair_type_dedup",
        )
        results["splits"][split] = split_report

    results["primary_reporting"] = {
        "primary": {
            "splits": ["val", "test"],
            "level": PRIMARY_LEVEL,
            "metrics": list(METRIC_KEYS),
        },
        "supplementary": {
            "level": SUPPLEMENTARY_LEVEL,
            "metrics": list(METRIC_KEYS),
        },
    }
    return results
