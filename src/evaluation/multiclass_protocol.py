"""Evaluation protocol for 955-class DDI interaction-type baselines.

Defines evaluation slices, aggregation levels, and metric reporting conventions
for ``data/processed/final_mimic_twosides_ml.csv``. Does not modify the dataset.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

from src.data.final_ml_dataset import TypeLabelEncoder, load_final_ml_dataset, prepare_multiclass_frame
from src.evaluation.multiclass import compute_multiclass_metrics

SliceName = Literal["full_955", "seen_class", "frequent_ge_20"]
EvalLevel = Literal["row", "pair_type_dedup"]
EvalSplit = Literal["val", "test"]

SLICE_DEFINITIONS: dict[SliceName, str] = {
    "full_955": "All evaluable rows with encoded labels (955 train-fit classes; unseen types excluded).",
    "seen_class": "Rows whose TWOSIDES type appears in the training split (label_index >= 0).",
    "frequent_ge_20": "Rows whose type has >= 20 training examples.",
}

LEVEL_DEFINITIONS: dict[EvalLevel, str] = {
    "row": "One prediction per dataset row (admission-level).",
    "pair_type_dedup": "One prediction per unique (pair_key, type); first row kept (features identical).",
}

PRIMARY_METRICS = ("top_3_accuracy", "top_5_accuracy", "macro_f1")
SUPPLEMENTARY_METRICS = ("top_1_accuracy", "micro_f1", "balanced_accuracy")


@dataclass(frozen=True)
class EvaluationSlice:
    name: SliceName
    level: EvalLevel
    split: EvalSplit

    @property
    def slice_id(self) -> str:
        return f"{self.split}_{self.level}_{self.name}"


def train_type_counts(frame: pd.DataFrame) -> pd.Series:
    """Return training-row counts keyed by TWOSIDES ``type``."""
    train = frame.loc[frame["split"] == "train"]
    return train.groupby("type").size()


def frequent_train_types(frame: pd.DataFrame, min_examples: int = 20) -> set[int]:
    counts = train_type_counts(frame)
    return {int(type_id) for type_id, count in counts.items() if int(count) >= min_examples}


def apply_slice_mask(
    eval_frame: pd.DataFrame,
    slice_name: SliceName,
    min_train_examples: int = 20,
    train_counts: pd.Series | None = None,
) -> pd.Series:
    """Boolean mask selecting rows for a named evaluation slice."""
    seen = eval_frame["label_index"].astype(int) >= 0
    if slice_name in ("full_955", "seen_class"):
        return seen
    if slice_name == "frequent_ge_20":
        if train_counts is None:
            raise ValueError("train_counts is required for the frequent_ge_20 slice.")
        frequent = {int(type_id) for type_id, count in train_counts.items() if int(count) >= min_train_examples}
        return eval_frame["type"].astype(int).isin(frequent) & seen
    raise ValueError(f"Unknown slice: {slice_name}")


def deduplicate_pair_type(
    frame: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray | None]:
    """Keep the first row for each (pair_key, type) combination."""
    if len(frame) != len(y_true) or len(frame) != len(y_pred):
        raise ValueError("Predictions must align with evaluation frame rows.")
    working = frame.copy()
    working["_y_true"] = np.asarray(y_true, dtype=int)
    working["_y_pred"] = np.asarray(y_pred, dtype=int)
    if y_prob is not None:
        working["_row_idx"] = np.arange(len(working))
    dedup = working.drop_duplicates(subset=["pair_key", "type"], keep="first").reset_index(drop=True)
    out_true = dedup["_y_true"].to_numpy(dtype=int)
    out_pred = dedup["_y_pred"].to_numpy(dtype=int)
    out_prob = None
    if y_prob is not None:
        row_idx = dedup["_row_idx"].to_numpy(dtype=int)
        out_prob = np.asarray(y_prob, dtype=float)[row_idx]
    return (
        dedup.drop(columns=["_y_true", "_y_pred"] + (["_row_idx"] if y_prob is not None else [])),
        out_true,
        out_pred,
        out_prob,
    )


def _format_metrics(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalize metric names for the protocol (top-1 = accuracy)."""
    out = {
        "n_samples": raw.get("n_samples"),
        "n_seen_label_samples": raw.get("n_seen_label_samples"),
        "n_unseen_label_samples": raw.get("n_unseen_label_samples"),
        "top_1_accuracy": raw.get("accuracy"),
        "top_3_accuracy": raw.get("top_3_accuracy"),
        "top_5_accuracy": raw.get("top_5_accuracy"),
        "macro_f1": raw.get("macro_f1"),
        "micro_f1": raw.get("micro_f1"),
        "balanced_accuracy": raw.get("balanced_accuracy"),
    }
    return out


def compute_slice_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None,
    n_classes: int,
) -> dict[str, Any]:
    raw = compute_multiclass_metrics(
        y_true,
        y_pred,
        y_prob,
        labels=list(range(n_classes)),
        top_k=(3, 5),
    )
    return _format_metrics(raw)


def describe_evaluation_slices(
    frame: pd.DataFrame,
    min_train_examples: int = 20,
) -> dict[str, Any]:
    """Report class and row counts for each protocol slice (no predictions required)."""
    enriched, encoder, _cache = prepare_multiclass_frame(frame)
    train_counts = train_type_counts(enriched)
    frequent = frequent_train_types(enriched, min_examples=min_train_examples)
    report: dict[str, Any] = {
        "dataset_rows": int(len(frame)),
        "encoder_n_classes": encoder.n_classes,
        "min_train_examples_for_frequent_slice": min_train_examples,
        "frequent_train_types": int(len(frequent)),
        "slices": {},
    }

    for split in ("val", "test"):
        split_frame = enriched.loc[enriched["split"] == split].reset_index(drop=True)
        evaluable = split_frame.loc[split_frame["label_index"] >= 0].reset_index(drop=True)
        for slice_name in ("full_955", "seen_class", "frequent_ge_20"):
            mask = apply_slice_mask(
                evaluable,
                slice_name,
                min_train_examples=min_train_examples,
                train_counts=train_counts,
            )
            subset = evaluable.loc[mask].reset_index(drop=True)
            dedup = subset.drop_duplicates(subset=["pair_key", "type"], keep="first")
            excluded_unseen = int((split_frame["label_index"] < 0).sum()) if slice_name != "frequent_ge_20" else None
            report["slices"][f"{split}_{slice_name}"] = {
                "row_level": {
                    "n_rows": int(len(subset)),
                    "n_classes": int(subset["type"].nunique()) if len(subset) else 0,
                    "excluded_unseen_type_rows": excluded_unseen,
                },
                "pair_type_dedup_level": {
                    "n_pair_types": int(len(dedup)),
                    "n_classes": int(dedup["type"].nunique()) if len(dedup) else 0,
                },
            }
    return report


def evaluate_multiclass_protocol(
    frame: pd.DataFrame,
    predictions_by_split: dict[str, pd.DataFrame],
    label_encoder: TypeLabelEncoder,
    min_train_examples: int = 20,
) -> dict[str, Any]:
    """Evaluate model predictions under the full protocol.

    ``predictions_by_split`` maps ``val``/``test`` to frames aligned with the
    corresponding split rows of ``frame`` after ``prepare_multiclass_frame``,
    containing columns: ``label_index``, ``y_pred``, and optionally ``y_prob``
    (as list/array per row or a 2-D numpy array stored column-wise — see below).

    Alternatively pass numpy arrays via ``predictions_by_split[split]`` as a dict with keys
    ``label_index``, ``y_pred``, ``y_prob``, ``eval_frame`` (aligned metadata rows).
    """
    required_cols = {"pair_key", "type", "split", "label_index"}
    results: dict[str, Any] = {
        "encoder_n_classes": label_encoder.n_classes,
        "min_train_examples_for_frequent_slice": min_train_examples,
        "primary_metrics": list(PRIMARY_METRICS),
        "supplementary_metrics": list(SUPPLEMENTARY_METRICS),
        "splits": {},
    }

    for split in ("val", "test"):
        payload = predictions_by_split[split]
        if isinstance(payload, pd.DataFrame):
            eval_frame = payload.copy()
            y_true = eval_frame["label_index"].astype(int).to_numpy()
            y_pred = np.asarray(eval_frame["y_pred"], dtype=int)
            y_prob = _extract_probabilities(eval_frame, label_encoder.n_classes)
        else:
            eval_frame = payload["eval_frame"].copy()
            y_true = np.asarray(payload["label_index"], dtype=int)
            y_pred = np.asarray(payload["y_pred"], dtype=int)
            y_prob = payload.get("y_prob")

        missing = required_cols - set(eval_frame.columns)
        if missing:
            raise ValueError(f"Split {split} eval_frame missing columns: {sorted(missing)}")

        base_eval = eval_frame.loc[eval_frame["label_index"] >= 0].reset_index(drop=True)
        base_true = base_eval["label_index"].astype(int).to_numpy()
        base_pred = y_pred[eval_frame["label_index"] >= 0]
        base_prob = y_prob[eval_frame["label_index"] >= 0] if y_prob is not None else None

        split_results: dict[str, Any] = {}
        train_counts = train_type_counts(frame if "split" in frame.columns else eval_frame)
        for slice_name in ("full_955", "seen_class", "frequent_ge_20"):
            mask = apply_slice_mask(
                base_eval,
                slice_name,
                min_train_examples=min_train_examples,
                train_counts=train_counts,
            )
            slice_frame = base_eval.loc[mask].reset_index(drop=True)
            slice_true = slice_frame["label_index"].astype(int).to_numpy()
            slice_pred = base_pred[mask.to_numpy()]
            slice_prob = base_prob[mask.to_numpy()] if base_prob is not None else None

            slice_report: dict[str, Any] = {
                "definition": SLICE_DEFINITIONS[slice_name],
                "row_level": _evaluate_level(
                    slice_frame,
                    slice_true,
                    slice_pred,
                    slice_prob,
                    label_encoder.n_classes,
                    level="row",
                ),
                "pair_type_dedup_level": None,
            }
            dedup_frame, dedup_true, dedup_pred, dedup_prob = deduplicate_pair_type(
                slice_frame, slice_true, slice_pred, slice_prob
            )
            slice_report["pair_type_dedup_level"] = _evaluate_level(
                dedup_frame,
                dedup_true,
                dedup_pred,
                dedup_prob,
                label_encoder.n_classes,
                level="pair_type_dedup",
            )
            split_results[slice_name] = slice_report

        results["splits"][split] = split_results

    results["primary_reporting"] = _primary_reporting_guide()
    return results


def _evaluate_level(
    frame: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None,
    n_classes: int,
    level: EvalLevel,
) -> dict[str, Any]:
    metrics = compute_slice_metrics(y_true, y_pred, y_prob, n_classes)
    sample_key = "n_samples" if level == "row" else "n_pair_types"
    return {
        "level": level,
        "level_definition": LEVEL_DEFINITIONS[level],
        sample_key: metrics.pop("n_samples"),
        "n_classes": int(frame["type"].nunique()) if len(frame) else 0,
        "metrics": metrics,
    }


def _extract_probabilities(frame: pd.DataFrame, n_classes: int) -> np.ndarray | None:
    if "y_prob" not in frame.columns:
        return None
    first = frame["y_prob"].iloc[0]
    if isinstance(first, (list, np.ndarray)):
        return np.vstack([np.asarray(row, dtype=float) for row in frame["y_prob"]])
    if frame["y_prob"].ndim == 2:
        return frame["y_prob"].to_numpy(dtype=float)
    raise ValueError("y_prob column must contain per-row probability vectors.")


def _primary_reporting_guide() -> dict[str, Any]:
    return {
        "primary": {
            "description": "Headline metrics for model comparison and reporting.",
            "recommended_slices": [
                "seen_class + pair_type_dedup",
                "frequent_ge_20 + pair_type_dedup",
            ],
            "metrics": list(PRIMARY_METRICS),
            "splits": ["val", "test"],
        },
        "supplementary": {
            "description": "Diagnostic metrics; not sufficient alone for 955-class conclusions.",
            "recommended_slices": [
                "full_955 + row",
                "seen_class + row",
            ],
            "metrics": list(SUPPLEMENTARY_METRICS),
            "notes": [
                "top_1_accuracy on 955 classes is near random baseline (~0.1%).",
                "micro_f1 is dominated by frequent interaction types.",
                "Row-level metrics count repeated admissions multiple times.",
            ],
        },
    }


def random_baseline_predictions(
    eval_frame: pd.DataFrame,
    n_classes: int,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate reproducible random predictions for protocol sanity checks."""
    rng = np.random.default_rng(seed)
    n = len(eval_frame)
    y_pred = rng.integers(0, n_classes, size=n)
    y_prob = rng.random((n, n_classes))
    y_prob /= y_prob.sum(axis=1, keepdims=True)
    return y_pred, y_prob
