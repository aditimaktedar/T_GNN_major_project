"""Resolve dataset paths and preparation helpers for multiclass formulations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.data.config import (
    FINAL_MIMIC_TWOSIDES_ML_FREQUENT363_PATH,
    FINAL_MIMIC_TWOSIDES_ML_PATH,
)
from src.data.exceptions import ConfigurationError
from src.data.final_ml_dataset import (
    FingerprintCache,
    TypeLabelEncoder,
    load_final_ml_dataset,
    prepare_multiclass_frame,
)
from src.data.frequent363_ml_dataset import (
    load_frequent363_ml_dataset,
    prepare_frequent363_multiclass_frame,
)

FORMULATION_955_ALIASES = frozenset({"955", "full", "955_class"})
FORMULATION_FREQUENT363_ALIASES = frozenset({"frequent363", "363", "frequent"})


def normalize_formulation(formulation: str) -> str:
    if formulation in FORMULATION_955_ALIASES:
        return "955"
    if formulation in FORMULATION_FREQUENT363_ALIASES:
        return "frequent363"
    raise ConfigurationError(
        f"Unknown formulation {formulation!r}. Use '955' or 'frequent363'."
    )


def resolve_formulation_path(formulation: str) -> Path:
    normalized = normalize_formulation(formulation)
    if normalized == "955":
        return FINAL_MIMIC_TWOSIDES_ML_PATH
    return FINAL_MIMIC_TWOSIDES_ML_FREQUENT363_PATH


def default_experiment_prefix(formulation: str) -> str:
    if normalize_formulation(formulation) == "frequent363":
        return "final_mimic_twosides_frequent363"
    return "final_mimic_twosides"


def load_multiclass_dataset(formulation: str, path: Path | str | None = None) -> pd.DataFrame:
    dataset_path = Path(path) if path is not None else resolve_formulation_path(formulation)
    if normalize_formulation(formulation) == "frequent363" or dataset_path == FINAL_MIMIC_TWOSIDES_ML_FREQUENT363_PATH:
        return load_frequent363_ml_dataset(dataset_path)
    return load_final_ml_dataset(dataset_path)


def prepare_multiclass_for_formulation(
    frame: pd.DataFrame,
    formulation: str,
    label_encoder: TypeLabelEncoder | None = None,
    cache: FingerprintCache | None = None,
) -> tuple[pd.DataFrame, TypeLabelEncoder, FingerprintCache]:
    if normalize_formulation(formulation) == "frequent363":
        return prepare_frequent363_multiclass_frame(frame, label_encoder=label_encoder, cache=cache)
    return prepare_multiclass_frame(frame, label_encoder=label_encoder, cache=cache)


def evaluate_protocol_for_formulation(
    formulation: str,
    frame: pd.DataFrame,
    predictions_by_split: dict[str, pd.DataFrame],
    label_encoder: TypeLabelEncoder,
    min_train_examples: int = 20,
) -> dict[str, Any]:
    if normalize_formulation(formulation) == "frequent363":
        from src.evaluation.frequent363_protocol import evaluate_frequent363_protocol

        return evaluate_frequent363_protocol(frame, predictions_by_split, label_encoder)
    from src.evaluation.multiclass_protocol import evaluate_multiclass_protocol

    return evaluate_multiclass_protocol(
        frame,
        predictions_by_split,
        label_encoder,
        min_train_examples=min_train_examples,
    )
