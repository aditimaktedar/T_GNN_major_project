"""Adapters between Member B evaluation and Member A model/XAI code.

Member A files inspected:
- src/models/tgnn.py  → stub (docstring only)
- src/xai/explain.py  → stub (docstring only)

These adapters do not pretend A's implementations exist.
"""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

EXPLANATION_REQUIRED_COLUMNS = (
    "patient_id",
    "drug_a",
    "drug_b",
    "probability",
    "explanation_mask",
    "masked_probability",
)

OPTIONAL_EXPLANATION_COLUMNS = (
    "prediction",
    "important_features",
    "important_nodes",
    "important_edges",
    "original_probability",
)


def member_a_tgnn_status() -> dict[str, Any]:
    """Report whether Member A's TGNN module exposes a usable model."""
    try:
        module = importlib.import_module("src.models.tgnn")
    except ImportError as exc:
        return {"available": False, "reason": str(exc)}
    source = inspect.getsource(module)
    if len(source.strip().splitlines()) <= 2:
        return {
            "available": False,
            "reason": "src/models/tgnn.py is a stub. TemporalDDI-GNN predictions must be supplied as a file.",
            "module": "src.models.tgnn",
        }
    return {"available": True, "module": "src.models.tgnn"}


def member_a_xai_status() -> dict[str, Any]:
    """Report whether Member A's explainer produces usable outputs."""
    try:
        module = importlib.import_module("src.xai.explain")
    except ImportError as exc:
        return {"available": False, "reason": str(exc)}
    source = inspect.getsource(module)
    if len(source.strip().splitlines()) <= 2:
        return {
            "available": False,
            "reason": "XAI explanation outputs are not available yet.",
            "module": "src.xai.explain",
        }
    return {"available": True, "module": "src.xai.explain"}


def normalize_explanation_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Map explanation inputs to the generic schema used by Member B evaluators."""
    working = frame.copy()
    if "original_probability" not in working.columns and "probability" in working.columns:
        working["original_probability"] = working["probability"]
    if "masked_probability" not in working.columns:
        raise ValueError(
            "Explanation data must include masked_probability (model output after mask is applied)."
        )
    if "explanation_mask" not in working.columns:
        if "important_features" in working.columns:
            working["explanation_mask"] = working["important_features"]
        else:
            raise ValueError(
                "Explanation data must include explanation_mask or important_features."
            )
    missing = [col for col in EXPLANATION_REQUIRED_COLUMNS if col not in working.columns]
    if missing:
        raise ValueError(f"Explanation data missing required columns: {missing}")
    return working


def load_explanations(path: Path) -> pd.DataFrame:
    """Load explanation outputs from CSV/Parquet/JSON produced by Member A."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Explanation file not found: {path}")
    if path.suffix == ".parquet":
        frame = pd.read_parquet(path)
    elif path.suffix == ".json":
        frame = pd.read_json(path)
    else:
        frame = pd.read_csv(path)
    return normalize_explanation_frame(frame)


def explanation_mask_to_array(value: object) -> np.ndarray:
    if isinstance(value, (list, tuple, np.ndarray)):
        return np.asarray(value, dtype=float)
    if isinstance(value, str):
        cleaned = value.strip("[]")
        if not cleaned:
            return np.array([], dtype=float)
        return np.asarray([float(x) for x in cleaned.split(",")], dtype=float)
    raise ValueError(f"Cannot parse explanation mask from {type(value)}")
