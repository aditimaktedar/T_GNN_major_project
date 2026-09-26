"""Run XAI metric evaluation on explanation outputs from Member A."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.config import METRICS_DIR
from src.data.io_utils import ensure_parent, write_json
from src.evaluation.adapters import (
    explanation_mask_to_array,
    load_explanations,
    member_a_xai_status,
    normalize_explanation_frame,
)
from src.evaluation.xai_metrics import evaluate_xai, fidelity_score, sparsity_score, stability_score


class XAIOutputsNotAvailableError(RuntimeError):
    """Raised when Member A has not supplied explanation outputs."""


def evaluate_explanations(
    explanations: pd.DataFrame,
    repeated_explanations: pd.DataFrame | None = None,
) -> tuple[dict, pd.DataFrame]:
    """Compute aggregate and per-example XAI metrics."""
    frame = normalize_explanation_frame(explanations)
    per_rows = []
    original_preds = []
    masked_preds = []
    masks = []
    repeated_masks = []

    repeated_lookup = {}
    if repeated_explanations is not None:
        rep = normalize_explanation_frame(repeated_explanations)
        for _, row in rep.iterrows():
            key = (str(row["patient_id"]), str(row["drug_a"]), str(row["drug_b"]))
            repeated_lookup[key] = explanation_mask_to_array(row["explanation_mask"])

    for _, row in frame.iterrows():
        mask = explanation_mask_to_array(row["explanation_mask"])
        orig = float(row.get("original_probability", row["probability"]))
        masked = float(row["masked_probability"])
        fid = fidelity_score(orig, masked)
        spar = sparsity_score(mask)
        key = (str(row["patient_id"]), str(row["drug_a"]), str(row["drug_b"]))
        stab = None
        if key in repeated_lookup:
            stab = stability_score(mask, repeated_lookup[key])
        per_rows.append(
            {
                "patient_id": row["patient_id"],
                "drug_a": row["drug_a"],
                "drug_b": row["drug_b"],
                "probability": orig,
                "masked_probability": masked,
                "fidelity": fid,
                "sparsity": spar,
                "stability": stab,
            }
        )
        original_preds.append(orig)
        masked_preds.append(masked)
        masks.append(mask)
        if key in repeated_lookup:
            repeated_masks.append(repeated_lookup[key])

    mask_array = np.stack(masks) if masks else np.empty((0, 0))
    repeated_array = np.stack(repeated_masks) if repeated_masks else None
    aggregate = evaluate_xai(
        np.asarray(original_preds),
        np.asarray(masked_preds),
        mask_array,
        repeated_array,
    )
    per_example = pd.DataFrame(per_rows)
    return aggregate, per_example


def run_xai_evaluation(
    explanations_path: Path | None = None,
    repeated_path: Path | None = None,
    synthetic_demo: bool = False,
    demo_explanations: pd.DataFrame | None = None,
) -> dict:
    """Evaluate XAI metrics. Real mode requires explanation file from Member A."""
    if explanations_path is None and demo_explanations is None:
        status = member_a_xai_status()
        if not status.get("available", False):
            raise XAIOutputsNotAvailableError(status.get("reason", "XAI explanation outputs are not available yet."))
        raise XAIOutputsNotAvailableError("XAI explanation outputs are not available yet.")

    if demo_explanations is not None:
        explanations = demo_explanations
    else:
        explanations = load_explanations(Path(explanations_path))

    repeated = load_explanations(repeated_path) if repeated_path else None
    aggregate, per_example = evaluate_explanations(explanations, repeated)

    payload = {
        "status": "ok",
        "synthetic_demo": synthetic_demo,
        "warning": "SYNTHETIC SOFTWARE VALIDATION ONLY" if synthetic_demo else None,
        "aggregate": aggregate,
        "n_examples": int(len(per_example)),
    }
    json_path = METRICS_DIR / "xai_metrics.json"
    csv_path = METRICS_DIR / "xai_per_example.csv"
    write_json(json_path, payload)
    ensure_parent(csv_path)
    per_example.to_csv(csv_path, index=False)
    payload["json_path"] = str(json_path)
    payload["csv_path"] = str(csv_path)
    return payload
