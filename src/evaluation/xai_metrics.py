"""XAI evaluation metrics: Fidelity, Sparsity, Stability.

Definitions (model-agnostic):
- Fidelity: fraction of original prediction preserved when using only the
  explanation-selected features/mask (higher = explanation captures behavior).
- Sparsity: fraction of features/edges NOT selected by the explanation
  (higher = smaller explanation relative to full input).
- Stability: 1 - normalized disagreement between two explanation masks
  (higher = more stable explanations under perturbation/repeat).
"""

from __future__ import annotations

from typing import Iterable

import numpy as np


def explanation_to_mask(explanation: Iterable[float] | np.ndarray, threshold: float = 0.0) -> np.ndarray:
    """Convert an explanation vector to a binary mask."""
    arr = np.asarray(explanation, dtype=float)
    return (arr > threshold).astype(float)


def fidelity_score(
    original_prediction: float,
    masked_prediction: float,
) -> float:
    """Compare original vs explanation-masked model output.

    Uses 1 - |p - p'| so identical behavior under the mask yields fidelity 1.
    """
    return float(max(0.0, 1.0 - abs(float(original_prediction) - float(masked_prediction))))


def sparsity_score(explanation_mask: np.ndarray) -> float:
    """Fraction of features/edges not selected by the explanation."""
    mask = np.asarray(explanation_mask, dtype=float)
    if mask.size == 0:
        return 0.0
    return float(1.0 - mask.mean())


def stability_score(explanation_a: np.ndarray, explanation_b: np.ndarray) -> float:
    """1 minus normalized Hamming distance between two explanation masks."""
    a = explanation_to_mask(explanation_a)
    b = explanation_to_mask(explanation_b)
    if a.shape != b.shape:
        raise ValueError("Explanation shapes must match for stability.")
    if a.size == 0:
        return 1.0
    disagreement = np.mean(a != b)
    return float(1.0 - disagreement)


def evaluate_xai(
    original_predictions: np.ndarray,
    masked_predictions: np.ndarray,
    explanation_masks: np.ndarray,
    repeated_explanations: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute aggregate XAI metrics from arrays."""
    original_predictions = np.asarray(original_predictions, dtype=float)
    masked_predictions = np.asarray(masked_predictions, dtype=float)
    explanation_masks = np.asarray(explanation_masks, dtype=float)
    fidelities = [
        fidelity_score(o, m) for o, m in zip(original_predictions, masked_predictions, strict=True)
    ]
    sparsities = [sparsity_score(mask) for mask in explanation_masks]
    result = {
        "fidelity_mean": float(np.mean(fidelities)) if fidelities else 0.0,
        "sparsity_mean": float(np.mean(sparsities)) if sparsities else 0.0,
    }
    if repeated_explanations is not None:
        repeated = np.asarray(repeated_explanations, dtype=float)
        stabilities = [
            stability_score(explanation_masks[i], repeated[i])
            for i in range(len(explanation_masks))
        ]
        result["stability_mean"] = float(np.mean(stabilities)) if stabilities else 0.0
    else:
        result["stability_mean"] = None
    return result
