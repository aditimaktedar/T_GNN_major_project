"""Deterministic drug-pair feature construction from molecular fingerprints.

Default method: ``concat`` (drug_a fingerprint followed by drug_b fingerprint).

Supported methods:
- concat: [fp_a, fp_b]
- abs_diff: |fp_a - fp_b|
- product: fp_a * fp_b (element-wise)
"""

from __future__ import annotations

from typing import Literal

import numpy as np

PairFeatureMethod = Literal["concat", "abs_diff", "product"]

VALID_METHODS: tuple[PairFeatureMethod, ...] = ("concat", "abs_diff", "product")
DEFAULT_PAIR_FEATURE_METHOD: PairFeatureMethod = "concat"


def _as_vector(value: object) -> np.ndarray:
    if value is None:
        raise ValueError("Fingerprint is missing.")
    if isinstance(value, float) and np.isnan(value):
        raise ValueError("Fingerprint is missing.")
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim != 1:
        raise ValueError(f"Expected 1-D fingerprint, got shape {arr.shape}.")
    return arr


def build_pair_features(
    fp_a: object,
    fp_b: object,
    method: PairFeatureMethod = DEFAULT_PAIR_FEATURE_METHOD,
) -> np.ndarray:
    """Build a pair feature vector from two molecular fingerprints."""
    if method not in VALID_METHODS:
        raise ValueError(f"Unknown pair feature method {method!r}. Valid: {VALID_METHODS}")
    a = _as_vector(fp_a)
    b = _as_vector(fp_b)
    if a.shape != b.shape:
        raise ValueError(f"Fingerprint dimension mismatch: {a.shape} vs {b.shape}.")
    if method == "concat":
        return np.concatenate([a, b], axis=0)
    if method == "abs_diff":
        return np.abs(a - b)
    return a * b


def pair_feature_dim(fingerprint_dim: int, method: PairFeatureMethod = DEFAULT_PAIR_FEATURE_METHOD) -> int:
    if method == "concat":
        return fingerprint_dim * 2
    return fingerprint_dim
