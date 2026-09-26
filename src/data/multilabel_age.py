"""Age feature processing and standardizer module.

Provides zero-leakage normalization for numerical patient age (`anchor_age`).
Fits mean and standard deviation solely on the training split.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

FeatureSetMode = Literal["drug_pair", "drug_pair_age"]


class AgeStandardizer:
    """Standardize patient age using mean and standard deviation fit on training data."""

    def __init__(self) -> None:
        self.mean: float | None = None
        self.std: float | None = None
        self.is_fitted: bool = False

    def fit(self, ages: pd.Series | np.ndarray | list[float]) -> "AgeStandardizer":
        arr = np.asarray(ages, dtype=float)
        valid = arr[~np.isnan(arr)]
        if len(valid) == 0:
            raise ValueError("Cannot fit AgeStandardizer on empty or all-NaN age values.")
        self.mean = float(np.mean(valid))
        self.std = float(np.std(valid))
        if self.std < 1e-8:
            self.std = 1.0  # Prevent division by zero
        self.is_fitted = True
        return self

    def transform(self, ages: pd.Series | np.ndarray | list[float]) -> np.ndarray:
        if not self.is_fitted or self.mean is None or self.std is None:
            raise RuntimeError("AgeStandardizer must be fit before calling transform.")
        arr = np.asarray(ages, dtype=float)
        # Fill missing with train mean if any
        arr_filled = np.where(np.isnan(arr), self.mean, arr)
        scaled = (arr_filled - self.mean) / self.std
        return scaled.astype(np.float32)

    def fit_transform(self, ages: pd.Series | np.ndarray | list[float]) -> np.ndarray:
        return self.fit(ages).transform(ages)

    def to_dict(self) -> dict[str, float]:
        if not self.is_fitted:
            raise RuntimeError("AgeStandardizer is not fitted.")
        return {"mean": float(self.mean), "std": float(self.std)}

    @classmethod
    def from_dict(cls, data: dict[str, float]) -> "AgeStandardizer":
        obj = cls()
        obj.mean = data["mean"]
        obj.std = data["std"]
        obj.is_fitted = True
        return obj


def prepare_age_feature(
    df: pd.DataFrame,
    split_col: str = "split",
    age_col: str = "anchor_age",
    standardizer: AgeStandardizer | None = None,
) -> tuple[np.ndarray, AgeStandardizer]:
    """Standardize `anchor_age` across dataframe splits without leakage.

    Fits standardizer on `split == 'train'` if standardizer is not provided.
    """
    if age_col not in df.columns:
        raise KeyError(f"Age column `{age_col}` not found in DataFrame.")

    if standardizer is None:
        standardizer = AgeStandardizer()
        if split_col in df.columns:
            train_ages = df.loc[df[split_col] == "train", age_col]
        else:
            train_ages = df[age_col]
        standardizer.fit(train_ages)

    scaled_ages = standardizer.transform(df[age_col])
    return scaled_ages, standardizer
