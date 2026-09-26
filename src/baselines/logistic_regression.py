"""Scikit-learn Logistic Regression baseline for drug-pair classification."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.data.pyg_dataset import LABEL_TO_INT, filter_trainable_rows
from src.evaluation.reproducibility import set_global_seed
from src.features.pair_features import DEFAULT_PAIR_FEATURE_METHOD, PairFeatureMethod, build_pair_features


class LogisticRegressionBaseline:
    """Train only on training patients; val/test are never used for fitting."""

    def __init__(
        self,
        C: float = 1.0,
        class_weight: str | dict | None = "balanced",
        seed: int = 42,
        pair_feature_method: PairFeatureMethod = DEFAULT_PAIR_FEATURE_METHOD,
        max_iter: int = 1000,
    ):
        self.C = C
        self.class_weight = class_weight
        self.seed = seed
        self.pair_feature_method = pair_feature_method
        self.max_iter = max_iter
        self.model = LogisticRegression(
            C=C,
            class_weight=class_weight,
            random_state=seed,
            max_iter=max_iter,
        )
        self.feature_dim: int | None = None

    def _build_matrix(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        rows = []
        labels = []
        for _, row in frame.iterrows():
            try:
                feat = build_pair_features(
                    row["drug_a_fingerprint"],
                    row["drug_b_fingerprint"],
                    method=self.pair_feature_method,
                )
            except ValueError:
                continue
            rows.append(feat)
            labels.append(LABEL_TO_INT[str(row["label"])])
        if not rows:
            raise ValueError("No valid training rows with fingerprints.")
        X = np.vstack(rows)
        y = np.asarray(labels, dtype=int)
        self.feature_dim = int(X.shape[1])
        return X, y

    def fit(self, train_frame: pd.DataFrame) -> "LogisticRegressionBaseline":
        set_global_seed(self.seed)
        train = filter_trainable_rows(train_frame)
        if train.empty:
            raise ValueError("Training frame is empty after filtering.")
        X, y = self._build_matrix(train)
        self.model.fit(X, y)
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        X, _ = self._build_matrix(filter_trainable_rows(frame))
        return self.model.predict(X)

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        X, _ = self._build_matrix(filter_trainable_rows(frame))
        return self.model.predict_proba(X)[:, 1]

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": self.model,
                "config": {
                    "C": self.C,
                    "class_weight": self.class_weight,
                    "seed": self.seed,
                    "pair_feature_method": self.pair_feature_method,
                    "feature_dim": self.feature_dim,
                },
            },
            path,
        )
        return path

    @classmethod
    def load(cls, path: Path) -> "LogisticRegressionBaseline":
        payload = joblib.load(path)
        config = payload["config"]
        obj = cls(
            C=config["C"],
            class_weight=config["class_weight"],
            seed=config["seed"],
            pair_feature_method=config["pair_feature_method"],
        )
        obj.model = payload["model"]
        obj.feature_dim = config.get("feature_dim")
        return obj
