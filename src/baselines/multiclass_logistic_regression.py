"""Multiclass Logistic Regression baseline for TWOSIDES interaction-type prediction."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.data.final_ml_dataset import TypeLabelEncoder, filter_trainable_rows
from src.evaluation.reproducibility import set_global_seed
from src.features.pair_features import DEFAULT_PAIR_FEATURE_METHOD, PairFeatureMethod, build_pair_features


class MulticlassLogisticRegressionBaseline:
    """Train on train split only; uses Morgan fingerprint pair features."""

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
            solver="lbfgs",
        )
        self.feature_dim: int | None = None
        self.label_encoder: TypeLabelEncoder | None = None

    def _build_matrix(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        rows = []
        labels = []
        for _, row in filter_trainable_rows(frame).iterrows():
            feat = build_pair_features(
                row["drug_a_fingerprint"],
                row["drug_b_fingerprint"],
                method=self.pair_feature_method,
            )
            rows.append(feat)
            labels.append(int(row["label_index"]))
        if not rows:
            raise ValueError("No valid rows with fingerprints and labels.")
        X = np.vstack(rows)
        y = np.asarray(labels, dtype=int)
        self.feature_dim = int(X.shape[1])
        return X, y

    def _expand_proba(self, proba: np.ndarray) -> np.ndarray:
        if self.label_encoder is None:
            raise ValueError("Label encoder is not set.")
        n_samples = proba.shape[0]
        full = np.zeros((n_samples, self.label_encoder.n_classes), dtype=float)
        for col_idx, class_idx in enumerate(self.model.classes_):
            full[:, int(class_idx)] = proba[:, col_idx]
        return full

    def fit(self, train_frame: pd.DataFrame, label_encoder: TypeLabelEncoder) -> "MulticlassLogisticRegressionBaseline":
        set_global_seed(self.seed)
        self.label_encoder = label_encoder
        train = filter_trainable_rows(train_frame)
        if train.empty:
            raise ValueError("Training frame is empty after filtering.")
        X, y = self._build_matrix(train)
        self.model.fit(X, y)
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return self.predict_proba(frame).argmax(axis=1)

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        X, _ = self._build_matrix(frame)
        return self._expand_proba(self.model.predict_proba(X))

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": self.model,
                "label_encoder": self.label_encoder,
                "config": {
                    "C": self.C,
                    "class_weight": self.class_weight,
                    "seed": self.seed,
                    "pair_feature_method": self.pair_feature_method,
                    "feature_dim": self.feature_dim,
                    "max_iter": self.max_iter,
                },
            },
            path,
        )
        return path

    @classmethod
    def load(cls, path: Path) -> "MulticlassLogisticRegressionBaseline":
        payload = joblib.load(path)
        config = payload["config"]
        obj = cls(
            C=config["C"],
            class_weight=config["class_weight"],
            seed=config["seed"],
            pair_feature_method=config["pair_feature_method"],
            max_iter=config.get("max_iter", 1000),
        )
        obj.model = payload["model"]
        obj.label_encoder = payload["label_encoder"]
        obj.feature_dim = config.get("feature_dim")
        return obj
