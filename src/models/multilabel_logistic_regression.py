"""Multi-label Logistic Regression baseline for DDI prediction.

Outputs N independent sigmoid probabilities per drug pair.
Uses BCEWithLogitsLoss for multi-label training.
Supports optional normalized patient age feature concatenation.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn


class MultilabelLogisticRegression(nn.Module):
    """Multi-label logistic regression: N independent sigmoid classifiers.

    Parameters
    ----------
    input_dim : int
        Dimensionality of input features (e.g., 2 * 2048 for concatenated
        Morgan fingerprint pair features, +1 if age is included).
    n_classes : int
        Number of output labels (e.g., 363 for FREQUENT363).
    """

    def __init__(self, input_dim: int, n_classes: int = 363):
        super().__init__()
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.linear = nn.Linear(input_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning raw logits (not sigmoid-activated).

        Parameters
        ----------
        x : torch.Tensor
            Input features of shape (batch_size, input_dim).

        Returns
        -------
        torch.Tensor
            Logits of shape (batch_size, n_classes).
        """
        return self.linear(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return sigmoid probabilities for inference.

        Parameters
        ----------
        x : torch.Tensor
            Input features of shape (batch_size, input_dim).

        Returns
        -------
        torch.Tensor
            Probabilities of shape (batch_size, n_classes).
        """
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits)


def build_fingerprint_features(
    smiles_a: list[str],
    smiles_b: list[str],
    n_bits: int = 2048,
    radius: int = 2,
    ages: np.ndarray | None = None,
) -> np.ndarray:
    """Build concatenated Morgan fingerprint pair features.

    Parameters
    ----------
    smiles_a : list[str]
        SMILES strings for drug A.
    smiles_b : list[str]
        SMILES strings for drug B.
    n_bits : int
        Morgan fingerprint bit length.
    radius : int
        Morgan fingerprint radius.
    ages : np.ndarray, optional
        Normalized age values to concatenate. Shape (n_samples,).

    Returns
    -------
    np.ndarray
        Feature matrix of shape (n_samples, 2*n_bits) or (n_samples, 2*n_bits+1).
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import rdFingerprintGenerator
    except ImportError:
        raise ImportError("RDKit is required for fingerprint feature generation.")

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    features = []
    for sa, sb in zip(smiles_a, smiles_b):
        mol_a = Chem.MolFromSmiles(sa)
        mol_b = Chem.MolFromSmiles(sb)
        if mol_a is None or mol_b is None:
            fp = np.zeros(2 * n_bits, dtype=np.float32)
        else:
            fp_a = np.array(generator.GetFingerprint(mol_a), dtype=np.float32)
            fp_b = np.array(generator.GetFingerprint(mol_b), dtype=np.float32)
            fp = np.concatenate([fp_a, fp_b])
        features.append(fp)

    X = np.vstack(features)

    if ages is not None:
        age_col = np.asarray(ages, dtype=np.float32).reshape(-1, 1)
        X = np.hstack([X, age_col])

    return X
