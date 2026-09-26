"""User-facing prediction interface for multi-label DDI prediction.

Takes Drug A, Drug B, and optional Age to output Top-K predicted
interaction scores with severity evidence.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


class DDIPredictor:
    """Static model prediction interface for drug-drug interaction prediction.

    Parameters
    ----------
    model : torch.nn.Module
        Trained multi-label model.
    label_mapping : dict[int, int]
        Mapping of type_id -> column_index.
    age_standardizer : Any, optional
        Fitted AgeStandardizer for normalizing age input.
    device : str
        Device for inference.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        label_mapping: dict[int, int],
        age_standardizer: Any = None,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.model.eval()
        self.label_mapping = label_mapping
        self.idx_to_type = {int(v): int(k) for k, v in label_mapping.items()}
        self.n_classes = len(label_mapping)
        self.age_standardizer = age_standardizer
        self.device = device

    def predict(
        self,
        drug_a_smiles: str,
        drug_b_smiles: str,
        age: float | None = None,
        top_k: int = 5,
        threshold: float = 0.5,
    ) -> dict[str, Any]:
        """Predict DDI interactions for a drug pair.

        Parameters
        ----------
        drug_a_smiles : str
            SMILES string for Drug A.
        drug_b_smiles : str
            SMILES string for Drug B.
        age : float, optional
            Patient age (raw, will be standardized if standardizer is available).
        top_k : int
            Number of top interactions to return.
        threshold : float
            Decision threshold for binary predictions.

        Returns
        -------
        dict
            Contains 'pair', 'top_k_interactions', 'all_probabilities', and warnings.
        """
        # Build features - this is model-dependent
        features = self._build_features(drug_a_smiles, drug_b_smiles, age)

        # Get probabilities
        with torch.no_grad():
            if isinstance(features, torch.Tensor):
                probs = torch.sigmoid(self.model(features.to(self.device)))
            else:
                # For graph-based models, features would be graph batches
                probs = self.model.predict_proba(*features)

        probs_np = probs.cpu().numpy().flatten()

        # Top-K ranked interactions
        top_k_indices = np.argsort(probs_np)[::-1][:top_k]
        top_interactions = []
        for rank, idx in enumerate(top_k_indices):
            type_id = self.idx_to_type.get(int(idx), int(idx))
            prob = float(probs_np[idx])
            top_interactions.append({
                "rank": rank + 1,
                "interaction_type": type_id,
                "probability": round(prob, 6),
                "predicted_positive": prob >= threshold,
            })

        # Count total predicted positives
        n_positive = int((probs_np >= threshold).sum())

        return {
            "drug_a_smiles": drug_a_smiles,
            "drug_b_smiles": drug_b_smiles,
            "age": age,
            "top_k": top_k,
            "threshold": threshold,
            "n_predicted_positive": n_positive,
            "n_classes": self.n_classes,
            "top_k_interactions": top_interactions,
        }

    def _build_features(
        self,
        drug_a_smiles: str,
        drug_b_smiles: str,
        age: float | None,
    ) -> torch.Tensor:
        """Build input features from SMILES and optional age.

        This default implementation uses Morgan fingerprints.
        Override for graph-based models.
        """
        try:
            from rdkit import Chem
            from rdkit.Chem import rdFingerprintGenerator
        except ImportError:
            raise ImportError("RDKit is required for fingerprint computation.")

        mol_a = Chem.MolFromSmiles(drug_a_smiles)
        mol_b = Chem.MolFromSmiles(drug_b_smiles)

        if mol_a is None:
            raise ValueError(f"Invalid SMILES for Drug A: {drug_a_smiles}")
        if mol_b is None:
            raise ValueError(f"Invalid SMILES for Drug B: {drug_b_smiles}")

        generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        fp_a = np.array(generator.GetFingerprint(mol_a), dtype=np.float32)
        fp_b = np.array(generator.GetFingerprint(mol_b), dtype=np.float32)

        features = np.concatenate([fp_a, fp_b])

        if age is not None and self.age_standardizer is not None:
            scaled_age = self.age_standardizer.transform([age])
            features = np.concatenate([features, scaled_age])

        return torch.tensor(features, dtype=torch.float32).unsqueeze(0)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Path | str,
        model: torch.nn.Module,
        label_mapping_path: Path | str,
        age_standardizer: Any = None,
        device: str = "cpu",
    ) -> "DDIPredictor":
        """Load a predictor from a saved checkpoint and label mapping.

        Parameters
        ----------
        checkpoint_path : Path or str
            Path to .pt checkpoint file.
        model : torch.nn.Module
            Uninitialized or fresh model instance.
        label_mapping_path : Path or str
            Path to label mapping JSON.
        age_standardizer : Any, optional
            Fitted AgeStandardizer.
        device : str
            Target device.

        Returns
        -------
        DDIPredictor
            Ready-to-use predictor instance.
        """
        checkpoint = torch.load(
            str(checkpoint_path), map_location=device, weights_only=False
        )
        model.load_state_dict(checkpoint["model_state_dict"])

        mapping_data = json.loads(
            Path(label_mapping_path).read_text(encoding="utf-8")
        )
        label_mapping = {int(k): int(v) for k, v in mapping_data["type_to_index"].items()}

        return cls(
            model=model,
            label_mapping=label_mapping,
            age_standardizer=age_standardizer,
            device=device,
        )
