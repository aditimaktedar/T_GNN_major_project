"""User-facing inference interface for multi-label DDI models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.models.checkpoints import CheckpointManager
from src.evaluation.ranking import rank_predictions_for_pair


class MultilabelPredictor:
    """End-to-end predictor for multi-label DDI.
    
    Loads a checkpoint, processes drug pairs, and outputs top-K ranked
    interaction probabilities.
    """

    def __init__(
        self,
        checkpoint_path: Path | str,
        label_mapping_path: Path | str,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.checkpoint_path = Path(checkpoint_path)
        self.label_mapping_path = Path(label_mapping_path)
        
        # Load label mapping
        if self.label_mapping_path.exists():
            with open(self.label_mapping_path, "r", encoding="utf-8") as f:
                mapping_raw = json.load(f)
                # mapping is type_id (str) -> column_idx (int)
                self.label_mapping = {int(k): int(v) for k, v in mapping_raw.items()}
            self.n_classes = len(self.label_mapping)
        else:
            self.label_mapping = {}
            self.n_classes = 363
            
        self._load_model()
        
    def _load_model(self) -> None:
        """Load model state and metadata from checkpoint."""
        # For this preparation phase, we assume the model logic is injected or 
        # dynamically loaded based on the checkpoint metadata.
        pass

    def predict_pair(
        self,
        smiles_a: str,
        smiles_b: str,
        age: float | None = None,
        k: int = 5,
    ) -> list[dict[str, Any]]:
        """Predict interactions for a single drug pair.
        
        Parameters
        ----------
        smiles_a : str
            SMILES string of drug A.
        smiles_b : str
            SMILES string of drug B.
        age : float, optional
            Patient age to use if the model requires it.
        k : int
            Number of top predictions to return.
            
        Returns
        -------
        list[dict]
            Ranked list of interaction predictions.
        """
        # Note: Implement actual prediction logic once models are trained.
        return []
