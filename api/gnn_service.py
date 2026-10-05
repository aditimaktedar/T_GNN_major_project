"""
GNN Prediction Service.
Loads model_v2 TemporalDDIGNN checkpoints (Folds 1-5), executes inference,
and computes presence and severity outputs with CDSS alert classifications.
"""

import os
from typing import Dict, Any, Optional, List, Tuple
from api.schemas import DrugEntity, GNNPredictionResult
from api.drug_resolver import DrugResolver
from src.inference import (
    StandaloneInferenceEngine,
    load_checkpoint_metadata,
    SEVERITY_CLASSES
)


class GNNPredictionService:
    def __init__(
        self,
        checkpoints_dir: Optional[str] = None,
        default_fold: int = 1,
        resolver: Optional[DrugResolver] = None
    ):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.checkpoints_dir = checkpoints_dir or os.path.join(base_dir, "checkpoints")
        self.default_fold = default_fold
        self.resolver = resolver or DrugResolver()

        # Cache of inference engines by fold
        self.engines: Dict[int, StandaloneInferenceEngine] = {}

        # Load default fold
        self._load_fold(default_fold)

    def _get_checkpoint_path(self, fold: int) -> str:
        return os.path.join(self.checkpoints_dir, f"severity_patient_fold_{fold}.pt")

    def _load_fold(self, fold: int) -> StandaloneInferenceEngine:
        if fold in self.engines:
            return self.engines[fold]

        ckpt_path = self._get_checkpoint_path(fold)
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint for fold {fold} not found at {ckpt_path}")

        engine = StandaloneInferenceEngine(ckpt_path)
        self.engines[fold] = engine

        # Update drug resolver with this engine's vocabulary
        self.resolver.update_model_vocab(engine.drug_to_idx)
        return engine

    def get_available_folds(self) -> List[int]:
        folds = []
        for i in range(1, 6):
            if os.path.exists(self._get_checkpoint_path(i)):
                folds.append(i)
        return folds

    def generate_alert_code(self, severity: str, prob: float) -> str:
        score = int(round(prob * 100))
        sev_upper = severity.upper()
        if sev_upper == "MAJOR" or (prob >= 0.85):
            return f"CDSS-DDI-CRITICAL-{score:03d}"
        elif sev_upper == "MODERATE" or (prob >= 0.50):
            return f"CDSS-DDI-MOD-{score:03d}"
        else:
            return f"CDSS-DDI-LOW-{score:03d}"

    def predict(
        self,
        drug_a: DrugEntity,
        drug_b: DrugEntity,
        fold: Optional[int] = None,
        edge_index: Optional[List[Tuple[int, int]]] = None
    ) -> GNNPredictionResult:
        """
        Executes GNN forward pass for drug pair.
        Supports in-vocabulary prediction and cold-start fallback handling.
        """
        fold_idx = fold or self.default_fold
        engine = self._load_fold(fold_idx)

        idx_a = drug_a.model_index
        idx_b = drug_b.model_index

        # If both drugs are present in the trained 38-drug vocabulary
        if idx_a is not None and idx_b is not None:
            raw = engine.predict_pair(idx_a, idx_b, edge_index=edge_index)
            prob = raw["presence_probability"]
            sev = raw["severity_label"].upper()
            sev_probs = raw["severity_probabilities"]
            source = f"TemporalDDI-GNN Clinical Engine v2 (Fold {fold_idx})"
        else:
            # Cold-start drug inference handling as documented in DEPLOYMENT_NOTES.md
            # Use baseline demographic distribution from MIMIC-IV Demo 2.2 benchmark
            is_a_cold = idx_a is None
            is_b_cold = idx_b is None
            source = (
                f"TemporalDDI-GNN v2 [Cold-Start: "
                f"{'Unseen ' + drug_a.name if is_a_cold else ''}"
                f"{' & ' if is_a_cold and is_b_cold else ''}"
                f"{'Unseen ' + drug_b.name if is_b_cold else ''}]"
            )
            # Default prior from MIMIC severity benchmark distribution
            prob = 0.74
            sev = "MODERATE"
            sev_probs = {
                "Minor": 0.20,
                "Moderate": 0.55,
                "Major": 0.25
            }
            raw = {
                "presence_probability": prob,
                "presence_label": "Yes",
                "severity_label": "Moderate",
                "severity_probabilities": sev_probs,
                "cold_start": True
            }

        interaction_status = "YES" if prob >= 0.5 else "NO"
        alert_code = self.generate_alert_code(sev, prob)

        return GNNPredictionResult(
            interaction=interaction_status,
            probability=round(prob, 4),
            severity=sev,
            severity_probabilities={k: round(v, 4) for k, v in sev_probs.items()},
            source=source,
            alertCode=alert_code,
            fold=fold_idx,
            raw_output=raw
        )
