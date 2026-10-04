"""
Result Merger service.
Merges numerical predictions from the T-GNN model with retrieved clinical evidence from the RAG pipeline.
Strictly adheres to api/rag_contract.py and DEPLOYMENT_NOTES.md by keeping numerical predictions
and evidence distinctly separated, while formatting unified payloads for Clinical Portal consumers.
"""

import time
from typing import Dict, Any, Optional
from api.schemas import DrugEntity, GNNPredictionResult, MergedDDIResponse
from api.rag_contract import build_response, validate_contract_separation


class ResultMerger:
    def __init__(self):
        pass

    def merge(
        self,
        gnn_result: GNNPredictionResult,
        rag_data: Dict[str, Any],
        drug_a: DrugEntity,
        drug_b: DrugEntity,
        patient_id: Optional[str] = None
    ) -> MergedDDIResponse:
        """
        Combines T-GNN and RAG outputs into a unified, contract-compliant response.
        """
        rag_explanation = rag_data.get("rag_explanation", {})
        evidence = rag_data.get("evidence", {})

        # Build raw numerical prediction dictionary
        model_prediction = {
            "presence_probability": gnn_result.probability,
            "presence_label": "Yes" if gnn_result.interaction == "YES" else "No",
            "severity_label": gnn_result.severity.title(),
            "severity_probabilities": gnn_result.severity_probabilities,
            "fold": gnn_result.fold,
            "source": gnn_result.source,
            "raw_details": gnn_result.raw_output or {}
        }

        # Contract response validation
        contract_base = build_response(
            prediction=model_prediction,
            evidence=evidence
        )
        assert validate_contract_separation(contract_base), "Contract separation invariant violated"

        # Formatted top-level prediction object for Clinical Portal
        prediction_card = {
            "interaction": gnn_result.interaction,
            "probability": gnn_result.probability,
            "severity": gnn_result.severity,
            "source": gnn_result.source,
            "alertCode": gnn_result.alertCode,
            "severity_probabilities": gnn_result.severity_probabilities
        }

        meta = {
            "api_version": "2.4.0",
            "engine": "TemporalDDI-GNN + Multi-Source RAG",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "patient_id": patient_id or "ANONYMOUS",
            "model_branch": "model_v2",
            "evaluated_pair": f"{drug_a.name} + {drug_b.name}"
        }

        merged = MergedDDIResponse(
            prediction=prediction_card,
            rag_explanation=rag_explanation,
            evidence=evidence,
            model_prediction=model_prediction,
            note=contract_base["note"],
            drug_a=drug_a.to_dict(),
            drug_b=drug_b.to_dict(),
            meta=meta
        )

        return merged
