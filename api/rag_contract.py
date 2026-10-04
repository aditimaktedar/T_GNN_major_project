"""
Contract module defining the strict separation between T-GNN numerical predictions and RAG evidence retrieval.
As established in DEPLOYMENT_NOTES.md:
RAG is intentionally outside the numerical T-GNN model.
Recommended flow: API -> drug resolver -> T-GNN -> prediction -> RAG -> response.
T-GNN prediction and RAG evidence must remain separate.
"""

from typing import Dict, Any


def build_rag_query(drug_a: str, drug_b: str) -> str:
    """Builds standard retrieval query for a pair of drugs."""
    return (
        "Drug-drug interaction evidence for "
        + str(drug_a)
        + " and "
        + str(drug_b)
    )


def build_response(
    prediction: Any,
    evidence: Any,
    additional_context: Any = None
) -> Dict[str, Any]:
    """
    Constructs the canonical contract response separating model prediction from retrieved evidence.
    """
    resp = {
        "model_prediction": prediction,
        "evidence": evidence,
        "note": (
            "T-GNN prediction and RAG evidence "
            "are reported separately."
        )
    }
    if additional_context:
        resp["context"] = additional_context
    return resp


def validate_contract_separation(response: Dict[str, Any]) -> bool:
    """
    Verifies that the response maintains architectural separation between
    model prediction and RAG evidence.
    """
    if "model_prediction" not in response:
        return False
    if "evidence" not in response:
        return False
    if "note" not in response:
        return False
    return True