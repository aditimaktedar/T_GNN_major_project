"""
Clinical API layer integrating TemporalDDI-GNN (model_v2) and multi-source RAG pipelines.
"""

from api.rag_contract import build_rag_query, build_response, validate_contract_separation
from api.schemas import DrugEntity, GNNPredictionResult, RAGExplanationResult, MultiSourceEvidenceResult, MergedDDIResponse
from api.drug_resolver import DrugResolver
from api.gnn_service import GNNPredictionService
from api.rag_service import RAGRetrievalService
from api.merger import ResultMerger
from api.service import ClinicalDDIOrchestrator

__all__ = [
    "build_rag_query",
    "build_response",
    "validate_contract_separation",
    "DrugEntity",
    "GNNPredictionResult",
    "RAGExplanationResult",
    "MultiSourceEvidenceResult",
    "MergedDDIResponse",
    "DrugResolver",
    "GNNPredictionService",
    "RAGRetrievalService",
    "ResultMerger",
    "ClinicalDDIOrchestrator"
]
