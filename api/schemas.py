"""
Data schemas for the T-GNN & RAG clinical interaction API.
Supports both dataclass and dictionary serialization for seamless use with or without Pydantic.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict


@dataclass
class DrugEntity:
    drugbank_id: str
    name: str
    pubchem_cid: Optional[str] = None
    model_index: Optional[int] = None
    in_model_vocab: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GNNPredictionResult:
    interaction: str  # "YES" | "NO"
    probability: float
    severity: str  # "MINOR" | "MODERATE" | "MAJOR"
    severity_probabilities: Dict[str, float]
    source: str
    alertCode: str
    fold: int = 1
    raw_output: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RAGExplanationResult:
    summary: str
    shared_targets: List[str] = field(default_factory=list)
    shared_mechanisms: List[str] = field(default_factory=list)
    shared_pk: List[str] = field(default_factory=list)
    shared_pd_effects: List[str] = field(default_factory=list)
    top_chunks: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MultiSourceEvidenceResult:
    drugbank: Dict[str, Any] = field(default_factory=dict)
    cpic: Dict[str, Any] = field(default_factory=dict)
    twosides: Dict[str, Any] = field(default_factory=dict)
    offsides: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MergedDDIResponse:
    # Top-level unified fields matching frontend expectations
    prediction: Dict[str, Any]
    rag_explanation: Dict[str, Any]
    evidence: Dict[str, Any]

    # Strict contract separation fields matching api/rag_contract.py
    model_prediction: Dict[str, Any]
    note: str = "T-GNN prediction and RAG evidence are reported separately."

    # Clinical and tracing metadata
    drug_a: Optional[Dict[str, Any]] = None
    drug_b: Optional[Dict[str, Any]] = None
    meta: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
