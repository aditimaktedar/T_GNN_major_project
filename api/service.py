"""
Clinical DDI Orchestrator Service.
Coordinates the end-to-end pipeline:
Drug Query -> Drug Resolver -> T-GNN Model Inference -> RAG Evidence Retrieval -> Result Merger.
"""

from typing import Dict, Any, List, Optional
from api.drug_resolver import DrugResolver
from api.gnn_service import GNNPredictionService
from api.rag_service import RAGRetrievalService
from api.merger import ResultMerger
from api.schemas import MergedDDIResponse


class ClinicalDDIOrchestrator:
    _instance: Optional["ClinicalDDIOrchestrator"] = None

    def __init__(self):
        self.resolver = DrugResolver()
        self.gnn = GNNPredictionService(resolver=self.resolver)
        self.rag = RAGRetrievalService()
        self.merger = ResultMerger()

    @classmethod
    def get_instance(cls) -> "ClinicalDDIOrchestrator":
        if cls._instance is None:
            cls._instance = ClinicalDDIOrchestrator()
        return cls._instance

    def evaluate_pair(
        self,
        drug_a_query: str,
        drug_b_query: str,
        fold: Optional[int] = None,
        top_k: int = 5,
        patient_id: Optional[str] = None
    ) -> MergedDDIResponse:
        """
        Evaluates interaction risk for a single drug pair through GNN and RAG layers.
        """
        drug_a = self.resolver.resolve(drug_a_query)
        drug_b = self.resolver.resolve(drug_b_query)

        # 1. Execute T-GNN numerical prediction
        gnn_result = self.gnn.predict(drug_a, drug_b, fold=fold)

        # 2. Retrieve multi-source clinical evidence independently
        rag_data = self.rag.retrieve_evidence(drug_a, drug_b, top_k=top_k)

        # 3. Merge results adhering to contract separation
        merged = self.merger.merge(
            gnn_result=gnn_result,
            rag_data=rag_data,
            drug_a=drug_a,
            drug_b=drug_b,
            patient_id=patient_id
        )

        return merged

    def evaluate_regimen(
        self,
        medications: List[Dict[str, Any]],
        fold: Optional[int] = None,
        patient_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluates all unique pairwise interactions within an active patient MAR regimen.
        """
        resolved_meds = []
        for m in medications:
            q = m.get("id") or m.get("drugbank_id") or m.get("name")
            if q:
                resolved_meds.append((self.resolver.resolve(q), m))

        evaluated_pairs = []
        n = len(resolved_meds)
        for i in range(n):
            for j in range(i + 1, n):
                d_a, m_a = resolved_meds[i]
                d_b, m_b = resolved_meds[j]
                res = self.evaluate_pair(
                    drug_a_query=d_a.drugbank_id,
                    drug_b_query=d_b.drugbank_id,
                    fold=fold,
                    patient_id=patient_id
                )
                evaluated_pairs.append({
                    "pair_label": f"{d_a.name} + {d_b.name}",
                    "drug_1": d_a.to_dict(),
                    "drug_2": d_b.to_dict(),
                    "prediction": res.prediction,
                    "evidence_summary": res.rag_explanation.get("summary", ""),
                    "full_response": res.to_dict()
                })

        # Rank pairs by severity (MAJOR > MODERATE > MINOR) and then probability
        severity_rank = {"MAJOR": 3, "MODERATE": 2, "MINOR": 1}
        evaluated_pairs.sort(
            key=lambda x: (
                severity_rank.get(x["prediction"]["severity"].upper(), 0),
                x["prediction"]["probability"]
            ),
            reverse=True
        )

        return {
            "patient_id": patient_id or "ANONYMOUS",
            "total_medications": len(medications),
            "total_pairs_evaluated": len(evaluated_pairs),
            "critical_alerts_count": sum(1 for p in evaluated_pairs if p["prediction"]["severity"].upper() == "MAJOR"),
            "evaluated_pairs": evaluated_pairs
        }

    def health(self) -> Dict[str, Any]:
        return {
            "status": "healthy",
            "model": "TemporalDDI-GNN (model_v2)",
            "available_folds": self.gnn.get_available_folds(),
            "indexed_drugs_count": len(self.resolver.all_drugs),
            "indexed_evidence_chunks": sum(len(c) for c in self.rag.metadata_by_drug.values()),
            "twosides_pairs_count": len(self.rag.twosides_pairs)
        }
