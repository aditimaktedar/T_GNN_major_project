"""
Comprehensive unit and integration tests for the T-GNN & RAG merged API layer.
"""

from api.drug_resolver import DrugResolver
from api.gnn_service import GNNPredictionService
from api.rag_service import RAGRetrievalService
from api.merger import ResultMerger
from api.service import ClinicalDDIOrchestrator
from api.rag_contract import build_rag_query, build_response, validate_contract_separation


def test_drug_resolver():
    resolver = DrugResolver()
    assert len(resolver.all_drugs) > 0

    # 1. Test DrugBank ID lookup
    panto = resolver.resolve("DB00213")
    assert panto.drugbank_id == "DB00213"
    assert "Pantoprazole" in panto.name

    # 2. Test Name lookup (case-insensitive)
    moda = resolver.resolve("modafinil")
    assert moda.drugbank_id == "DB00745"

    # 3. Test PubChem CID lookup
    amlo = resolver.resolve("2162")
    assert amlo.drugbank_id == "DB00381"
    assert "Amlodipine" in amlo.name

    # 4. Test Autocomplete / list
    results = resolver.list_all_drugs(query="panto", limit=10)
    assert len(results) > 0
    assert any("Pantoprazole" in r["name"] for r in results)

    # 5. Unknown drug fallback
    unk = resolver.resolve("NonExistentDrugXYZ")
    assert unk.name == "NonExistentDrugXYZ"


def test_gnn_prediction_service():
    resolver = DrugResolver()
    service = GNNPredictionService(resolver=resolver, default_fold=1)

    # Check folds
    folds = service.get_available_folds()
    assert 1 in folds
    assert len(folds) >= 1

    # In-vocabulary prediction
    # Both Pantoprazole (4679) and Amlodipine (2162) are in vocabulary
    panto = resolver.resolve("Pantoprazole")
    amlo = resolver.resolve("Amlodipine")
    assert panto.model_index is not None
    assert amlo.model_index is not None

    res = service.predict(panto, amlo, fold=1)
    assert res.interaction in ("YES", "NO")
    assert 0.0 <= res.probability <= 1.0
    assert res.severity in ("MINOR", "MODERATE", "MAJOR")
    assert "Minor" in res.severity_probabilities
    assert "Moderate" in res.severity_probabilities
    assert "Major" in res.severity_probabilities
    assert res.alertCode.startswith("CDSS-DDI-")

    # Cold-start prediction
    moda = resolver.resolve("Modafinil")
    res_cold = service.predict(panto, moda, fold=1)
    assert res_cold.interaction in ("YES", "NO")
    assert "Cold-Start" in res_cold.source


def test_rag_retrieval_service():
    resolver = DrugResolver()
    rag = RAGRetrievalService()

    panto = resolver.resolve("Pantoprazole")
    moda = resolver.resolve("Modafinil")

    evidence_data = rag.retrieve_evidence(panto, moda, top_k=5)
    assert "rag_explanation" in evidence_data
    assert "evidence" in evidence_data

    expl = evidence_data["rag_explanation"]
    assert len(expl["summary"]) > 20
    assert isinstance(expl["shared_pk"], list)
    assert isinstance(expl["shared_targets"], list)
    assert isinstance(expl["shared_mechanisms"], list)
    assert isinstance(expl["top_chunks"], list)

    ev = evidence_data["evidence"]
    assert "drugbank" in ev
    assert "cpic" in ev
    assert "twosides" in ev
    assert "offsides" in ev
    assert ev["drugbank"]["drug_1_count"] > 0


def test_rag_contract_and_merger():
    resolver = DrugResolver()
    gnn = GNNPredictionService(resolver=resolver)
    rag = RAGRetrievalService()
    merger = ResultMerger()

    panto = resolver.resolve("Pantoprazole")
    moda = resolver.resolve("Modafinil")

    gnn_out = gnn.predict(panto, moda)
    rag_out = rag.retrieve_evidence(panto, moda)

    merged = merger.merge(gnn_out, rag_out, panto, moda, patient_id="MRN-1234")

    # Contract check
    assert merged.note == "T-GNN prediction and RAG evidence are reported separately."
    assert "model_prediction" in merged.to_dict()
    assert "evidence" in merged.to_dict()
    assert validate_contract_separation(merged.to_dict())

    # Unified schema fields for Frontend
    assert merged.prediction["interaction"] in ("YES", "NO")
    assert merged.prediction["severity"] in ("MINOR", "MODERATE", "MAJOR")
    assert merged.rag_explanation["summary"] is not None
    assert merged.evidence["drugbank"] is not None


def test_orchestrator_end_to_end():
    orch = ClinicalDDIOrchestrator.get_instance()

    # Health
    health = orch.health()
    assert health["status"] == "healthy"
    assert health["model"] == "TemporalDDI-GNN (model_v2)"

    # Pair evaluation
    res = orch.evaluate_pair("Pantoprazole", "Modafinil", patient_id="TEST-PATIENT")
    assert res.prediction["interaction"] in ("YES", "NO")
    assert res.meta["patient_id"] == "TEST-PATIENT"

    # Regimen evaluation
    meds = [
        {"id": "DB00213", "name": "Pantoprazole"},
        {"id": "DB00745", "name": "Modafinil"},
        {"id": "DB00381", "name": "Amlodipine"}
    ]
    regimen_out = orch.evaluate_regimen(meds, patient_id="TEST-REGIMEN")
    assert regimen_out["total_medications"] == 3
    # 3 medications -> 3 unique pairs: (0,1), (0,2), (1,2)
    assert regimen_out["total_pairs_evaluated"] == 3
    assert len(regimen_out["evaluated_pairs"]) == 3


if __name__ == "__main__":
    print("Running API Layer test suite...")
    test_drug_resolver()
    print("[PASS] test_drug_resolver")
    test_gnn_prediction_service()
    print("[PASS] test_gnn_prediction_service")
    test_rag_retrieval_service()
    print("[PASS] test_rag_retrieval_service")
    test_rag_contract_and_merger()
    print("[PASS] test_rag_contract_and_merger")
    test_orchestrator_end_to_end()
    print("[PASS] test_orchestrator_end_to_end")
    print("ALL API LAYER TESTS PASSED SUCCESSFULLY!")
