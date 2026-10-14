"""Tests for portal evaluation payload adapter."""

from __future__ import annotations

from api.portal_adapter import orchestrator_regimen_to_portal


def test_orchestrator_regimen_to_portal_shape() -> None:
    selected = [
        {"id": "DB00213", "name": "Pantoprazole"},
        {"id": "DB00745", "name": "Modafinil"},
    ]
    orchestrator_result = {
        "total_pairs_evaluated": 1,
        "critical_alerts_count": 0,
        "evaluated_pairs": [
            {
                "pair_label": "Pantoprazole + Modafinil",
                "drug_1": {"drugbank_id": "DB00213", "name": "Pantoprazole"},
                "drug_2": {"drugbank_id": "DB00745", "name": "Modafinil"},
                "prediction": {
                    "interaction": "YES",
                    "probability": 0.74,
                    "severity": "MODERATE",
                    "source": "TemporalDDI-GNN v2",
                    "alertCode": "CDSS-DDI-MOD-074",
                },
                "evidence_summary": "Example summary",
                "full_response": {
                    "rag_explanation": {
                        "summary": "Mechanistic overlap",
                        "shared_targets": ["CYP2C19"],
                        "shared_mechanisms": ["Enzyme inhibition"],
                        "shared_pk": ["metabolism"],
                        "shared_pd_effects": ["Bioavailability shift"],
                    },
                    "evidence": {
                        "drugbank": {"total": 10},
                        "cpic": {"total": 2},
                        "twosides": {"observed": True},
                    },
                },
            }
        ],
    }

    payload = orchestrator_regimen_to_portal(
        selected,
        orchestrator_result,
        engine_meta={"status": "healthy", "model": "TemporalDDI-GNN (model_v2)"},
    )

    assert payload["pairCount"] == 1
    assert payload["prediction"]["probability"] == 0.74
    assert payload["pairs"][0]["severity"] == "MODERATE"
    assert payload["rag_explanation"]["shared_pk"] == ["metabolism"]
    assert payload["evidence"]["twosides"]["observed"] is True


def test_portal_resolves_clinical_name_when_resolver_returns_drugbank_id() -> None:
    selected = [
        {"id": "DB00213", "name": "Pantoprazole"},
        {"id": "DB00722", "name": "Lisinopril"},
    ]
    orchestrator_result = {
        "total_pairs_evaluated": 1,
        "evaluated_pairs": [
            {
                "pair_label": "Pantoprazole + DB00722",
                "drug_1": {"drugbank_id": "DB00213", "name": "Pantoprazole"},
                "drug_2": {"drugbank_id": "DB00722", "name": "DB00722"},
                "prediction": {
                    "interaction": "NO",
                    "probability": 0.21,
                    "severity": "LOW",
                    "source": "TemporalDDI-GNN v2",
                },
                "full_response": {"rag_explanation": {}, "evidence": {}},
            }
        ],
    }

    payload = orchestrator_regimen_to_portal(selected, orchestrator_result)
    pair = payload["pairs"][0]
    assert pair["drug1"]["name"] == "Pantoprazole"
    assert pair["drug1"]["id"] == "DB00213"
    assert pair["drug2"]["name"] == "Lisinopril"
    assert pair["drug2"]["id"] == "DB00722"
    assert payload["expectedPairCount"] == 1
