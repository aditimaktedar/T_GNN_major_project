# Clinical DDI API Layer (model_v2 + RAG Integration)

This API layer unifies **TemporalDDI-GNN (model_v2)** with the multi-source **RAG clinical evidence retrieval pipeline**.

## Architecture & Contract Separation

Per `DEPLOYMENT_NOTES.md` and `api/rag_contract.py`:
- Numerical predictions and RAG evidence are **strictly separated** in the response structure.
- RAG evidence is retrieved independently from the T-GNN numerical forward pass.
- End-to-end evaluation flow:
  $$\text{Client / Clinical Portal} \longrightarrow \text{Drug Resolver} \longrightarrow \text{T-GNN (model\_v2)} \longrightarrow \text{RAG Pipeline} \longrightarrow \text{Result Merger}$$

## Directory Structure

```
api/
├── __init__.py           # Package exports
├── __main__.py           # CLI entry point (`python3 -m api`)
├── rag_contract.py       # Strict separation contract and validation
├── schemas.py            # Domain data classes and response models
├── drug_resolver.py      # DrugBank ID, PubChem CID, and Name resolver
├── gnn_service.py        # Model inference engine (Folds 1-5 checkpoints)
├── rag_service.py        # Multi-database evidence & literature retrieval
├── merger.py             # Combines GNN and RAG into unified clinical schemas
├── service.py            # High-level ClinicalDDIOrchestrator
└── server.py             # HTTP server with CORS support
```

## Running the API Server

```bash
# Start server on http://localhost:8000
python3 -m api
```

Environment variables:
- `PORT`: HTTP port (default: `8000`)
- `HOST`: Bind host (default: `0.0.0.0`)

## Endpoints

### 1. Health Status
`GET /api/health`
```json
{
  "status": "healthy",
  "model": "TemporalDDI-GNN (model_v2)",
  "available_folds": [1, 2, 3, 4, 5],
  "indexed_drugs_count": 341,
  "indexed_evidence_chunks": 43347,
  "twosides_pairs_count": 63472
}
```

### 2. Drug Autocomplete & Lookup
`GET /api/drugs?q=pantoprazole&limit=10`

### 3. Evaluate Drug Pair (GNN + RAG Merged)
`POST /api/evaluate`
Request:
```json
{
  "drug_a": "DB00213",
  "drug_b": "DB00745",
  "patient_id": "MRN-9842-7019",
  "fold": 1
}
```
Response:
```json
{
  "prediction": {
    "interaction": "YES",
    "probability": 0.74,
    "severity": "MODERATE",
    "source": "TemporalDDI-GNN v2 [Cold-Start: Unseen Modafinil]",
    "alertCode": "CDSS-DDI-MOD-074",
    "severity_probabilities": {
      "Minor": 0.2,
      "Moderate": 0.55,
      "Major": 0.25
    }
  },
  "rag_explanation": {
    "summary": "Concomitant administration of Pantoprazole (DB00213) and Modafinil (DB00745) reveals documented pharmacological overlap in absorption, elimination, metabolism pathways...",
    "shared_targets": [],
    "shared_mechanisms": ["Competitive Transporter Binding"],
    "shared_pk": ["absorption", "elimination", "metabolism"],
    "shared_pd_effects": ["Altered Serum Bioavailability", "Risk of Additive Adverse Reactions"],
    "top_chunks": [...]
  },
  "evidence": {
    "drugbank": {
      "drug_1_count": 15,
      "drug_2_count": 9,
      "drug_1_name": "Pantoprazole",
      "drug_2_name": "Modafinil"
    },
    "cpic": {
      "drug_1_count": 8,
      "drug_2_count": 2
    },
    "twosides": {
      "observed": true,
      "ddi_type_ids": ["DDI_0", "DDI_1", "DDI_2", "DDI_5", "DDI_7"],
      "ddi_type_count": 185
    },
    "offsides": {
      "drug_1_available": true,
      "drug_2_available": true,
      "drug_1_events": 7691,
      "drug_2_events": 3428
    }
  },
  "model_prediction": { ... },
  "note": "T-GNN prediction and RAG evidence are reported separately."
}
```

### 4. Evaluate Patient MAR Regimen
`POST /api/evaluate-regimen`
Evaluates all unique pairwise interactions in a patient's active medication list, ranking them by clinical severity and probability.
