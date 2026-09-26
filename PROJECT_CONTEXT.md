# TemporalDDI-GNN — Project Context

This repository is the **TemporalDDI-GNN** university research project: a temporal graph neural network for drug-drug interaction (DDI) analysis.

This file records Member B responsibilities, the approved data sources, and project rules. It is documentation only. It does not implement the data pipeline.

## Member B

Member B is responsible for:

- MIMIC-IV prescription data processing
- Drug name normalization to RxNorm
- DDI label integration
- Drug-pair preparation
- PubChem SMILES retrieval
- RDKit molecular fingerprints
- Patient-level train/validation/test splitting
- PyTorch Geometric Dataset/DataLoader
- Logistic Regression baseline
- Static GAT baseline
- Classification metrics:
  - Accuracy
  - Precision
  - Recall
  - F1
  - AUROC
  - PR-AUC
  - MCC
- XAI metrics:
  - Fidelity
  - Sparsity
  - Stability
- DrugBank mechanism sanity checks
- RAG-related ablation numbers
- Reproducibility and documentation

Other existing folders such as `src/graph/`, `src/models/`, `src/xai/`, `outputs/`, and `data/demo/` belong to the current repository and must be preserved.

## Approved data sources

Use only these sources. Do not substitute another dataset.

### 1. MIMIC-IV

- URL: https://www.kaggle.com/datasets/mangeshwagle/mimic-iv-2-1
- Role: main clinical/prescription source for the Member B pipeline

### 2. TWOSIDES

- URL: https://github.com/jcsun-00/Twosides
- Role: drug-drug interaction information/labels

### 3. DrugBank Full Database

- URL: https://go.drugbank.com/releases/5-1-16/downloads/all-full-database
- Role: drug information and known mechanisms/relationships, including later sanity checks
- Team-supplied download command (placeholders only; never put a real email or password here or in any project file):

```bash
curl -Lfv -o filename.zip -u EMAIL:PASSWORD https://go.drugbank.com/releases/5-1-16/downloads/all-full-database
```

DrugBank is licensed/private. Credentials must never be stored in source code, committed to Git, or written into this file. Do not run this download until the team explicitly starts that stage.

### 4. PharmGKB / CPIC Clinical Pharmacogenetics

- URL: https://www.clinpgx.org/downloads
- Role: pharmacogenetics/clinical pharmacology information and project corpus/RAG-related work

### 5. OFFSIDES

- URL: https://github.com/tatonetti-lab/offsides
- Role: adverse drug-event information and DDI-related processing according to the project methodology

### 6. PubChem PUG REST

- URL: https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest
- Role: retrieve chemical information such as SMILES for project drugs, later used with RDKit

### 7. RxNorm

RxNorm is the required normalization target for MIMIC drug names.

The exact RxNorm source/mapping procedure has **not** been established. Do not invent a mapping source. That decision will be made when drug normalization is implemented.

## High-level Member B pipeline

```
MIMIC-IV
→ prescription processing
→ drug normalization to RxNorm
→ drug-pair construction
→ TWOSIDES/OFFSIDES DDI integration
→ PubChem SMILES
→ RDKit molecular features
→ patient-level split
→ PyTorch Geometric dataset
→ Logistic Regression
→ Static GAT
→ evaluation metrics
→ XAI evaluation
→ DrugBank sanity checks
```

This pipeline is not implemented yet. Work proceeds one stage at a time.

## Data handling rules

- Do not modify raw datasets. Keep originals in `data/raw/`.
- Put in-progress files in `data/interim/`.
- Put finished pipeline outputs in `data/processed/`.
- Do not invent dataset columns, labels, or drug identifiers.
- Do not assume a dataset’s structure before inspecting the actual files.
- Do not silently substitute another dataset for an approved source.
- Never hard-code passwords or API keys. Never commit credentials.

## Related existing folders (do not replace)

- `src/graph/`, `src/models/`, `src/xai/`: existing model/graph/XAI stubs
- `outputs/`: existing output placeholders (checkpoints, explanations, graphs)
- `data/demo/`: existing demo-data placeholder
- `SCHEMA.md`: existing schema document (to be filled later, not replaced here)
- `notebooks/01_graph_demo.ipynb` and `notebooks/02_graph_validation.ipynb`: existing notebooks
