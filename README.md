# TemporalDDI-GNN

A temporal graph neural network project for drug-drug interaction analysis.

This repository is a university research project. Member B owns the clinical data pipeline, baselines, and evaluation work described below. Existing graph/model/XAI folders from the rest of the team are preserved.

## Member B responsibilities

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
- Classification metrics: Accuracy, Precision, Recall, F1, AUROC, PR-AUC, MCC
- XAI metrics: Fidelity, Sparsity, Stability
- DrugBank mechanism sanity checks
- RAG-related ablation numbers
- Reproducibility and documentation

## Environment

Use the project virtual environment. Do **not** use the macOS system Python 3.9.6.

Python 3.12 was not installed on this Mac. The environment uses **Python 3.14.2** because it was the only non-3.9 interpreter available, and PyTorch Geometric 2.8 documents Python 3.10–3.14 with PyTorch 2.9–2.12.

```bash
source .venv/bin/activate
python -m pytest
```

Create the environment again later with:

```bash
/Library/Frameworks/Python.framework/Versions/3.14/bin/python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

`.venv/` is gitignored. Do not commit it.

## Approved data sources

Use only these sources. Do not substitute another dataset.

| Source | URL | Role |
| --- | --- | --- |
| MIMIC-IV | https://www.kaggle.com/datasets/mangeshwagle/mimic-iv-2-1 | Clinical/prescription data |
| TWOSIDES | https://github.com/jcsun-00/Twosides | DDI information/labels |
| DrugBank Full Database | https://go.drugbank.com/releases/5-1-16/downloads/all-full-database | Drug information and mechanism sanity checks |
| PharmGKB / CPIC | https://www.clinpgx.org/downloads | Pharmacogenetics / RAG-related corpus |
| OFFSIDES | https://github.com/tatonetti-lab/offsides | Adverse drug-event / DDI-related processing |
| PubChem PUG REST | https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest | SMILES and chemical information |
| RxNorm | *source not chosen yet* | Required normalization target for MIMIC drug names |

See `PROJECT_CONTEXT.md` and `DATA_SOURCE_NOTES.md` for source notes.

## Data setup

Place original files under `data/raw/` in source-specific folders when they are obtained later:

- `data/raw/mimic/`
- `data/raw/twosides/`
- `data/raw/offsides/`
- `data/raw/pharmgkb/`
- `data/raw/drugbank/`

Rules:

- **Do not modify raw datasets.** Keep originals in `data/raw/`. Write cleaned files to `data/interim/` or `data/processed/`.
- **Never commit credentials**, passwords, API keys, or DrugBank login details.
- **DrugBank is licensed/private.** Do not download it until that stage, and never store DrugBank usernames or passwords in this repository.
- **MIMIC-IV may require appropriate access/credentials** (PhysioNet/Kaggle access). Do not commit those credentials.
- **PubChem** will later be queried programmatically through PUG REST and cached locally. Do not hard-code API keys.
- **RxNorm mapping will be implemented later**, after the exact mapping strategy is established. Do not invent a mapping source now.
- `data/demo/` contains **synthetic** files for software testing only. They are not real medical data and must not be used for training or results.

Inspect a local file only after it exists on disk. Do not assume columns from the filename.

### Inspect a file

From the repository root, with the virtual environment activated:

```bash
python -m src.data.inspect_data path/to/file.csv
```

Example using the synthetic demo file (not real medical data):

```bash
python -m src.data.inspect_data data/demo/synthetic_drug_pairs.csv
```

Optional arguments:

```bash
python -m src.data.inspect_data path/to/file.csv --sample-rows 5 --chunk-size 100000
```

### Inventory local data folders

This lists files under `data/raw/`, `data/interim/`, and `data/processed/` only. It does not download anything.

```bash
python -m src.data.inventory
```

JSON output:

```bash
python -m src.data.inventory --json
```

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

This pipeline is implemented as independent CLI stages. Real MIMIC/TWOSIDES/OFFSIDES files are still required for a full run.

## Pipeline commands

Activate the environment first:

```bash
source .venv/bin/activate
```

### Dataset placement

Put unmodified approved files here:

- MIMIC-IV → `data/raw/mimic/`
- TWOSIDES → `data/raw/twosides/` (decompress `.7z` to CSV first)
- OFFSIDES → `data/raw/offsides/` if a local table exists
- RxNorm mapping CSV/TSV → `data/raw/rxnorm/` or pass `--mapping`

### Data inspection

```bash
python -m src.data.inspect_data path/to/file.csv
python -m src.data.inspect_data data/demo/synthetic_drug_pairs.csv
python -m src.data.inventory
python -m src.data.pipeline inspect
python -m src.data.pipeline inspect data/demo/synthetic_drug_pairs.csv
```

### Each pipeline stage

```bash
python -m src.data.pipeline ingest-mimic
python -m src.data.pipeline clean
python -m src.data.pipeline normalize --mapping path/to/rxnorm_mapping.csv
python -m src.data.pipeline pairs --rule interval_overlap
python -m src.data.pipeline twosides
python -m src.data.pipeline offsides
python -m src.data.pipeline pubchem-candidates
python -m src.data.pipeline labels-twosides
python -m src.data.pipeline prepare-ml-twosides
python -m src.data.pipeline labels --negative-mode none
python -m src.data.pipeline pubchem
python -m src.data.pipeline features --radius 2 --n-bits 2048
python -m src.data.pipeline splits --train 0.70 --val 0.15 --test 0.15 --seed 42
python -m src.data.pipeline build
```

`labels --negative-mode none` keeps unmatched MIMIC pairs as `unknown`. It does **not** treat absence from TWOSIDES as a true negative.

`prepare-ml-twosides` builds a streaming ML dataset from the real MIMIC ↔ TWOSIDES join: deduplicated positives (963 interaction types preserved), deterministically sampled negatives from real unmatched MIMIC pairs with verified CIDs, and patient-level train/val/test splits. Default `--negative-ratio 1.0 --seed 42`.

Pair rule `interval_overlap` requires `start_time` and `end_time`. Use `--rule same_admission` only if `admission_id` exists.

### Full pipeline

Runs only if required local sources exist. Missing files stop the run; they are not invented.

```bash
python -m src.data.pipeline run --mapping path/to/rxnorm_mapping.csv
```

### Tests

```bash
source .venv/bin/activate
python -m pytest
```

## Machine learning baselines

**REAL DATA REQUIRED FOR SCIENTIFIC RESULTS.** Synthetic demo runs are software validation only.

Configuration: `configs/baselines.json`

### Synthetic demo (software test only)

```bash
source .venv/bin/activate
python -m src.baselines.run_baselines --demo
```

Outputs:
- `results/baselines/synthetic_demo_logistic_regression.joblib`
- `results/baselines/synthetic_demo_static_gat.pt`
- `results/metrics/synthetic_demo_*`

### Real processed dataset

After `python -m src.data.pipeline build`:

```bash
python -m src.baselines.run_baselines --data data/processed/ml_dataset.parquet --experiment-prefix real_run
```

Run one model only:

```bash
python -m src.baselines.run_baselines --demo --models logistic_regression
python -m src.baselines.run_baselines --demo --models static_gat
```

Pair feature method (Logistic Regression): set `pair_feature_method` in `configs/baselines.json` to `concat`, `abs_diff`, or `product`.

## Evaluation pipeline

Member B evaluation integrates baseline metrics, Member A explanations, DrugBank sanity checks, and Member C RAG ablation results.

See `docs/MEMBER_B_EVALUATION.md` and `docs/XAI_EVALUATION.md` for full workflow and metric definitions.

### Synthetic demo (software validation only)

```bash
source .venv/bin/activate
python -m src.baselines.run_baselines --demo
python -m src.evaluation.run_evaluation --demo
```

All demo outputs are labeled `SYNTHETIC SOFTWARE VALIDATION ONLY`. Do not use them in the paper.

### Real evaluation (when data and team outputs exist)

```bash
python -m src.evaluation.run_evaluation \
  --baseline-summary results/metrics/real_run_run_summary.json \
  --dataset data/processed/ml_dataset.parquet \
  --explanations path/to/member_a_explanations.csv \
  --tgnn-metrics path/to/tgnn_metrics.json \
  --drugbank-path path/to/local_drugbank_export.csv \
  --without-rag path/to/c_without_rag.json \
  --with-rag path/to/c_with_rag.json
```

Missing optional inputs are skipped with a clear status message.

### Paper-ready outputs

| Path | Description |
| --- | --- |
| `results/metrics/model_comparison.csv` | LR, Static GAT, optional TemporalDDI-GNN |
| `results/metrics/xai_metrics.json` | Fidelity, Sparsity, Stability |
| `results/metrics/drugbank_sanity.json` | Local DrugBank match report |
| `results/metrics/rag_ablation.json` | Member C WITH vs WITHOUT RAG |
| `results/metrics/data_audit.json` | Dataset counts and leakage check |
| `results/metrics/reproducibility.json` | Environment and git metadata |
| `results/tables/baseline_comparison.csv` | Paper table export |
| `results/tables/xai_comparison.csv` | XAI table export |

### Member A / C integration status

- `src/models/tgnn.py` — stub; supply `--tgnn-metrics` when Member A delivers predictions
- `src/xai/explain.py` — stub; supply `--explanations` when Member A delivers XAI outputs
- Member C RAG — supply `--without-rag` and `--with-rag` result JSON files

Adapters: `src/evaluation/adapters.py`

## Repository layout

```
data/raw/              original datasets (do not modify)
data/interim/          in-progress processing files
data/processed/        finished pipeline outputs
data/demo/             synthetic demo files for software tests only

src/data/              Member B data pipeline (ingest, clean, normalize, pairs, labels, splits)
src/features/          PubChem retrieval and RDKit fingerprints
src/baselines/         Logistic Regression and Static GAT baselines
src/evaluation/        Classification, XAI, DrugBank, RAG ablation, audit, export
src/graph/             existing graph-construction code (preserved)
src/models/            existing model code (preserved)
src/xai/               existing XAI code (preserved)

results/baselines/     baseline result files
results/metrics/       metric result files
results/figures/       plots and figures
outputs/               existing output placeholders (preserved)

notebooks/             notebooks (existing graph notebooks preserved)
tests/                 tests (existing graph tests preserved)
```

## Important rules

- **Do not modify raw datasets.** Keep originals in `data/raw/` and write cleaned versions to `data/interim/` or `data/processed/`.
- **Never commit credentials**, passwords, API keys, or DrugBank login details. Use a local `.env` file if needed; `.gitignore` is set to exclude it.
- Do not invent columns, labels, drug identifiers, or experimental results.
- Do not assume a dataset’s structure before inspecting the actual files.

## Basic future workflow

1. Keep existing team files.
2. Obtain an approved dataset into `data/raw/` (later stages).
3. Run the inventory and inspection commands above.
4. Add one pipeline stage at a time.
5. Record how to run each stage and how to check that it worked.

## Further documentation

- `PROJECT_CONTEXT.md` — Member B scope, approved sources, and rules
- `DATA_SOURCE_NOTES.md` — what public source docs currently establish
- `SCHEMA.md` — data schema notes (existing file; to be filled later)
- `docs/MEMBER_B_EVALUATION.md` — evaluation workflow and integration with Members A/C
- `docs/XAI_EVALUATION.md` — operational XAI metric definitions
