# Member B Evaluation Workflow

This document describes how Member B evaluation integrates with the rest of TemporalDDI-GNN.

## Workflow

```
RAW DATA
→ processed dataset (src/data/pipeline build)
→ baseline models (Logistic Regression, Static GAT)
→ predictions / metrics
→ classification metrics
→ Member A explanations (TemporalDDI-GNN + src/xai)
→ XAI metrics (Fidelity, Sparsity, Stability)
→ DrugBank sanity check (local licensed export)
→ Member C RAG results
→ ablation comparison (WITH vs WITHOUT RAG)
→ paper-ready tables (results/metrics, results/tables)
```

## Member A integration

| Module | Status | Member B adapter |
| --- | --- | --- |
| `src/models/tgnn.py` | Stub | Supply TGNN metrics JSON via `--tgnn-metrics` |
| `src/xai/explain.py` | Stub | Supply explanation CSV/Parquet via `--explanations` |

Member B does **not** overwrite Member A code. Adapters live in `src/evaluation/adapters.py`.

## Commands

### Synthetic demo (software validation only)

```bash
source .venv/bin/activate
python -m src.baselines.run_baselines --demo
python -m src.evaluation.run_evaluation --demo
```

All demo outputs are labeled `SYNTHETIC SOFTWARE VALIDATION ONLY`.

### Real data (when available)

```bash
python -m src.data.pipeline run --mapping path/to/rxnorm_mapping.csv
python -m src.baselines.run_baselines --data data/processed/ml_dataset.parquet --experiment-prefix real_run
python -m src.evaluation.run_evaluation \
  --baseline-summary results/metrics/real_run_run_summary.json \
  --dataset data/processed/ml_dataset.parquet \
  --explanations path/to/a_explanations.csv \
  --tgnn-metrics path/to/tgnn_metrics.json \
  --drugbank-path path/to/local_drugbank_export.csv \
  --without-rag path/to/c_without_rag.json \
  --with-rag path/to/c_with_rag.json
```

Missing optional components are skipped with a clear status message.

## Output artifacts (paper-ready)

| File | Content |
| --- | --- |
| `results/metrics/MASTER_RESULTS.json` | **Single source of truth** for 955-class and frequent363 model metrics |
| `results/metrics/model_comparison.csv` | Baseline vs TGNN metrics (binary Member B eval path) |
| `results/metrics/xai_metrics.json` | Aggregate XAI metrics |
| `results/metrics/drugbank_sanity.json` | DrugBank match report |
| `results/metrics/rag_ablation.json` | Member C ablation comparison |
| `results/metrics/data_audit.json` | Dataset audit |
| `results/metrics/reproducibility.json` | Environment reproducibility |
| `results/tables/baseline_comparison.csv` | Paper table export |
| `results/tables/xai_comparison.csv` | XAI table export |

## What belongs in the paper

- Metrics from **real** processed MIMIC/TWOSIDES pipeline runs only
- XAI metrics from **real** Member A explanation outputs
- DrugBank sanity checks from **licensed local** DrugBank exports
- RAG ablation from **real** Member C experiment files

Do **not** include synthetic demo numbers in the paper.
