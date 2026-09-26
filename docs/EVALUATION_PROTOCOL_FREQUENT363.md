# Evaluation Protocol — Frequent363 (363-Class) Formulation

Protocol for evaluating multiclass TWOSIDES interaction-type prediction on:

`data/processed/final_mimic_twosides_ml_frequent363.csv`

**Status:** Active for Phase 2 (post-formulation-change) baselines.

Implementation: `src/evaluation/frequent363_protocol.py`  
CLI: `python -m src.evaluation.run_frequent363_protocol`

**Canonical scores:** `results/metrics/MASTER_RESULTS.json` (`formulations.frequent363`).  
See `docs/MASTER_RESULTS.md`. New training overwrites the model entry in that file.

The 955-class protocol in `docs/EVALUATION_PROTOCOL.md` remains unchanged.

---

## 1. Task and label space

| Item | Value |
|------|------:|
| Target column | `type` (TWOSIDES interaction code) |
| Eligible types | **363** (≥ 20 train rows) |
| Model output dimension | 363 logits (indices 0–362) |
| `"other"` class | **None** — rare types excluded at dataset build |

Label encoder is **fit on training split only**. All val/test types in the derived dataset appear in train.

---

## 2. Evaluation splits

Use the existing `split` column from the source dataset. **Do not re-split.**

| Split | Purpose |
|-------|---------|
| `train` | Training and encoder fitting |
| `val` | Model selection |
| `test` | Final reporting |

Pair-level splitting is preserved: no `pair_key` spans multiple splits.

---

## 3. Metrics

| Metric key | Definition |
|------------|------------|
| `top_1_accuracy` | Exact-match accuracy |
| `top_3_accuracy` | True class in top-3 predictions |
| `top_5_accuracy` | True class in top-5 predictions |
| `macro_f1` | Unweighted mean F1 across classes in slice |
| `micro_f1` | Global F1 over all TP/FP/FN |
| `balanced_accuracy` | Mean per-class recall |

---

## 4. Evaluation slice

### `full_363`

- **Definition:** All rows in the frequent363 derived dataset.
- **Rationale:** Dataset build already excludes rare types; no additional slice filtering needed.

---

## 5. Aggregation levels

| Level | Definition | Role |
|-------|------------|------|
| `pair_type_dedup` | One prediction per unique `(pair_key, type)`; first row kept | **Primary** |
| `row` | One prediction per admission-level row | Supplementary |

---

## 6. Required reporting

Each run must report:

- Number of eligible classes (363)
- Train / val / test row counts
- Train / val / test unique `(pair_key, type)` counts
- Train class frequency summary
- Excluded type count (592 from source train)
- Pair-level leakage check (expect 0)
- Val/test labels absent from train encoder (expect none)

Report JSON: `results/metrics/final_mimic_twosides_ml_frequent363_report.json`

---

## 7. Checkpoint naming

Frequent363 models use prefix `final_mimic_twosides_frequent363` to avoid overwriting 955-class checkpoints:

```
results/baselines/final_mimic_twosides_frequent363_multiclass_logistic_regression.joblib
results/baselines/final_mimic_twosides_frequent363_multiclass_static_gat.pt
results/baselines/final_mimic_twosides_frequent363_multiclass_molecular_gnn.pt
```
