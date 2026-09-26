# Frequent363 Target Formulation

**Status:** Fully implemented, trained, and evaluated.

**Source dataset (read-only):** `data/processed/final_mimic_twosides_ml.csv`

**Derived dataset:** `data/processed/final_mimic_twosides_ml_frequent363.csv`

**Report:** `results/metrics/final_mimic_twosides_ml_frequent363_report.json`

**Canonical experiment scores:** `results/metrics/MASTER_RESULTS.json` (`formulations.frequent363`)

---

## Definition

Frequency-filtered multiclass TWOSIDES interaction-type prediction:

- Keep only types with **≥ 20 training rows** (threshold derived from train split only).
- **No `"other"` class** — rare types are excluded, not merged.
- Preserve the existing pair-level train/validation/test split (101 / 21 / 23 pairs).
- Label encoder fit on **train split only**.

Build command:

```bash
python -m src.data.prepare_frequent363_ml_dataset
```

---

## Verified Dataset Counts

| Metric | Value |
|--------|------:|
| Eligible types | **363** |
| Excluded train types (<20 rows) | **592** |
| Train rows retained | **13,458** (101 unique pairs) |
| Val rows retained | **2,841** (21 unique pairs) |
| Test rows retained | **2,584** (23 unique pairs) |
| Test deduplicated `(pair_key, type)` | **2,074** |
| Pair-level split leakage | **0** |
| Val types absent from encoder | **0** |
| Test types absent from encoder | **0** |

---

## Experimental Results (Test Split)

All models evaluated on identical splits without pair leakage. Scores extracted from canonical `results/metrics/MASTER_RESULTS.json`:

### Primary Metric Level: `pair_type_dedup` (Test Split)

| Model | Top-1 Acc | Top-3 Acc | Top-5 Acc | Macro F1 | Micro F1 | Balanced Acc |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Guessing Baseline** (Theoretical) | 0.28% | 0.83% | 1.38% | — | 0.28% | 0.28% |
| **Logistic Regression** (Morgan FP) | **0.34%** | **1.11%** | **2.12%** | **0.00038** | **0.34%** | 0.25% |
| **Static GAT** (Pair graph) | 0.24% | 1.01% | 1.88% | 0.00027 | 0.24% | **0.29%** |
| **Molecular GNN** (Dual atom-level GIN) | 0.24% | 0.77% | 1.30% | 0.00009 | 0.24% | 0.22% |

### Supplementary Metric Level: `row_level` (Test Split)

| Model | Accuracy (Top-1) | Top-3 Acc | Top-5 Acc | Macro F1 | Micro F1 | Balanced Acc |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **0.39%** | **1.16%** | **2.21%** | **0.00037** | **0.39%** | 0.31% |
| **Static GAT** | 0.31% | 1.08% | 1.97% | 0.00027 | 0.31% | **0.34%** |
| **Molecular GNN** | 0.23% | 0.74% | 1.24% | 0.00009 | 0.23% | 0.22% |

---

## Key Scientific Findings

1. **Model Ranking**: Simple Logistic Regression with Morgan fingerprints outperforms both Static GAT and Molecular GNN across all primary ranking metrics (Top-1, Top-3, Top-5, and Macro F1).
2. **Comparison with Random Guessing**:
   - The Molecular GNN performs **worse than random guessing** across Top-1 (0.24% vs 0.28%), Top-3 (0.77% vs 0.83%), and Top-5 (1.30% vs 1.38%).
   - Static GAT is below random on Top-1 and marginally above random on Top-3 and Top-5.
   - All models display near-zero Macro F1 ($< 0.0004$), indicating virtually no discriminative power across individual classes.
3. **The Data Formulation Bottleneck**:
   - **Holdout Pair Scarcity**: With 0 pair leakage, the test set contains only **23 unique drug pairs**.
   - **Label Multiplicity / Saturated Ground Truth**: Across those 23 test pairs, there are 2,074 distinct `(pair_key, type)` tuples—an average of **~90 true interaction types per drug pair**.
   - **Task Mismatch**: Formulating the problem as single-label multiclass classification (where each admission row requires selecting exactly 1 of 363 classes) creates an artificial and ill-posed objective when dozens of side effects simultaneously occur for that pair in TWOSIDES.
   - **Conclusion**: Added model architecture complexity (e.g. Temporal GNN or deeper GNNs) is not the solution. The bottleneck is the formulation and data structure itself.

