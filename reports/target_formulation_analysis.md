# Technical Report: Target Formulation Analysis for TemporalDDI-GNN

**Status:** Technical Analysis & Dataset Audit  
**Date:** September 21, 2026  
**Scope:** Investigation of Target Definition & Multi-Label Dataset Support  
**Primary Finding:** 100% of drug pairs in the dataset have multiple co-occurring TWOSIDES interaction types (mean: 129.94 types/pair; median: 106 types/pair). Single-label multiclass classification is mathematically ill-posed and discards 99.23% of true DDI associations.

---

## 1. Executive Summary

This report investigates why all existing baseline models (Logistic Regression, Static GAT, Molecular GNN) achieved near-random accuracy ($\approx 0.1\% - 0.3\%$ Top-1) under both the 955-class and frequent363-class multiclass formulations.

Our empirical audit of `data/selected/final_mimic_twosides_dataset.csv`, `data/processed/final_mimic_twosides_ml.csv`, and `data/processed/final_mimic_twosides_ml_frequent363.csv` reveals that **the performance bottleneck is a target-definition problem, not a model architecture failure**:

1. **Zero Single-Label Pairs**: Not a single drug pair in the entire dataset is associated with only one TWOSIDES interaction type.
2. **Massive Co-Occurrence**: Drug pairs carry between 6 and 402 distinct interaction types simultaneously (mean: 129.94 types per pair in full dataset; 87.81 in frequent363).
3. **Severe Information Loss**: Formulating the problem as single-label multiclass (one target per admission row) discards **99.23% (18,696 out of 18,841)** of true interaction label associations in the full dataset.
4. **Conflicting Supervision**: Feeding the exact same drug pair $(SMILES_A, SMILES_B)$ with 100+ different single-label targets causes CrossEntropy Loss gradients to cancel out across rows, forcing model probabilities toward uniform random distribution ($\sim 1/955 \approx 0.10\%$).

---

## 2. Dataset Statistics & Target Quantification

We quantified interaction type distributions across all unique drug pairs (`pair_key`) separately for the Full ML Dataset (955-class candidate set) and the Frequent363 Dataset (363-class candidate set):

### Summary Overview

| Metric | Full ML Dataset (955-class) | Frequent363 ML Dataset (363-class) |
| :--- | :---: | :---: |
| **Total Rows** | 26,597 | 18,883 |
| **Unique Drug Pairs** | 145 | 145 |
| **Unique TWOSIDES Types** | 962 (955 in train) | 363 (363 in train) |
| **Unique Pair-Type Associations** | 18,841 | 12,732 |
| **Train Pairs / Rows** | 101 pairs / 18,709 rows | 101 pairs / 13,458 rows |
| **Val Pairs / Rows** | 21 pairs / 3,916 rows | 21 pairs / 2,841 rows |
| **Test Pairs / Rows** | 23 pairs / 3,972 rows | 23 pairs / 2,584 rows |

### Distribution of Interaction Types per Drug Pair

| Statistic / Bin | Full ML Dataset (955-class) | Frequent363 ML Dataset (363-class) |
| :--- | :---: | :---: |
| **Min Types per Pair** | 6 | 6 |
| **Max Types per Pair** | 402 | 216 |
| **Mean Types per Pair** | **129.94** | **87.81** |
| **Median Types per Pair** | **106.0** | **82.0** |
| **Standard Deviation** | 90.01 | 49.25 |
| **25th Percentile** | 62.0 | 50.0 |
| **75th Percentile** | 184.0 | 132.0 |
| **90th Percentile** | 271.4 | 152.0 |
| **95th Percentile** | 307.2 | 171.6 |
| **Pairs with exactly 1 type** | **0 (0.0%)** | **0 (0.0%)** |
| **Pairs with exactly 2 types** | **0 (0.0%)** | **0 (0.0%)** |
| **Pairs with exactly 3 types** | **0 (0.0%)** | **0 (0.0%)** |
| **Pairs with $\ge 5$ types** | **145 (100.0%)** | **145 (100.0%)** |
| **Pairs with $\ge 10$ types** | **144 (99.3%)** | **143 (98.6%)** |
| **Pairs with $\ge 20$ types** | **139 (95.9%)** | **136 (93.8%)** |
| **Pairs with $\ge 50$ types** | **117 (80.7%)** | **109 (75.2%)** |
| **Pairs with $\ge 100$ types** | **78 (53.8%)** | **55 (37.9%)** |

---

## 3. Evidence of Information Loss

### Quantitative Information Loss Metrics

1. **Percentage of Multi-Type Pairs**: **100.0%** (145 of 145 drug pairs have $\ge 6$ interaction types).
2. **Percentage of Rows in Multi-Type Pairs**: **100.0%** (26,597 of 26,597 rows belong to multi-type pairs).
3. **Average Labels per Pair**: **129.94** (full) vs **87.81** (frequent363).
4. **Discarded Label Associations**: If only 1 label per pair is retained, **18,696 out of 18,841 true interaction associations (99.23%)** are discarded in the full dataset, and **12,587 out of 12,732 (98.86%)** are discarded in the frequent363 dataset.

### Technical Explanation for Supervised Learning Failure

- **Violation of Mutually Exclusive Softmax Assumption**: Single-label multiclass classification assumes that classes are mutually exclusive (i.e. $\sum P(y=c) = 1$ where exactly one class $c$ is true). In TWOSIDES, side effects co-occur simultaneously.
- **Conflicting Supervision Signals**: In the row-level dataset, pair `CID000001983|CID000002244` appears in 276 rows with 276 different `type` labels. When fitting a model, row 1 demands predicting `type=0`, row 2 demands predicting `type=1`, etc. Softmax loss penalizes the model for outputting high probability for `type=0` when processing row 2, even though `type=0` is a valid side effect for that pair.
- **Gradient Cancellation**: The loss gradients from conflicting rows cancel out, driving output predictions toward a uniform random distribution ($\sim 1/955 \approx 0.10\%$).

---

## 4. Train / Validation / Test Split Audit

We audited the split integrity of `final_mimic_twosides_ml.csv` and `final_mimic_twosides_ml_frequent363.csv`:

### Split Integrity & Leakage Check

| Metric / Check | Train Split | Validation Split | Test Split |
| :--- | :---: | :---: | :---: |
| **Unique Pairs (`pair_key`)** | 101 | 21 | 23 |
| **Total Rows (Full Dataset)** | 18,709 | 3,916 | 3,972 |
| **Unique Types (Full Dataset)** | 955 | 738 | 843 |
| **Unique Pair-Type Combos (Full)** | 12,966 | 2,644 | 3,231 |
| **Total Rows (Frequent363)** | 13,458 | 2,841 | 2,584 |
| **Unique Types (Frequent363)** | 363 | 357 | 359 |
| **Unique Pair-Type Combos (363)** | 8,800 | 1,858 | 2,074 |

- **Pair Leakage Check**:
  - `train` $\cap$ `val` pair overlap = **0 pairs**
  - `train` $\cap$ `test` pair overlap = **0 pairs**
  - `val` $\cap$ `test` pair overlap = **0 pairs**
- **Co-occurrence Preservation**: All rows and all associated interaction types for any given `pair_key` reside 100% within the assigned split. No pair labels leak across split boundaries.

---

## 5. Comparison of Possible Target Formulations

| Aspect | Formulation A: Single-Label Multiclass | Formulation B: Pair-Level Multi-Label (Full 962) | Formulation C: Frequency-Filtered Pair Multi-Label (363) |
| :--- | :--- | :--- | :--- |
| **Prediction Target** | 1 class index out of 955/363 | Binary vector $\mathbf{y} \in \{0,1\}^{962}$ | Binary vector $\mathbf{y} \in \{0,1\}^{363}$ |
| **Input Unit** | 1 admission row (26,597 rows) | 1 unique drug pair (145 pairs) | 1 unique drug pair (145 pairs) |
| **Information Retention** | 0.77% (99.23% discarded) | **100.0%** (0% discarded) | **67.6%** (rare types dropped, 0% pair-type loss for frequent) |
| **Supervision Validity** | **Invalid** (conflicting row labels) | **Valid** (multi-hot binary vector) | **Valid** (multi-hot binary vector over frequent types) |
| **LR Feasibility** | Done (Top-1 = 0.34%) | Feasible (Multi-output Binary LR) | Feasible (Multi-output Binary LR) |
| **Static GAT Feasibility** | Done (Top-1 = 0.24%) | Feasible (BCEWithLogitsLoss head) | Feasible (BCEWithLogitsLoss head) |
| **Molecular GNN Feasibility** | Done (Top-1 = 0.24%) | Feasible (BCEWithLogitsLoss head) | Feasible (BCEWithLogitsLoss head) |
| **TGNN Compatibility** | Compatible (if timestamped) | Compatible (if timestamped) | Compatible (if timestamped) |
| **Evaluation Metrics** | Top-k Acc, Macro F1, Micro F1 | Micro/Macro F1, Hamming Loss, Jaccard, AP@k | Micro/Macro F1, Hamming Loss, Jaccard, AP@k |
| **Scientific Validity** | **Low** (ill-posed target) | **High** (true DDI profile) | **High** (statistically balanced DDI profile) |

---

## 6. Audit of Demographics (Role of Age)

We inspected `anchor_age` across all datasets:
- **Availability**: Present in `data/selected/final_mimic_twosides_dataset.csv` and `data/processed/final_mimic_twosides_ml.csv` (26,597 rows, 0 missing values).
- **Distribution**: Integer age range $23 - 91$, mean = $58.19$, std = $16.65$.
- **Current Feature Pipeline Audit**:
  - `src/features/pair_features.py`: **NOT USED** (computes Morgan fingerprint features only).
  - `src/features/rdkit_features.py`: **NOT USED**.
  - `MulticlassLogisticRegressionBaseline`: **NOT USED**.
  - `MulticlassStaticGATTrainer`: **NOT USED**.
  - `DualMolecularGNN` / `MulticlassMolecularGNNTrainer`: **NOT USED**.
- **Conclusion**: `anchor_age` exists as a column in the processed CSV but is **100% unused** by all existing ML models and feature builders.

---

## 7. Pair-Level Multi-Label Schema Definition

To support multi-label training, the dataset should be transformed into **one row per unique `pair_key`** (145 rows total; 101 train / 21 val / 23 test):

```json
{
  "pair_key": "CID000001983|CID000002244",
  "drug_a": "CID000001983",
  "drug_b": "CID000002244",
  "smiles_a": "CC(=O)NC1=CC=C(C=C1)O",
  "smiles_b": "CC(=O)OC1=CC=CC=C1C(=O)O",
  "split": "train",
  "anchor_age": 58.2,
  "multi_label_types": [0, 1, 2, 5, 8, 13, 14, 21, 23, 26, ...]
}
```

This reduces the dataset size from 26,597 admission rows to **145 unique pair profile vectors**, eliminating duplicate row conflicts and matching true DDI multi-label structure.

---

## 8. Technical Recommendations & Direct Answers

1. **Is single-label multiclass still scientifically appropriate for this dataset?**  
   **No**. Single-label multiclass is mathematically and scientifically invalid because 100% of drug pairs have multiple co-occurring interaction types (mean ~130 types per pair).
2. **Does the dataset support pair-level multi-label prediction?**  
   **Yes**. The dataset contains 145 unique drug pairs with complete SMILES and multi-hot label sets.
3. **If yes, what exactly should the target look like?**  
   A multi-hot binary vector $\mathbf{y} \in \{0, 1\}^K$ ($K=363$ for frequent types or $K=962$ for all types) for each unique `pair_key`.
4. **Should the 363 frequent classes be retained, changed, or reconsidered?**  
   **Retain the 363 frequent classes** as the primary multi-label output space ($K=363$) because rare classes ($<20$ train rows) have insufficient positive examples to evaluate multi-label precision/recall reliably.
5. **What should the next experiment be?**  
   Transform dataset to pair-level multi-label format and train Multi-Label Logistic Regression (BCE loss on Morgan fingerprints) as the new multi-label baseline.
6. **Which existing models should be reused after the formulation change?**  
   Reuse `LogisticRegression`, `StaticGAT`, and `DualMolecularGNN` by swapping the final Softmax/CrossEntropy head with a Sigmoid/BCEWithLogitsLoss head over $K$ binary labels.
7. **Which evaluation metrics should be used for multi-label prediction?**  
   Micro F1, Macro F1, Hamming Loss, Jaccard Similarity Coefficient, and Mean Average Precision (mAP / Ranking AP@k).
8. **What should NOT be worked on yet?**  
   Do NOT implement Temporal GNN, do NOT add new complex GNN architectures, do NOT run hyperparameter tuning, do NOT overwrite `MASTER_RESULTS.json`.
