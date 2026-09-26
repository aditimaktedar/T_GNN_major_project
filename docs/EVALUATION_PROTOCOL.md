# Evaluation Protocol — 955-Class DDI Baselines

Protocol for evaluating multiclass TWOSIDES interaction-type prediction on:

`data/processed/final_mimic_twosides_ml.csv`

**Status:** Active for Phase 1 baselines (Multiclass LR, Static GAT).

This document defines **how** to report results. It does **not** modify the dataset, remove classes, or merge labels.

Implementation: `src/evaluation/multiclass_protocol.py`  
CLI utilities: `python -m src.evaluation.run_multiclass_protocol`

**Canonical scores:** `results/metrics/MASTER_RESULTS.json` (see `docs/MASTER_RESULTS.md`).  
Do not add per-run summary files for new 955-class training.

---

## 1. Task and label space

| Item | Value |
|------|------:|
| Target column | `type` (TWOSIDES interaction code) |
| Total unique types in dataset | 962 |
| Train-fitted encoder classes | **955** (7 types absent from train) |
| Model output dimension | 955 logits (indices 0–954) |

The label encoder is **fit on the training split only**. Types that never appear in train map to `label_index = -1` and are excluded from metric computation (they are not learnable).

---

## 2. Evaluation splits

Use the existing `split` column. **Do not re-split.**

| Split | Purpose |
|-------|---------|
| `train` | Model training and label-encoder fitting only |
| `val` | Model selection / hyperparameter tuning |
| `test` | Final held-out reporting |

Pair-level splitting is enforced: no `pair_key` appears in more than one split.

---

## 3. Metrics

All metrics are computed on **encoded class indices** (0–954).

| Metric key | Definition |
|------------|------------|
| `top_1_accuracy` | Exact-match accuracy (argmax prediction) |
| `top_3_accuracy` | Fraction of samples where true class is in top-3 predicted classes |
| `top_5_accuracy` | Fraction of samples where true class is in top-5 predicted classes |
| `macro_f1` | Unweighted mean F1 across classes present in the evaluation slice |
| `micro_f1` | Global F1 aggregated over all TP/FP/FN in the slice |
| `balanced_accuracy` | Mean per-class recall (supplementary) |

Top-k metrics require predicted class probabilities (`y_prob` shape: `[n_samples, 955]`).

---

## 4. Evaluation slices

Three slices are defined. **No classes are removed from the dataset**; slices filter which **evaluation rows** are scored.

### 4.1 `full_955`

- **Definition:** All evaluable rows with `label_index >= 0` (types seen during encoder fitting).
- **Purpose:** Reports performance under the full 955-class model head.
- **Note:** At evaluation time this selects the same rows as `seen_class` because unseen train types cannot be encoded. The name documents the model output size.

### 4.2 `seen_class`

- **Definition:** Rows whose TWOSIDES `type` appears in the training split (`label_index >= 0`).
- **Purpose:** Explicitly excludes the 7 types never seen in train (6 in val, 4 in test; 14 rows total).
- **Unseen types (never in train):** 115, 438, 618, 762, 852, 898, 951

### 4.3 `frequent_ge_20`

- **Definition:** Rows whose `type` has **≥ 20 training examples**.
- **Purpose:** Restricts evaluation to types with enough training support for meaningful classification.
- **Threshold:** 20 train rows (configurable via `min_train_examples`).
- **Train types meeting threshold:** 363 of 955

---

## 5. Aggregation levels

Each slice is reported at two levels:

### 5.1 Row level (`row`)

- One sample per dataset row (admission-level).
- Matches training input granularity.
- **Caveat:** 7,756 rows are duplicate `(pair_key, type)` observations; counting them multiple times may inflate confidence in metrics.

### 5.2 Deduplicated pair–type level (`pair_type_dedup`)

- One sample per unique `(pair_key, type)` within the split.
- Keeps the **first** row when duplicates exist (features and label are identical across duplicates).
- **Recommended primary reporting level** for generalization assessment.

---

## 6. Slice inventory (measured on current dataset)

Counts below exclude rows with invalid SMILES fingerprints (14 rows total) and rows with unseen train types.

### Validation

| Slice | Level | Samples | Classes |
|-------|-------|--------:|--------:|
| `full_955` | row | 3,907 | 732 |
| `full_955` | pair_type_dedup | 2,636 | 732 |
| `seen_class` | row | 3,907 | 732 |
| `seen_class` | pair_type_dedup | 2,636 | 732 |
| `frequent_ge_20` | row | 2,841 | 357 |
| `frequent_ge_20` | pair_type_dedup | 1,858 | 357 |

Excluded from val: **9 rows** with types absent from train.

### Test

| Slice | Level | Samples | Classes |
|-------|-------|--------:|--------:|
| `full_955` | row | 3,967 | 839 |
| `full_955` | pair_type_dedup | 3,227 | 839 |
| `seen_class` | row | 3,967 | 839 |
| `seen_class` | pair_type_dedup | 3,227 | 839 |
| `frequent_ge_20` | row | 2,584 | 359 |
| `frequent_ge_20` | pair_type_dedup | 2,074 | 359 |

Excluded from test: **5 rows** with types absent from train.

Regenerate counts:

```bash
python -m src.evaluation.run_multiclass_protocol --describe-slices
```

Output: `results/metrics/multiclass_evaluation_protocol_slices.json`

---

## 7. Primary vs supplementary metrics

### Primary (headline results)

Report these on **validation and test**, using **deduplicated `(pair_key, type)`** level:

| Priority | Slice | Metrics |
|----------|-------|---------|
| **1** | `seen_class` + `pair_type_dedup` | `top_3_accuracy`, `top_5_accuracy` |
| **2** | `frequent_ge_20` + `pair_type_dedup` | `top_3_accuracy`, `top_5_accuracy`, `macro_f1` |

**Rationale:** Top-k metrics are more informative than exact 955-way classification for this label space. Deduplicated pair–type evaluation avoids inflating scores via repeated admissions. The frequent-class slice covers types with sufficient training evidence.

### Supplementary (diagnostics)

| Slice | Level | Metrics | Notes |
|-------|-------|---------|-------|
| `full_955` | row | `top_1_accuracy`, `micro_f1` | Dominated by frequent types; row duplication |
| `seen_class` | row | all metrics | Admission-level view |
| any | any | `balanced_accuracy` | Per-class recall average |

**Do not use row-level `top_1_accuracy` on `full_955` as the sole success criterion.** Random baseline ≈ 0.1% (1/955).

---

## 8. Evaluation procedure

After model training, for each split (`val`, `test`):

1. Load split rows from the processed CSV (do not modify the file).
2. Attach fingerprints and encode labels with the **train-fitted** `TypeLabelEncoder`.
3. Generate predictions: `y_pred` (int, shape `[n]`) and `y_prob` (float, shape `[n, 955]`).
4. Call `evaluate_multiclass_protocol()` from `src/evaluation/multiclass_protocol.py`.
5. Save the full protocol JSON alongside model checkpoints.

```python
from src.evaluation.multiclass_protocol import evaluate_multiclass_protocol

results = evaluate_multiclass_protocol(
    frame=enriched_frame,
    predictions_by_split={
        "val": val_predictions_df,   # columns: pair_key, type, split, label_index, y_pred, y_prob
        "test": test_predictions_df,
    },
    label_encoder=encoder,
    min_train_examples=20,
)
```

---

## 9. Reporting template

For each model, report at minimum:

```
Model: <name>
Split: test
Primary:
  seen_class / pair_type_dedup — top_3: X.XX, top_5: X.XX
  frequent_ge_20 / pair_type_dedup — top_3: X.XX, top_5: X.XX, macro_f1: X.XX
Supplementary:
  seen_class / row — top_1: X.XX, micro_f1: X.XX
Slice sizes:
  seen_class / pair_type_dedup — n=3227, classes=839
  frequent_ge_20 / pair_type_dedup — n=2074, classes=359
```

---

## 10. Sanity checks

Verify the protocol without training:

```bash
python -m src.evaluation.run_multiclass_protocol --sanity-check
```

This command:
- Computes slice counts on the real dataset
- Runs random baseline predictions through the full metric pipeline
- Validates that all metrics are finite and top-5 ≥ top-1

Output: `results/metrics/multiclass_evaluation_protocol_sanity.json`

---

## 11. Related documents

- `docs/ML_DESIGN.md` — model architecture and training design
- `src/evaluation/multiclass.py` — low-level metric functions
- `src/data/final_ml_dataset.py` — dataset adapter and label encoding
