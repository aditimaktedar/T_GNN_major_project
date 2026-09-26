# Target Formulation Analysis

**Status:** design/analysis only — no dataset changes, no new training.

**Dataset:** `data/processed/final_mimic_twosides_ml.csv` (26,597 rows, 145 pairs, 962 types)

**Context:** Three baselines (Logistic Regression, Static GAT, Molecular GNN) were evaluated under the current **955-class** formulation. Results in `docs/BASELINE_COMPARISON.md` show all models perform near the random baseline for top-1 accuracy (~0.1%) with no clear gain from added complexity.

This document investigates two alternative formulations **before** any further modeling.

**Machine-readable output:** `results/metrics/target_formulation_analysis.json`

---

## 1. Current formulation (baseline)

| Aspect | Definition |
|--------|------------|
| **Unit** | One row = one admission-level `(pair, type)` observation |
| **Input** | `smiles_a`, `smiles_b` |
| **Target** | Single TWOSIDES interaction type (955 train-fit classes) |
| **Split** | Pair-level (101 train / 21 val / 23 test pairs) |

**Problem:** 955 classes from 145 pairs → extreme label sparsity; baselines achieve &lt;1.3% top-5 on primary test slices.

---

## 2. Option 1 — Frequency-filtered multiclass (≥20 train examples)

### 2.1 Definition

Keep only interaction types with **≥20 training rows**. Types below the threshold are **not used as separate classes** (either excluded from training/evaluation or optionally collapsed into an `"other"` bucket — see §2.5).

Threshold matches the existing `frequent_ge_20` evaluation slice in `docs/EVALUATION_PROTOCOL.md`.

### 2.2 Which types qualify?

| Category | Count |
|----------|------:|
| **Frequent types (≥20 train rows)** | **363** |
| Rare types in train (&lt;20 rows) | 592 |
| Types never appearing in train | 7 (IDs: 115, 438, 618, 762, 852, 898, 951) |

### 2.3 Remaining data by split

#### Row-level

| Split | All rows | Frequent-only rows | Retention |
|-------|--------:|-------------------:|----------:|
| Train | 18,709 | **13,458** | 71.9% |
| Val | 3,916 | **2,841** | 72.5% |
| Test | 3,972 | **2,584** | 65.1% |

#### Deduplicated `(pair_key, type)` level

| Split | All pair-types | Frequent-only pair-types | Types |
|-------|---------------:|-------------------------:|------:|
| Train | 12,966 | **8,800** | 363 |
| Val | 2,644 | **1,858** | 357 |
| Test | 3,231 | **2,074** | 359 |

All 101 train / 21 val / 23 test **pairs** remain represented in the frequent-only subset (rare types co-occur on the same pairs).

### 2.4 What happens to types below the threshold?

| Treatment | Train impact | Val/test impact |
|-----------|--------------|-----------------|
| **Exclude** | 5,251 rows (592 types) removed from training | 1,066 val + 1,383 test rows excluded from eval |
| **Map to `other`** | 5,251 rows relabeled `other` | 1,075 val + 1,388 test rows relabeled `other` |

Types with ≥20 train rows but &lt;20 **pair-level** observations still appear — row threshold is on admission rows, not unique pairs.

### 2.5 Is an `"other"` class appropriate?

| Consideration | Assessment |
|---------------|------------|
| **Scientific meaning** | **Weak.** TWOSIDES types are specific pharmacological interaction categories. Collapsing 592 distinct types into one `"other"` bucket mixes unrelated mechanisms. |
| **Class imbalance** | `"other"` would be **28% of train rows** (5,251 / 18,709) — a dominant meta-class hiding fine-grained biology. |
| **Methodological use** | Acceptable only as a **secondary exploratory analysis**, not as the primary research target. |
| **Recommended handling** | **Exclude** rare types from training and primary evaluation; report exclusion counts transparently. |

### 2.6 Split leakage check

| Check | Result |
|-------|--------|
| `(pair_key, type)` spanning multiple splits | **0** (unchanged from current split) |
| Frequent filter applied using **train counts only** | No leakage — threshold derived from train split |

### 2.7 Formulation details

| Question | Answer |
|----------|--------|
| **What is predicted?** | One TWOSIDES interaction type from the **363 frequent classes** |
| **One training sample?** | One row (or one deduplicated `(pair_key, type)`) with `smiles_a`, `smiles_b` |
| **Label** | Integer class index over 363 frequent types |
| **Architecture** | Reuse existing baselines (LR, Static GAT, Molecular GNN) with 363-way head |
| **Metrics** | Top-1/3/5, macro/micro F1, balanced accuracy; pair_type_dedup primary |
| **Statistical limitations** | Still 363 classes; median ~37 train rows/class (13,458/363); 21 val / 23 test pairs |
| **More defensible than 955-class?** | **Yes** — better per-class support, aligns with existing `frequent_ge_20` slice where GNN showed relative gains |

---

## 3. Option 2 — Pair-level multi-label

### 3.1 Definition

One sample = **one unique drug pair** (`pair_key`). The label is the **set of all TWOSIDES interaction types** ever observed for that pair in the dataset.

### 3.2 Pair and label statistics

| Statistic | Value |
|-----------|------:|
| Unique drug pairs | **145** |
| All pairs have multiple types? | **Yes** (min **6**, max **402**, median **106**, mean **130**) |
| Label space size | 962 TWOSIDES types |
| Positive labels (unique pair–type combos) | 18,841 |
| Label matrix density | **13.5%** (~130 of 962 labels active per pair on average) |

#### Labels per pair — distribution highlights

| Percentile | Labels per pair |
|------------|----------------:|
| Min | 6 |
| 25th | 62 |
| Median | 106 |
| 75th | 184 |
| 90th | 271 |
| Max | 402 |

Every pair carries a **large** multi-label vector — this is not a sparse few-label problem.

### 3.3 Split at pair level

| Split | Pairs | Median labels/pair | Total positive labels |
|-------|------:|-------------------:|----------------------:|
| Train | **101** | 102 | 12,966 |
| Val | **21** | 119 | 2,644 |
| Test | **23** | 96 | 3,231 |

### 3.4 Type support (pair-level co-occurrence)

| Metric | Value |
|--------|------:|
| Types appearing on ≥20 pairs (global) | 384 |
| Types on ≥20 **train** pairs | 222 |
| Types on exactly 1 pair | 7 |
| Median pairs per type | 15 |

### 3.5 Formulation details

| Question | Answer |
|----------|--------|
| **What is predicted?** | Binary vector **y ∈ {0,1}^962** — which interaction types apply to this drug pair |
| **One training sample?** | One `pair_key` → `(smiles_a, smiles_b)` |
| **Label** | Multi-hot vector over 962 types (avg ~130 positives) |
| **Architecture** | Pair encoder (LR / GAT / GNN) + sigmoid output layer; binary cross-entropy per label with pos-weighting |
| **Metrics** | Example-based F1, micro/macro F1, subset accuracy, Hamming loss, label ranking AP@k |
| **Statistical limitations** | **Severe:** only **23 test pairs**; 962-dimensional output; labels highly correlated within pair; impossible to estimate per-label metrics reliably |
| **More defensible than 955-class?** | **Conceptually yes** (matches pair-level DDI question) but **statistically no** with current sample size |

### 3.6 Advantages

- Matches the natural unit of DDI knowledge: **“what interaction types can occur for this drug pair?”**
- Eliminates duplicate admission rows as separate conflicting single-label samples
- Aligns with pair-level split already in use
- Reduces sample count inflation (145 true units vs 26,597 rows)

### 3.7 Limitations

- **Only 23 test pairs** — metrics have enormous variance; no stable model comparison
- **~130 positive labels per pair** — predicting a 962-bit vector from 101 training pairs is underdetermined
- Labels for the same pair are **correlated** (co-reporting in MIMIC/TWOSIDES) — violates naive independence assumptions
- Cannot evaluate rare types per pair reliably
- Admission-level clinical context (patient, timing) is discarded

### 3.8 Feasibility verdict

**Not statistically feasible** as a primary research task with 145 pairs. Would require substantially more unique drug pairs or external validation data.

---

## 4. Side-by-side comparison

| Criterion | Current 955-class | Option 1 (≥20 filter) | Option 2 (multi-label) |
|-----------|------------------:|----------------------:|------------------------:|
| **Classes / label dim** | 955 | **363** | 962 (multi-hot) |
| **Train samples** | 18,709 rows | 13,458 rows | **101 pairs** |
| **Test eval units** | 3,227 pair-types | **2,074 pair-types** | **23 pairs** |
| **Question asked** | Which type for this admission? | Which frequent type? | Which types for this pair? |
| **Biological clarity** | Low (sparse) | **Moderate** | **High** |
| **Statistical power** | Very low | **Low–moderate** | **Very low** |
| **Implementation effort** | Done | **Low** (filter + retrain) | Medium (new pipeline) |
| **Split leakage risk** | None | None | None |
| **Defensible as primary task?** | No | **Best among three** | No (too few pairs) |

---

## 5. Recommendation

### Primary recommendation: **Adopt Option 1 — frequency-filtered multiclass (exclude rare types)**

In simple terms:

> Stop trying to predict all 955 interaction types. Focus the research task on the **363 interaction types that actually have enough training examples** (at least 20 each). Drop the rest from training and main evaluation rather than lumping them into a vague “other” category.

**Why this over the current 955-class task?**

- Keeps the same model pipeline and split
- Retains **~72% of data** with **2.6× fewer classes**
- Baselines already perform relatively best on the `frequent_ge_20` slice
- Methodologically easy to explain and reproduce

**Why not Option 2 (multi-label)?**

- Only **23 drug pairs** in the test set — far too few to draw research conclusions
- Each pair already has **dozens to hundreds** of interaction types, making the prediction problem enormous relative to sample size
- Better suited as a **future direction** if the dataset gains many more unique pairs

**Why not keep the current 955-class formulation as the headline task?**

- All three models score near random for exact match
- Increasing model complexity did not help overall
- Reviewers will correctly challenge a 955-way classifier built from 145 pairs

### Secondary recommendation

- Continue reporting **955-class results as supplementary/exploratory** (already completed)
- Do **not** use an `"other"` class as the primary target unless explicitly justified for error analysis
- Do **not** implement Temporal GNN or further architecture changes until the target formulation is updated

### Suggested next steps (design only — not executed here)

1. Document the 363-class label list and exclusion policy
2. Extend `TypeLabelEncoder` to support frequent-only mapping (in a future implementation phase)
3. Re-evaluate existing checkpoints on the frequent slice only (no retraining required for initial comparison)
4. Plan retraining on 363-class task if approved

---

## 6. Reproducibility

Regenerate analysis:

```bash
python -m src.evaluation.run_target_formulation_analysis
```

Output: `results/metrics/target_formulation_analysis.json`

---

*Analysis performed without modifying `data/processed/final_mimic_twosides_ml.csv`, splits, checkpoints, or baseline results.*
