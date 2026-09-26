# Model-Ready Temporal Multi-Label Dataset Validation Report

> [!IMPORTANT]
> **Dataset Generation & Validation Verification:**
> - **Total Observations (Pair + Admission):** `84`
> - **Target Dimension:** `363` multi-label classes
> - **Unique Temporal DDI Pairs:** `62`
> - **Unique Patients (`subject_id`):** `49`
> - **Unique Admissions (`hadm_id`):** `57`
> - **Target Consistency:** `100% verified (Zero encoding or dimension errors)`
> - **Split Leakage:** `0 pairs across multiple splits (Verified Disjoint)`
> - **Total Missing Values:** `0`

---

## 1. Dataset Overview

| Metric | Value |
| :--- | :---: |
| Total Temporal Observations | 84 |
| Target Dimension ($C$) | 363 |
| Total Multi-Label Associations | 7663 |
| Unique Temporal Pairs | 62 |
| Unique Subjects | 49 |
| Unique Hospital Admissions | 57 |
| Missing Values Across All Columns | 0 |

---

## 2. Multi-Label Target Distribution per Observation

- **Min Active Labels / Observation:** `11`
- **Median Active Labels / Observation:** `90.0`
- **Mean Active Labels / Observation:** `91.2262`
- **Max Active Labels / Observation:** `188`

---

## 3. Preserved Split Distribution

| Split | Observations Count | Observations % | Unique Pairs | Unique Pairs % |
| :--- | :---: | :---: | :---: | :---: |
| `train` | 58 | 69.05% | 40 | 64.52% |
| `val` | 16 | 19.05% | 12 | 19.35% |
| `test` | 10 | 11.90% | 10 | 16.13% |

> [!NOTE]
> **Split Preservation:** Pair-level partition strictly preserved from `multilabel_frequent363_dataset.csv`. Zero overlap between train, val, and test splits.

---

## 4. Temporal Interval ($\Delta t$) Distribution

| Metric | Min (Hours) | Median (Hours) | Mean (Hours) | Max (Hours) |
| :--- | :---: | :---: | :---: | :---: |
| **Minimum $\Delta t$ (`min_delta_hours`)** | 0.0000 | 0.0000 | 1.7994 | 35.8667 |
| **Median $\Delta t$ (`median_delta_hours`)** | 0.0000 | 34.7542 | 49.5033 | 543.0417 |

---

## 5. Initial Administration Order Distribution

| Initial Administration Event | Count | Percentage |
| :--- | :---: | :---: |
| Drug A Administered First (`a_before_b`) | 36 | 42.86% |
| Drug B Administered First (`b_before_a`) | 25 | 29.76% |
| Simultaneous Initial Administration (`same_timestamp`) | 23 | 27.38% |

---

## 6. Missingness Audit

| Column Name | Missing Count | Status |
| :--- | :---: | :--- |
| `subject_id` | 0 | Clean (0 missing) |
| `hadm_id` | 0 | Clean (0 missing) |
| `pair_key` | 0 | Clean (0 missing) |
| `drug_a` | 0 | Clean (0 missing) |
| `drug_b` | 0 | Clean (0 missing) |
| `smiles_a` | 0 | Clean (0 missing) |
| `smiles_b` | 0 | Clean (0 missing) |
| `anchor_age` | 0 | Clean (0 missing) |
| `num_a_events` | 0 | Clean (0 missing) |
| `num_b_events` | 0 | Clean (0 missing) |
| `num_total_pair_events` | 0 | Clean (0 missing) |
| `first_a_time` | 0 | Clean (0 missing) |
| `first_b_time` | 0 | Clean (0 missing) |
| `min_delta_hours` | 0 | Clean (0 missing) |
| `median_delta_hours` | 0 | Clean (0 missing) |
| `a_before_b` | 0 | Clean (0 missing) |
| `b_before_a` | 0 | Clean (0 missing) |
| `same_timestamp` | 0 | Clean (0 missing) |
| `labels` | 0 | Clean (0 missing) |
| `target` | 0 | Clean (0 missing) |
| `split` | 0 | Clean (0 missing) |
