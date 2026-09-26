# Temporal Pair Feature Dataset Validation Report

> [!IMPORTANT]
> **Dataset Generation & Validation Verification:**
> - **Total Temporal Pair/Admission Observations:** `84`
> - **Unique Temporal DDI Pairs:** `62`
> - **Unique Patients (`subject_id`):** `49`
> - **Unique Admissions (`hadm_id`):** `57`
> - **Split Leakage:** `0 pairs across multiple splits (Verified Disjoint)`
> - **Total Missing Values:** `0`

---

## 1. Summary Overview

| Metric | Count |
| :--- | :---: |
| Total Temporal Observations (Pair + Admission) | 84 |
| Unique Temporal Pairs | 62 |
| Unique Subjects | 49 |
| Unique Hospital Admissions | 57 |
| Missing Values Across All Columns | 0 |

---

## 2. Train / Validation / Test Split Distribution

Preserved strictly from the existing pair-level split (`multilabel_frequent363_dataset.csv` / `multilabel_pair_splits.json`).

| Split | Observations Count | Observations % | Unique Pairs | Unique Pairs % |
| :--- | :---: | :---: | :---: | :---: |
| `train` | 58 | 69.05% | 40 | 64.52% |
| `val` | 16 | 19.05% | 12 | 19.35% |
| `test` | 10 | 11.90% | 10 | 16.13% |

> [!NOTE]
> **Split Integrity:** 100% of temporal pairs belong strictly to their predetermined split partition. Zero leakage between train, validation, and test sets.

---

## 3. Temporal Interval ($\Delta t$) Distribution

| Metric | Min (Hours) | Median (Hours) | Mean (Hours) | Max (Hours) |
| :--- | :---: | :---: | :---: | :---: |
| **Minimum $\Delta t$ (`min_delta_hours`)** | 0.0000 | 0.0000 | 1.7994 | 35.8667 |
| **Median $\Delta t$ (`median_delta_hours`)** | 0.0000 | 34.7542 | 49.5033 | 543.0417 |

---

## 4. Initial Administration Order Distribution

| Initial Administration Event | Count | Percentage |
| :--- | :---: | :---: |
| Drug A Administered First (`a_before_b`) | 36 | 42.86% |
| Drug B Administered First (`b_before_a`) | 25 | 29.76% |
| Simultaneous Initial Administration (`same_timestamp`) | 23 | 27.38% |

---

## 5. Repeated Observations per Pair

- **Min Observations / Pair:** `1`
- **Median Observations / Pair:** `1.0`
- **Mean Observations / Pair:** `1.3548`
- **Max Observations / Pair:** `7`
- **Single-Observation Pairs:** `50` (80.65%)
- **Repeated-Observation Pairs (>1 admission):** `12` (19.35%)

### Frequency Breakdown

| Observations per Pair ($k$) | Number of Pairs with $k$ Observations |
| :---: | :---: |
| 1 | 50 |
| 2 | 6 |
| 3 | 5 |
| 7 | 1 |

---

## 6. Missingness Audit

| Column Name | Missing Count | Status |
| :--- | :---: | :--- |
| `subject_id` | 0 | Clean (0 missing) |
| `hadm_id` | 0 | Clean (0 missing) |
| `pair_key` | 0 | Clean (0 missing) |
| `drug_a` | 0 | Clean (0 missing) |
| `drug_b` | 0 | Clean (0 missing) |
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
| `split` | 0 | Clean (0 missing) |
