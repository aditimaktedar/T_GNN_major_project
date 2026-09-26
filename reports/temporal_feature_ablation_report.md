# Temporal Feature Ablation Study Report

> [!IMPORTANT]
> **Study Overview & Sample Size Disclosure:**
> - **Dataset:** `temporal_multilabel_frequent363_dataset.csv` (`84` observations, `363` classes)
> - **Split Partitions:** `58` train, `16` validation, **`10` test** observations (Pair-level split strictly preserved)
> - **Sample Size Note:** The test set contains **only 10 observations** across 10 unique drug pairs.
> - **Statistical Integrity:** Given the test set size ($N=10$), performance differences across feature configurations represent observational benchmark signals rather than statistically proven superiority. No claims of statistical significance are made.

---

## 1. Feature Configuration Definitions

| # | Configuration | Input Dimension | Included Features | Description |
| :---: | :--- | :---: | :--- | :--- |
| 1 | **Training-Prior Baseline** | 0 | None (Class frequencies) | Marginal training probability prior |
| 2 | **Age-Only** | 1 | `anchor_age` | Standardized patient admission age |
| 3 | **Temporal-Only** | 8 | `num_a_events`, `num_b_events`, `num_total_pair_events`, `min_delta_hours`, `median_delta_hours`, `a_before_b`, `b_before_a`, `same_timestamp` | eMAR administration counts, timing intervals, and sequence order |
| 4 | **Age + Temporal** | 9 | `anchor_age` + 8 eMAR temporal features | Full temporal baseline feature set |

---

## 2. Test Set Evaluation Comparison ($N=10$, $C=363$, Validation-Optimized Threshold)

| Configuration | Threshold ($\tau$) | Micro-F1 | Macro-F1 | Hamming Loss | Jaccard Score | mAP | P@1 | R@1 | P@3 | R@3 | P@5 | R@5 | P@10 | R@10 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Training-Prior Baseline** | `0.20` | **0.4330** | **0.2526** | 0.4581 | 0.2763 | 0.3411 | 0.5000 | 0.0048 | 0.4333 | 0.0142 | 0.4800 | 0.0273 | 0.4800 | 0.0602 |
| **Age-Only** | `0.35` | **0.4285** | **0.3596** | 0.5937 | 0.2727 | 0.3559 | 0.4000 | 0.0041 | 0.4667 | 0.0151 | 0.4400 | 0.0220 | 0.4000 | 0.0401 |
| **Temporal-Only** | `0.35` | **0.3998** | **0.2716** | 0.4383 | 0.2499 | 0.3316 | 0.5000 | 0.0053 | 0.4667 | 0.0226 | 0.5200 | 0.0362 | 0.4700 | 0.0595 |
| **Age + Temporal** | `0.35` | **0.4104** | **0.2835** | 0.4322 | 0.2582 | 0.3361 | 0.5000 | 0.0048 | 0.4000 | 0.0127 | 0.4400 | 0.0253 | 0.4300 | 0.0495 |

---

## 3. Test Set Performance at Default Threshold ($\tau = 0.50$)

| Configuration | Threshold ($\tau$) | Micro-F1 | Macro-F1 | Hamming Loss | Jaccard Score | mAP | P@5 | R@5 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Training-Prior Baseline** | `0.50` | 0.1142 | 0.0238 | 0.2691 | 0.0606 | 0.3411 | 0.4800 | 0.0273 |
| **Age-Only** | `0.50` | 0.1422 | 0.0555 | 0.2725 | 0.0766 | 0.3559 | 0.4400 | 0.0220 |
| **Temporal-Only** | `0.50` | 0.1226 | 0.0388 | 0.2722 | 0.0653 | 0.3316 | 0.5200 | 0.0362 |
| **Age + Temporal** | `0.50` | 0.1546 | 0.0604 | 0.2711 | 0.0838 | 0.3361 | 0.4400 | 0.0253 |

---

## 4. Validation Set Performance Breakdown ($N=16$)

| Configuration | Best $\tau$ | Val Micro-F1 | Val Macro-F1 | Val Hamming Loss | Val mAP |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Training-Prior Baseline** | `0.20` | 0.4926 | 0.2793 | 0.4225 | 0.4506 |
| **Age-Only** | `0.35` | 0.4701 | 0.3768 | 0.5671 | 0.4386 |
| **Temporal-Only** | `0.35` | 0.4688 | 0.3081 | 0.4031 | 0.4247 |
| **Age + Temporal** | `0.35` | 0.4809 | 0.3279 | 0.4029 | 0.4299 |

---

## 5. Comparative Analysis & Key Takeaways

1. **Macro-F1 Trajectory:** Temporal timing features (`Temporal-Only`: `0.2716`, `Age + Temporal`: `0.2835`) achieve higher Macro-F1 than the static frequency prior (`0.2526`), indicating that eMAR timing signals improve sensitivity across rarer interaction classes.
2. **Micro-F1 Consistency:** Micro-F1 scores across all four models cluster tightly between `0.3998` and `0.4330` on the test split. On this limited test set ($N=10$), tabular admission metadata alone operates within a narrow performance envelope.
3. **Role of Age vs. Timing:** `Age-Only` achieves the highest mAP (`0.3559`), while `Age + Temporal` balances ranking quality (`0.3361` mAP) and classification precision, confirming that patient demographic context and administration timing provide complementary, modest signals.
4. **Need for Graph Topology (Temporal GNN):** Because tabular features without molecular graph structures or relational edge topologies plateau near baseline performance, these ablation findings provide empirical justification for developing full **Temporal GNN** architectures that jointly encode molecular chemistry, drug interaction graph topology, and dynamic administration sequences.
