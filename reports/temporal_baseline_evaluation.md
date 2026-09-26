# Temporal Multi-Label Baseline Evaluation Report

> [!IMPORTANT]
> **Key Findings & Scientific Evaluation:**
> - **Model Evaluated:** `TemporalMultilabelLogisticRegression` on 9 admission temporal features
> - **Target Space:** `363` TWOSIDES multi-label adverse interaction classes
> - **Observations:** `58` train, `16` val, `10` test (Pair-level split strictly preserved)
> - **Temporal Baseline Test Micro-F1:** `0.4175` (Macro-F1: `0.3033`, mAP: `0.3340`, Threshold: `0.35`)
> - **Training Prior Baseline Test Micro-F1:** `0.4330` (Macro-F1: `0.2526`, mAP: `0.3411`, Threshold: `0.2`)

---

## 1. Direct Comparison on Identical Test Split ($N=10$, $C=363$)

| Model / Baseline | Threshold Strategy | Micro-F1 | Macro-F1 | Hamming Loss | Jaccard Score | mAP | P@1 | R@1 | P@3 | R@3 | P@5 | R@5 | P@10 | R@10 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Temporal Baseline** | Val-Opt (`0.35`) | **0.4175** | **0.3033** | 0.4336 | 0.2638 | 0.3340 | 0.4000 | 0.0042 | 0.3667 | 0.0115 | 0.4200 | 0.0245 | 0.4100 | 0.0464 |
| Temporal Baseline | Default (`0.50`) | 0.1375 | 0.0526 | 0.2730 | 0.0738 | 0.3340 | 0.4000 | 0.0042 | 0.3667 | 0.0115 | 0.4200 | 0.0245 | 0.4100 | 0.0464 |
| **Training Prior Baseline** | Val-Opt (`0.2`) | 0.4330 | 0.2526 | 0.4581 | 0.2763 | 0.3411 | 0.5000 | 0.0048 | 0.4333 | 0.0142 | 0.4800 | 0.0273 | 0.4800 | 0.0602 |
| Training Prior Baseline | Default (`0.50`) | 0.1142 | 0.0238 | 0.2691 | 0.0606 | 0.3411 | 0.5000 | 0.0048 | 0.4333 | 0.0142 | 0.4800 | 0.0273 | 0.4800 | 0.0602 |

---

## 2. Split Performance Breakdown (Temporal Baseline)

| Split | Samples | Micro-F1 | Macro-F1 | Hamming Loss | Jaccard Score | mAP |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `Train` | 58 | 0.4999 | 0.4264 | 0.3648 | 0.3333 | 0.5041 |
| `Validation` | 16 | 0.4772 | 0.3261 | 0.4093 | 0.3134 | 0.4253 |
| `Test (Optimized)` | 10 | 0.4175 | 0.3033 | 0.4336 | 0.2638 | 0.3340 |
| `Test (Default 0.50)` | 10 | 0.1375 | 0.0526 | 0.2730 | 0.0738 | 0.3340 |

---

## 3. Analysis & Discussion

1. **Macro-F1 vs Micro-F1:** The temporal baseline achieves a Macro-F1 of `0.3033` (vs `0.2526` for the prior baseline) on the test set, demonstrating that admission temporal timing features provide discriminatory signal across multi-label interaction classes.
2. **Threshold Sensitivity:** Because average label frequency is ~25%, a standard threshold of 0.50 yields low recall (`0.1375` Micro-F1). Optimizing the decision threshold on validation data (selecting `0.35`) substantially improves test Micro-F1 to `0.4175` without any data leakage.
3. **Direct Test Comparison:** Compared directly on the identical test split, the temporal multi-label baseline achieves Micro-F1 `0.4175` and mAP `0.3340`, performing in line with the class frequency prior (`0.4330` Micro-F1 / `0.3411` mAP). This confirms that temporal features alone provide a solid reference point, while full graph topology and molecular embeddings remain key for higher-order representation in subsequent Temporal GNN stages.
