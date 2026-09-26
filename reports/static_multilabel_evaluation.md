# Static Multi-Label Baseline (Pipeline 1) -- Evaluation Report

> [!IMPORTANT]
> **Key Findings & Scientific Evaluation:**
> - **Model:** StaticMultilabelLR on Morgan fingerprint pair features (bits=2048, radius=2, age=included)
> - **Input Dimension:** 4097  -- **Target Space:** 363 TWOSIDES multi-label classes
> - **Observations:** 101 train, 21 val, 23 test (Pair-level split preserved)
> - **Static LR Test Micro-F1:** 0.4388 (Macro-F1: 0.3218, mAP: 0.3949, Threshold: 0.35)
> - **Training Prior Baseline Micro-F1:** 0.4301 (Threshold: 0.2)

---

## 1. Test Set Comparison

| Model | Threshold | Micro-F1 | Macro-F1 | Hamming Loss | Jaccard | mAP | P@1 | R@1 | P@5 | R@5 | P@10 | R@10 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Static LR (Fingerprint + Age)** | 0.35 | **0.4388** | **0.3218** | 0.4500 | 0.2811 | 0.3949 | 0.6957 | 0.0099 | 0.5565 | 0.0340 | 0.5348 | 0.0664 |
| Static LR (Default 0.50) | 0.50 | 0.1415 | 0.0416 | 0.2427 | 0.0762 | 0.3949 | 0.6957 | 0.0099 | 0.5565 | 0.0340 | 0.5348 | 0.0664 |
| Training Prior Baseline | 0.2 | 0.4301 | 0.2588 | 0.4634 | 0.2740 | 0.3658 | 0.4348 | 0.0044 | 0.5652 | 0.0362 | 0.5391 | 0.0685 |

---

## 2. Split Performance Breakdown

| Split | N | Micro-F1 | Macro-F1 | Hamming Loss | Jaccard | mAP |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Train | 101 | 0.4364 | 0.3290 | 0.4491 | 0.2791 | 0.3865 |
| Validation | 21 | 0.4355 | 0.2981 | 0.4601 | 0.2784 | 0.3726 |
| Test (Optimized) | 23 | 0.4388 | 0.3218 | 0.4500 | 0.2811 | 0.3949 |
| Test (Default 0.50) | 23 | 0.1415 | 0.0416 | 0.2427 | 0.0762 | 0.3949 |

---

## 3. Training Summary

- **Best Epoch:** 20 / 200
- **Best Val Loss:** 0.576751
- **Final Train Loss:** 0.588105
- **Learning Rate:** 0.05,  **Weight Decay:** 0.001
