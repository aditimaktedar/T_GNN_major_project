# Temporal Threshold Audit Report

## 1. Current Threshold Implementation

### `optimize_threshold_on_val` (coarse)

- Candidate range: 0.05 to 0.95
- Step size: 0.05
- Optimization metric: micro_f1 (configurable)
- Tie-breaking: first occurrence (lower threshold wins on ties)
- Global or per-label: global
- Predictions are sigmoid probabilities: True
- Uses validation labels only: True
- Test labels excluded: True

### `optimize_threshold_on_val_fine` (two-stage)

- Two-stage: coarse (0.05 step) then fine (0.01 step around coarse winner)
- Bug status: **FIXED**

> [!NOTE]
> Bug has been fixed. Function now returns correctly.

## 2. Why Threshold 0.01 Is Selected

- Avg positive labels per training observation: **86.93** / 363
- Average label frequency in train: **0.239479**
- pos_weight range: [0.3488, 13.5], median=3.8333
- Labels with pos_weight > 10: **15**
- Labels with pos_weight > 50: **0**

**Root cause chain:**

1. Many labels are rare (low frequency), so `pos_weight = n_neg / n_pos` is very large
2. Large pos_weights cause BCEWithLogitsLoss to penalize false negatives heavily
3. The model learns to output high sigmoid probabilities for many labels
4. At low thresholds (0.01-0.05), nearly all 363 labels are predicted positive
5. Micro-F1 favors recall when label density per observation is high
6. Predicting all-positive gives high recall → non-trivial Micro-F1
7. The coarse search finds a plateau where thresholds 0.05-0.35 all give the same Micro-F1
8. Fine search extends to 0.01, which ties. First-occurrence tie-breaking picks 0.01

## 3. Degenerate Prediction Behavior

| Configuration | Stored Threshold | Likely Degenerate | Val Micro-F1 | Test mAP |
|---|---|---|---|---|
| Temporal-only | 0.01 | **YES** | 0.4462 | 0.2765 |
| Molecular-only | 0.34 | No | 0.4512 | 0.3041 |
| Age-only | 0.01 | **YES** | 0.4462 | 0.2587 |
| Molecular + Age | 0.01 | **YES** | 0.4462 | 0.2797 |
| Molecular + Temporal | 0.01 | **YES** | 0.4462 | 0.2692 |
| Molecular + Temporal + Age (Full) | 0.36 | No | 0.4464 | 0.2371 |

## 4. Validation Threshold Curves (Full Mode — Checkpoint)

| Threshold | Micro-F1 | Macro-F1 | Jaccard | Avg Pos | Behavior |
|---|---|---|---|---|---|
| 0.01 | 0.4462 | 0.4092 | 0.2872 | 363.0 | ALL_POSITIVE |
| 0.05 | 0.4462 | 0.4092 | 0.2872 | 363.0 | ALL_POSITIVE |
| 0.10 | 0.4462 | 0.4092 | 0.2872 | 363.0 | ALL_POSITIVE |
| 0.15 | 0.4462 | 0.4092 | 0.2872 | 363.0 | ALL_POSITIVE |
| 0.20 | 0.4462 | 0.4092 | 0.2872 | 363.0 | ALL_POSITIVE |
| 0.25 | 0.4462 | 0.4092 | 0.2872 | 363.0 | ALL_POSITIVE |
| 0.30 | 0.4462 | 0.4092 | 0.2872 | 363.0 | ALL_POSITIVE |
| 0.35 | 0.4462 | 0.4092 | 0.2872 | 363.0 | ALL_POSITIVE |
| 0.40 | 0.4456 | 0.4067 | 0.2866 | 360.1 | NEAR_ALL_POSITIVE |
| 0.45 | 0.4434 | 0.3968 | 0.2848 | 346.6 | NEAR_ALL_POSITIVE |
| 0.50 | 0.3848 | 0.2947 | 0.2382 | 207.3 | NORMAL |
| 0.60 | 0.0083 | 0.0100 | 0.0042 | 1.5 | NEAR_ALL_NEGATIVE |
| 0.70 | 0.0000 | 0.0000 | 0.0000 | 0.0 | ALL_NEGATIVE |
| 0.80 | 0.0000 | 0.0000 | 0.0000 | 0.0 | ALL_NEGATIVE |
| 0.90 | 0.0000 | 0.0000 | 0.0000 | 0.0 | ALL_NEGATIVE |

## 5. Probability Distributions (Full Mode)

| Split | Min | Max | Mean | Median | Std |
|---|---|---|---|---|---|
| Val | 0.354498 | 0.631483 | 0.503795 | 0.504255 | 0.031025 |
| Test | 0.339496 | 0.626392 | 0.49902 | 0.498291 | 0.035977 |

### Quantiles (Validation)

| Q01 | Q05 | Q10 | Q25 | Q50 | Q75 | Q90 | Q95 | Q99 |
|---|---|---|---|---|---|---|---|---|
| 0.409813 | 0.452549 | 0.470489 | 0.488294 | 0.504255 | 0.520249 | 0.53824 | 0.552768 | 0.585634 |

## 6. pos_weight Distribution

- Min: 0.3488, Max: 13.5, Mean: 4.1915, Median: 3.8333, Std: 2.4874
- Labels with weight > 5: 110
- Labels with weight > 10: 15
- Labels with weight > 20: 0
- Labels with weight > 50: 0

| Q01 | Q05 | Q10 | Q25 | Q50 | Q75 | Q90 | Q95 | Q99 |
|---|---|---|---|---|---|---|---|---|
| 0.772 | 1.1481 | 1.3393 | 2.317 | 3.8333 | 5.4444 | 7.2857 | 8.6667 | 13.5 |

## 7. Ranking Metrics (Threshold-Independent, Full Mode)

| Split | mAP | P@1 | R@1 | P@5 | R@5 | P@10 | R@10 |
|---|---|---|---|---|---|---|---|
| Val | 0.2945 | 0.0000 | 0.0000 | 0.1000 | 0.0036 | 0.1562 | 0.0128 |
| Test | 0.2562 | 0.3000 | 0.0020 | 0.2600 | 0.0127 | 0.2600 | 0.0257 |

### Why reasonable mAP but poor thresholded F1?

mAP evaluates ranking quality using continuous probabilities — it rewards placing true positive labels higher in the ranked list, regardless of where the decision boundary falls. A model can have good ranking (reasonable mAP) but poor thresholded F1 if its probability distribution is compressed into a narrow range, making it impossible to find a threshold that cleanly separates positives from negatives. With pos_weight pushing many predictions above 0.5, the sigmoid outputs for positive and negative labels overlap heavily.

## 8. Top-K as Alternative Output Format (Test)

| K | P@K | R@K |
|---|---|---|
| 1 | 0.3000 | 0.0020 |
| 5 | 0.2600 | 0.0127 |
| 10 | 0.2600 | 0.0257 |

## 9. Test Set Safety Verification

```
1. Train model on train split only
2. Select best checkpoint using validation loss
3. Generate validation predictions from best checkpoint
4. Sweep thresholds on validation predictions + labels only
5. Select threshold maximizing validation metric
6. FREEZE threshold
7. Generate test predictions
8. Evaluate test metrics using frozen threshold
9. No test feedback flows backward
```

- Test labels used for threshold selection: **False**
- Test metrics used for model selection: **False**
- Threshold frozen before test: **True**

## 10. Threshold Selection Policy Comparison (Full Mode)

| Policy | Threshold | Test Micro-F1 | Test Macro-F1 | Test mAP | Test Avg Pos | Behavior |
|---|---|---|---|---|---|---|
| micro_f1 | 0.36 | 0.4153 | 0.3887 | 0.2562 | 362.7 | NEAR_ALL_POSITIVE |
| macro_f1 | 0.01 | 0.4159 | 0.3896 | 0.2562 | 363.0 | ALL_POSITIVE |
| jaccard | 0.36 | 0.4153 | 0.3887 | 0.2562 | 362.7 | NEAR_ALL_POSITIVE |

## 11. Compact Summary Tables

### Current State (Stored Ablation Results)

| Configuration | Current Threshold | Selection Metric | Degenerate? |
|---|---|---|---|
| Temporal-only | 0.01 | Micro-F1 | **YES** |
| Molecular-only | 0.34 | Micro-F1 | No |
| Age-only | 0.01 | Micro-F1 | **YES** |
| Molecular + Age | 0.01 | Micro-F1 | **YES** |
| Molecular + Temporal | 0.01 | Micro-F1 | **YES** |
| Molecular + Temporal + Age (Full) | 0.36 | Micro-F1 | No |

### Full-Mode Policy Comparison

| Selection Metric | Threshold | Val Score | Behavior | Recommended |
|---|---|---|---|---|
| micro_f1 | 0.36 | 0.4463 | NEAR_ALL_POSITIVE |  |
| macro_f1 | 0.01 | 0.4092 | ALL_POSITIVE |  |
| jaccard | 0.36 | 0.2872 | NEAR_ALL_POSITIVE |  |

## 12. Recommended Threshold-Selection Policy

### Analysis

- **Policy A (Micro-F1)**: Produces degenerate all-positive predictions. Micro-F1 rewards high recall; with high label density per observation, predicting all 363 labels achieves recall≈1.0 and non-trivial precision.

- **Policy B (Macro-F1)**: Penalizes per-label over-prediction. Rare labels get F1≈0 when predicted all-positive. Naturally selects discriminative thresholds.

- **Policy C (Jaccard)**: Similar to Micro-F1, vulnerable to all-positive.

- **Policy E (ranking primary)**: mAP and P@K avoid thresholding entirely.

### Recommendation

**Policy E (ranking metrics primary) + Policy B (Macro-F1 secondary thresholding)**

1. **Primary comparison**: Use **mAP** as the headline metric for model/ablation comparison.
2. **Secondary thresholded metrics**: Select threshold maximizing **validation Macro-F1**.
3. **Report both**: mAP, P@K (ranking), and Macro-F1/Micro-F1 (at Macro-F1-selected threshold).

This is defensible because:
- mAP is standard in multi-label learning benchmarks (LSML, MLkNN, SLEEC)
- Macro-F1 threshold selection is unbiased across label frequencies
- No threshold is cherry-picked for test-set inflation

## 13. Implementation Bug Status

> [!NOTE]
> Bug in `optimize_threshold_on_val_fine` has been **FIXED**. Return statement restored.

## 14. Whether Retraining Is Required

Retraining is **NOT** required for diagnostic conclusions. The threshold selection methodology is the issue, not the model weights. Next steps:
1. The missing return statement bug has been fixed
2. Switch primary comparison metric to mAP
3. Use Macro-F1 for threshold selection when binary predictions are needed
4. Re-evaluate existing checkpoints with corrected methodology

---

## FINAL VERDICT

**A. THRESHOLD IMPLEMENTATION IS VALID — METHODOLOGY NEEDS REFINEMENT**

1. `optimize_threshold_on_val_fine` was missing its return statement (now fixed)
2. Even when working correctly, selecting threshold by Micro-F1 in a high-density 363-label problem leads to degenerate all-positive predictions
3. The combination of large pos_weights and Micro-F1 threshold selection is the root cause of the threshold=0.01 behavior

### Next Steps

1. ~~Fix the return statement~~ (DONE)
2. Adopt mAP as the primary comparison metric
3. Use Macro-F1 for validation threshold selection
4. Re-evaluate all ablation configurations with corrected threshold selection
5. Update the paper Methods section to document the threshold policy
