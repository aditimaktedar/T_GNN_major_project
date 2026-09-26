# Multi-Label DDI Experiment Results

> **Status**: AWAITING TRAINING — No experiments have been run yet.
> Dataset finalization is in progress. Results will be populated after training.

## Label Space
- **FREQUENT363**: 363 interaction types from TWOSIDES
- **Target**: 363-dimensional multi-hot binary vector per unique drug pair

## Models

| # | Model | Description |
|---|-------|-------------|
| 1 | MultilabelLogisticRegression | N independent sigmoid classifiers on Morgan FP pair features |
| 2 | MultilabelStaticGAT | Graph Attention Network on 2-node drug-pair graphs |
| 3 | MultilabelMolecularGNN | Dual GIN encoder with pair fusion on molecular graphs |

## Feature Configurations

| Config | Features | Age |
|--------|----------|-----|
| A: drug_pair | Concatenated Morgan fingerprints (2×2048) | ✗ |
| B: drug_pair_age | Concatenated Morgan fingerprints + normalized anchor_age | ✓ |

## Evaluation Metrics (13 total)

| Metric | Description |
|--------|-------------|
| Micro F1 | F1 averaged over all label predictions |
| Macro F1 | F1 averaged over each label independently |
| Hamming Loss | Fraction of incorrect label predictions |
| Jaccard Score | Intersection/union of predicted and true label sets |
| mAP | Mean Average Precision across all labels |
| Precision@K | Fraction of top-K predictions that are true (K=1,3,5,10) |
| Recall@K | Fraction of true labels found in top-K predictions (K=1,3,5,10) |

## Results Table

> ⚠️ **No results yet.** This table will be populated after model training.

| Model | Feature Set | Micro F1 | Macro F1 | Hamming Loss | Jaccard | mAP | P@1 | R@1 | P@3 | R@3 | P@5 | R@5 | P@10 | R@10 |
|-------|-------------|----------|----------|--------------|---------|-----|-----|-----|-----|-----|-----|-----|------|------|
| — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |

## Age Ablation Analysis

> ⚠️ **Pending.** Ablation comparing Config A vs Config B will be added after training.

## Notes
- Split strategy: Pair-level, seed=42, ratios 70/15/15
- Loss function: BCEWithLogitsLoss for all models
- Threshold: Default 0.50 (optimized on validation split only)
