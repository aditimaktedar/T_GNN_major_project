# Manual Training Workflow Guide (PowerShell)

This guide documents how to manually train, evaluate, and inspect all DDI models across Pipeline 1 (Static) and Pipeline 2 (Temporal) directly from PowerShell.

---

## Quick Reference Commands

### Pipeline 1: Static DDI Modeling

All static models are trained on `data/processed/multilabel_frequent363_dataset.csv` (101 train / 21 val / 23 test unique pairs) across 363 TWOSIDES adverse interaction labels.

```powershell
# 1. Multi-label Logistic Regression Baseline (Morgan FP + Patient Age)
python -m src.models.train_static_baseline

# 2. Multi-label Static Graph Attention Network (2-Node Drug Pair GAT)
python -m src.models.train_static_gat

# 3. Dual Molecular Graph Neural Network (GIN Encoder + Pair Fusion)
python -m src.models.train_molecular_gnn
```

### Pipeline 2: Temporal DDI Modeling

All temporal models operate on `data/processed/temporal_multilabel_frequent363_dataset.csv` (58 train / 16 val / 10 test observations) using 9 standardized admission timing and event features.

```powershell
# 1. Temporal Multi-label Logistic Regression Baseline
python -m src.models.train_temporal_baseline

# 2. Temporal GNN (T-GNN) Framework Status & Readiness Check
python -m src.models.train_temporal_gnn
```

---

## Command Options & Hyperparameters

Each training script accepts standard CLI arguments:

| Argument | Default | Description |
| :--- | :---: | :--- |
| `--epochs` | `200` | Number of training epochs |
| `--lr` | `0.05` (LR) / `0.001` (GNN/GAT) | Learning rate |
| `--weight-decay` | `1e-3` (LR) / `1e-4` (GNN/GAT) | Weight decay (L2 regularization) |
| `--seed` | `42` | Random seed for reproducibility |
| `--no-age` | `False` | Exclude normalized `anchor_age` feature |
| `--checkpoint` | `results/checkpoints/<model>.pt` | Target checkpoint path |
| `--results-json` | `results/metrics/<model>.json` | Target JSON metrics path |

#### Example Custom Run:
```powershell
python -m src.models.train_static_baseline --epochs 300 --lr 0.01 --seed 123
```

---

## Artifact Locations

### Checkpoints (`results/checkpoints/`)
Trained PyTorch model state dictionaries and preprocessing metadata are saved to:
- `results/checkpoints/static_logistic_regression.pt`
- `results/checkpoints/static_gat.pt`
- `results/checkpoints/static_molecular_gnn.pt`
- `results/checkpoints/temporal_baseline.pt`

### Results JSON (`results/metrics/`)
Machine-readable evaluation summaries containing complete dataset sizes, unique pair counts, training loss histories, validation-selected thresholds, and test set metrics:
- `results/metrics/static_logistic_regression.json`
- `results/metrics/static_gat.json`
- `results/metrics/static_molecular_gnn.json`
- `results/metrics/temporal_baseline_results.json`

---

## Terminal Performance Report Explained

Upon completion of training, each script prints a standardized terminal report:

```text
============================================================
MODEL TRAINING COMPLETE
============================================================

Pipeline: Static
Model: Molecular GNN
Dataset: Frequent363
Test observations: 23
Test unique pairs: 23
Number of labels: 363

Training:
  Epochs: 200
  Best epoch: 42
  Best validation loss: 0.123456
  Best validation Micro-F1: 0.654321

Threshold:
  Selected threshold: 0.35
  Selection set: validation

TEST PERFORMANCE
------------------------------------------------------------
Micro-F1       : 0.543210
Macro-F1       : 0.123456
Hamming Loss   : 0.054321
Jaccard        : 0.432100
mAP            : 0.612345

Precision@1    : 0.826087
Recall@1       : 0.012345

Precision@5    : 0.652174
Recall@5       : 0.054321

Precision@10   : 0.521739
Recall@10      : 0.098765

------------------------------------------------------------
CHECKPOINT
Path: results/checkpoints/static_molecular_gnn.pt

RESULTS
Path: results/metrics/static_molecular_gnn.json

============================================================
```

### What Each Metric Means

- **Micro-F1**: Global harmonic mean of precision and recall pooled across all instances and all 363 classes. Reflects overall positive prediction quality under severe class imbalance.
- **Macro-F1**: Unweighted arithmetic average of F1 scores computed per class. Sensitive to rare adverse interaction types.
- **Hamming Loss**: Fraction of incorrectly predicted individual label decisions (lower is better; 0.0 is perfect).
- **Jaccard Score (Micro)**: Intersection over Union (IoU) between true multi-hot label vectors and predicted binary label vectors.
- **mAP (Mean Average Precision)**: Area under the Precision-Recall curve across all classes without threshold discretization.
- **Precision@K / Recall@K**: Evaluation of top-$K$ ranked predictions ($K \in \{1, 5, 10\}$), assessing how accurately the highest-confidence predicted interaction types match ground truth.

---

## How to Evaluate Model Performance

1. **Compare against the Prior Baseline**:
   - The training class-frequency prior achieves ~0.49 to 0.51 Micro-F1. A well-performing model must exceed this baseline on test data.
2. **Examine Macro-F1 vs. Micro-F1**:
   - Because TWOSIDES adverse interaction types exhibit power-law frequency distributions, higher Macro-F1 indicates the model is learning distinct molecular/temporal signals rather than simply memorizing the common interaction classes.
3. **Verify Precision/Recall at Top-K**:
   - High Precision@1 and Precision@5 (>0.70) indicate that clinicians can trust the top-ranked warning types generated by the system.

---

## Crucial Rule: Validation-Only Threshold Selection

> [!IMPORTANT]
> **Why the test set must NEVER be used for threshold optimization:**
> 1. In multi-label DDI classification with 363 sparse classes, testing multiple probability thresholds (e.g. searching $0.05 \dots 0.95$) directly on the test set is a form of **data leakage** that produces overly optimistic, ungeneralizable performance claims.
> 2. All training workflows in this project strictly execute grid search for the optimal decision threshold on the **validation split only**.
> 3. Once the threshold is locked, the final model evaluates **exactly once** on the untouched test split.
