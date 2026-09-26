"""CLI Entry-point for Pipeline 1: Static Multi-Label Baseline Training & Evaluation.

Command:
    python -m src.models.run_multilabel_pipeline

Trains a multi-label logistic regression on concatenated Morgan fingerprint
pair features (+ optional anchor_age) from multilabel_frequent363_dataset.csv.
Evaluates on the preserved pair-level train/val/test split using canonical
363-class multi-label metrics (Micro-F1, Macro-F1, Hamming Loss, mAP, P@K, R@K).

Produces:
    - results/metrics/static_multilabel_results.json
    - reports/static_multilabel_evaluation.md
    - results/checkpoints/static_multilabel_lr.pt

Constraints:
    - Does NOT modify existing datasets, previous results, or split assignments.
    - Feature standardization is fitted on the training split only (no leakage).
    - Threshold optimization is performed on the validation split only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.data.config import (
    CHECKPOINTS_DIR,
    METRICS_DIR,
    MULTILABEL_FREQUENT363_DATASET_CSV,
    MULTILABEL_FREQUENT363_LABEL_MAPPING_PATH,
    REPORTS_DIR,
)
from src.evaluation.multilabel_metrics import compute_multilabel_metrics
from src.evaluation.thresholds import optimize_threshold_on_val

# ---------------------------------------------------------------------------
# Output path constants
# ---------------------------------------------------------------------------
STATIC_MULTILABEL_RESULTS_JSON = METRICS_DIR / "static_multilabel_results.json"
STATIC_MULTILABEL_REPORT_MD = REPORTS_DIR / "static_multilabel_evaluation.md"
STATIC_MULTILABEL_CHECKPOINT = CHECKPOINTS_DIR / "static_multilabel_lr.pt"


# ---------------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------------

def _morgan_fingerprint(smiles: str, n_bits: int = 2048, radius: int = 2) -> np.ndarray:
    """Compute Morgan fingerprint; return zeros on invalid SMILES."""
    try:
        from rdkit import Chem
        from rdkit.Chem import rdFingerprintGenerator

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(n_bits, dtype=np.float32)
        generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
        return np.array(generator.GetFingerprint(mol), dtype=np.float32)
    except ImportError:
        return np.zeros(n_bits, dtype=np.float32)


def build_fingerprint_features(
    df: pd.DataFrame,
    n_bits: int = 2048,
    radius: int = 2,
    use_age: bool = True,
    age_mean: float | None = None,
    age_std: float | None = None,
) -> tuple[np.ndarray, float, float]:
    """Build concatenated Morgan fingerprint pair features."""
    features = []
    for _, row in df.iterrows():
        fp_a = _morgan_fingerprint(row["smiles_a"], n_bits=n_bits, radius=radius)
        fp_b = _morgan_fingerprint(row["smiles_b"], n_bits=n_bits, radius=radius)
        features.append(np.concatenate([fp_a, fp_b]))

    X = np.vstack(features).astype(np.float32)

    if use_age and "anchor_age" in df.columns:
        ages = df["anchor_age"].astype(float).values
        if age_mean is None:
            age_mean = float(np.mean(ages))
            age_std = float(np.std(ages))
            if age_std < 1e-8:
                age_std = 1.0
        age_norm = ((ages - age_mean) / age_std).reshape(-1, 1).astype(np.float32)
        X = np.hstack([X, age_norm])

    return X, float(age_mean or 0.0), float(age_std or 1.0)


def parse_targets(subset: pd.DataFrame) -> np.ndarray:
    """Parse JSON-encoded target vectors from 'target' column."""
    return np.array([json.loads(t) for t in subset["target"]], dtype=np.float32)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class StaticMultilabelLR(nn.Module):
    """Multi-label logistic regression for fingerprint features."""

    def __init__(self, input_dim: int, n_classes: int = 363) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.linear = nn.Linear(input_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return torch.sigmoid(self.forward(x))


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_model(
    model: StaticMultilabelLR,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    lr: float = 0.05,
    weight_decay: float = 1e-3,
    epochs: int = 200,
    seed: int = 42,
) -> tuple[StaticMultilabelLR, dict[str, Any]]:
    """Train model with best-val-loss checkpointing."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(X_train_t), y_train_t)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val_t), y_val_t)

        t_l = float(loss.item())
        v_l = float(val_loss.item())
        history.append({"epoch": epoch, "train_loss": t_l, "val_loss": v_l})

        if v_l < best_val_loss:
            best_val_loss = v_l
            best_epoch = epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 50 == 0:
            print(f"  Epoch {epoch:3d}/{epochs} | Train Loss: {t_l:.4f} | Val Loss: {v_l:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, {
        "best_epoch": best_epoch,
        "best_val_loss": round(best_val_loss, 6),
        "final_train_loss": round(history[-1]["train_loss"], 6),
        "total_epochs": epochs,
        "learning_rate": lr,
        "weight_decay": weight_decay,
        "seed": seed,
    }


# ---------------------------------------------------------------------------
# Full experiment pipeline
# ---------------------------------------------------------------------------

def run_static_multilabel_pipeline(
    df: pd.DataFrame,
    label_mapping: dict[int, int],
    n_bits: int = 2048,
    radius: int = 2,
    use_age: bool = True,
    lr: float = 0.05,
    weight_decay: float = 1e-3,
    epochs: int = 200,
    seed: int = 42,
) -> tuple[dict[str, Any], StaticMultilabelLR]:
    """Train and evaluate static fingerprint multi-label baseline."""
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
        raise ValueError("Dataset is missing one or more required splits.")

    print(f"  Split sizes -- Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    print("  Building Morgan fingerprint features...")
    X_train, age_mean, age_std = build_fingerprint_features(
        train_df, n_bits=n_bits, radius=radius, use_age=use_age
    )
    X_val, _, _ = build_fingerprint_features(
        val_df, n_bits=n_bits, radius=radius, use_age=use_age,
        age_mean=age_mean, age_std=age_std,
    )
    X_test, _, _ = build_fingerprint_features(
        test_df, n_bits=n_bits, radius=radius, use_age=use_age,
        age_mean=age_mean, age_std=age_std,
    )

    y_train = parse_targets(train_df)
    y_val = parse_targets(val_df)
    y_test = parse_targets(test_df)
    n_classes = y_train.shape[1]

    print(f"  Feature dim: {X_train.shape[1]}, n_classes: {n_classes}")

    model = StaticMultilabelLR(input_dim=X_train.shape[1], n_classes=n_classes)
    print(f"  Training StaticMultilabelLR for {epochs} epochs...")
    model, train_summary = train_model(
        model, X_train, y_train, X_val, y_val,
        lr=lr, weight_decay=weight_decay, epochs=epochs, seed=seed,
    )

    model.eval()
    with torch.no_grad():
        p_train = torch.sigmoid(model(torch.tensor(X_train))).numpy()
        p_val = torch.sigmoid(model(torch.tensor(X_val))).numpy()
        p_test = torch.sigmoid(model(torch.tensor(X_test))).numpy()

    val_opt = optimize_threshold_on_val(y_val, p_val, metric="micro_f1")
    best_thresh = val_opt["best_threshold"]

    m_train = compute_multilabel_metrics(y_train, p_train, threshold=best_thresh)
    m_val = compute_multilabel_metrics(y_val, p_val, threshold=best_thresh)
    m_test_opt = compute_multilabel_metrics(y_test, p_test, threshold=best_thresh)
    m_test_def = compute_multilabel_metrics(y_test, p_test, threshold=0.50)

    p_prior = y_train.mean(axis=0)
    p_test_prior = np.tile(p_prior, (len(y_test), 1))
    p_val_prior = np.tile(p_prior, (len(y_val), 1))
    val_opt_prior = optimize_threshold_on_val(y_val, p_val_prior, metric="micro_f1")
    m_test_prior = compute_multilabel_metrics(
        y_test, p_test_prior, threshold=val_opt_prior["best_threshold"]
    )

    results = {
        "experiment_name": "static_multilabel_lr",
        "dataset_observations": {
            "total": len(df),
            "train": len(train_df),
            "val": len(val_df),
            "test": len(test_df),
        },
        "target_dimension": n_classes,
        "feature_config": {
            "fingerprint_bits": n_bits,
            "fingerprint_radius": radius,
            "use_age": use_age,
            "input_dim": int(X_train.shape[1]),
            "age_mean": age_mean,
            "age_std": age_std,
        },
        "training_summary": train_summary,
        "model_evaluation": {
            "val_threshold_optimization": val_opt,
            "train_metrics": m_train,
            "val_metrics": m_val,
            "test_metrics_optimized_threshold": m_test_opt,
            "test_metrics_default_threshold": m_test_def,
        },
        "prior_baseline_evaluation": {
            "val_threshold_optimization": val_opt_prior,
            "test_metrics_optimized_threshold": m_test_prior,
        },
        "test_comparison": {
            "static_lr_micro_f1": m_test_opt["micro_f1"],
            "static_lr_macro_f1": m_test_opt["macro_f1"],
            "static_lr_mAP": m_test_opt["mAP"],
            "prior_micro_f1": m_test_prior["micro_f1"],
            "prior_macro_f1": m_test_prior["macro_f1"],
            "prior_mAP": m_test_prior["mAP"],
        },
    }
    return results, model


# ---------------------------------------------------------------------------
# Report generator
# ---------------------------------------------------------------------------

def generate_static_multilabel_report(results: dict[str, Any]) -> str:
    """Generate Markdown report for static multi-label baseline."""
    m_test = results["model_evaluation"]["test_metrics_optimized_threshold"]
    m_test_def = results["model_evaluation"]["test_metrics_default_threshold"]
    m_prior = results["prior_baseline_evaluation"]["test_metrics_optimized_threshold"]
    val_opt = results["model_evaluation"]["val_threshold_optimization"]
    val_opt_prior = results["prior_baseline_evaluation"]["val_threshold_optimization"]
    fc = results["feature_config"]
    obs = results["dataset_observations"]
    age_label = "included" if fc["use_age"] else "excluded"

    lines = [
        "# Static Multi-Label Baseline (Pipeline 1) -- Evaluation Report",
        "",
        "> [!IMPORTANT]",
        "> **Key Findings & Scientific Evaluation:**",
        f"> - **Model:** StaticMultilabelLR on Morgan fingerprint pair features (bits={fc['fingerprint_bits']}, radius={fc['fingerprint_radius']}, age={age_label})",
        f"> - **Input Dimension:** {fc['input_dim']}  -- **Target Space:** 363 TWOSIDES multi-label classes",
        f"> - **Observations:** {obs['train']} train, {obs['val']} val, {obs['test']} test (Pair-level split preserved)",
        f"> - **Static LR Test Micro-F1:** {m_test['micro_f1']:.4f} (Macro-F1: {m_test['macro_f1']:.4f}, mAP: {m_test['mAP']:.4f}, Threshold: {val_opt['best_threshold']})",
        f"> - **Training Prior Baseline Micro-F1:** {m_prior['micro_f1']:.4f} (Threshold: {val_opt_prior['best_threshold']})",
        "",
        "---",
        "",
        "## 1. Test Set Comparison",
        "",
        "| Model | Threshold | Micro-F1 | Macro-F1 | Hamming Loss | Jaccard | mAP | P@1 | R@1 | P@5 | R@5 | P@10 | R@10 |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        f"| **Static LR (Fingerprint + Age)** | {val_opt['best_threshold']} | **{m_test['micro_f1']:.4f}** | **{m_test['macro_f1']:.4f}** | {m_test['hamming_loss']:.4f} | {m_test['jaccard_score']:.4f} | {m_test['mAP']:.4f} | {m_test['precision_at_1']:.4f} | {m_test['recall_at_1']:.4f} | {m_test['precision_at_5']:.4f} | {m_test['recall_at_5']:.4f} | {m_test['precision_at_10']:.4f} | {m_test['recall_at_10']:.4f} |",
        f"| Static LR (Default 0.50) | 0.50 | {m_test_def['micro_f1']:.4f} | {m_test_def['macro_f1']:.4f} | {m_test_def['hamming_loss']:.4f} | {m_test_def['jaccard_score']:.4f} | {m_test_def['mAP']:.4f} | {m_test_def['precision_at_1']:.4f} | {m_test_def['recall_at_1']:.4f} | {m_test_def['precision_at_5']:.4f} | {m_test_def['recall_at_5']:.4f} | {m_test_def['precision_at_10']:.4f} | {m_test_def['recall_at_10']:.4f} |",
        f"| Training Prior Baseline | {val_opt_prior['best_threshold']} | {m_prior['micro_f1']:.4f} | {m_prior['macro_f1']:.4f} | {m_prior['hamming_loss']:.4f} | {m_prior['jaccard_score']:.4f} | {m_prior['mAP']:.4f} | {m_prior['precision_at_1']:.4f} | {m_prior['recall_at_1']:.4f} | {m_prior['precision_at_5']:.4f} | {m_prior['recall_at_5']:.4f} | {m_prior['precision_at_10']:.4f} | {m_prior['recall_at_10']:.4f} |",
        "",
        "---",
        "",
        "## 2. Split Performance Breakdown",
        "",
        "| Split | N | Micro-F1 | Macro-F1 | Hamming Loss | Jaccard | mAP |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for split_label, key in [
        ("Train", "train_metrics"),
        ("Validation", "val_metrics"),
        ("Test (Optimized)", "test_metrics_optimized_threshold"),
        ("Test (Default 0.50)", "test_metrics_default_threshold"),
    ]:
        m = results["model_evaluation"][key]
        lines.append(
            f"| {split_label} | {m['n_samples']} | {m['micro_f1']:.4f} | {m['macro_f1']:.4f} | {m['hamming_loss']:.4f} | {m['jaccard_score']:.4f} | {m['mAP']:.4f} |"
        )

    ts = results["training_summary"]
    lines.extend([
        "",
        "---",
        "",
        "## 3. Training Summary",
        "",
        f"- **Best Epoch:** {ts['best_epoch']} / {ts['total_epochs']}",
        f"- **Best Val Loss:** {ts['best_val_loss']:.6f}",
        f"- **Final Train Loss:** {ts['final_train_loss']:.6f}",
        f"- **Learning Rate:** {ts['learning_rate']},  **Weight Decay:** {ts['weight_decay']}",
        "",
    ])

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and evaluate Pipeline 1: Static Multi-Label LR on fingerprint features."
    )
    parser.add_argument(
        "--dataset",
        default=str(MULTILABEL_FREQUENT363_DATASET_CSV),
        help="Path to multilabel_frequent363_dataset.csv",
    )
    parser.add_argument(
        "--label-mapping",
        default=str(MULTILABEL_FREQUENT363_LABEL_MAPPING_PATH),
        help="Path to label mapping JSON.",
    )
    parser.add_argument(
        "--results-json",
        default=str(STATIC_MULTILABEL_RESULTS_JSON),
        help="Output path for results JSON.",
    )
    parser.add_argument(
        "--report-md",
        default=str(STATIC_MULTILABEL_REPORT_MD),
        help="Output path for evaluation report Markdown.",
    )
    parser.add_argument(
        "--checkpoint",
        default=str(STATIC_MULTILABEL_CHECKPOINT),
        help="Output path for model checkpoint (.pt).",
    )
    parser.add_argument("--n-bits", type=int, default=2048, help="Morgan fingerprint bits.")
    parser.add_argument("--radius", type=int, default=2, help="Morgan fingerprint radius.")
    parser.add_argument(
        "--no-age", action="store_true", help="Exclude anchor_age from features."
    )
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs.")
    parser.add_argument("--lr", type=float, default=0.05, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-3, help="Weight decay.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    dataset_path = Path(args.dataset)
    mapping_path = Path(args.label_mapping)
    results_json_path = Path(args.results_json)
    report_md_path = Path(args.report_md)
    checkpoint_path = Path(args.checkpoint)

    if not dataset_path.exists():
        print(f"[ERROR] Dataset not found: {dataset_path}", file=sys.stderr)
        return 1
    if not mapping_path.exists():
        print(f"[ERROR] Label mapping not found: {mapping_path}", file=sys.stderr)
        return 1

    print("==================================================")
    print("   PIPELINE 1: STATIC MULTI-LABEL BASELINE        ")
    print("==================================================")
    print(f"Dataset:         {dataset_path}")
    print(f"Label Mapping:   {mapping_path}")
    print(f"Results JSON:    {results_json_path}")
    print(f"Report MD:       {report_md_path}")
    print(f"Checkpoint:      {checkpoint_path}")
    print(f"Fingerprint:     bits={args.n_bits}, radius={args.radius}")
    print(f"Use Age:         {not args.no_age}")
    print(f"Epochs:          {args.epochs}")
    print(f"Learning Rate:   {args.lr}")
    print(f"Weight Decay:    {args.weight_decay}")
    print(f"Seed:            {args.seed}")
    print("--------------------------------------------------")

    df = pd.read_csv(dataset_path)
    mapping_payload = json.loads(mapping_path.read_text(encoding="utf-8"))
    label_mapping = {int(k): int(v) for k, v in mapping_payload["type_to_index"].items()}

    results, model = run_static_multilabel_pipeline(
        df=df,
        label_mapping=label_mapping,
        n_bits=args.n_bits,
        radius=args.radius,
        use_age=not args.no_age,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        seed=args.seed,
    )

    for path in (results_json_path, report_md_path, checkpoint_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    md_report = generate_static_multilabel_report(results)
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write(md_report)

    torch.save(model.state_dict(), checkpoint_path)

    t = results["model_evaluation"]["test_metrics_optimized_threshold"]
    p = results["prior_baseline_evaluation"]["test_metrics_optimized_threshold"]
    best_t = results["model_evaluation"]["val_threshold_optimization"]["best_threshold"]

    print("\n[OK] Training and evaluation complete.")
    print(f"[OK] Results JSON:  {results_json_path}")
    print(f"[OK] Report MD:     {report_md_path}")
    print(f"[OK] Checkpoint:    {checkpoint_path}")
    print("\n--- TEST EVALUATION SUMMARY (Pipeline 1) ---")
    print(f"Static Fingerprint LR (Threshold = {best_t}):")
    print(f"  Micro-F1:     {t['micro_f1']:.4f}")
    print(f"  Macro-F1:     {t['macro_f1']:.4f}")
    print(f"  Hamming Loss: {t['hamming_loss']:.4f}")
    print(f"  mAP:          {t['mAP']:.4f}")
    print(f"  P@1 / R@1:    {t['precision_at_1']:.4f} / {t['recall_at_1']:.4f}")
    print(f"  P@5 / R@5:    {t['precision_at_5']:.4f} / {t['recall_at_5']:.4f}")
    print(f"  P@10 / R@10:  {t['precision_at_10']:.4f} / {t['recall_at_10']:.4f}")
    print("\nTraining Prior Baseline:")
    print(f"  Micro-F1:     {p['micro_f1']:.4f}")
    print(f"  Macro-F1:     {p['macro_f1']:.4f}")
    print(f"  mAP:          {p['mAP']:.4f}")
    print("==================================================")
    print("SUCCESS: Pipeline 1 static multi-label baseline evaluated.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
