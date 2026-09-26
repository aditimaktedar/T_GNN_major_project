"""CLI Entry-point for Pipeline 1: Static GAT Training & Evaluation.

Command:
    python -m src.models.train_static_gat

Trains a Graph Attention Network (MultilabelStaticGAT) on 2-node drug-pair
graphs with Morgan fingerprint node features (+ optional patient age).
Evaluates on the preserved pair-level train/val/test split using canonical
363-class multi-label metrics (Micro-F1, Macro-F1, Hamming Loss, mAP, P@K, R@K).
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

try:
    from torch_geometric.data import Batch, Data
    from torch_geometric.loader import DataLoader
    HAS_PYG = True
except ImportError:
    HAS_PYG = False

from src.data.config import (
    CHECKPOINTS_DIR,
    METRICS_DIR,
    MULTILABEL_FREQUENT363_DATASET_CSV,
    MULTILABEL_FREQUENT363_LABEL_MAPPING_PATH,
)
from src.evaluation.multilabel_metrics import compute_multilabel_metrics
from src.evaluation.thresholds import optimize_threshold_on_val, optimize_threshold_on_val_fine
from src.evaluation.training_report import (
    build_result_payload,
    format_terminal_report,
    save_json_results,
)

DEFAULT_RESULTS_JSON = METRICS_DIR / "static_gat.json"
DEFAULT_CHECKPOINT = CHECKPOINTS_DIR / "static_gat.pt"


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


def parse_targets(subset: pd.DataFrame) -> np.ndarray:
    """Parse JSON-encoded target vectors from 'target' column."""
    return np.array([json.loads(t) for t in subset["target"]], dtype=np.float32)


def build_pair_graphs(
    df: pd.DataFrame,
    n_bits: int = 2048,
    radius: int = 2,
    use_age: bool = True,
    age_mean: float | None = None,
    age_std: float | None = None,
) -> tuple[list[Data], torch.Tensor | None, float, float]:
    """Build PyG Data objects for 2-node drug pairs with edge index [[0,1],[1,0]]."""
    if not HAS_PYG:
        raise ImportError("torch_geometric is required for GAT training. Please install PyG.")

    graphs: list[Data] = []
    for _, row in df.iterrows():
        fp_a = _morgan_fingerprint(str(row["smiles_a"]), n_bits=n_bits, radius=radius)
        fp_b = _morgan_fingerprint(str(row["smiles_b"]), n_bits=n_bits, radius=radius)
        x = torch.tensor(np.stack([fp_a, fp_b], axis=0), dtype=torch.float32)
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        graphs.append(Data(x=x, edge_index=edge_index))

    age_tensor = None
    if use_age and "anchor_age" in df.columns:
        ages = df["anchor_age"].astype(float).values
        if age_mean is None:
            age_mean = float(np.mean(ages))
            age_std = float(np.std(ages))
            if age_std < 1e-8:
                age_std = 1.0
        age_norm = ((ages - age_mean) / age_std).astype(np.float32)
        age_tensor = torch.tensor(age_norm, dtype=torch.float32).unsqueeze(1)

    return graphs, age_tensor, float(age_mean or 0.0), float(age_std or 1.0)


def _compute_pos_weight(y_train: np.ndarray) -> torch.Tensor:
    """Compute per-label pos_weight = n_neg / n_pos for BCEWithLogitsLoss."""
    n_samples = y_train.shape[0]
    pos_counts = y_train.sum(axis=0).astype(np.float32)
    neg_counts = n_samples - pos_counts
    pos_counts = np.maximum(pos_counts, 1.0)
    weights = neg_counts / pos_counts
    return torch.tensor(weights, dtype=torch.float32)


def train_static_gat(
    model: nn.Module,
    train_graphs: list[Data],
    train_y: np.ndarray,
    train_age: torch.Tensor | None,
    val_graphs: list[Data],
    val_y: np.ndarray,
    val_age: torch.Tensor | None,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    epochs: int = 200,
    seed: int = 42,
    verbose_interval: int = 50,
    patience: int = 30,
    use_pos_weight: bool = True,
) -> tuple[nn.Module, dict[str, Any]]:
    """Train MultilabelStaticGAT with LR scheduling, early stopping, and gradient clipping."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    batch_train = Batch.from_data_list(train_graphs)
    batch_val = Batch.from_data_list(val_graphs)

    y_train_t = torch.tensor(train_y, dtype=torch.float32)
    y_val_t = torch.tensor(val_y, dtype=torch.float32)

    if use_pos_weight:
        pw = _compute_pos_weight(train_y)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
    else:
        criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )

    best_val_loss = float("inf")
    best_val_micro_f1 = 0.0
    best_state = None
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []

    print("\nStarting training...")
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(batch_train, age=train_age)
        loss = criterion(logits, y_train_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(batch_val, age=val_age)
            val_loss = criterion(val_logits, y_val_t)
            val_probs = torch.sigmoid(val_logits).numpy()

        t_l = float(loss.item())
        v_l = float(val_loss.item())
        val_m = compute_multilabel_metrics(val_y, val_probs, threshold=0.50)
        v_micro = float(val_m["micro_f1"])
        v_macro = float(val_m["macro_f1"])

        scheduler.step(v_l)

        history.append({
            "epoch": epoch,
            "train_loss": t_l,
            "val_loss": v_l,
            "val_micro_f1": v_micro,
            "val_macro_f1": v_macro,
        })

        if v_l < best_val_loss:
            best_val_loss = v_l
            best_val_micro_f1 = v_micro
            best_epoch = epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch == 1 or epoch % verbose_interval == 0 or epoch == epochs:
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch:3d}/{epochs} | "
                f"Train Loss: {t_l:.4f} | "
                f"Val Loss: {v_l:.4f} | "
                f"Val Micro-F1: {v_micro:.4f} | "
                f"Val Macro-F1: {v_macro:.4f} | "
                f"LR: {current_lr:.6f}"
            )

        if patience > 0 and epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs).")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Recompute train loss with restored best-state model
    model.eval()
    with torch.no_grad():
        restored_train_loss = float(criterion(model(batch_train, age=train_age), y_train_t).item())

    return model, {
        "best_epoch": best_epoch,
        "best_val_loss": round(best_val_loss, 6),
        "best_val_micro_f1": round(best_val_micro_f1, 6),
        "final_train_loss": round(restored_train_loss, 6),
        "total_epochs": epoch,
        "max_epochs": epochs,
        "learning_rate": lr,
        "weight_decay": weight_decay,
        "seed": seed,
        "early_stopped": epochs_without_improvement >= patience if patience > 0 else False,
        "use_pos_weight": use_pos_weight,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and evaluate Pipeline 1: Static Multi-Label GAT."
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
        default=str(DEFAULT_RESULTS_JSON),
        help="Output path for results JSON.",
    )
    parser.add_argument(
        "--checkpoint",
        default=str(DEFAULT_CHECKPOINT),
        help="Output path for model checkpoint (.pt).",
    )
    parser.add_argument("--n-bits", type=int, default=2048, help="Morgan fingerprint bits.")
    parser.add_argument("--radius", type=int, default=2, help="Morgan fingerprint radius.")
    parser.add_argument(
        "--no-age", action="store_true", help="Exclude anchor_age from features."
    )
    parser.add_argument("--hidden-dim", type=int, default=64, help="GAT hidden dimension.")
    parser.add_argument("--heads", type=int, default=4, help="Number of GAT attention heads.")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate.")
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--patience", type=int, default=30, help="Early stopping patience (0 to disable).")
    parser.add_argument(
        "--no-pos-weight", action="store_true",
        help="Disable pos_weight in BCEWithLogitsLoss (use uniform weighting).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if not HAS_PYG:
        print("[ERROR] PyTorch Geometric (torch_geometric) is required to run Static GAT.", file=sys.stderr)
        print("Please install torch_geometric or activate your GNN environment.", file=sys.stderr)
        return 1

    from src.models.multilabel_static_gat import MultilabelStaticGAT

    dataset_path = Path(args.dataset)
    results_json_path = Path(args.results_json)
    checkpoint_path = Path(args.checkpoint)

    if not dataset_path.exists():
        print(f"[ERROR] Dataset not found: {dataset_path}", file=sys.stderr)
        return 1

    df = pd.read_csv(dataset_path)

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
        print("[ERROR] Missing required splits (train/val/test).", file=sys.stderr)
        return 1

    train_graphs, train_age, age_mean, age_std = build_pair_graphs(
        train_df, n_bits=args.n_bits, radius=args.radius, use_age=not args.no_age
    )
    val_graphs, val_age, _, _ = build_pair_graphs(
        val_df, n_bits=args.n_bits, radius=args.radius, use_age=not args.no_age,
        age_mean=age_mean, age_std=age_std,
    )
    test_graphs, test_age, _, _ = build_pair_graphs(
        test_df, n_bits=args.n_bits, radius=args.radius, use_age=not args.no_age,
        age_mean=age_mean, age_std=age_std,
    )

    y_train = parse_targets(train_df)
    y_val = parse_targets(val_df)
    y_test = parse_targets(test_df)
    n_classes = y_train.shape[1]

    model = MultilabelStaticGAT(
        input_dim=args.n_bits,
        n_classes=n_classes,
        hidden_dim=args.hidden_dim,
        heads=args.heads,
        dropout=args.dropout,
        use_age=not args.no_age,
    )

    model, train_summary = train_static_gat(
        model=model,
        train_graphs=train_graphs,
        train_y=y_train,
        train_age=train_age,
        val_graphs=val_graphs,
        val_y=y_val,
        val_age=val_age,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        seed=args.seed,
        patience=args.patience,
        use_pos_weight=not args.no_pos_weight,
    )

    # Probabilities
    model.eval()
    with torch.no_grad():
        batch_val = Batch.from_data_list(val_graphs)
        batch_test = Batch.from_data_list(test_graphs)
        p_val = torch.sigmoid(model(batch_val, age=val_age)).numpy()
        p_test = torch.sigmoid(model(batch_test, age=test_age)).numpy()

    # Threshold selection ONLY on validation set (two-stage fine search)
    val_opt = optimize_threshold_on_val_fine(y_val, p_val, metric="micro_f1")
    selected_threshold = float(val_opt["best_threshold"])

    val_m_default = compute_multilabel_metrics(y_val, p_val, threshold=0.50)
    train_summary["val_micro_f1_selected_threshold"] = float(val_opt["best_score"])
    train_summary["val_micro_f1_default_threshold"] = float(val_m_default["micro_f1"])

    # Final evaluation ONCE on untouched test set
    test_metrics = compute_multilabel_metrics(y_test, p_test, threshold=selected_threshold)

    # Save checkpoint
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": args.n_bits,
            "n_classes": n_classes,
            "hidden_dim": args.hidden_dim,
            "heads": args.heads,
            "dropout": args.dropout,
            "use_age": not args.no_age,
            "feature_config": {
                "n_bits": args.n_bits,
                "radius": args.radius,
                "age_mean": age_mean,
                "age_std": age_std,
            },
            "selected_threshold": selected_threshold,
            "train_summary": train_summary,
        },
        checkpoint_path,
    )

    # Build and save result payload
    payload = build_result_payload(
        pipeline="Static",
        model="Static GAT",
        dataset="Frequent363",
        number_of_labels=n_classes,
        train_size=len(train_df),
        val_size=len(val_df),
        test_size=len(test_df),
        unique_train_pairs=int(train_df["pair_key"].nunique()),
        unique_val_pairs=int(val_df["pair_key"].nunique()),
        unique_test_pairs=int(test_df["pair_key"].nunique()),
        training_config={
            "epochs": args.epochs,
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
            "hidden_dim": args.hidden_dim,
            "heads": args.heads,
            "dropout": args.dropout,
            "n_bits": args.n_bits,
            "radius": args.radius,
            "use_age": not args.no_age,
        },
        training_summary=train_summary,
        selected_threshold=selected_threshold,
        threshold_selection_method="validation_micro_f1",
        test_metrics=test_metrics,
        checkpoint_path=str(checkpoint_path),
        extra={
            "val_threshold_optimization": val_opt,
        },
    )

    save_json_results(payload, results_json_path)

    # Print final terminal report
    report_data = {
        "pipeline": "Static",
        "model": "Static GAT",
        "dataset": "Frequent363",
        "test_size": len(test_df),
        "unique_test_pairs": int(test_df["pair_key"].nunique()),
        "number_of_labels": n_classes,
        "training_summary": train_summary,
        "selected_threshold": selected_threshold,
        "threshold_selection_set": "validation",
        "test_metrics": test_metrics,
        "checkpoint_path": str(checkpoint_path),
        "results_path": str(results_json_path),
    }

    print("\n" + format_terminal_report(report_data))
    return 0


if __name__ == "__main__":
    sys.exit(main())
