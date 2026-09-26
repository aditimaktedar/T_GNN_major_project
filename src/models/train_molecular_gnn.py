"""CLI Entry-point for Pipeline 1: Molecular GNN Training & Evaluation.

Command:
    python -m src.models.train_molecular_gnn

Trains a Dual Molecular Graph Neural Network (MultilabelMolecularGNN) on
atom-level molecular graphs using a shared GIN encoder with pair fusion
(+ optional patient age).
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

DEFAULT_RESULTS_JSON = METRICS_DIR / "static_molecular_gnn.json"
DEFAULT_CHECKPOINT = CHECKPOINTS_DIR / "static_molecular_gnn.pt"


def parse_targets(subset: pd.DataFrame) -> np.ndarray:
    """Parse JSON-encoded target vectors from 'target' column."""
    return np.array([json.loads(t) for t in subset["target"]], dtype=np.float32)


def _compute_pos_weight(y_train: np.ndarray) -> torch.Tensor:
    """Compute per-label pos_weight = n_neg / n_pos for BCEWithLogitsLoss."""
    n_samples = y_train.shape[0]
    pos_counts = y_train.sum(axis=0).astype(np.float32)
    neg_counts = n_samples - pos_counts
    pos_counts = np.maximum(pos_counts, 1.0)
    weights = neg_counts / pos_counts
    return torch.tensor(weights, dtype=torch.float32)


def build_molecular_batches(
    df: pd.DataFrame,
    use_age: bool = True,
    age_mean: float | None = None,
    age_std: float | None = None,
) -> tuple[Batch, Batch, torch.Tensor | None, float, float]:
    """Convert smiles_a and smiles_b columns to PyG Batches."""
    if not HAS_PYG:
        raise ImportError("torch_geometric is required for Molecular GNN. Please install PyG.")

    try:
        from src.graph.molecular_graph import MolecularGraphCache, ATOM_FEATURE_DIM
    except ImportError:
        ATOM_FEATURE_DIM = 30
        MolecularGraphCache = None

    cache = MolecularGraphCache() if MolecularGraphCache else None

    graphs_a: list[Data] = []
    graphs_b: list[Data] = []

    for _, row in df.iterrows():
        sa = str(row["smiles_a"])
        sb = str(row["smiles_b"])

        ga = cache.get(sa) if cache else None
        gb = cache.get(sb) if cache else None

        if ga is None:
            # Fallback 1-node graph if parsing fails
            ga = Data(
                x=torch.zeros((1, ATOM_FEATURE_DIM), dtype=torch.float32),
                edge_index=torch.tensor([[0], [0]], dtype=torch.long),
            )
        if gb is None:
            gb = Data(
                x=torch.zeros((1, ATOM_FEATURE_DIM), dtype=torch.float32),
                edge_index=torch.tensor([[0], [0]], dtype=torch.long),
            )

        graphs_a.append(ga)
        graphs_b.append(gb)

    batch_a = Batch.from_data_list(graphs_a)
    batch_b = Batch.from_data_list(graphs_b)

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

    return batch_a, batch_b, age_tensor, float(age_mean or 0.0), float(age_std or 1.0)


def train_molecular_gnn(
    model: nn.Module,
    train_a: Batch,
    train_b: Batch,
    train_y: np.ndarray,
    train_age: torch.Tensor | None,
    val_a: Batch,
    val_b: Batch,
    val_y: np.ndarray,
    val_age: torch.Tensor | None,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    epochs: int = 200,
    seed: int = 42,
    verbose_interval: int = 50,
    patience: int = 0,
    use_pos_weight: bool = True,
) -> tuple[nn.Module, dict[str, Any]]:
    """Train MultilabelMolecularGNN with validation tracking."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    y_train_t = torch.tensor(train_y, dtype=torch.float32)
    y_val_t = torch.tensor(val_y, dtype=torch.float32)

    # Loss with optional pos_weight for class imbalance
    if use_pos_weight:
        pw = _compute_pos_weight(train_y)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
    else:
        criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # Learning rate scheduler on validation loss
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
        logits = model(train_a, train_b, age=train_age)
        loss = criterion(logits, y_train_t)
        # Gradient clipping for stability
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(val_a, val_b, age=val_age)
            val_loss = criterion(val_logits, y_val_t)
            val_probs = torch.sigmoid(val_logits).numpy()

        t_l = float(loss.item())
        v_l = float(val_loss.item())
        val_m = compute_multilabel_metrics(val_y, val_probs, threshold=0.50)
        v_micro = float(val_m["micro_f1"])
        v_macro = float(val_m["macro_f1"])

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
            print(
                f"Epoch {epoch:3d}/{epochs} | "
                f"Train Loss: {t_l:.4f} | "
                f"Val Loss: {v_l:.4f} | "
                f"Val Micro-F1: {v_micro:.4f} | "
                f"Val Macro-F1: {v_macro:.4f}"
            )

        # Early stopping
        if patience > 0 and epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs).")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Recompute train loss with restored best-state model for accurate reporting
    model.eval()
    with torch.no_grad():
        restored_train_loss = float(criterion(model(train_a, train_b, age=train_age), y_train_t).item())

    return model, {
        "best_epoch": best_epoch,
        "best_val_loss": round(best_val_loss, 6),
        "best_val_micro_f1": round(best_val_micro_f1, 6),
        "final_train_loss": round(restored_train_loss, 6),
        "total_epochs": epochs,
        "learning_rate": lr,
        "weight_decay": weight_decay,
        "seed": seed,
        "early_stopped": epochs_without_improvement >= patience if patience > 0 else False,
        "use_pos_weight": use_pos_weight,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and evaluate Pipeline 1: Molecular GNN (Dual GIN encoder)."
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
    parser.add_argument(
        "--no-age", action="store_true", help="Exclude anchor_age from features."
    )
    parser.add_argument("--hidden-dim", type=int, default=64, help="GNN hidden dimension.")
    parser.add_argument("--num-layers", type=int, default=3, help="Number of GIN layers.")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate.")
    parser.add_argument("--fusion-hidden", type=int, default=128, help="Pair fusion hidden dimension.")
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--patience", type=int, default=0, help="Early stopping patience (0 disables).")
    parser.add_argument("--no-pos-weight", action="store_true", help="Disable pos_weight in loss.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if not HAS_PYG:
        print("[ERROR] PyTorch Geometric (torch_geometric) is required to run Molecular GNN.", file=sys.stderr)
        print("Please install torch_geometric or activate your GNN environment.", file=sys.stderr)
        return 1

    from src.models.multilabel_molecular_gnn import MultilabelMolecularGNN
    from src.graph.molecular_graph import ATOM_FEATURE_DIM

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

    train_a, train_b, train_age, age_mean, age_std = build_molecular_batches(
        train_df, use_age=not args.no_age
    )
    val_a, val_b, val_age, _, _ = build_molecular_batches(
        val_df, use_age=not args.no_age, age_mean=age_mean, age_std=age_std
    )
    test_a, test_b, test_age, _, _ = build_molecular_batches(
        test_df, use_age=not args.no_age, age_mean=age_mean, age_std=age_std
    )

    y_train = parse_targets(train_df)
    y_val = parse_targets(val_df)
    y_test = parse_targets(test_df)
    n_classes = y_train.shape[1]

    model = MultilabelMolecularGNN(
        n_classes=n_classes,
        input_dim=ATOM_FEATURE_DIM,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        fusion_hidden=args.fusion_hidden,
        use_age=not args.no_age,
    )

    model, train_summary = train_molecular_gnn(
        model=model,
        train_a=train_a,
        train_b=train_b,
        train_y=y_train,
        train_age=train_age,
        val_a=val_a,
        val_b=val_b,
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
        p_val = torch.sigmoid(model(val_a, val_b, age=val_age)).numpy()
        p_test = torch.sigmoid(model(test_a, test_b, age=test_age)).numpy()

    # Threshold selection strictly on validation set
    val_opt = optimize_threshold_on_val(y_val, p_val, metric="micro_f1")
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
            "n_classes": n_classes,
            "input_dim": ATOM_FEATURE_DIM,
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "fusion_hidden": args.fusion_hidden,
            "use_age": not args.no_age,
            "feature_config": {
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
        model="Molecular GNN",
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
            "num_layers": args.num_layers,
            "fusion_hidden": args.fusion_hidden,
            "dropout": args.dropout,
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
        "model": "Molecular GNN",
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
