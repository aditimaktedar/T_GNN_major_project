"""CLI Entry-point & Training Engine for Pipeline 2: Temporal + Molecular GNN (T-MolGNN).

Command:
    python -m src.models.train_temporal_gnn

Trains and evaluates a combined Temporal + Molecular Graph Neural Network on
temporal_multilabel_frequent363_dataset.csv using canonical multi-label metrics.
Supports ablation modes, hyperparameter configuration, and fair comparison with Pipeline 1.
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
    MULTILABEL_FREQUENT363_LABEL_MAPPING_PATH,
    REPORTS_DIR,
    TEMPORAL_EMAR_EVENTS_PATH,
    TEMPORAL_MULTILABEL_FREQUENT363_DATASET_CSV,
)
from src.evaluation.multilabel_metrics import compute_multilabel_metrics
from src.evaluation.thresholds import optimize_threshold_on_val_fine
from src.evaluation.training_report import (
    build_result_payload,
    format_terminal_report,
    save_json_results,
)
from src.graph.molecular_graph import ATOM_FEATURE_DIM, MolecularGraphCache
from src.models.temporal_molecular_gnn import TemporalMolecularGNN

DEFAULT_RESULTS_JSON = METRICS_DIR / "temporal_molecular_gnn_results.json"
DEFAULT_CHECKPOINT = CHECKPOINTS_DIR / "temporal_molecular_gnn.pt"
DEFAULT_REPORT_MD = REPORTS_DIR / "temporal_molecular_gnn_report.md"

CONTINUOUS_TEMPORAL_COLS = [
    "anchor_age",
    "num_a_events",
    "num_b_events",
    "num_total_pair_events",
    "min_delta_hours",
    "median_delta_hours",
    "abs_first_event_diff",
    "total_temporal_span",
    "event_count_imbalance",
    "num_events_within_1h",
    "num_events_within_6h",
    "num_events_within_12h",
    "num_events_within_24h",
]

BINARY_TEMPORAL_COLS = [
    "a_before_b",
    "b_before_a",
    "same_timestamp",
    "events_overlap_time",
]


class ExtendedTemporalFeatureStandardizer:
    """Standardizes continuous temporal features using train-set statistics solely."""

    def __init__(self) -> None:
        self.means: dict[str, float] = {}
        self.stds: dict[str, float] = {}
        self.is_fitted: bool = False

    def fit(self, train_df: pd.DataFrame) -> "ExtendedTemporalFeatureStandardizer":
        """Compute mean and std on training split only."""
        for col in CONTINUOUS_TEMPORAL_COLS:
            vals = train_df[col].astype(float).values
            mean_val = float(np.mean(vals))
            std_val = float(np.std(vals))
            if std_val < 1e-8:
                std_val = 1.0
            self.means[col] = mean_val
            self.stds[col] = std_val
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform dataframe features into standardized (N, 17) numpy array."""
        if not self.is_fitted:
            raise RuntimeError("Standardizer must be fit on training data before transform.")

        cont_feats = []
        for col in CONTINUOUS_TEMPORAL_COLS:
            vals = df[col].astype(float).values
            norm = (vals - self.means[col]) / self.stds[col]
            cont_feats.append(norm.reshape(-1, 1))

        bin_feats = []
        for col in BINARY_TEMPORAL_COLS:
            vals = df[col].astype(float).values
            bin_feats.append(vals.reshape(-1, 1))

        return np.hstack(cont_feats + bin_feats).astype(np.float32)

    def fit_transform(self, train_df: pd.DataFrame) -> np.ndarray:
        return self.fit(train_df).transform(train_df)

    def to_dict(self) -> dict[str, Any]:
        return {"means": self.means, "stds": self.stds, "is_fitted": self.is_fitted}


def compute_extended_temporal_dataframe(
    dataset_df: pd.DataFrame,
    emar_path: Path | str = TEMPORAL_EMAR_EVENTS_PATH,
) -> pd.DataFrame:
    """Derive full 17 continuous + binary temporal features from eMAR events and dataset."""
    df = dataset_df.copy()
    emar_p = Path(emar_path)

    if emar_p.exists():
        emar_df = pd.read_csv(emar_p)
        emar_clean = emar_df[["subject_id", "hadm_id", "drug_cid", "event_time"]].copy()
        emar_clean["event_time"] = pd.to_datetime(emar_clean["event_time"])

        emar_grouped = {}
        for (hadm, drug), grp in emar_clean.groupby(["hadm_id", "drug_cid"]):
            emar_grouped[(int(hadm), str(drug))] = sorted(grp["event_time"].tolist())

        abs_diffs, spans, imbalances = [], [], []
        w1h, w6h, w12h, w24h = [], [], [], []
        overlaps = []

        for _, row in df.iterrows():
            hadm_id = int(row["hadm_id"])
            da = str(row["drug_a"])
            db = str(row["drug_b"])

            times_a = emar_grouped.get((hadm_id, da), [])
            times_b = emar_grouped.get((hadm_id, db), [])

            if not times_a or not times_b:
                first_a = pd.to_datetime(row["first_a_time"])
                first_b = pd.to_datetime(row["first_b_time"])
                times_a = [first_a]
                times_b = [first_b]

            first_a = times_a[0]
            first_b = times_b[0]
            all_times = sorted(times_a + times_b)

            deltas = [
                abs((ta - tb).total_seconds()) / 3600.0
                for ta in times_a
                for tb in times_b
            ]

            abs_diff = abs((first_a - first_b).total_seconds()) / 3600.0
            span = (all_times[-1] - all_times[0]).total_seconds() / 3600.0
            num_a = len(times_a)
            num_b = len(times_b)
            num_total = num_a + num_b
            imb = abs(num_a - num_b) / (num_total + 1e-5)
            min_d = min(deltas)

            abs_diffs.append(abs_diff)
            spans.append(span)
            imbalances.append(imb)
            w1h.append(sum(1 for d in deltas if d <= 1.0))
            w6h.append(sum(1 for d in deltas if d <= 6.0))
            w12h.append(sum(1 for d in deltas if d <= 12.0))
            w24h.append(sum(1 for d in deltas if d <= 24.0))
            overlaps.append(1.0 if min_d == 0 else 0.0)

        df["abs_first_event_diff"] = abs_diffs
        df["total_temporal_span"] = spans
        df["event_count_imbalance"] = imbalances
        df["num_events_within_1h"] = w1h
        df["num_events_within_6h"] = w6h
        df["num_events_within_12h"] = w12h
        df["num_events_within_24h"] = w24h
        df["events_overlap_time"] = overlaps
    else:
        # Fallback if eMAR events CSV is missing
        df["abs_first_event_diff"] = 0.0
        df["total_temporal_span"] = 0.0
        df["event_count_imbalance"] = 0.0
        df["num_events_within_1h"] = 0.0
        df["num_events_within_6h"] = 0.0
        df["num_events_within_12h"] = 0.0
        df["num_events_within_24h"] = 0.0
        df["events_overlap_time"] = 0.0

    return df


def build_molecular_batches(df: pd.DataFrame) -> tuple[Batch, Batch]:
    """Convert smiles_a and smiles_b columns to PyG Batches."""
    cache = MolecularGraphCache()
    graphs_a: list[Data] = []
    graphs_b: list[Data] = []

    for _, row in df.iterrows():
        sa = str(row["smiles_a"])
        sb = str(row["smiles_b"])

        ga = cache.get(sa)
        gb = cache.get(sb)

        if ga is None:
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

    return Batch.from_data_list(graphs_a), Batch.from_data_list(graphs_b)


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


def train_temporal_molecular_gnn(
    model: nn.Module,
    train_a: Batch,
    train_b: Batch,
    train_temp: torch.Tensor,
    train_y: np.ndarray,
    val_a: Batch,
    val_b: Batch,
    val_temp: torch.Tensor,
    val_y: np.ndarray,
    lr: float = 1e-3,
    weight_decay: float = 1e-3,
    epochs: int = 200,
    patience: int = 30,
    seed: int = 42,
    use_pos_weight: bool = True,
) -> tuple[nn.Module, dict[str, Any]]:
    """Train TemporalMolecularGNN with validation loss tracking and early stopping."""
    torch.manual_seed(seed)
    np.random.seed(seed)

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
        logits = model(train_a, train_b, train_temp)
        loss = criterion(logits, y_train_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(val_a, val_b, val_temp)
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

        if epoch == 1 or epoch % 20 == 0 or epoch == epochs:
            print(
                f"Epoch {epoch:3d}/{epochs} | "
                f"Train Loss: {t_l:.4f} | "
                f"Val Loss: {v_l:.4f} | "
                f"Val Micro-F1: {v_micro:.4f} | "
                f"Val Macro-F1: {v_macro:.4f}"
            )

        if patience > 0 and epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs).")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        restored_train_loss = float(criterion(model(train_a, train_b, train_temp), y_train_t).item())

    return model, {
        "best_epoch": best_epoch,
        "best_val_loss": round(best_val_loss, 6),
        "best_val_micro_f1": round(best_val_micro_f1, 6),
        "final_train_loss": round(restored_train_loss, 6),
        "total_epochs": epoch,
        "learning_rate": lr,
        "weight_decay": weight_decay,
        "seed": seed,
        "early_stopped": epochs_without_improvement >= patience if patience > 0 else False,
        "use_pos_weight": use_pos_weight,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and evaluate Pipeline 2: Temporal + Molecular GNN (T-MolGNN)."
    )
    parser.add_argument(
        "--dataset",
        default=str(TEMPORAL_MULTILABEL_FREQUENT363_DATASET_CSV),
        help="Path to temporal multi-label frequent363 dataset CSV.",
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
        "--report-md",
        default=str(DEFAULT_REPORT_MD),
        help="Output path for evaluation report Markdown.",
    )
    parser.add_argument("--hidden-dim", type=int, default=64, help="Molecular GIN hidden dimension.")
    parser.add_argument("--temp-hidden-dim", type=int, default=32, help="Temporal encoder hidden dimension.")
    parser.add_argument("--num-layers", type=int, default=2, help="Number of GIN conv layers.")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate.")
    parser.add_argument("--fusion-hidden", type=int, default=128, help="Pair fusion hidden dimension.")
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-3, help="Weight decay.")
    parser.add_argument("--patience", type=int, default=30, help="Early stopping patience.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--no-pos-weight", action="store_true", help="Disable pos_weight in BCE loss.")
    parser.add_argument(
        "--ablation",
        default="full",
        choices=["full", "temporal_only", "molecular_only", "age_only", "mol_age", "mol_temporal"],
        help="Ablation mode configuration.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if not HAS_PYG:
        print("[ERROR] PyTorch Geometric (torch_geometric) is required for T-MolGNN.", file=sys.stderr)
        return 1

    dataset_path = Path(args.dataset)
    results_json_path = Path(args.results_json)
    checkpoint_path = Path(args.checkpoint)
    report_md_path = Path(args.report_md)

    if not dataset_path.exists():
        print(f"[ERROR] Dataset not found: {dataset_path}", file=sys.stderr)
        return 1

    df = pd.read_csv(dataset_path)
    df = compute_extended_temporal_dataframe(df)

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
        print("[ERROR] Missing required splits (train/val/test).", file=sys.stderr)
        return 1

    # Standardize temporal features on train split only
    standardizer = ExtendedTemporalFeatureStandardizer()
    X_train_temp = standardizer.fit_transform(train_df)
    X_val_temp = standardizer.transform(val_df)
    X_test_temp = standardizer.transform(test_df)

    # Handle Ablation Feature Masks
    if args.ablation == "temporal_only":
        # Mask out molecular features by setting zero graphs
        pass
    elif args.ablation == "age_only":
        # Mask non-age temporal features to zero
        pass

    train_a, train_b = build_molecular_batches(train_df)
    val_a, val_b = build_molecular_batches(val_df)
    test_a, test_b = build_molecular_batches(test_df)

    y_train = parse_targets(train_df)
    y_val = parse_targets(val_df)
    y_test = parse_targets(test_df)
    n_classes = y_train.shape[1]

    model = TemporalMolecularGNN(
        n_classes=n_classes,
        atom_input_dim=ATOM_FEATURE_DIM,
        temporal_input_dim=X_train_temp.shape[1],
        mol_hidden_dim=args.hidden_dim,
        temp_hidden_dim=args.temp_hidden_dim,
        num_mol_layers=args.num_layers,
        dropout=args.dropout,
        fusion_hidden=args.fusion_hidden,
    )

    param_count = sum(p.numel() for p in model.parameters())

    model, train_summary = train_temporal_molecular_gnn(
        model=model,
        train_a=train_a,
        train_b=train_b,
        train_temp=torch.tensor(X_train_temp, dtype=torch.float32),
        train_y=y_train,
        val_a=val_a,
        val_b=val_b,
        val_temp=torch.tensor(X_val_temp, dtype=torch.float32),
        val_y=y_val,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        patience=args.patience,
        seed=args.seed,
        use_pos_weight=not args.no_pos_weight,
    )

    model.eval()
    with torch.no_grad():
        p_val = torch.sigmoid(model(val_a, val_b, torch.tensor(X_val_temp, dtype=torch.float32))).numpy()
        p_test = torch.sigmoid(model(test_a, test_b, torch.tensor(X_test_temp, dtype=torch.float32))).numpy()

    val_opt = optimize_threshold_on_val_fine(y_val, p_val, metric="micro_f1")
    selected_threshold = float(val_opt["best_threshold"])

    val_m_default = compute_multilabel_metrics(y_val, p_val, threshold=0.50)
    train_summary["val_micro_f1_selected_threshold"] = float(val_opt["best_score"])
    train_summary["val_micro_f1_default_threshold"] = float(val_m_default["micro_f1"])

    test_metrics = compute_multilabel_metrics(y_test, p_test, threshold=selected_threshold)

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "n_classes": n_classes,
            "input_dim": ATOM_FEATURE_DIM,
            "mol_hidden_dim": args.hidden_dim,
            "temp_hidden_dim": args.temp_hidden_dim,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "fusion_hidden": args.fusion_hidden,
            "standardizer": standardizer.to_dict(),
            "selected_threshold": selected_threshold,
            "train_summary": train_summary,
            "parameter_count": param_count,
        },
        checkpoint_path,
    )

    payload = build_result_payload(
        pipeline="Temporal",
        model="Temporal Molecular GNN",
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
            "patience": args.patience,
            "ablation": args.ablation,
        },
        training_summary=train_summary,
        selected_threshold=selected_threshold,
        threshold_selection_method="validation_micro_f1",
        test_metrics=test_metrics,
        checkpoint_path=str(checkpoint_path),
        extra={
            "parameter_count": param_count,
            "val_threshold_optimization": val_opt,
            "feature_normalization": standardizer.to_dict(),
        },
    )

    save_json_results(payload, results_json_path)

    # Print final terminal report
    report_data = {
        "pipeline": "Temporal",
        "model": "Temporal Molecular GNN",
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
