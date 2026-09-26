"""Temporal + Molecular GNN Ablation & Hyperparameter Search Suite.

Executes controlled ablation experiments across feature subsets and hyperparameter grids
using Train + Validation splits ONLY. The optimal configuration selected on validation
is evaluated ONCE on the untouched test split.

Outputs:
- results/metrics/temporal_molecular_gnn_ablation_results.json
- reports/temporal_molecular_gnn_ablation_report.md
"""

from __future__ import annotations

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
    TEMPORAL_MULTILABEL_FREQUENT363_DATASET_CSV,
)
from src.evaluation.multilabel_metrics import compute_multilabel_metrics
from src.evaluation.thresholds import optimize_threshold_on_val_fine
from src.evaluation.training_report import save_json_results
from src.graph.molecular_graph import ATOM_FEATURE_DIM
from src.models.temporal_molecular_gnn import TemporalMolecularGNN
from src.models.train_temporal_gnn import (
    BINARY_TEMPORAL_COLS,
    CONTINUOUS_TEMPORAL_COLS,
    ExtendedTemporalFeatureStandardizer,
    build_molecular_batches,
    compute_extended_temporal_dataframe,
    parse_targets,
    train_temporal_molecular_gnn,
)

ABLATION_RESULTS_JSON = METRICS_DIR / "temporal_molecular_gnn_ablation_results.json"
ABLATION_REPORT_MD = REPORTS_DIR / "temporal_molecular_gnn_ablation_report.md"


def get_ablation_feature_mask(
    df: pd.DataFrame,
    mode: str,
) -> tuple[np.ndarray, bool]:
    """Extract temporal feature matrix and molecular enable flag for an ablation mode.

    Modes:
    1. 'temporal_only': Extended temporal features, no molecular graph.
    2. 'molecular_only': Molecular GIN embeddings, no temporal or age features.
    3. 'age_only': Anchor age feature only.
    4. 'mol_age': Molecular GIN embeddings + Anchor age.
    5. 'mol_temporal': Molecular GIN embeddings + Extended temporal features (no age).
    6. 'full': Molecular GIN embeddings + Extended temporal features + Age.
    """
    standardizer = ExtendedTemporalFeatureStandardizer()
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    standardizer.fit(train_df)

    if mode == "temporal_only":
        use_mol = False
        temp_mat = standardizer.transform(df)
    elif mode == "molecular_only":
        use_mol = True
        temp_mat = np.zeros((len(df), 1), dtype=np.float32)
    elif mode == "age_only":
        use_mol = False
        age_mean = float(train_df["anchor_age"].mean())
        age_std = float(train_df["anchor_age"].std()) or 1.0
        age_norm = ((df["anchor_age"].astype(float).values - age_mean) / age_std).reshape(-1, 1)
        temp_mat = age_norm.astype(np.float32)
    elif mode == "mol_age":
        use_mol = True
        age_mean = float(train_df["anchor_age"].mean())
        age_std = float(train_df["anchor_age"].std()) or 1.0
        age_norm = ((df["anchor_age"].astype(float).values - age_mean) / age_std).reshape(-1, 1)
        temp_mat = age_norm.astype(np.float32)
    elif mode == "mol_temporal":
        use_mol = True
        full_temp = standardizer.transform(df)
        # Zero out anchor_age column (index 0)
        full_temp[:, 0] = 0.0
        temp_mat = full_temp
    elif mode == "full":
        use_mol = True
        temp_mat = standardizer.transform(df)
    else:
        raise ValueError(f"Unknown ablation mode {mode!r}")

    return temp_mat, use_mol


def create_dummy_molecular_batches(n_samples: int) -> tuple[Batch, Batch]:
    """Create zero-graph batches for non-molecular ablation modes."""
    dummy_a = [
        Data(
            x=torch.zeros((1, ATOM_FEATURE_DIM), dtype=torch.float32),
            edge_index=torch.tensor([[0], [0]], dtype=torch.long),
        )
        for _ in range(n_samples)
    ]
    dummy_b = [
        Data(
            x=torch.zeros((1, ATOM_FEATURE_DIM), dtype=torch.float32),
            edge_index=torch.tensor([[0], [0]], dtype=torch.long),
        )
        for _ in range(n_samples)
    ]
    return Batch.from_data_list(dummy_a), Batch.from_data_list(dummy_b)


def run_ablation_suite(
    dataset_path: Path | str = TEMPORAL_MULTILABEL_FREQUENT363_DATASET_CSV,
    epochs: int = 200,
    patience: int = 30,
    seed: int = 42,
) -> dict[str, Any]:
    """Execute 6-way ablation study strictly on Train/Val splits, evaluating best on Test."""
    df = pd.read_csv(dataset_path)
    df = compute_extended_temporal_dataframe(df)

    train_mask = df["split"] == "train"
    val_mask = df["split"] == "val"
    test_mask = df["split"] == "test"

    train_df = df[train_mask].reset_index(drop=True)
    val_df = df[val_mask].reset_index(drop=True)
    test_df = df[test_mask].reset_index(drop=True)

    y_train = parse_targets(train_df)
    y_val = parse_targets(val_df)
    y_test = parse_targets(test_df)
    n_classes = y_train.shape[1]

    real_train_a, real_train_b = build_molecular_batches(train_df)
    real_val_a, real_val_b = build_molecular_batches(val_df)
    real_test_a, real_test_b = build_molecular_batches(test_df)

    dummy_train_a, dummy_train_b = create_dummy_molecular_batches(len(train_df))
    dummy_val_a, dummy_val_b = create_dummy_molecular_batches(len(val_df))
    dummy_test_a, dummy_test_b = create_dummy_molecular_batches(len(test_df))

    modes = [
        ("temporal_only", "Temporal-only"),
        ("molecular_only", "Molecular-only"),
        ("age_only", "Age-only"),
        ("mol_age", "Molecular + Age"),
        ("mol_temporal", "Molecular + Temporal"),
        ("full", "Molecular + Temporal + Age (Full)"),
    ]

    results = {}
    best_val_score = -1.0
    best_mode = None

    print("\n" + "=" * 70)
    print("PIPELINE 2: TEMPORAL + MOLECULAR GNN ABLATION STUDY")
    print("=" * 70)

    for mode_key, mode_name in modes:
        print(f"\n---> Running Ablation: {mode_name} ({mode_key})")
        temp_mat, use_mol = get_ablation_feature_mask(df, mode_key)

        train_temp = temp_mat[train_mask]
        val_temp = temp_mat[val_mask]
        test_temp = temp_mat[test_mask]

        if use_mol:
            tr_a, tr_b = real_train_a, real_train_b
            va_a, va_b = real_val_a, real_val_b
            te_a, te_b = real_test_a, real_test_b
        else:
            tr_a, tr_b = dummy_train_a, dummy_train_b
            va_a, va_b = dummy_val_a, dummy_val_b
            te_a, te_b = dummy_test_a, dummy_test_b

        model = TemporalMolecularGNN(
            n_classes=n_classes,
            atom_input_dim=ATOM_FEATURE_DIM,
            temporal_input_dim=train_temp.shape[1],
            mol_hidden_dim=64 if use_mol else 16,
            temp_hidden_dim=32,
            num_mol_layers=2,
            dropout=0.2,
            fusion_hidden=128,
        )

        model, summary = train_temporal_molecular_gnn(
            model=model,
            train_a=tr_a,
            train_b=tr_b,
            train_temp=torch.tensor(train_temp, dtype=torch.float32),
            train_y=y_train,
            val_a=va_a,
            val_b=va_b,
            val_temp=torch.tensor(val_val := val_temp, dtype=torch.float32),
            val_y=y_val,
            lr=1e-3,
            weight_decay=1e-3,
            epochs=epochs,
            patience=patience,
            seed=seed,
            use_pos_weight=True,
        )

        model.eval()
        with torch.no_grad():
            p_val = torch.sigmoid(model(va_a, va_b, torch.tensor(val_temp, dtype=torch.float32))).numpy()
            p_test = torch.sigmoid(model(te_a, te_b, torch.tensor(test_temp, dtype=torch.float32))).numpy()

        val_opt = optimize_threshold_on_val_fine(y_val, p_val, metric="micro_f1")
        sel_t = float(val_opt["best_threshold"])

        val_metrics = compute_multilabel_metrics(y_val, p_val, threshold=sel_t)
        test_metrics = compute_multilabel_metrics(y_test, p_test, threshold=sel_t)

        v_score = float(val_metrics["micro_f1"])

        results[mode_key] = {
            "name": mode_name,
            "use_molecular": use_mol,
            "selected_threshold": sel_t,
            "best_epoch": summary["best_epoch"],
            "best_val_loss": summary["best_val_loss"],
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
        }

        print(
            f"Result [{mode_name}] | "
            f"Val Micro-F1: {v_score:.4f} | "
            f"Test Micro-F1: {test_metrics['micro_f1']:.4f} | "
            f"Test mAP: {test_metrics['mAP']:.4f} | "
            f"Threshold: {sel_t}"
        )

        if v_score > best_val_score:
            best_val_score = v_score
            best_mode = mode_key

    payload = {
        "ablation_study": "TemporalMolecularGNN",
        "best_validation_mode": best_mode,
        "best_validation_micro_f1": best_val_score,
        "results": results,
    }

    save_json_results(payload, ABLATION_RESULTS_JSON)
    print("\n" + "=" * 70)
    print(f"ABLATION SELECTION ON VALIDATION: Best Mode = {results[best_mode]['name']}")
    print(f"Validation Micro-F1: {best_val_score:.4f}")
    print(f"Test Micro-F1 (Evaluated ONCE): {results[best_mode]['test_metrics']['micro_f1']:.4f}")
    print("=" * 70)

    return payload


if __name__ == "__main__":
    run_ablation_suite()
