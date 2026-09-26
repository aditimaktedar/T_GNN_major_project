"""Tests for dual molecular GNN baseline."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

from src.data.molecular_pair_dataset import (
    MulticlassMolecularPairDataset,
    attach_molecular_graphs,
    collate_molecular_pairs,
    filter_molecular_trainable_rows,
    prepare_molecular_multiclass_frame,
)
from src.models.dual_molecular_gnn import DualMolecularGNN, MulticlassMolecularGNNTrainer


def _build_tiny_frame() -> pd.DataFrame:
    rows = []
    specs = [
        ("PAIR0", "CCO", "CCC", 0, "train"),
        ("PAIR0", "CCO", "CCC", 0, "train"),
        ("PAIR1", "CCCC", "CC", 1, "train"),
        ("PAIR2", "CCO", "CCCC", 2, "train"),
        ("PAIR3", "CCC", "CC", 0, "val"),
        ("PAIR3", "CCC", "CC", 1, "val"),
    ]
    for i, (pair_key, smiles_a, smiles_b, type_id, split) in enumerate(specs):
        rows.append(
            {
                "subject_id": 100 + i,
                "hadm_id": i,
                "drug_a": f"A{i}",
                "drug_b": f"B{i}",
                "smiles_a": smiles_a,
                "smiles_b": smiles_b,
                "pair_key": pair_key,
                "type": type_id,
                "split": split,
            }
        )
    return pd.DataFrame(rows)


def test_attach_molecular_graphs_and_dataset() -> None:
    enriched, encoder, _fp, cache = prepare_molecular_multiclass_frame(_build_tiny_frame())
    assert enriched["graph_valid"].all()
    ds = MulticlassMolecularPairDataset(enriched, graph_cache=cache)
    assert len(ds) == len(filter_molecular_trainable_rows(enriched))
    sample = ds[0]
    assert sample.mol_a.x.ndim == 2
    assert sample.mol_b.x.ndim == 2
    mol_a, mol_b, y = collate_molecular_pairs([sample, ds[1]])
    assert mol_a.num_graphs == 2
    assert mol_b.num_graphs == 2
    assert y.shape[0] == 2


def test_dual_molecular_gnn_forward() -> None:
    enriched, encoder, _fp, cache = prepare_molecular_multiclass_frame(_build_tiny_frame())
    ds = MulticlassMolecularPairDataset(enriched, graph_cache=cache)
    mol_a, mol_b, y = collate_molecular_pairs([ds[0], ds[1]])
    model = DualMolecularGNN(n_classes=encoder.n_classes, hidden_dim=32, num_layers=2)
    logits = model(mol_a, mol_b)
    assert logits.shape == (2, encoder.n_classes)


def test_molecular_gnn_train_and_checkpoint(tmp_path: Path) -> None:
    enriched, encoder, _fp, cache = prepare_molecular_multiclass_frame(_build_tiny_frame())
    train = enriched.loc[enriched["split"] == "train"]
    val = enriched.loc[enriched["split"] == "val"]
    trainer = MulticlassMolecularGNNTrainer(
        n_classes=encoder.n_classes,
        hidden_dim=32,
        num_layers=2,
        epochs=2,
        seed=42,
        graph_cache=cache,
    )
    ckpt = tmp_path / "mol_gnn.pt"
    history = trainer.fit(train, val, encoder, batch_size=2, checkpoint_path=ckpt)
    assert ckpt.exists()
    assert "best_val_macro_f1" in history
    probs = trainer.predict_proba(val, batch_size=2)
    assert probs.shape[0] == len(filter_molecular_trainable_rows(val))
    loaded = MulticlassMolecularGNNTrainer.load(ckpt, graph_cache=cache)
    assert len(loaded.predict(val, batch_size=2)) == len(filter_molecular_trainable_rows(val))
