"""PyG dataset for drug-pair classification with atom-level molecular graphs."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data

from src.data.final_ml_dataset import (
    TypeLabelEncoder,
    filter_trainable_rows,
    prepare_multiclass_frame,
)
from src.data.frequent363_ml_dataset import prepare_frequent363_multiclass_frame
from src.graph.molecular_graph import MolecularGraphCache


@dataclass
class MolecularPairSample:
    mol_a: Data
    mol_b: Data
    y: torch.Tensor


def attach_molecular_graphs(
    frame: pd.DataFrame,
    cache: MolecularGraphCache | None = None,
) -> pd.DataFrame:
    """Return a copy with molecular graph validity flags (does not modify source CSV)."""
    graph_cache = cache or MolecularGraphCache()
    out = frame.copy()
    valid = []
    for _, row in out.iterrows():
        graph_a = graph_cache.get(row["smiles_a"])
        graph_b = graph_cache.get(row["smiles_b"])
        valid.append(graph_a is not None and graph_b is not None)
    out["graph_valid"] = valid
    return out


def filter_molecular_trainable_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep rows with valid fingerprints, encoded labels, and molecular graphs."""
    working = frame.copy()
    if "graph_valid" not in working.columns:
        working = attach_molecular_graphs(working)
    mask = pd.Series(True, index=working.index)
    if "fingerprint_valid" in working.columns:
        mask &= working["fingerprint_valid"].astype(bool)
    mask &= working["graph_valid"].astype(bool)
    if "label_index" in working.columns:
        mask &= working["label_index"].astype(int) >= 0
    return working.loc[mask].reset_index(drop=True)


def prepare_molecular_multiclass_frame(
    frame: pd.DataFrame,
    label_encoder: TypeLabelEncoder | None = None,
    graph_cache: MolecularGraphCache | None = None,
):
    enriched, encoder, fp_cache = prepare_multiclass_frame(frame, label_encoder=label_encoder)
    mol_cache = graph_cache or MolecularGraphCache()
    enriched = attach_molecular_graphs(enriched, cache=mol_cache)
    return enriched, encoder, fp_cache, mol_cache


def prepare_molecular_frequent363_frame(
    frame: pd.DataFrame,
    label_encoder: TypeLabelEncoder | None = None,
    graph_cache: MolecularGraphCache | None = None,
    min_train_rows: int | None = None,
):
    kwargs = {"label_encoder": label_encoder}
    if min_train_rows is not None:
        kwargs["min_train_rows"] = min_train_rows
    enriched, encoder, fp_cache = prepare_frequent363_multiclass_frame(frame, **kwargs)
    mol_cache = graph_cache or MolecularGraphCache()
    enriched = attach_molecular_graphs(enriched, cache=mol_cache)
    return enriched, encoder, fp_cache, mol_cache


def collate_molecular_pairs(batch: list[MolecularPairSample]) -> tuple[Batch, Batch, torch.Tensor]:
    mol_a = Batch.from_data_list([item.mol_a for item in batch])
    mol_b = Batch.from_data_list([item.mol_b for item in batch])
    y = torch.stack([item.y for item in batch]).view(-1)
    return mol_a, mol_b, y


class MulticlassMolecularPairDataset(Dataset):
    """One sample = (mol_graph_a, mol_graph_b, multiclass label)."""

    def __init__(self, frame: pd.DataFrame, graph_cache: MolecularGraphCache | None = None):
        self.cache = graph_cache or MolecularGraphCache()
        self.frame = filter_molecular_trainable_rows(frame)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> MolecularPairSample:
        row = self.frame.iloc[index]
        graph_a = self.cache.get(row["smiles_a"])
        graph_b = self.cache.get(row["smiles_b"])
        if graph_a is None or graph_b is None:
            raise ValueError("Invalid molecular graph for trainable row.")
        mol_a = graph_a.clone()
        mol_b = graph_b.clone()
        y = torch.tensor([int(row["label_index"])], dtype=torch.long)
        return MolecularPairSample(mol_a=mol_a, mol_b=mol_b, y=y)
