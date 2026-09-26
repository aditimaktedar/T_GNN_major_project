"""Tests for atom-level molecular graph construction."""

from __future__ import annotations

import torch
from rdkit import Chem

from src.graph.molecular_graph import (
    ATOM_FEATURE_DIM,
    EDGE_FEATURE_DIM,
    MolecularGraphCache,
    mol_to_pyg_data,
    smiles_to_pyg_data,
)


def test_smiles_to_pyg_data_valid() -> None:
    data = smiles_to_pyg_data("CCO")
    assert data is not None
    assert data.x.shape[1] == ATOM_FEATURE_DIM
    assert data.edge_index.ndim == 2
    assert data.edge_attr.shape[1] == EDGE_FEATURE_DIM
    assert data.num_nodes == data.x.size(0)


def test_smiles_to_pyg_data_invalid() -> None:
    assert smiles_to_pyg_data("") is None
    assert smiles_to_pyg_data("not_a_smiles") is None


def test_single_atom_graph_has_self_loop() -> None:
    mol = Chem.MolFromSmiles("[Na+]")
    data = mol_to_pyg_data(mol)
    assert data.num_nodes == 1
    assert data.edge_index.shape[1] >= 1


def test_molecular_graph_cache_reuses_graphs() -> None:
    cache = MolecularGraphCache()
    first = cache.get("CCO")
    second = cache.get("CCO")
    assert first is not None
    assert second is not None
    assert cache.n_cached == 1
    assert torch.equal(first.x, second.x)
