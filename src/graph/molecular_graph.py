"""Atom-level molecular graph construction from SMILES using RDKit + PyG."""

from __future__ import annotations

from rdkit import Chem
from rdkit.Chem.rdchem import BondType, HybridizationType
import torch
from torch_geometric.data import Data

from src.features.rdkit_features import parse_smiles

HYBRIDIZATION_TYPES: tuple[HybridizationType, ...] = (
    HybridizationType.SP,
    HybridizationType.SP2,
    HybridizationType.SP3,
    HybridizationType.SP3D,
    HybridizationType.SP3D2,
    HybridizationType.S,
    HybridizationType.UNSPECIFIED,
)

BOND_TYPES: tuple[BondType, ...] = (
    BondType.SINGLE,
    BondType.DOUBLE,
    BondType.TRIPLE,
    BondType.AROMATIC,
)

ATOM_SCALAR_FEATURES = 6
ATOM_FEATURE_DIM = ATOM_SCALAR_FEATURES + len(HYBRIDIZATION_TYPES)
EDGE_FEATURE_DIM = len(BOND_TYPES)


def _one_hot(value: object, choices: tuple) -> list[float]:
    encoding = [0.0] * len(choices)
    for index, choice in enumerate(choices):
        if value == choice:
            encoding[index] = 1.0
            break
    return encoding


def atom_feature_vector(atom: Chem.rdchem.Atom) -> list[float]:
    """Standard atom features for GNN message passing."""
    return [
        atom.GetAtomicNum() / 100.0,
        atom.GetDegree() / 6.0,
        atom.GetFormalCharge() / 5.0,
        float(atom.GetIsAromatic()),
        atom.GetTotalNumHs() / 4.0,
        float(atom.IsInRing()),
        *_one_hot(atom.GetHybridization(), HYBRIDIZATION_TYPES),
    ]


def bond_feature_vector(bond: Chem.rdchem.Bond) -> list[float]:
    return _one_hot(bond.GetBondType(), BOND_TYPES)


def _add_self_loops(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    loops = torch.arange(num_nodes, dtype=torch.long)
    self_loops = torch.stack([loops, loops], dim=0)
    if edge_index.numel() == 0:
        return self_loops
    return torch.cat([edge_index, self_loops], dim=1)


def mol_to_pyg_data(mol: Chem.rdchem.Mol) -> Data:
    """Convert an RDKit molecule to a PyG ``Data`` object (heavy atoms only)."""
    atom_features = []
    for atom in mol.GetAtoms():
        atom_features.append(atom_feature_vector(atom))
    x = torch.tensor(atom_features, dtype=torch.float32)

    edge_indices: list[list[int]] = [[], []]
    edge_attrs: list[list[float]] = []
    for bond in mol.GetBonds():
        start = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        feat = bond_feature_vector(bond)
        edge_indices[0].extend([start, end])
        edge_indices[1].extend([end, start])
        edge_attrs.extend([feat, feat])

    num_nodes = int(x.size(0))
    if edge_indices[0]:
        edge_index = torch.tensor(edge_indices, dtype=torch.long)
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float32)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, EDGE_FEATURE_DIM), dtype=torch.float32)

    edge_index = _add_self_loops(edge_index, num_nodes)
    if num_nodes:
        loop_attr = torch.zeros((num_nodes, EDGE_FEATURE_DIM), dtype=torch.float32)
        loop_attr[:, 0] = 1.0
        edge_attr = torch.cat([edge_attr, loop_attr], dim=0) if edge_attr.numel() else loop_attr

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=num_nodes)


def smiles_to_pyg_data(smiles: object) -> Data | None:
    """Parse SMILES and return a molecular graph, or ``None`` if invalid."""
    mol = parse_smiles(smiles)
    if mol is None:
        return None
    return mol_to_pyg_data(mol)


class MolecularGraphCache:
    """Cache PyG molecular graphs keyed by SMILES string."""

    def __init__(self) -> None:
        self._cache: dict[str, Data | None] = {}

    def get(self, smiles: object) -> Data | None:
        key = str(smiles).strip()
        if key in self._cache:
            return self._cache[key]
        graph = smiles_to_pyg_data(key)
        if graph is not None:
            graph = graph.clone()
        self._cache[key] = graph
        return graph

    @property
    def n_cached(self) -> int:
        return len(self._cache)

    @property
    def n_valid(self) -> int:
        return sum(value is not None for value in self._cache.values())
