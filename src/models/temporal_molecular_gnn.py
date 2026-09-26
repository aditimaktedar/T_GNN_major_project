"""Pipeline 2: Temporal + Molecular Graph Neural Network (T-MolGNN).

Encodes two molecular graphs with a shared GIN encoder, processes extended
admission temporal features, fuses pair representations, and outputs N independent
sigmoid logits for multi-label DDI prediction.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.nn import GINConv, global_mean_pool

try:
    from src.graph.molecular_graph import ATOM_FEATURE_DIM
except ImportError:
    ATOM_FEATURE_DIM = 30  # fallback


class MolecularGINEncoder(nn.Module):
    """Shared GIN encoder with global mean pooling for molecular graphs."""

    def __init__(
        self,
        input_dim: int = ATOM_FEATURE_DIM,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for layer_idx in range(num_layers):
            in_dim = input_dim if layer_idx == 0 else hidden_dim
            mlp = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINConv(mlp))
            self.norms.append(nn.BatchNorm1d(hidden_dim))

    def forward(self, data: Batch) -> torch.Tensor:
        """Encode molecular graph batch to fixed-size embedding vector."""
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return global_mean_pool(x, batch)


class TemporalFeatureEncoder(nn.Module):
    """Encoder for admission-level temporal and age features."""

    def __init__(
        self,
        input_dim: int = 17,
        hidden_dim: int = 32,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class TemporalMolecularGNN(nn.Module):
    """Dual Molecular GNN + Admission Temporal Feature Multi-Label Predictor.

    Parameters
    ----------
    n_classes : int, default=363
        Number of output target classes.
    atom_input_dim : int, default=ATOM_FEATURE_DIM
        Atom node feature dimension.
    temporal_input_dim : int, default=17
        Number of input temporal + age features.
    mol_hidden_dim : int, default=64
        Molecular GIN hidden dimension.
    temp_hidden_dim : int, default=32
        Temporal encoder hidden dimension.
    num_mol_layers : int, default=2
        Number of GIN conv layers.
    dropout : float, default=0.2
        Dropout probability.
    fusion_hidden : int, default=128
        Pair fusion MLP hidden dimension.
    """

    def __init__(
        self,
        n_classes: int = 363,
        atom_input_dim: int = ATOM_FEATURE_DIM,
        temporal_input_dim: int = 17,
        mol_hidden_dim: int = 64,
        temp_hidden_dim: int = 32,
        num_mol_layers: int = 2,
        dropout: float = 0.2,
        fusion_hidden: int = 128,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.mol_hidden_dim = mol_hidden_dim
        self.temp_hidden_dim = temp_hidden_dim

        self.mol_encoder = MolecularGINEncoder(
            input_dim=atom_input_dim,
            hidden_dim=mol_hidden_dim,
            num_layers=num_mol_layers,
            dropout=dropout,
        )

        self.temp_encoder = TemporalFeatureEncoder(
            input_dim=temporal_input_dim,
            hidden_dim=temp_hidden_dim,
            dropout=dropout,
        )

        # Pair representation: [h_a, h_b, |h_a - h_b|, h_a * h_b, h_temp]
        fusion_in = mol_hidden_dim * 4 + temp_hidden_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, fusion_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, fusion_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(fusion_hidden, n_classes)

    def encode_pair(
        self,
        mol_a: Batch,
        mol_b: Batch,
        temporal_feats: torch.Tensor,
    ) -> torch.Tensor:
        """Encode molecular pair graphs and temporal features into fused representation."""
        h_a = self.mol_encoder(mol_a)
        h_b = self.mol_encoder(mol_b)
        h_temp = self.temp_encoder(temporal_feats)

        pair_mol = torch.cat([h_a, h_b, torch.abs(h_a - h_b), h_a * h_b], dim=1)
        return torch.cat([pair_mol, h_temp], dim=1)

    def forward(
        self,
        mol_a: Batch,
        mol_b: Batch,
        temporal_feats: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass returning raw class logits."""
        fused = self.encode_pair(mol_a, mol_b, temporal_feats)
        return self.classifier(self.fusion(fused))

    def predict_proba(
        self,
        mol_a: Batch,
        mol_b: Batch,
        temporal_feats: torch.Tensor,
    ) -> torch.Tensor:
        """Return sigmoid class probabilities for inference."""
        with torch.no_grad():
            logits = self.forward(mol_a, mol_b, temporal_feats)
            return torch.sigmoid(logits)
