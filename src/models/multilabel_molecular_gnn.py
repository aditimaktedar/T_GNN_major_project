"""Multi-label Dual Molecular GNN for DDI prediction.

Encodes two molecular graphs with a shared GIN encoder, fuses pair representations,
and outputs N independent sigmoid logits. Uses BCEWithLogitsLoss.
Fusion: [h_a, h_b, |h_a - h_b|, h_a ⊙ h_b]
Supports optional age vector concatenation.
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


class MultilabelMolecularGNNEncoder(nn.Module):
    """Shared GIN encoder with global mean pooling for molecular graphs."""

    def __init__(
        self,
        input_dim: int = ATOM_FEATURE_DIM,
        hidden_dim: int = 64,
        num_layers: int = 3,
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
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return global_mean_pool(x, batch)


class MultilabelMolecularGNN(nn.Module):
    """Dual molecular GNN with multi-label sigmoid output.

    Parameters
    ----------
    n_classes : int
        Number of output labels (e.g., 363 for FREQUENT363).
    input_dim : int
        Atom feature dimensionality.
    hidden_dim : int
        GIN hidden dimension.
    num_layers : int
        Number of GIN layers.
    dropout : float
        Dropout probability.
    fusion_hidden : int
        Hidden dimension for pair fusion MLP.
    use_age : bool
        Whether to expect and use a concatenated age feature.
    """

    def __init__(
        self,
        n_classes: int = 363,
        input_dim: int = ATOM_FEATURE_DIM,
        hidden_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.2,
        fusion_hidden: int = 128,
        use_age: bool = False,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.use_age = use_age

        self.encoder = MultilabelMolecularGNNEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )

        # Fusion: [h_a, h_b, |h_a - h_b|, h_a * h_b]
        fusion_in = hidden_dim * 4 + (1 if use_age else 0)
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, fusion_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, fusion_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(fusion_hidden, n_classes)

    def encode_pair(self, mol_a: Batch, mol_b: Batch) -> torch.Tensor:
        """Encode a pair of molecular graphs and fuse their representations."""
        h_a = self.encoder(mol_a)
        h_b = self.encoder(mol_b)
        return torch.cat([h_a, h_b, torch.abs(h_a - h_b), h_a * h_b], dim=1)

    def forward(
        self,
        mol_a: Batch,
        mol_b: Batch,
        age: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass returning raw logits.

        Parameters
        ----------
        mol_a : Batch
            PyG Batch for drug A molecular graphs.
        mol_b : Batch
            PyG Batch for drug B molecular graphs.
        age : torch.Tensor, optional
            Normalized age values of shape (batch_size, 1).

        Returns
        -------
        torch.Tensor
            Logits of shape (batch_size, n_classes).
        """
        pair_features = self.encode_pair(mol_a, mol_b)

        if self.use_age and age is not None:
            if age.dim() == 1:
                age = age.unsqueeze(1)
            pair_features = torch.cat([pair_features, age], dim=1)

        return self.classifier(self.fusion(pair_features))

    def predict_proba(
        self,
        mol_a: Batch,
        mol_b: Batch,
        age: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return sigmoid probabilities for inference."""
        with torch.no_grad():
            logits = self.forward(mol_a, mol_b, age=age)
            return torch.sigmoid(logits)
