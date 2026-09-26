"""Multi-label Static GAT model for DDI prediction.

Outputs N independent sigmoid logits per drug pair using Graph Attention Networks.
Uses BCEWithLogitsLoss for multi-label training.
Supports optional age vector concatenation at the graph readout level.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.nn import GATConv, global_mean_pool


class MultilabelStaticGAT(nn.Module):
    """Static GAT on a 2-node drug-pair graph with multi-label sigmoid output.

    Parameters
    ----------
    input_dim : int
        Dimensionality of node features.
    n_classes : int
        Number of output labels (e.g., 363 for FREQUENT363).
    hidden_dim : int
        Hidden dimension for GAT layers.
    heads : int
        Number of attention heads.
    dropout : float
        Dropout probability.
    use_age : bool
        Whether to expect and use a concatenated age feature.
    """

    def __init__(
        self,
        input_dim: int,
        n_classes: int = 363,
        hidden_dim: int = 64,
        heads: int = 4,
        dropout: float = 0.2,
        use_age: bool = False,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.use_age = use_age
        self.dropout = dropout

        self.conv1 = GATConv(input_dim, hidden_dim, heads=heads, dropout=dropout)
        self.conv2 = GATConv(
            hidden_dim * heads, hidden_dim, heads=1, concat=False, dropout=dropout
        )

        classifier_in = hidden_dim + (1 if use_age else 0)
        self.classifier = nn.Sequential(
            nn.Linear(classifier_in, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(
        self, data: Batch, age: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Forward pass returning raw logits.

        Parameters
        ----------
        data : Batch
            PyG Batch with x, edge_index, batch.
        age : torch.Tensor, optional
            Normalized age values of shape (batch_size, 1).

        Returns
        -------
        torch.Tensor
            Logits of shape (batch_size, n_classes).
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)

        if self.use_age and age is not None:
            if age.dim() == 1:
                age = age.unsqueeze(1)
            x = torch.cat([x, age], dim=1)

        return self.classifier(x)

    def predict_proba(
        self, data: Batch, age: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Return sigmoid probabilities for inference."""
        with torch.no_grad():
            logits = self.forward(data, age=age)
            return torch.sigmoid(logits)
