import torch
import torch.nn as nn


class SafeGraphConv(nn.Module):

    def __init__(self, in_dim, out_dim, use_norm=False):
        super().__init__()

        self.lin = nn.Linear(in_dim, out_dim)
        self.use_norm = use_norm
        if use_norm:
            self.norm = nn.LayerNorm(out_dim)
        else:
            self.norm = None

    def forward(self, x, edge_index=None):
        if edge_index is None or edge_index.numel() == 0:
            agg = torch.zeros_like(x)
        else:
            src, dst = edge_index

            agg = torch.zeros_like(x)
            agg.index_add_(
                0,
                dst,
                x[src]
            )

            degree = torch.zeros(
                x.size(0),
                device=x.device,
                dtype=x.dtype
            )
            degree.index_add_(
                0,
                dst,
                torch.ones_like(
                    dst,
                    dtype=x.dtype
                )
            )
            agg = agg / degree.clamp_min(1).unsqueeze(1)

        out = self.lin(agg)
        if self.norm is not None:
            out = self.norm(out)

        return torch.relu(out)


class TemporalDDIGNN(nn.Module):

    def __init__(
        self,
        num_drugs=38,
        embedding_dim=32,
        num_severity_classes=3,
        use_norm=False
    ):
        super().__init__()

        self.emb = nn.Embedding(
            num_drugs,
            embedding_dim
        )
        self.embedding = self.emb

        self.conv1 = SafeGraphConv(
            embedding_dim,
            embedding_dim,
            use_norm=use_norm
        )

        self.conv2 = SafeGraphConv(
            embedding_dim,
            embedding_dim,
            use_norm=use_norm
        )

        pair_dim = embedding_dim * 5

        self.shared = nn.Sequential(
            nn.Linear(pair_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.10)
        )

        self.presence_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        self.severity_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_severity_classes)
        )

    def encode_graph(self, edge_index=None):
        x = self.emb.weight

        x1 = self.conv1(
            x,
            edge_index
        )

        x2 = self.conv2(
            x1,
            edge_index
        )

        return x + x1 + x2

    def pair_representation(
        self,
        node_repr,
        drug_a,
        drug_b
    ):
        h_a = node_repr[drug_a]
        h_b = node_repr[drug_b]

        return torch.cat(
            [
                h_a,
                h_b,
                h_a * h_b,
                torch.abs(h_a - h_b),
                h_a + h_b
            ],
            dim=-1
        )

    def forward(
        self,
        drug_a,
        drug_b,
        edge_index=None
    ):
        node_repr = self.encode_graph(
            edge_index
        )

        pair_repr = self.pair_representation(
            node_repr,
            drug_a,
            drug_b
        )

        shared = self.shared(
            pair_repr
        )

        return {
            "presence_logits":
                self.presence_head(shared),

            "severity_logits":
                self.severity_head(shared),

            "node_repr":
                node_repr
        }