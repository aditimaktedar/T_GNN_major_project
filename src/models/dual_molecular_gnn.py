"""Dual molecular GNN for multiclass drug-drug interaction type prediction."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from torch_geometric.nn import GINConv, global_mean_pool

from src.data.final_ml_dataset import TypeLabelEncoder
from src.data.molecular_pair_dataset import (
    MulticlassMolecularPairDataset,
    collate_molecular_pairs,
    filter_molecular_trainable_rows,
)
from src.evaluation.multiclass import compute_multiclass_metrics
from src.evaluation.reproducibility import get_device, set_global_seed
from src.graph.molecular_graph import ATOM_FEATURE_DIM, MolecularGraphCache


class MolecularGNNEncoder(nn.Module):
    """Shared GIN encoder with global mean pooling."""

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


class DualMolecularGNN(nn.Module):
    """Encode two molecular graphs, fuse embeddings, predict interaction type."""

    def __init__(
        self,
        n_classes: int,
        input_dim: int = ATOM_FEATURE_DIM,
        hidden_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.2,
        fusion_hidden: int = 128,
    ):
        super().__init__()
        self.encoder = MolecularGNNEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        fusion_in = hidden_dim * 4
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
        h_a = self.encoder(mol_a)
        h_b = self.encoder(mol_b)
        return torch.cat([h_a, h_b, torch.abs(h_a - h_b), h_a * h_b], dim=1)

    def forward(self, mol_a: Batch, mol_b: Batch) -> torch.Tensor:
        pair_features = self.encode_pair(mol_a, mol_b)
        return self.classifier(self.fusion(pair_features))


class MulticlassMolecularGNNTrainer:
    def __init__(
        self,
        n_classes: int,
        input_dim: int = ATOM_FEATURE_DIM,
        hidden_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.2,
        fusion_hidden: int = 128,
        learning_rate: float = 1e-3,
        epochs: int = 20,
        seed: int = 42,
        class_weight: str | None = "balanced",
        graph_cache: MolecularGraphCache | None = None,
    ):
        self.n_classes = n_classes
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.fusion_hidden = fusion_hidden
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.seed = seed
        self.class_weight = class_weight
        self.graph_cache = graph_cache or MolecularGraphCache()
        self.device = get_device()
        set_global_seed(seed)
        self.model = DualMolecularGNN(
            n_classes=n_classes,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            fusion_hidden=fusion_hidden,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion: nn.CrossEntropyLoss | None = None
        self.label_encoder: TypeLabelEncoder | None = None

    def _compute_class_weights(self, train_frame: pd.DataFrame) -> torch.Tensor | None:
        if self.class_weight != "balanced":
            return None
        labels = filter_molecular_trainable_rows(train_frame)["label_index"].astype(int).to_numpy()
        counts = np.bincount(labels, minlength=self.n_classes)
        counts = np.maximum(counts, 1)
        weights = len(labels) / (self.n_classes * counts.astype(float))
        return torch.tensor(weights, dtype=torch.float, device=self.device)

    def _batch_loss(self, mol_a: Batch, mol_b: Batch, targets: torch.Tensor) -> torch.Tensor:
        logits = self.model(mol_a, mol_b)
        assert self.criterion is not None
        return self.criterion(logits, targets)

    def _make_loader(self, frame: pd.DataFrame, batch_size: int, shuffle: bool) -> DataLoader:
        dataset = MulticlassMolecularPairDataset(frame, graph_cache=self.graph_cache)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_molecular_pairs,
        )

    def train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total = 0.0
        n = 0
        for mol_a, mol_b, targets in loader:
            mol_a = mol_a.to(self.device)
            mol_b = mol_b.to(self.device)
            targets = targets.to(self.device)
            self.optimizer.zero_grad()
            loss = self._batch_loss(mol_a, mol_b, targets)
            loss.backward()
            self.optimizer.step()
            total += float(loss.item()) * targets.size(0)
            n += targets.size(0)
        return total / max(n, 1)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> tuple[float, dict]:
        self.model.eval()
        losses: list[float] = []
        y_true: list[int] = []
        y_pred: list[int] = []
        y_prob: list[np.ndarray] = []
        for mol_a, mol_b, targets in loader:
            mol_a = mol_a.to(self.device)
            mol_b = mol_b.to(self.device)
            targets = targets.to(self.device)
            logits = self.model(mol_a, mol_b)
            loss = self._batch_loss(mol_a, mol_b, targets)
            losses.append(float(loss.item()))
            prob = torch.softmax(logits, dim=1).cpu().numpy()
            y_prob.append(prob)
            y_pred.extend(prob.argmax(axis=1).tolist())
            y_true.extend(targets.cpu().numpy().astype(int).tolist())
        metrics = compute_multiclass_metrics(
            np.asarray(y_true, dtype=int),
            np.asarray(y_pred, dtype=int),
            np.vstack(y_prob) if y_prob else None,
            labels=list(range(self.n_classes)),
        )
        return float(np.mean(losses)) if losses else 0.0, metrics

    def fit(
        self,
        train_frame: pd.DataFrame,
        val_frame: pd.DataFrame,
        label_encoder: TypeLabelEncoder,
        batch_size: int = 32,
        checkpoint_path: Path | None = None,
    ) -> dict:
        self.label_encoder = label_encoder
        class_weights = self._compute_class_weights(train_frame)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)

        train_loader = self._make_loader(train_frame, batch_size=batch_size, shuffle=True)
        val_loader = self._make_loader(val_frame, batch_size=batch_size, shuffle=False)
        history = {"train_loss": [], "val_loss": [], "val_metrics": []}
        best_macro_f1 = -1.0
        best_state = None
        for _epoch in range(self.epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss, val_metrics = self.evaluate(val_loader)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_metrics"].append(val_metrics)
            macro_f1 = val_metrics.get("macro_f1") or 0.0
            if macro_f1 >= best_macro_f1:
                best_macro_f1 = macro_f1
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
        if best_state is not None:
            self.model.load_state_dict(best_state)
        if checkpoint_path:
            self.save(checkpoint_path)
        history["best_val_macro_f1"] = best_macro_f1
        return history

    @torch.no_grad()
    def predict_proba(self, frame: pd.DataFrame, batch_size: int = 32) -> np.ndarray:
        loader = self._make_loader(frame, batch_size=batch_size, shuffle=False)
        self.model.eval()
        probs = []
        for mol_a, mol_b, _targets in loader:
            mol_a = mol_a.to(self.device)
            mol_b = mol_b.to(self.device)
            logits = self.model(mol_a, mol_b)
            probs.append(torch.softmax(logits, dim=1).cpu().numpy())
        return np.vstack(probs) if probs else np.empty((0, self.n_classes))

    def predict(self, frame: pd.DataFrame, batch_size: int = 32) -> np.ndarray:
        return self.predict_proba(frame, batch_size=batch_size).argmax(axis=1)

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "config": {
                    "n_classes": self.n_classes,
                    "input_dim": self.input_dim,
                    "hidden_dim": self.hidden_dim,
                    "num_layers": self.num_layers,
                    "dropout": self.dropout,
                    "fusion_hidden": self.fusion_hidden,
                    "learning_rate": self.learning_rate,
                    "epochs": self.epochs,
                    "seed": self.seed,
                    "class_weight": self.class_weight,
                },
                "label_encoder": self.label_encoder,
            },
            path,
        )
        return path

    @classmethod
    def load(cls, path: Path, graph_cache: MolecularGraphCache | None = None) -> "MulticlassMolecularGNNTrainer":
        payload = torch.load(path, map_location=get_device(), weights_only=False)
        trainer = cls(**payload["config"], graph_cache=graph_cache)
        trainer.model.load_state_dict(payload["state_dict"])
        trainer.label_encoder = payload["label_encoder"]
        return trainer
