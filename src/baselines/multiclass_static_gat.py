"""Multiclass Static GAT baseline on 2-node fingerprint graphs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.nn import GATConv, global_mean_pool

from src.data.final_ml_dataset import (
    MulticlassDrugPairGraphDataset,
    TypeLabelEncoder,
    filter_trainable_rows,
)
from src.evaluation.multiclass import compute_multiclass_metrics
from src.evaluation.reproducibility import get_device, set_global_seed


class MulticlassStaticGAT(nn.Module):
    """Static GAT on a 2-node drug-pair graph with molecular fingerprint node features."""

    def __init__(
        self,
        input_dim: int,
        n_classes: int,
        hidden_dim: int = 64,
        heads: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.conv1 = GATConv(input_dim, hidden_dim, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden_dim * heads, hidden_dim, heads=1, concat=False, dropout=dropout)
        self.dropout = dropout
        self.classifier = nn.Linear(hidden_dim, n_classes)

    def forward(self, data: Batch) -> torch.Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)
        return self.classifier(x)


class MulticlassStaticGATTrainer:
    def __init__(
        self,
        input_dim: int,
        n_classes: int,
        hidden_dim: int = 64,
        heads: int = 4,
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
        epochs: int = 20,
        seed: int = 42,
        class_weight: str | None = "balanced",
    ):
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.seed = seed
        self.class_weight = class_weight
        self.device = get_device()
        set_global_seed(seed)
        self.model = MulticlassStaticGAT(
            input_dim=input_dim,
            n_classes=n_classes,
            hidden_dim=hidden_dim,
            heads=heads,
            dropout=dropout,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion: nn.CrossEntropyLoss | None = None
        self.label_encoder: TypeLabelEncoder | None = None

    def _compute_class_weights(self, train_frame: pd.DataFrame) -> torch.Tensor | None:
        if self.class_weight != "balanced":
            return None
        labels = filter_trainable_rows(train_frame)["label_index"].astype(int).to_numpy()
        counts = np.bincount(labels, minlength=self.n_classes)
        counts = np.maximum(counts, 1)
        weights = len(labels) / (self.n_classes * counts.astype(float))
        return torch.tensor(weights, dtype=torch.float, device=self.device)

    def _batch_loss(self, batch: Batch) -> torch.Tensor:
        logits = self.model(batch)
        targets = batch.y.view(-1)
        assert self.criterion is not None
        return self.criterion(logits, targets)

    def train_epoch(self, loader: PyGDataLoader) -> float:
        self.model.train()
        total = 0.0
        n = 0
        for batch in loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()
            loss = self._batch_loss(batch)
            loss.backward()
            self.optimizer.step()
            total += float(loss.item()) * batch.num_graphs
            n += batch.num_graphs
        return total / max(n, 1)

    @torch.no_grad()
    def evaluate(self, loader: PyGDataLoader) -> tuple[float, dict]:
        self.model.eval()
        losses = []
        y_true = []
        y_pred = []
        y_prob = []
        for batch in loader:
            batch = batch.to(self.device)
            logits = self.model(batch)
            loss = self._batch_loss(batch)
            losses.append(float(loss.item()))
            prob = torch.softmax(logits, dim=1).cpu().numpy()
            pred = prob.argmax(axis=1)
            y_prob.append(prob)
            y_pred.extend(pred.tolist())
            y_true.extend(batch.y.view(-1).cpu().numpy().astype(int).tolist())
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

        train_loader = PyGDataLoader(
            MulticlassDrugPairGraphDataset(train_frame),
            batch_size=batch_size,
            shuffle=True,
        )
        val_loader = PyGDataLoader(
            MulticlassDrugPairGraphDataset(val_frame),
            batch_size=batch_size,
            shuffle=False,
        )
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
    def predict(self, frame: pd.DataFrame, batch_size: int = 32) -> np.ndarray:
        loader = PyGDataLoader(MulticlassDrugPairGraphDataset(frame), batch_size=batch_size, shuffle=False)
        self.model.eval()
        preds = []
        for batch in loader:
            batch = batch.to(self.device)
            logits = self.model(batch)
            preds.extend(logits.argmax(dim=1).cpu().numpy().tolist())
        return np.asarray(preds, dtype=int)

    @torch.no_grad()
    def predict_proba(self, frame: pd.DataFrame, batch_size: int = 32) -> np.ndarray:
        loader = PyGDataLoader(MulticlassDrugPairGraphDataset(frame), batch_size=batch_size, shuffle=False)
        self.model.eval()
        probs = []
        for batch in loader:
            batch = batch.to(self.device)
            logits = self.model(batch)
            probs.append(torch.softmax(logits, dim=1).cpu().numpy())
        return np.vstack(probs) if probs else np.empty((0, self.n_classes))

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "config": {
                    "input_dim": self.input_dim,
                    "n_classes": self.n_classes,
                    "hidden_dim": self.hidden_dim,
                    "heads": self.heads,
                    "dropout": self.dropout,
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
    def load(cls, path: Path) -> "MulticlassStaticGATTrainer":
        payload = torch.load(path, map_location=get_device(), weights_only=False)
        config = payload["config"]
        trainer = cls(**config)
        trainer.model.load_state_dict(payload["state_dict"])
        trainer.label_encoder = payload["label_encoder"]
        return trainer
