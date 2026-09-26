"""Static Graph Attention Network baseline (not temporal)."""

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

from src.data.pyg_dataset import DrugPairGraphDataset, filter_trainable_rows
from src.evaluation.classification import compute_classification_metrics
from src.evaluation.reproducibility import get_device, set_global_seed


class StaticGAT(nn.Module):
    """Static GAT on a 2-node drug-pair graph with molecular node features."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        heads: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.conv1 = GATConv(input_dim, hidden_dim, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden_dim * heads, hidden_dim, heads=1, concat=False, dropout=dropout)
        self.dropout = dropout
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, data: Batch) -> torch.Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)
        return self.classifier(x).view(-1)


class StaticGATTrainer:
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        heads: int = 4,
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
        epochs: int = 20,
        seed: int = 42,
        class_weight: str | None = "balanced",
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.seed = seed
        self.class_weight = class_weight
        self.device = get_device()
        set_global_seed(seed)
        self.model = StaticGAT(input_dim, hidden_dim, heads, dropout).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.pos_weight: torch.Tensor | None = None

    def _compute_pos_weight(self, train_frame) -> None:
        train = filter_trainable_rows(train_frame)
        labels = train["label"].map({"positive": 1, "negative": 0}).astype(int)
        pos = max(int((labels == 1).sum()), 1)
        neg = max(int((labels == 0).sum()), 1)
        if self.class_weight == "balanced":
            self.pos_weight = torch.tensor([neg / pos], dtype=torch.float, device=self.device)
        else:
            self.pos_weight = None

    def _batch_loss(self, batch: Batch) -> torch.Tensor:
        logits = self.model(batch)
        targets = batch.y.view(-1)
        if self.pos_weight is not None:
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        else:
            loss_fn = nn.BCEWithLogitsLoss()
        return loss_fn(logits, targets)

    def train_epoch(self, loader) -> float:
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
    def evaluate(self, loader) -> tuple[float, dict]:
        self.model.eval()
        losses = []
        y_true = []
        y_prob = []
        for batch in loader:
            batch = batch.to(self.device)
            logits = self.model(batch)
            loss = self._batch_loss(batch)
            losses.append(float(loss.item()))
            prob = torch.sigmoid(logits).cpu().numpy()
            y_prob.extend(prob.tolist())
            y_true.extend(batch.y.view(-1).cpu().numpy().astype(int).tolist())
        y_pred = (np.asarray(y_prob) >= 0.5).astype(int)
        metrics = compute_classification_metrics(np.asarray(y_true), y_pred, np.asarray(y_prob))
        return float(np.mean(losses)) if losses else 0.0, metrics

    def fit(
        self,
        train_frame,
        val_frame,
        batch_size: int = 32,
        checkpoint_path: Path | None = None,
    ) -> dict:
        self._compute_pos_weight(train_frame)
        train_dataset = DrugPairGraphDataset(train_frame)
        val_dataset = DrugPairGraphDataset(val_frame)
        train_loader = PyGDataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        val_loader = PyGDataLoader(
            val_dataset, batch_size=batch_size, shuffle=False
        )
        history = {"train_loss": [], "val_loss": [], "val_metrics": []}
        best_f1 = -1.0
        best_state = None
        for _epoch in range(self.epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss, val_metrics = self.evaluate(val_loader)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_metrics"].append(val_metrics)
            f1 = val_metrics.get("f1", 0.0) or 0.0
            if f1 >= best_f1:
                best_f1 = f1
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
        if best_state is not None:
            self.model.load_state_dict(best_state)
        if checkpoint_path:
            self.save(checkpoint_path)
        history["best_val_f1"] = best_f1
        return history

    def predict_proba(self, frame, batch_size: int = 32) -> np.ndarray:
        dataset = DrugPairGraphDataset(frame)
        loader = PyGDataLoader(dataset, batch_size=batch_size, shuffle=False)
        self.model.eval()
        probs = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                logits = self.model(batch)
                probs.extend(torch.sigmoid(logits).cpu().numpy().tolist())
        return np.asarray(probs, dtype=float)

    def predict(self, frame, batch_size: int = 32) -> np.ndarray:
        prob = self.predict_proba(frame, batch_size=batch_size)
        return (prob >= 0.5).astype(int)

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "config": {
                    "input_dim": self.input_dim,
                    "hidden_dim": self.hidden_dim,
                    "heads": self.heads,
                    "dropout": self.dropout,
                    "learning_rate": self.learning_rate,
                    "epochs": self.epochs,
                    "seed": self.seed,
                    "class_weight": self.class_weight,
                },
            },
            path,
        )
        return path

    @classmethod
    def load(cls, path: Path) -> "StaticGATTrainer":
        payload = torch.load(path, map_location=get_device(), weights_only=False)
        config = payload["config"]
        trainer = cls(**config)
        trainer.model.load_state_dict(payload["state_dict"])
        return trainer


# End of module