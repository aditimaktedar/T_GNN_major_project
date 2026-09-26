"""Checkpoint manager for multi-label DDI model artifacts.

Records model metadata, version, feature set, label mapping version,
split ID, seed, hyperparameters, and best validation metrics.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


class CheckpointManager:
    """Manage model checkpoints with rich metadata for reproducibility.

    Parameters
    ----------
    checkpoint_dir : Path or str
        Directory where checkpoints are saved.
    """

    def __init__(self, checkpoint_dir: Path | str):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        epoch: int = 0,
        model_name: str = "",
        feature_set: str = "drug_pair",
        label_mapping_version: str = "FREQUENT363_v1",
        n_classes: int = 363,
        split_seed: int = 42,
        hyperparameters: dict[str, Any] | None = None,
        val_metrics: dict[str, Any] | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> Path:
        """Save model checkpoint with metadata.

        Parameters
        ----------
        model : torch.nn.Module
            Trained model.
        optimizer : torch.optim.Optimizer, optional
            Optimizer state.
        epoch : int
            Current epoch number.
        model_name : str
            Human-readable model identifier.
        feature_set : str
            Feature configuration ('drug_pair' or 'drug_pair_age').
        label_mapping_version : str
            Label mapping version string.
        n_classes : int
            Number of output labels.
        split_seed : int
            Random seed used for data splitting.
        hyperparameters : dict, optional
            Model hyperparameters to record.
        val_metrics : dict, optional
            Best validation metrics at checkpoint time.
        extra_metadata : dict, optional
            Any additional metadata to store.

        Returns
        -------
        Path
            Path to saved checkpoint file.
        """
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"{model_name}_{feature_set}_epoch{epoch}_{timestamp}.pt"
        filepath = self.checkpoint_dir / filename

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "metadata": {
                "model_name": model_name,
                "feature_set": feature_set,
                "label_mapping_version": label_mapping_version,
                "n_classes": n_classes,
                "split_seed": split_seed,
                "hyperparameters": hyperparameters or {},
                "val_metrics": val_metrics or {},
                "timestamp_utc": timestamp,
                **(extra_metadata or {}),
            },
        }

        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()

        torch.save(checkpoint, filepath)

        # Save companion metadata JSON for easy inspection
        meta_path = filepath.with_suffix(".json")
        meta_path.write_text(
            json.dumps(checkpoint["metadata"], indent=2, default=str),
            encoding="utf-8",
        )

        return filepath

    def load_checkpoint(
        self,
        filepath: Path | str,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        map_location: str = "cpu",
    ) -> dict[str, Any]:
        """Load model checkpoint from file.

        Parameters
        ----------
        filepath : Path or str
            Path to checkpoint file.
        model : torch.nn.Module
            Model to load state into.
        optimizer : torch.optim.Optimizer, optional
            Optimizer to load state into.
        map_location : str
            Device to map tensors to.

        Returns
        -------
        dict
            Checkpoint metadata.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Checkpoint not found: {filepath}")

        checkpoint = torch.load(filepath, map_location=map_location, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])

        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        return checkpoint.get("metadata", {})

    def list_checkpoints(self) -> list[dict[str, Any]]:
        """List all checkpoints with their metadata.

        Returns
        -------
        list[dict]
            Sorted list of checkpoint info (newest first).
        """
        entries = []
        for pt_file in sorted(self.checkpoint_dir.glob("*.pt"), reverse=True):
            meta_file = pt_file.with_suffix(".json")
            meta = {}
            if meta_file.exists():
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
            entries.append({
                "path": str(pt_file),
                "filename": pt_file.name,
                "size_bytes": pt_file.stat().st_size,
                **meta,
            })
        return entries
