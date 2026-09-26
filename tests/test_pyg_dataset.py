"""Tests for PyG dataset construction."""

from __future__ import annotations

import pandas as pd
import pytest
import torch
from torch_geometric.data import Batch

from src.baselines.run_baselines import build_demo_ml_dataset
from src.data.pyg_dataset import (
    assert_no_patient_leakage,
    create_dataloaders,
    filter_trainable_rows,
    row_to_pyg_data,
    validate_ml_dataset_frame,
)


def test_demo_dataset_validates_and_creates_pyg_data() -> None:
    frame = build_demo_ml_dataset()
    validate_ml_dataset_frame(frame)
    assert_no_patient_leakage(frame)
    data = row_to_pyg_data(frame.iloc[0])
    assert data.x.shape[0] == 2
    assert data.edge_index.shape == (2, 2)
    assert data.y.item() in (0.0, 1.0)


def test_dataloader() -> None:
    frame = build_demo_ml_dataset()
    loaders = create_dataloaders(frame, batch_size=2)
    batch = next(iter(loaders["train"]))
    assert isinstance(batch, Batch)


def test_filter_trainable_rows_excludes_unknown() -> None:
    frame = build_demo_ml_dataset()
    frame = pd.concat(
        [frame, pd.DataFrame([{**frame.iloc[0].to_dict(), "label": "unknown"}])],
        ignore_index=True,
    )
    filtered = filter_trainable_rows(frame)
    assert "unknown" not in set(filtered["label"])


def test_patient_leakage_detection() -> None:
    frame = build_demo_ml_dataset()
    leaked = frame.copy()
    leaked.loc[0, "split"] = "test"
    with pytest.raises(Exception):
        assert_no_patient_leakage(leaked)
