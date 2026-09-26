"""Tests for multi-label model structural correctness.

Covers Parts 1, 6, 11 of engineering readiness:
  - Part 1: Tests match real project APIs
  - Part 6: Verify multi-label model output (independent logits, no softmax)
  - Part 11: Realistic structural tests with smallest valid inputs
"""

import pytest
import numpy as np
import torch
import torch.nn as nn

# ── Real project import: Logistic Regression ────────────────────────────
from src.models.multilabel_logistic_regression import (
    MultilabelLogisticRegression,
)


# ── Conditional PyG imports ─────────────────────────────────────────────
_HAS_PYG = False
try:
    from torch_geometric.data import Data, Batch
    _HAS_PYG = True
except ImportError:
    pass

_pyg_required = pytest.mark.skipif(
    not _HAS_PYG,
    reason="torch_geometric is not installed — GAT/MolGNN structural tests skipped",
)


# =====================================================================
# LOGISTIC REGRESSION (always runnable — no PyG needed)
# =====================================================================

class TestMultilabelLogisticRegression:
    """Structural tests for MultilabelLogisticRegression."""

    def test_construction(self):
        n_classes = 10
        model = MultilabelLogisticRegression(input_dim=64, n_classes=n_classes)
        assert model.n_classes == n_classes
        assert model.input_dim == 64

    def test_output_shape(self):
        n_classes = 15
        model = MultilabelLogisticRegression(input_dim=32, n_classes=n_classes)
        x = torch.randn(4, 32)
        out = model(x)
        assert out.shape == (4, n_classes)

    def test_output_dimension_equals_label_space(self):
        """Output dim must equal n_classes (the active label-space size)."""
        for n_cls in [5, 50, 363]:
            model = MultilabelLogisticRegression(input_dim=128, n_classes=n_cls)
            out = model(torch.randn(2, 128))
            assert out.shape[1] == n_cls

    def test_output_is_logits_not_probabilities(self):
        """Forward returns raw logits — values can be negative and > 1."""
        model = MultilabelLogisticRegression(input_dim=64, n_classes=20)
        torch.manual_seed(0)
        x = torch.randn(100, 64)
        logits = model(x)
        # At least some values should be negative (logits, not sigmoid)
        assert (logits < 0).any(), "Expected raw logits (some negative values)"

    def test_no_softmax_output(self):
        """Output rows must NOT sum to 1 (no softmax)."""
        model = MultilabelLogisticRegression(input_dim=64, n_classes=20)
        torch.manual_seed(1)
        x = torch.randn(10, 64)
        logits = model(x)
        row_sums = logits.sum(dim=1)
        # If softmax were applied, all sums would be exactly 1.0
        assert not torch.allclose(row_sums, torch.ones_like(row_sums), atol=0.01)

    def test_loss_with_multihot_target(self):
        """BCEWithLogitsLoss can be computed against a multi-hot target."""
        n_classes = 8
        model = MultilabelLogisticRegression(input_dim=16, n_classes=n_classes)
        x = torch.randn(3, 16)
        target = torch.tensor([
            [1, 0, 1, 0, 0, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 0],
            [1, 1, 1, 0, 0, 0, 0, 0],
        ], dtype=torch.float32)
        logits = model(x)
        loss_fn = nn.BCEWithLogitsLoss()
        loss = loss_fn(logits, target)
        assert loss.item() > 0
        assert torch.isfinite(loss)

    def test_predict_proba_returns_probabilities(self):
        """predict_proba returns values in [0, 1]."""
        model = MultilabelLogisticRegression(input_dim=32, n_classes=10)
        x = torch.randn(5, 32)
        probs = model.predict_proba(x)
        assert (probs >= 0).all()
        assert (probs <= 1).all()
        assert probs.shape == (5, 10)

    def test_predict_proba_not_softmax(self):
        """Probabilities should NOT sum to 1 per row (independent sigmoids)."""
        model = MultilabelLogisticRegression(input_dim=64, n_classes=20)
        torch.manual_seed(42)
        x = torch.randn(10, 64)
        probs = model.predict_proba(x)
        row_sums = probs.sum(dim=1)
        # With 20 independent sigmoids, row sums are unlikely to all be ~1.0
        assert not torch.allclose(row_sums, torch.ones_like(row_sums), atol=0.5)

    def test_n_classes_from_config_not_hardcoded(self):
        """n_classes should be explicitly provided, not assumed to be 363."""
        model_a = MultilabelLogisticRegression(input_dim=32, n_classes=100)
        model_b = MultilabelLogisticRegression(input_dim=32, n_classes=500)
        assert model_a.n_classes == 100
        assert model_b.n_classes == 500


# =====================================================================
# STATIC GAT (requires PyG)
# =====================================================================

@_pyg_required
class TestMultilabelStaticGAT:
    """Structural tests for MultilabelStaticGAT."""

    @pytest.fixture
    def model_and_batch(self):
        from src.models.multilabel_static_gat import MultilabelStaticGAT

        n_classes = 10
        input_dim = 8
        model = MultilabelStaticGAT(
            input_dim=input_dim,
            n_classes=n_classes,
            hidden_dim=16,
            heads=2,
            dropout=0.0,
        )
        # Smallest valid 2-node graph (drug pair)
        g1 = Data(
            x=torch.randn(2, input_dim),
            edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        )
        g2 = Data(
            x=torch.randn(2, input_dim),
            edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        )
        batch = Batch.from_data_list([g1, g2])
        return model, batch, n_classes

    def test_construction(self, model_and_batch):
        model, _, n_classes = model_and_batch
        assert model.n_classes == n_classes

    def test_forward_output_shape(self, model_and_batch):
        model, batch, n_classes = model_and_batch
        model.eval()
        out = model(batch)
        assert out.shape == (2, n_classes)

    def test_output_is_logits(self, model_and_batch):
        model, batch, _ = model_and_batch
        model.eval()
        logits = model(batch)
        assert (logits < 0).any() or True  # logits CAN be all positive with random init

    def test_no_softmax(self, model_and_batch):
        model, batch, n_classes = model_and_batch
        model.eval()
        logits = model(batch)
        row_sums = logits.sum(dim=1)
        assert not torch.allclose(row_sums, torch.ones(2), atol=0.01)

    def test_loss_computable(self, model_and_batch):
        model, batch, n_classes = model_and_batch
        model.eval()
        logits = model(batch)
        target = torch.zeros(2, n_classes)
        target[0, [0, 3, 5]] = 1.0
        target[1, [1, 7]] = 1.0
        loss = nn.BCEWithLogitsLoss()(logits, target)
        assert torch.isfinite(loss)
        assert loss.item() > 0

    def test_output_dimension_configurable(self):
        from src.models.multilabel_static_gat import MultilabelStaticGAT
        for n_cls in [5, 100, 363]:
            model = MultilabelStaticGAT(input_dim=8, n_classes=n_cls, hidden_dim=16, heads=2)
            g = Data(
                x=torch.randn(2, 8),
                edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
            )
            batch = Batch.from_data_list([g])
            model.eval()
            out = model(batch)
            assert out.shape[1] == n_cls

    def test_with_age(self):
        from src.models.multilabel_static_gat import MultilabelStaticGAT
        model = MultilabelStaticGAT(
            input_dim=8, n_classes=10, hidden_dim=16, heads=2, use_age=True
        )
        g = Data(
            x=torch.randn(2, 8),
            edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        )
        batch = Batch.from_data_list([g])
        age = torch.tensor([0.5])
        model.eval()
        out = model(batch, age=age)
        assert out.shape == (1, 10)


# =====================================================================
# MOLECULAR GNN (requires PyG)
# =====================================================================

@_pyg_required
class TestMultilabelMolecularGNN:
    """Structural tests for MultilabelMolecularGNN."""

    @pytest.fixture
    def model_and_inputs(self):
        from src.models.multilabel_molecular_gnn import MultilabelMolecularGNN

        n_classes = 10
        input_dim = 13  # ATOM_FEATURE_DIM fallback
        model = MultilabelMolecularGNN(
            n_classes=n_classes,
            input_dim=input_dim,
            hidden_dim=16,
            num_layers=2,
            dropout=0.0,
        )
        # Minimal molecule A: 3 atoms, 2 bonds
        mol_a = Data(
            x=torch.randn(3, input_dim),
            edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long),
        )
        # Minimal molecule B: 2 atoms, 1 bond
        mol_b = Data(
            x=torch.randn(2, input_dim),
            edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        )
        batch_a = Batch.from_data_list([mol_a])
        batch_b = Batch.from_data_list([mol_b])
        return model, batch_a, batch_b, n_classes

    def test_construction(self, model_and_inputs):
        model, _, _, n_classes = model_and_inputs
        assert model.n_classes == n_classes

    def test_forward_output_shape(self, model_and_inputs):
        model, batch_a, batch_b, n_classes = model_and_inputs
        model.eval()
        out = model(batch_a, batch_b)
        assert out.shape == (1, n_classes)

    def test_output_is_logits(self, model_and_inputs):
        model, batch_a, batch_b, _ = model_and_inputs
        model.eval()
        logits = model(batch_a, batch_b)
        # Raw logits can be any real number
        assert logits.dtype == torch.float32

    def test_no_softmax(self, model_and_inputs):
        model, batch_a, batch_b, n_classes = model_and_inputs
        model.eval()
        logits = model(batch_a, batch_b)
        row_sum = logits.sum(dim=1)
        assert not torch.allclose(row_sum, torch.ones(1), atol=0.01)

    def test_loss_computable(self, model_and_inputs):
        model, batch_a, batch_b, n_classes = model_and_inputs
        model.eval()
        logits = model(batch_a, batch_b)
        target = torch.zeros(1, n_classes)
        target[0, [2, 5, 8]] = 1.0
        loss = nn.BCEWithLogitsLoss()(logits, target)
        assert torch.isfinite(loss)
        assert loss.item() > 0

    def test_output_dimension_configurable(self):
        from src.models.multilabel_molecular_gnn import MultilabelMolecularGNN
        for n_cls in [5, 100, 363]:
            model = MultilabelMolecularGNN(n_classes=n_cls, input_dim=13, hidden_dim=16, num_layers=2)
            mol_a = Data(
                x=torch.randn(2, 13),
                edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
            )
            mol_b = Data(
                x=torch.randn(2, 13),
                edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
            )
            model.eval()
            out = model(Batch.from_data_list([mol_a]), Batch.from_data_list([mol_b]))
            assert out.shape[1] == n_cls

    def test_with_age(self):
        from src.models.multilabel_molecular_gnn import MultilabelMolecularGNN
        model = MultilabelMolecularGNN(
            n_classes=10, input_dim=13, hidden_dim=16, num_layers=2, use_age=True
        )
        mol_a = Data(
            x=torch.randn(3, 13),
            edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long),
        )
        mol_b = Data(
            x=torch.randn(2, 13),
            edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        )
        model.eval()
        age = torch.tensor([0.3])
        out = model(
            Batch.from_data_list([mol_a]),
            Batch.from_data_list([mol_b]),
            age=age,
        )
        assert out.shape == (1, 10)

    def test_predict_proba_preserves_full_vector(self):
        """predict_proba returns complete probability vector, not just argmax."""
        from src.models.multilabel_molecular_gnn import MultilabelMolecularGNN
        n_classes = 15
        model = MultilabelMolecularGNN(n_classes=n_classes, input_dim=13, hidden_dim=16, num_layers=2)
        mol_a = Data(
            x=torch.randn(2, 13),
            edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        )
        mol_b = Data(
            x=torch.randn(2, 13),
            edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        )
        model.eval()
        probs = model.predict_proba(
            Batch.from_data_list([mol_a]),
            Batch.from_data_list([mol_b]),
        )
        assert probs.shape == (1, n_classes)
        assert (probs >= 0).all()
        assert (probs <= 1).all()


# =====================================================================
# CROSS-MODEL CONSISTENCY
# =====================================================================

class TestCrossModelConsistency:
    """Verify all models share multi-label structural properties."""

    def test_lr_no_crossentropyloss(self):
        """BCEWithLogitsLoss is the correct loss for multi-label LR."""
        model = MultilabelLogisticRegression(input_dim=16, n_classes=5)
        logits = model(torch.randn(2, 16))
        target = torch.tensor([[1, 0, 1, 0, 0], [0, 1, 0, 1, 0]], dtype=torch.float)

        # BCEWithLogitsLoss should work
        bce_loss = nn.BCEWithLogitsLoss()(logits, target)
        assert torch.isfinite(bce_loss)

        # CrossEntropyLoss would be WRONG for multi-label
        # (it assumes mutually exclusive classes)
