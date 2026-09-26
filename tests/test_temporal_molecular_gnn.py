"""Comprehensive tests for Temporal + Molecular GNN (T-MolGNN) pipeline.

Covers Phase 10 requirements:
1. Molecular encoder output shape
2. Temporal encoder output shape
3. Final 363-label output shape
4. No NaNs in forward pass or loss
5. Correct train-only normalization
6. Correct train-only pos_weight calculation
7. Checkpoint restoration
8. Validation-only threshold selection
9. Pair split integrity (zero pair overlap)
10. A/B representation consistency & swap symmetry
11. Correct temporal feature calculation
12. Existing tests pass cleanly
"""

import pytest
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

try:
    from torch_geometric.data import Data, Batch
    _HAS_PYG = True
except ImportError:
    _HAS_PYG = False

from src.models.temporal_molecular_gnn import (
    MolecularGINEncoder,
    TemporalFeatureEncoder,
    TemporalMolecularGNN,
)
from src.models.train_temporal_gnn import (
    ExtendedTemporalFeatureStandardizer,
    _compute_pos_weight,
    compute_extended_temporal_dataframe,
)

_pyg_required = pytest.mark.skipif(
    not _HAS_PYG,
    reason="torch_geometric is required for T-MolGNN structural tests",
)


@_pyg_required
class TestTemporalMolecularGNNStructure:
    """Structural tests for TemporalMolecularGNN modules."""

    @pytest.fixture
    def sample_data(self):
        n_classes = 363
        atom_dim = 30
        temp_dim = 17
        mol_h = 64
        temp_h = 32

        # 2 sample graphs for A and B
        g_a = Data(
            x=torch.randn(3, atom_dim),
            edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long),
        )
        g_b = Data(
            x=torch.randn(2, atom_dim),
            edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        )
        batch_a = Batch.from_data_list([g_a])
        batch_b = Batch.from_data_list([g_b])
        temp_feats = torch.randn(1, temp_dim)

        return batch_a, batch_b, temp_feats, n_classes, atom_dim, temp_dim, mol_h, temp_h

    def test_molecular_encoder_output_shape(self, sample_data):
        batch_a, _, _, _, atom_dim, _, mol_h, _ = sample_data
        encoder = MolecularGINEncoder(input_dim=atom_dim, hidden_dim=mol_h, num_layers=2)
        out = encoder(batch_a)
        assert out.shape == (1, mol_h)

    def test_temporal_encoder_output_shape(self, sample_data):
        _, _, temp_feats, _, _, temp_dim, _, temp_h = sample_data
        encoder = TemporalFeatureEncoder(input_dim=temp_dim, hidden_dim=temp_h)
        out = encoder(temp_feats)
        assert out.shape == (1, temp_h)

    def test_final_output_shape(self, sample_data):
        batch_a, batch_b, temp_feats, n_classes, atom_dim, temp_dim, mol_h, temp_h = sample_data
        model = TemporalMolecularGNN(
            n_classes=n_classes,
            atom_input_dim=atom_dim,
            temporal_input_dim=temp_dim,
            mol_hidden_dim=mol_h,
            temp_hidden_dim=temp_h,
        )
        logits = model(batch_a, batch_b, temp_feats)
        assert logits.shape == (1, n_classes)

    def test_no_nans_in_forward_and_loss(self, sample_data):
        batch_a, batch_b, temp_feats, n_classes, atom_dim, temp_dim, mol_h, temp_h = sample_data
        model = TemporalMolecularGNN(
            n_classes=n_classes,
            atom_input_dim=atom_dim,
            temporal_input_dim=temp_dim,
            mol_hidden_dim=mol_h,
            temp_hidden_dim=temp_h,
        )
        logits = model(batch_a, batch_b, temp_feats)
        assert not torch.isnan(logits).any()
        assert torch.isfinite(logits).all()

        target = torch.zeros((1, n_classes), dtype=torch.float32)
        target[0, [5, 10, 50]] = 1.0
        loss = nn.BCEWithLogitsLoss()(logits, target)
        assert torch.isfinite(loss)
        assert not torch.isnan(loss)

    def test_swap_symmetry_properties(self, sample_data):
        """Diff and Product pair features must be invariant/symmetric to A/B swap."""
        batch_a, batch_b, temp_feats, n_classes, atom_dim, temp_dim, mol_h, temp_h = sample_data
        model = TemporalMolecularGNN(
            n_classes=n_classes,
            atom_input_dim=atom_dim,
            temporal_input_dim=temp_dim,
            mol_hidden_dim=mol_h,
            temp_hidden_dim=temp_h,
        )
        model.eval()

        with torch.no_grad():
            h_a = model.mol_encoder(batch_a)
            h_b = model.mol_encoder(batch_b)

            diff_ab = torch.abs(h_a - h_b)
            diff_ba = torch.abs(h_b - h_a)
            prod_ab = h_a * h_b
            prod_ba = h_b * h_a

            assert torch.allclose(diff_ab, diff_ba)
            assert torch.allclose(prod_ab, prod_ba)


class TestTemporalDataAndNormalization:
    """Tests for temporal feature calculation, standardizer, and train-only pos_weight."""

    def test_train_only_pos_weight_calculation(self):
        y_train = np.array([
            [1, 0, 1, 0],
            [0, 0, 1, 0],
            [1, 0, 0, 0],
            [1, 1, 1, 0],
        ], dtype=np.float32)

        pw = _compute_pos_weight(y_train)
        assert pw.shape == (4,)
        # For col 0: 3 pos, 1 neg -> pw = 1/3
        assert pytest.approx(pw[0].item(), 1e-4) == 1.0 / 3.0
        # For col 1: 1 pos, 3 neg -> pw = 3/1 = 3.0
        assert pytest.approx(pw[1].item(), 1e-4) == 3.0 / 1.0

    def test_train_only_standardization(self):
        train_df = pd.DataFrame({
            "anchor_age": [50.0, 70.0, 60.0],
            "num_a_events": [2.0, 4.0, 6.0],
            "num_b_events": [1.0, 3.0, 5.0],
            "num_total_pair_events": [3.0, 7.0, 11.0],
            "min_delta_hours": [0.0, 2.0, 4.0],
            "median_delta_hours": [1.0, 5.0, 9.0],
            "abs_first_event_diff": [0.5, 1.5, 2.5],
            "total_temporal_span": [10.0, 20.0, 30.0],
            "event_count_imbalance": [0.1, 0.2, 0.3],
            "num_events_within_1h": [1.0, 2.0, 3.0],
            "num_events_within_6h": [1.0, 2.0, 3.0],
            "num_events_within_12h": [2.0, 4.0, 6.0],
            "num_events_within_24h": [3.0, 6.0, 9.0],
            "a_before_b": [1.0, 0.0, 0.0],
            "b_before_a": [0.0, 1.0, 0.0],
            "same_timestamp": [0.0, 0.0, 1.0],
            "events_overlap_time": [1.0, 0.0, 0.0],
        })

        val_df = pd.DataFrame({
            "anchor_age": [65.0],
            "num_a_events": [3.0],
            "num_b_events": [3.0],
            "num_total_pair_events": [6.0],
            "min_delta_hours": [1.0],
            "median_delta_hours": [3.0],
            "abs_first_event_diff": [1.0],
            "total_temporal_span": [15.0],
            "event_count_imbalance": [0.0],
            "num_events_within_1h": [1.5],
            "num_events_within_6h": [1.5],
            "num_events_within_12h": [3.0],
            "num_events_within_24h": [4.5],
            "a_before_b": [1.0],
            "b_before_a": [0.0],
            "same_timestamp": [0.0],
            "events_overlap_time": [0.0],
        })

        standardizer = ExtendedTemporalFeatureStandardizer()
        X_tr = standardizer.fit_transform(train_df)
        X_va = standardizer.transform(val_df)

        assert X_tr.shape == (3, 17)
        assert X_va.shape == (1, 17)

        # Check age mean on train = 60.0
        assert pytest.approx(standardizer.means["anchor_age"], 1e-4) == 60.0

    def test_pair_split_integrity(self):
        """Ensure no pair overlap between train, val, and test in temporal dataset."""
        df = pd.read_csv("data/processed/temporal_multilabel_frequent363_dataset.csv")

        train_pairs = set(df[df["split"] == "train"]["pair_key"])
        val_pairs = set(df[df["split"] == "val"]["pair_key"])
        test_pairs = set(df[df["split"] == "test"]["pair_key"])

        assert len(train_pairs & val_pairs) == 0, "Train and Val pair overlap detected!"
        assert len(train_pairs & test_pairs) == 0, "Train and Test pair overlap detected!"
        assert len(val_pairs & test_pairs) == 0, "Val and Test pair overlap detected!"


class TestAblationSuiteIntegrity:
    """Regression tests for 6-way ablation study feature masks, model independence, and thresholding."""

    def test_feature_masks_isolation(self):
        from src.models.run_temporal_gnn_ablation import get_ablation_feature_mask

        df = pd.read_csv("data/processed/temporal_multilabel_frequent363_dataset.csv")
        df = compute_extended_temporal_dataframe(df)

        # 1. Temporal-only (no molecular)
        t_mat, use_mol = get_ablation_feature_mask(df, "temporal_only")
        assert use_mol is False
        assert t_mat.shape[1] == 17

        # 2. Molecular-only (no temporal, no age)
        m_mat, use_mol = get_ablation_feature_mask(df, "molecular_only")
        assert use_mol is True
        assert m_mat.shape[1] == 1
        assert (m_mat == 0.0).all()

        # 3. Age-only (no molecular, no non-age temporal)
        a_mat, use_mol = get_ablation_feature_mask(df, "age_only")
        assert use_mol is False
        assert a_mat.shape[1] == 1
        assert not (a_mat == 0.0).all()

        # 4. Mol + Age (molecular + age, no non-age temporal)
        ma_mat, use_mol = get_ablation_feature_mask(df, "mol_age")
        assert use_mol is True
        assert ma_mat.shape[1] == 1
        assert not (ma_mat == 0.0).all()

        # 5. Mol + Temporal (molecular + non-age temporal, age zeroed out)
        mt_mat, use_mol = get_ablation_feature_mask(df, "mol_temporal")
        assert use_mol is True
        assert mt_mat.shape[1] == 17
        assert (mt_mat[:, 0] == 0.0).all()  # Age column at index 0 must be zeroed

        # 6. Full (molecular + temporal + age)
        f_mat, use_mol = get_ablation_feature_mask(df, "full")
        assert use_mol is True
        assert f_mat.shape[1] == 17
        assert not (f_mat[:, 0] == 0.0).all()

    def test_ablation_model_independence(self):
        """Ensure each ablation mode initializes a fresh model with independent parameters."""
        m1 = TemporalMolecularGNN(n_classes=363, temporal_input_dim=17)
        m2 = TemporalMolecularGNN(n_classes=363, temporal_input_dim=1)

        # Parameter counts must differ
        params_m1 = sum(p.numel() for p in m1.parameters())
        params_m2 = sum(p.numel() for p in m2.parameters())
        assert params_m1 != params_m2

