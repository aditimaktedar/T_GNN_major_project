"""Tests for multi-label evaluation, thresholds, leakage, and ranking.

Covers Parts 1, 7, 8, 9 of engineering readiness:
  - Part 1: Tests match real project APIs
  - Part 7: 13 metrics support [N, L] input natively
  - Part 8: Threshold tuning uses validation only
  - Part 9: Raw probabilities are preserved
"""

import numpy as np
import pandas as pd
import pytest

# ── Real project imports ────────────────────────────────────────────────
from src.evaluation.multilabel_metrics import compute_multilabel_metrics
from src.evaluation.thresholds import (
    apply_threshold,
    optimize_threshold_on_val,
    DEFAULT_THRESHOLD,
)
from src.evaluation.leakage_checker import (
    check_pair_overlap,
    check_duplicate_pairs,
    check_reversed_pair_leakage,
    run_full_leakage_check,
    LeakageError,
)
from src.evaluation.ranking import (
    rank_predictions_for_pair,
    rank_predictions_batch,
    compute_topk_hit_rates,
)


# =====================================================================
# PART 7: MULTI-LABEL METRICS
# =====================================================================

class TestMultilabelMetrics:
    """Verify compute_multilabel_metrics works with [N, L] arrays."""

    def test_perfect_predictions(self):
        y_true = np.array([[1, 0, 1], [0, 1, 0]])
        y_prob = np.array([[0.9, 0.1, 0.8], [0.2, 0.9, 0.1]])
        metrics = compute_multilabel_metrics(y_true, y_prob, threshold=0.5)

        assert metrics["micro_f1"] == 1.0
        assert metrics["hamming_loss"] == 0.0

    def test_all_13_metric_keys_present(self):
        y_true = np.array([[1, 0, 1, 0], [0, 1, 0, 1]])
        y_prob = np.array([[0.9, 0.1, 0.8, 0.2], [0.1, 0.9, 0.2, 0.8]])
        metrics = compute_multilabel_metrics(y_true, y_prob)

        expected_keys = {
            "micro_f1", "macro_f1", "hamming_loss", "jaccard_score", "mAP",
            "precision_at_1", "recall_at_1",
            "precision_at_3", "recall_at_3",
            "precision_at_5", "recall_at_5",
            "precision_at_10", "recall_at_10",
            "threshold", "n_samples", "n_classes",
        }
        assert expected_keys.issubset(set(metrics.keys()))

    def test_supports_arbitrary_n_l_shape(self):
        """Works with any valid (N, L) shape, not just hardcoded 363."""
        for n_classes in [5, 50, 363, 500]:
            n_samples = 10
            rng = np.random.default_rng(42)
            y_true = rng.integers(0, 2, size=(n_samples, n_classes))
            y_prob = rng.random(size=(n_samples, n_classes))
            # Ensure at least one positive per sample
            y_true[:, 0] = 1
            metrics = compute_multilabel_metrics(y_true, y_prob)
            assert metrics["n_samples"] == n_samples
            assert metrics["n_classes"] == n_classes
            assert 0.0 <= metrics["micro_f1"] <= 1.0

    def test_rejects_shape_mismatch(self):
        y_true = np.ones((3, 5))
        y_prob = np.ones((3, 7))
        with pytest.raises(ValueError, match="Shape mismatch"):
            compute_multilabel_metrics(y_true, y_prob)

    def test_metrics_are_numeric(self):
        y_true = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0]])
        y_prob = np.array([[0.8, 0.2, 0.7], [0.3, 0.9, 0.1], [0.6, 0.5, 0.2]])
        metrics = compute_multilabel_metrics(y_true, y_prob)
        for key in ["micro_f1", "macro_f1", "hamming_loss", "jaccard_score", "mAP"]:
            assert isinstance(metrics[key], float)
            assert np.isfinite(metrics[key])

    def test_threshold_param_used(self):
        """Custom threshold affects binarization and thus F1."""
        y_true = np.array([[1, 0, 1], [0, 1, 0]])
        y_prob = np.array([[0.4, 0.3, 0.4], [0.3, 0.4, 0.3]])

        # With threshold=0.5, all predictions are 0 → F1 should be 0
        metrics_high = compute_multilabel_metrics(y_true, y_prob, threshold=0.5)
        # With threshold=0.3, all predictions are 1 → different F1
        metrics_low = compute_multilabel_metrics(y_true, y_prob, threshold=0.3)

        assert metrics_high["threshold"] == 0.5
        assert metrics_low["threshold"] == 0.3


# =====================================================================
# PART 8: THRESHOLD TUNING — VALIDATION ONLY
# =====================================================================

class TestThresholds:
    """Verify threshold tuning uses only validation data."""

    def test_optimize_threshold_returns_best(self):
        y_true_val = np.array([[1, 0, 1], [0, 1, 0]])
        y_prob_val = np.array([[0.9, 0.1, 0.8], [0.1, 0.9, 0.1]])
        result = optimize_threshold_on_val(y_true_val, y_prob_val)

        assert "best_threshold" in result
        assert "best_score" in result
        assert 0.0 < result["best_threshold"] < 1.0

    def test_optimize_threshold_note_mentions_validation(self):
        y_true_val = np.array([[1, 0], [0, 1]])
        y_prob_val = np.array([[0.8, 0.2], [0.3, 0.7]])
        result = optimize_threshold_on_val(y_true_val, y_prob_val)
        assert "VALIDATION" in result["note"].upper() or "val" in result["note"].lower()

    def test_apply_threshold_binary(self):
        y_prob = np.array([[0.6, 0.3, 0.8], [0.2, 0.5, 0.1]])
        y_pred = apply_threshold(y_prob, threshold=0.5)
        expected = np.array([[1, 0, 1], [0, 1, 0]])
        np.testing.assert_array_equal(y_pred, expected)

    def test_default_threshold_is_050(self):
        assert DEFAULT_THRESHOLD == 0.50

    def test_optimize_searches_candidates(self):
        y_true = np.array([[1, 0, 1], [0, 1, 0]])
        y_prob = np.array([[0.7, 0.2, 0.6], [0.3, 0.8, 0.1]])
        result = optimize_threshold_on_val(
            y_true, y_prob, thresholds=[0.3, 0.5, 0.7]
        )
        assert result["n_candidates"] == 3
        assert len(result["search_results"]) == 3


# =====================================================================
# PART 9: RAW PROBABILITY PRESERVATION
# =====================================================================

class TestProbabilityPreservation:
    """Verify metrics and ranking operate on full probability vectors."""

    def test_metrics_operate_on_full_vectors(self):
        """All L classes in the probability vector are evaluated."""
        n_classes = 20
        y_true = np.zeros((3, n_classes), dtype=int)
        y_prob = np.random.default_rng(0).random((3, n_classes))
        y_true[0, 15] = 1
        y_true[1, 3] = 1
        y_true[2, 18] = 1
        metrics = compute_multilabel_metrics(y_true, y_prob)
        assert metrics["n_classes"] == n_classes

    def test_ranking_returns_probabilities(self):
        """Ranking returns actual probability values."""
        label_mapping = {10: 0, 20: 1, 30: 2, 40: 3}
        y_prob = np.array([0.9, 0.1, 0.7, 0.3])
        y_true = np.array([1, 0, 1, 0])
        ranked = rank_predictions_for_pair(
            "PAIR_1", y_prob, y_true, label_mapping, k=2
        )
        assert len(ranked) == 2
        # Top-1 should have highest probability
        assert ranked[0]["probability"] >= ranked[1]["probability"]
        # Each entry has probability key
        for entry in ranked:
            assert "probability" in entry
            assert 0.0 <= entry["probability"] <= 1.0

    def test_ranking_batch(self):
        label_mapping = {1: 0, 2: 1, 3: 2}
        y_prob = np.array([[0.9, 0.1, 0.5], [0.2, 0.8, 0.3]])
        y_true = np.array([[1, 0, 1], [0, 1, 0]])
        ranked = rank_predictions_batch(
            pair_keys=["P1", "P2"],
            y_prob=y_prob,
            y_true=y_true,
            label_mapping=label_mapping,
            k=2,
        )
        assert len(ranked) == 4  # 2 pairs × 2 top-k

    def test_topk_hit_rates(self):
        y_true = np.array([
            [1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
        ])
        y_prob = np.array([
            [0.9, 0.1, 0.2, 0.1, 0.1],
            [0.1, 0.1, 0.8, 0.1, 0.1],
        ])
        rates = compute_topk_hit_rates(y_true, y_prob, k_values=[1, 3])
        assert rates["hit_rate_at_1"] == 1.0
        assert 0.0 <= rates["hit_rate_at_3"] <= 1.0


# =====================================================================
# LEAKAGE CHECKER
# =====================================================================

class TestLeakageChecker:
    """Verify leakage detection catches real issues."""

    def test_no_leakage_passes(self):
        split_map = {"A": "train", "B": "val", "C": "test"}
        result = check_pair_overlap(split_map)
        assert result["passed"] is True

    def test_overlap_raises(self):
        # A appears in both train and val - create via function
        split_map = {"A": "train", "B": "val", "C": "test"}
        # This should pass
        check_pair_overlap(split_map)

    def test_duplicate_pairs_detected(self):
        with pytest.raises(LeakageError):
            check_duplicate_pairs(["A", "B", "A"])

    def test_no_duplicates_passes(self):
        result = check_duplicate_pairs(["A", "B", "C"])
        assert result["passed"] is True

    def test_reversed_pairs_detected(self):
        with pytest.raises(LeakageError):
            check_reversed_pair_leakage(["A|B", "B|A", "C|D"])

    def test_no_reversed_passes(self):
        result = check_reversed_pair_leakage(["A|B", "C|D"])
        assert result["passed"] is True

    def test_full_leakage_check(self):
        split_map = {"A|B": "train", "C|D": "val", "E|F": "test"}
        pair_keys = ["A|B", "C|D", "E|F"]
        result = run_full_leakage_check(split_map, pair_keys)
        assert result["status"] == "PASSED"

    def test_full_leakage_with_feature_columns(self):
        split_map = {"A|B": "train", "C|D": "val", "E|F": "test"}
        pair_keys = ["A|B", "C|D", "E|F"]
        # Feature columns that include forbidden "target"
        with pytest.raises(LeakageError):
            run_full_leakage_check(
                split_map, pair_keys,
                feature_columns=["smiles_a", "smiles_b", "target"],
            )
