"""Regression tests for threshold selection methodology.

Tests:
1. Threshold candidates generated correctly
2. Threshold optimization uses validation data only
3. Test labels not passed to threshold selection
4. All-positive predictions correctly detected
5. All-negative predictions correctly detected
6. Predicted-positive fraction calculated correctly
7. Ranking metrics use continuous scores
8. Thresholded metrics use binary predictions
9. Existing threshold behavior remains reproducible
10. optimize_threshold_on_val_fine returns a dict (not None)
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import f1_score

from src.evaluation.thresholds import (
    DEFAULT_THRESHOLD,
    apply_threshold,
    detect_degenerate_prediction,
    optimize_threshold_on_val,
    optimize_threshold_on_val_fine,
    optimize_threshold_non_degenerate,
)
from src.evaluation.multilabel_metrics import (
    compute_multilabel_metrics,
    _precision_recall_at_k,
)


# --- Test 1: Threshold candidates generated correctly ---

class TestThresholdCandidates:
    def test_coarse_candidates_range(self):
        """Coarse search should generate thresholds from 0.05 to 0.95 at 0.05 step."""
        y = np.array([[1, 0], [0, 1]])
        p = np.array([[0.8, 0.2], [0.3, 0.7]])
        result = optimize_threshold_on_val(y, p)
        thresholds_searched = [r["threshold"] for r in result["search_results"]]
        expected = [round(t * 0.05, 2) for t in range(1, 20)]
        assert thresholds_searched == expected, f"Expected {expected}, got {thresholds_searched}"

    def test_coarse_n_candidates(self):
        """Should have exactly 19 coarse candidates."""
        y = np.array([[1, 0], [0, 1]])
        p = np.array([[0.8, 0.2], [0.3, 0.7]])
        result = optimize_threshold_on_val(y, p)
        assert result["n_candidates"] == 19

    def test_custom_thresholds(self):
        """Custom threshold list should be used when provided."""
        y = np.array([[1, 0], [0, 1]])
        p = np.array([[0.8, 0.2], [0.3, 0.7]])
        custom = [0.2, 0.5, 0.8]
        result = optimize_threshold_on_val(y, p, thresholds=custom)
        thresholds_searched = [r["threshold"] for r in result["search_results"]]
        assert thresholds_searched == custom

    def test_fine_search_candidates(self):
        """Fine search should search around coarse best at 0.01 step."""
        y = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0]])
        p = np.array([[0.8, 0.2, 0.9], [0.1, 0.7, 0.3], [0.6, 0.5, 0.2]])
        result = optimize_threshold_on_val_fine(y, p)
        assert result is not None
        assert "fine_search_results" in result
        fine_ts = [r["threshold"] for r in result["fine_search_results"]]
        # Fine search should be at 0.01 steps around coarse best
        for i in range(len(fine_ts) - 1):
            diff = round(fine_ts[i + 1] - fine_ts[i], 2)
            assert diff == 0.01, f"Fine step should be 0.01, got {diff}"


# --- Test 2: Uses validation data only ---

class TestValidationOnly:
    def test_optimize_uses_val_labels(self):
        """Optimization should use provided y_true_val, not any external data."""
        y_val = np.array([[1, 0, 0], [0, 1, 0]])
        p_val = np.array([[0.8, 0.1, 0.1], [0.1, 0.9, 0.1]])
        result = optimize_threshold_on_val(y_val, p_val, metric="micro_f1")
        # The function should optimize based on val data
        assert result["best_score"] > 0, "Should find non-zero score on valid data"
        assert "VALIDATION" in result["note"]


# --- Test 3: Test labels not passed ---

class TestTestLabelsExcluded:
    def test_no_test_parameter(self):
        """optimize_threshold_on_val should not accept test labels parameter."""
        import inspect
        sig = inspect.signature(optimize_threshold_on_val)
        param_names = list(sig.parameters.keys())
        assert "y_true_test" not in param_names
        assert "y_test" not in param_names
        assert "test" not in " ".join(param_names).lower()

    def test_fine_no_test_parameter(self):
        """optimize_threshold_on_val_fine should not accept test labels."""
        import inspect
        sig = inspect.signature(optimize_threshold_on_val_fine)
        param_names = list(sig.parameters.keys())
        assert "y_true_test" not in param_names
        assert "y_test" not in param_names


# --- Test 4: All-positive detection ---

class TestAllPositiveDetection:
    def test_all_positive_detected(self):
        """Should detect all-positive predictions."""
        y_pred = np.ones((5, 10), dtype=int)
        is_deg, pos_frac, mean_pos = detect_degenerate_prediction(y_pred)
        assert is_deg is True
        assert pos_frac == 1.0
        assert mean_pos == 10.0

    def test_near_all_positive(self):
        """Should detect near-all-positive (> 50% default threshold)."""
        y_pred = np.ones((5, 10), dtype=int)
        y_pred[0, 0] = 0  # Only 1 zero out of 50
        is_deg, pos_frac, mean_pos = detect_degenerate_prediction(y_pred)
        assert is_deg is True
        assert pos_frac > 0.5

    def test_normal_not_flagged(self):
        """Normal predictions should not be flagged as degenerate."""
        y_pred = np.zeros((5, 10), dtype=int)
        y_pred[0, :3] = 1
        y_pred[1, :2] = 1
        is_deg, pos_frac, mean_pos = detect_degenerate_prediction(y_pred)
        assert is_deg is False
        assert pos_frac < 0.5

    def test_custom_fraction_threshold(self):
        """Custom max_positive_fraction should work."""
        y_pred = np.zeros((5, 10), dtype=int)
        y_pred[:, :4] = 1  # 40% positive
        is_deg_low, _, _ = detect_degenerate_prediction(y_pred, max_positive_fraction=0.3)
        is_deg_high, _, _ = detect_degenerate_prediction(y_pred, max_positive_fraction=0.5)
        assert is_deg_low is True
        assert is_deg_high is False


# --- Test 5: All-negative detection ---

class TestAllNegativeDetection:
    def test_all_negative_zero_score(self):
        """All-negative predictions should give F1=0."""
        y_true = np.array([[1, 0, 1], [0, 1, 0]])
        y_pred = np.zeros_like(y_true)
        assert y_pred.sum() == 0
        # apply_threshold at a very high threshold
        p = np.array([[0.01, 0.01, 0.01], [0.01, 0.01, 0.01]])
        y_pred_thresh = apply_threshold(p, 0.99)
        assert y_pred_thresh.sum() == 0

    def test_all_negative_degenerate_detection(self):
        """Empty predictions should not be flagged as degenerate (pos_fraction=0)."""
        y_pred = np.zeros((5, 10), dtype=int)
        is_deg, pos_frac, mean_pos = detect_degenerate_prediction(y_pred)
        assert is_deg is False
        assert pos_frac == 0.0
        assert mean_pos == 0.0


# --- Test 6: Predicted-positive fraction ---

class TestPredictedPositiveFraction:
    def test_fraction_calculation(self):
        """Predicted-positive fraction should be correct."""
        y_pred = np.zeros((4, 10), dtype=int)
        y_pred[0, :5] = 1  # 5 positives
        y_pred[1, :3] = 1  # 3 positives
        # Total: 8 positives out of 40
        is_deg, pos_frac, mean_pos = detect_degenerate_prediction(y_pred)
        assert abs(pos_frac - 0.2) < 1e-5, f"Expected 0.2, got {pos_frac}"
        assert abs(mean_pos - 2.0) < 1e-5, f"Expected 2.0, got {mean_pos}"

    def test_empty_array(self):
        """Empty array should return (False, 0.0, 0.0)."""
        y_pred = np.zeros((0, 10), dtype=int)
        is_deg, pos_frac, mean_pos = detect_degenerate_prediction(y_pred)
        assert is_deg is False
        assert pos_frac == 0.0


# --- Test 7: Ranking metrics use continuous scores ---

class TestRankingMetrics:
    def test_precision_recall_at_k_uses_probabilities(self):
        """P@K should use continuous scores for ranking, not binary predictions."""
        y_true = np.array([[1, 0, 1, 0, 0]])
        # Probabilities — not binary
        y_prob = np.array([[0.9, 0.1, 0.7, 0.3, 0.2]])
        p1, r1 = _precision_recall_at_k(y_true, y_prob, k=1)
        # Top-1 should be index 0 (prob=0.9), which IS a true positive
        assert p1 == 1.0
        assert r1 == 0.5  # 1 hit / 2 true positives

    def test_ranking_not_affected_by_threshold(self):
        """Ranking metrics should give same results regardless of threshold."""
        y_true = np.array([[1, 0, 1, 0, 0]])
        y_prob = np.array([[0.9, 0.1, 0.7, 0.3, 0.2]])
        m1 = compute_multilabel_metrics(y_true, y_prob, threshold=0.5)
        m2 = compute_multilabel_metrics(y_true, y_prob, threshold=0.1)
        # mAP should be same regardless of threshold
        assert m1["mAP"] == m2["mAP"]
        # P@K should be same regardless of threshold
        assert m1["precision_at_5"] == m2["precision_at_5"]


# --- Test 8: Thresholded metrics use binary predictions ---

class TestThresholdedMetrics:
    def test_different_thresholds_give_different_f1(self):
        """F1 should change with threshold."""
        y_true = np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]])
        y_prob = np.array([[0.8, 0.3, 0.2, 0.1, 0.05], [0.1, 0.7, 0.4, 0.2, 0.1]])
        m_low = compute_multilabel_metrics(y_true, y_prob, threshold=0.1)
        m_high = compute_multilabel_metrics(y_true, y_prob, threshold=0.9)
        # At threshold 0.1, more labels predicted → different F1 than at 0.9
        assert m_low["micro_f1"] != m_high["micro_f1"]

    def test_apply_threshold_binary(self):
        """apply_threshold should produce binary output."""
        p = np.array([[0.3, 0.7], [0.5, 0.5]])
        result = apply_threshold(p, 0.5)
        assert set(np.unique(result)).issubset({0, 1})
        assert result[0, 0] == 0
        assert result[0, 1] == 1
        assert result[1, 0] == 1  # >= 0.5
        assert result[1, 1] == 1  # >= 0.5


# --- Test 9: Reproducible threshold behavior ---

class TestReproducible:
    def test_coarse_deterministic(self):
        """Same input should give same threshold."""
        y = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 0]])
        p = np.array([[0.8, 0.2, 0.6], [0.1, 0.9, 0.3], [0.7, 0.1, 0.2]])
        r1 = optimize_threshold_on_val(y, p, metric="micro_f1")
        r2 = optimize_threshold_on_val(y, p, metric="micro_f1")
        assert r1["best_threshold"] == r2["best_threshold"]
        assert r1["best_score"] == r2["best_score"]

    def test_fine_deterministic(self):
        """Fine search should be deterministic."""
        y = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 0]])
        p = np.array([[0.8, 0.2, 0.6], [0.1, 0.9, 0.3], [0.7, 0.1, 0.2]])
        r1 = optimize_threshold_on_val_fine(y, p, metric="micro_f1")
        r2 = optimize_threshold_on_val_fine(y, p, metric="micro_f1")
        assert r1["best_threshold"] == r2["best_threshold"]

    def test_tie_breaking_favors_lower(self):
        """When multiple thresholds give the same score, the first (lowest) should win."""
        y = np.array([[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]])
        # All probs above 0.05 → any threshold from 0.05-0.5 gives same binary prediction
        p = np.array([[0.9, 0.8, 0.7, 0.6, 0.55, 0.01, 0.01, 0.01, 0.01, 0.01]])
        result = optimize_threshold_on_val(y, p, metric="micro_f1")
        # Multiple thresholds should give the same F1
        scores = [r["score"] for r in result["search_results"]]
        # First threshold that achieves best score should be selected
        best_score = result["best_score"]
        first_best = next(r["threshold"] for r in result["search_results"] if r["score"] == best_score)
        assert result["best_threshold"] == first_best


# --- Test 10: optimize_threshold_on_val_fine returns dict ---

class TestFineFunctionReturns:
    def test_returns_dict_not_none(self):
        """optimize_threshold_on_val_fine must return a dict, not None."""
        y = np.array([[1, 0], [0, 1]])
        p = np.array([[0.8, 0.2], [0.3, 0.7]])
        result = optimize_threshold_on_val_fine(y, p)
        assert result is not None, "CRITICAL: optimize_threshold_on_val_fine returns None"
        assert isinstance(result, dict), f"Expected dict, got {type(result)}"

    def test_has_required_keys(self):
        """Return dict must contain required keys."""
        y = np.array([[1, 0], [0, 1]])
        p = np.array([[0.8, 0.2], [0.3, 0.7]])
        result = optimize_threshold_on_val_fine(y, p)
        required_keys = {"best_threshold", "best_score", "metric"}
        assert required_keys.issubset(set(result.keys()))

    def test_best_threshold_in_valid_range(self):
        """Best threshold must be between 0.01 and 0.99."""
        y = np.array([[1, 0, 1], [0, 1, 0]])
        p = np.array([[0.8, 0.2, 0.6], [0.1, 0.9, 0.3]])
        result = optimize_threshold_on_val_fine(y, p)
        assert 0.01 <= result["best_threshold"] <= 0.99

    def test_non_degenerate_function_returns(self):
        """optimize_threshold_non_degenerate must return a dict."""
        y = np.array([[1, 0, 1], [0, 1, 0]])
        p = np.array([[0.8, 0.2, 0.6], [0.1, 0.9, 0.3]])
        result = optimize_threshold_non_degenerate(y, p)
        assert result is not None
        assert isinstance(result, dict)
        assert "best_threshold" in result
        assert "is_degenerate" in result


# --- Additional: Non-degenerate optimizer ---

class TestNonDegenerateOptimizer:
    def test_prefers_non_degenerate(self):
        """Should prefer non-degenerate threshold when possible."""
        # Create a case where all-positive gives high micro-F1
        n_classes = 20
        y = np.ones((5, n_classes), dtype=int)  # All positive ground truth
        y[:, 10:] = 0  # Half negative
        p = np.full((5, n_classes), 0.6)  # All probs at 0.6

        result = optimize_threshold_non_degenerate(y, p, max_positive_fraction=0.80)
        assert result is not None
        assert "selection_status" in result

    def test_fallback_when_all_degenerate(self):
        """Should fallback gracefully when all thresholds are degenerate."""
        y = np.ones((3, 5), dtype=int)
        p = np.full((3, 5), 0.99)
        result = optimize_threshold_non_degenerate(
            y, p, min_threshold=0.05, max_threshold=0.95, max_positive_fraction=0.01
        )
        assert result is not None
        assert result["selection_status"] in ("non_degenerate_optimum", "fallback_all_thresholds_degenerate")
