"""
Tests for quant_sim/calibration/brier.py
"""

import numpy as np
import pytest
from quant_sim.calibration.brier import BrierScorer


@pytest.fixture
def scorer():
    return BrierScorer()


class TestBrierScore:
    def test_perfect_forecast_score_zero(self, scorer):
        y_true = np.array([0, 0, 1, 1, 0, 1], dtype=float)
        y_pred = y_true.copy()  # perfect predictions
        assert scorer.score(y_true, y_pred) == pytest.approx(0.0)

    def test_worst_forecast_score_one(self, scorer):
        y_true = np.array([0, 0, 1, 1], dtype=float)
        y_pred = 1.0 - y_true  # always wrong
        assert scorer.score(y_true, y_pred) == pytest.approx(1.0)

    def test_constant_half_forecast(self, scorer):
        """Predicting 0.5 always: BS = mean((0.5 - y)^2)."""
        y_true = np.array([0, 0, 0, 1, 1, 1], dtype=float)
        y_pred = np.full(6, 0.5)
        expected = np.mean((0.5 - y_true) ** 2)
        assert scorer.score(y_true, y_pred) == pytest.approx(expected)

    def test_score_nonnegative(self, scorer):
        rng = np.random.default_rng(0)
        y_true = rng.integers(0, 2, 1000).astype(float)
        y_pred = rng.uniform(0, 1, 1000)
        assert scorer.score(y_true, y_pred) >= 0.0

    def test_better_forecast_lower_score(self, scorer):
        rng = np.random.default_rng(1)
        true_probs = rng.uniform(0.1, 0.9, 2000)
        y_true = (rng.uniform(size=2000) < true_probs).astype(float)

        y_good = np.clip(true_probs + rng.normal(0, 0.02, 2000), 0.01, 0.99)
        y_bad = np.clip(true_probs + rng.normal(0, 0.30, 2000), 0.01, 0.99)

        assert scorer.score(y_true, y_good) < scorer.score(y_true, y_bad)


class TestBrierDecompose:
    def test_decomposition_identity(self, scorer):
        """BS = Uncertainty - Resolution + Reliability (Murphy 1973)."""
        rng = np.random.default_rng(42)
        true_probs = rng.uniform(0.1, 0.9, 2000)
        y_true = (rng.uniform(size=2000) < true_probs).astype(float)
        y_pred = np.clip(true_probs + rng.normal(0, 0.05, 2000), 0.01, 0.99)

        result = scorer.decompose(y_true, y_pred)
        reconstructed = result["uncertainty"] - result["resolution"] + result["reliability"]
        # Murphy (1973) decomposition uses bin-mean predictions, so there is an inherent
        # within-bin approximation error. The identity holds to within ~1% of the BS value.
        assert abs(reconstructed - result["brier_score"]) < 0.005

    def test_uncertainty_is_base_rate_variance(self, scorer):
        """Uncertainty = o_bar * (1 - o_bar)."""
        rng = np.random.default_rng(10)
        y_true = rng.integers(0, 2, 500).astype(float)
        y_pred = rng.uniform(0, 1, 500)
        result = scorer.decompose(y_true, y_pred)
        o_bar = np.mean(y_true)
        expected_uncertainty = o_bar * (1 - o_bar)
        assert abs(result["uncertainty"] - expected_uncertainty) < 1e-10

    def test_all_components_nonnegative(self, scorer):
        rng = np.random.default_rng(20)
        y_true = rng.integers(0, 2, 1000).astype(float)
        y_pred = rng.uniform(0, 1, 1000)
        result = scorer.decompose(y_true, y_pred)
        assert result["uncertainty"] >= 0
        assert result["resolution"] >= 0
        assert result["reliability"] >= 0

    def test_perfect_forecast_zero_reliability(self, scorer):
        """A perfectly calibrated forecast has reliability = 0."""
        rng = np.random.default_rng(30)
        true_probs = rng.uniform(0.1, 0.9, 5000)
        y_true = (rng.uniform(size=5000) < true_probs).astype(float)
        # Use true probs as predictions — perfectly calibrated in expectation
        result = scorer.decompose(y_true, true_probs, n_bins=10)
        assert result["reliability"] < 0.01  # small due to finite sample noise

    def test_returns_arrays_for_bins(self, scorer):
        rng = np.random.default_rng(40)
        y_true = rng.integers(0, 2, 500).astype(float)
        y_pred = rng.uniform(0, 1, 500)
        result = scorer.decompose(y_true, y_pred, n_bins=10)
        assert len(result["bin_centers"]) > 0
        assert len(result["empirical_freqs"]) == len(result["bin_centers"])

    def test_empirical_freqs_between_zero_and_one(self, scorer):
        rng = np.random.default_rng(50)
        y_true = rng.integers(0, 2, 1000).astype(float)
        y_pred = rng.uniform(0, 1, 1000)
        result = scorer.decompose(y_true, y_pred)
        assert np.all(result["empirical_freqs"] >= 0)
        assert np.all(result["empirical_freqs"] <= 1)


class TestSkillScore:
    def test_perfect_forecast_skill_positive(self, scorer):
        rng = np.random.default_rng(60)
        true_probs = rng.uniform(0.1, 0.9, 2000)
        y_true = (rng.uniform(size=2000) < true_probs).astype(float)
        y_pred = np.clip(true_probs + rng.normal(0, 0.01, 2000), 0.01, 0.99)
        assert scorer.skill_score(y_true, y_pred) > 0

    def test_climatology_forecast_skill_zero(self, scorer):
        """Predicting the base rate always should give BSS ≈ 0."""
        rng = np.random.default_rng(70)
        y_true = rng.integers(0, 2, 2000).astype(float)
        y_pred = np.full(2000, np.mean(y_true))
        bss = scorer.skill_score(y_true, y_pred)
        assert abs(bss) < 0.01

    def test_skill_score_bounded_above_by_one(self, scorer):
        rng = np.random.default_rng(80)
        y_true = rng.integers(0, 2, 500).astype(float)
        y_pred = rng.uniform(0, 1, 500)
        bss = scorer.skill_score(y_true, y_pred)
        assert bss <= 1.0 + 1e-10
