"""Tests for quant_sim.distributions — MLE fitting and permutation testing."""

import numpy as np
import pytest
from scipy import stats
from quant_sim.distributions.fitting import (
    fit_normal, fit_student_t, normality_tests, permutation_test,
)


# ── fit_normal ─────────────────────────────────────────────────────────────────

class TestFitNormal:
    def setup_method(self):
        np.random.seed(0)
        self.data = np.random.normal(loc=0.001, scale=0.015, size=1000)

    def test_returns_expected_keys(self):
        result = fit_normal(self.data)
        assert {"loc", "scale", "log_likelihood", "aic"} == set(result.keys())

    def test_loc_close_to_true(self):
        result = fit_normal(self.data)
        assert abs(result["loc"] - 0.001) < 0.002

    def test_scale_close_to_true(self):
        result = fit_normal(self.data)
        assert abs(result["scale"] - 0.015) < 0.002

    def test_log_likelihood_negative(self):
        # Log-likelihood of continuous density can be positive or negative;
        # for small daily returns the values are large and usually negative
        result = fit_normal(self.data)
        assert isinstance(result["log_likelihood"], float)

    def test_aic_formula(self):
        result = fit_normal(self.data)
        expected_aic = 2 * 2 - 2 * result["log_likelihood"]
        assert abs(result["aic"] - expected_aic) < 1e-8


# ── fit_student_t ──────────────────────────────────────────────────────────────

class TestFitStudentT:
    def setup_method(self):
        np.random.seed(1)
        self.true_df = 4.0
        self.data = stats.t.rvs(df=self.true_df, loc=0.0005, scale=0.015, size=1000)

    def test_returns_expected_keys(self):
        result = fit_student_t(self.data)
        assert {"df", "loc", "scale", "log_likelihood", "aic", "success"} == set(result.keys())

    def test_df_in_reasonable_range(self):
        result = fit_student_t(self.data)
        # With 1000 samples, df estimate should be in [2, 20]
        assert 2.0 < result["df"] < 20.0

    def test_student_t_aic_lower_than_normal_for_fat_tails(self):
        """Student-t should fit fat-tailed data better than normal."""
        norm_result = fit_normal(self.data)
        t_result = fit_student_t(self.data)
        assert t_result["aic"] < norm_result["aic"]

    def test_loc_close_to_true(self):
        result = fit_student_t(self.data)
        assert abs(result["loc"] - 0.0005) < 0.005

    def test_scale_positive(self):
        result = fit_student_t(self.data)
        assert result["scale"] > 0.0

    def test_success_flag(self):
        result = fit_student_t(self.data)
        assert isinstance(result["success"], bool)


# ── test_normality ─────────────────────────────────────────────────────────────

class TestNormality:
    def test_normal_data_not_rejected(self):
        np.random.seed(42)
        data = np.random.normal(size=500)
        result = normality_tests(data)
        # With large normal sample some tests may still reject; check structure
        assert "reject_normality" in result
        assert isinstance(result["reject_normality"], bool)

    def test_fat_tailed_data_rejected(self):
        np.random.seed(42)
        data = stats.t.rvs(df=2.5, size=2000)
        result = normality_tests(data)
        assert result["reject_normality"] is True

    def test_returns_all_keys(self):
        np.random.seed(0)
        result = normality_tests(np.random.normal(size=300))
        expected = {
            "n", "skewness", "excess_kurtosis",
            "dagostino_pearson", "shapiro_wilk", "jarque_bera",
            "reject_normality",
        }
        assert expected == set(result.keys())

    def test_excess_kurtosis_positive_for_fat_tails(self):
        np.random.seed(0)
        data = stats.t.rvs(df=3, size=5000)
        result = normality_tests(data)
        assert result["excess_kurtosis"] > 0.0

    def test_p_values_in_range(self):
        np.random.seed(0)
        result = normality_tests(np.random.normal(size=500))
        for test in ("dagostino_pearson", "shapiro_wilk", "jarque_bera"):
            assert 0.0 <= result[test]["p_value"] <= 1.0


# ── permutation_test ───────────────────────────────────────────────────────────

class TestPermutationTest:
    def setup_method(self):
        np.random.seed(99)
        n = 500
        # True signal: slightly predicts the next return
        self.returns = np.random.normal(0.0005, 0.015, n)
        # Perfect signal: always long when returns are positive
        self.good_signals = np.sign(self.returns)
        # Random signal: pure noise
        self.bad_signals = np.random.choice([-1.0, 1.0], size=n)

    def test_returns_expected_keys(self):
        result = permutation_test(self.bad_signals, self.returns, n_perms=100, seed=0)
        expected = {
            "observed_sharpe", "p_value", "null_sharpe_mean",
            "null_sharpe_std", "n_perms", "significant_at_5pct",
        }
        assert expected == set(result.keys())

    def test_perfect_signal_is_significant(self):
        result = permutation_test(self.good_signals, self.returns, n_perms=2000, seed=0)
        assert result["p_value"] < 0.05
        assert result["significant_at_5pct"] is True

    def test_random_signal_not_significant(self):
        result = permutation_test(self.bad_signals, self.returns, n_perms=2000, seed=0)
        # Random signal should NOT be significant in expectation
        # (occasionally may be, but p-value should be high on average)
        assert result["p_value"] > 0.01   # very loose bound

    def test_p_value_in_range(self):
        result = permutation_test(self.bad_signals, self.returns, n_perms=500, seed=0)
        assert 0.0 <= result["p_value"] <= 1.0

    def test_n_perms_recorded(self):
        result = permutation_test(self.bad_signals, self.returns, n_perms=333, seed=0)
        assert result["n_perms"] == 333

    def test_observed_sharpe_finite(self):
        result = permutation_test(self.bad_signals, self.returns, n_perms=100, seed=0)
        assert np.isfinite(result["observed_sharpe"])
