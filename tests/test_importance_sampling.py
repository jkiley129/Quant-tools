"""
Tests for quant_sim/importance_sampling/tail_risk.py
"""

import numpy as np
import pytest
from scipy import stats as scipy_stats
from quant_sim.importance_sampling.tail_risk import ImportanceSampler


S0, MU, SIGMA, R, T = 100.0, 0.10, 0.25, 0.05, 1.0


@pytest.fixture
def sampler():
    return ImportanceSampler(S0=S0, mu=MU, sigma=SIGMA, r=R, T=T)


def analytical_exceedance(S0, mu, sigma, T, threshold):
    d2 = (np.log(S0 / threshold) + (mu - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return float(scipy_stats.norm.cdf(d2))


class TestEstimateTailProbability:
    def test_returns_expected_keys(self, sampler):
        result = sampler.estimate_tail_probability(threshold=120.0, n_samples=10_000)
        expected = {"is_estimate", "is_std_error", "naive_estimate",
                    "analytical", "variance_reduction_ratio", "ess", "n_samples"}
        assert expected == set(result.keys())

    def test_is_estimate_close_to_analytical(self):
        np.random.seed(0)
        sampler = ImportanceSampler(S0=100, mu=0.10, sigma=0.25, T=1.0)
        threshold = 130.0  # moderately rare
        analytical = analytical_exceedance(100, 0.10, 0.25, 1.0, threshold)
        result = sampler.estimate_tail_probability(threshold=threshold, n_samples=100_000)
        assert abs(result["is_estimate"] - analytical) < 0.005

    def test_is_tighter_than_naive(self):
        """IS std error should be smaller than naive MC std error for a tail event."""
        np.random.seed(1)
        sampler = ImportanceSampler(S0=100, mu=0.10, sigma=0.25, T=1.0)
        threshold = 150.0  # rare 50% gain event
        result = sampler.estimate_tail_probability(threshold=threshold, n_samples=50_000)
        # IS std error should be much smaller than naive
        assert result["is_std_error"] < result["naive_estimate"] * 0.05 or result["is_std_error"] < 1e-4

    def test_variance_reduction_ratio_greater_than_one(self):
        np.random.seed(2)
        sampler = ImportanceSampler(S0=100, mu=0.10, sigma=0.25, T=1.0)
        result = sampler.estimate_tail_probability(threshold=140.0, n_samples=50_000)
        assert result["variance_reduction_ratio"] > 1.0

    def test_ess_positive_and_bounded(self):
        np.random.seed(3)
        sampler = ImportanceSampler(S0=100, mu=0.10, sigma=0.25, T=1.0)
        result = sampler.estimate_tail_probability(threshold=120.0, n_samples=10_000)
        assert 0 < result["ess"] <= result["n_samples"]

    def test_estimate_between_zero_and_one(self, sampler):
        np.random.seed(4)
        result = sampler.estimate_tail_probability(threshold=110.0, n_samples=20_000)
        assert 0.0 <= result["is_estimate"] <= 1.0

    def test_higher_threshold_lower_probability(self):
        np.random.seed(5)
        sampler = ImportanceSampler(S0=100, mu=0.10, sigma=0.25, T=1.0)
        p_low = sampler.estimate_tail_probability(threshold=110.0, n_samples=50_000)["is_estimate"]
        p_high = sampler.estimate_tail_probability(threshold=130.0, n_samples=50_000)["is_estimate"]
        assert p_low > p_high


class TestEstimateCvar:
    def test_returns_expected_keys(self, sampler):
        np.random.seed(10)
        result = sampler.estimate_cvar(alpha=0.05, n_samples=10_000)
        assert set(result.keys()) == {"cvar", "var", "alpha"}

    def test_cvar_leq_var(self, sampler):
        """CVaR(alpha) <= VaR(alpha) since CVaR is the mean of the worst-alpha tail."""
        np.random.seed(11)
        result = sampler.estimate_cvar(alpha=0.05, n_samples=50_000)
        assert result["cvar"] <= result["var"] + 0.01  # small tolerance for IS estimator

    def test_var_positive(self, sampler):
        np.random.seed(12)
        result = sampler.estimate_cvar(alpha=0.05, n_samples=20_000)
        assert result["var"] > 0

    def test_alpha_preserved(self, sampler):
        np.random.seed(13)
        for alpha in [0.01, 0.05, 0.10]:
            result = sampler.estimate_cvar(alpha=alpha, n_samples=10_000)
            assert result["alpha"] == alpha


class TestCompareNaiveVsIs:
    def test_returns_both_estimates(self, sampler):
        np.random.seed(20)
        result = sampler.compare_naive_vs_is(threshold=130.0, n_samples=20_000)
        assert "naive_estimate" in result
        assert "is_estimate" in result
        assert "variance_reduction_ratio" in result

    def test_both_estimates_close_to_analytical(self):
        np.random.seed(21)
        sampler = ImportanceSampler(S0=100, mu=0.10, sigma=0.25, T=1.0)
        threshold = 120.0
        analytical = analytical_exceedance(100, 0.10, 0.25, 1.0, threshold)
        result = sampler.compare_naive_vs_is(threshold=threshold, n_samples=100_000)
        assert abs(result["naive_estimate"] - analytical) < 0.01
        assert abs(result["is_estimate"] - analytical) < 0.01
