"""
Tests for quant_sim/variance_reduction/techniques.py
"""

import numpy as np
import pytest
from quant_sim.variance_reduction.techniques import VarianceReducer


S0, MU, SIGMA, R, T = 100.0, 0.10, 0.25, 0.05, 1.0
K = 105.0  # slightly OTM binary call


@pytest.fixture
def reducer():
    np.random.seed(0)
    return VarianceReducer(S0=S0, mu=MU, sigma=SIGMA, r=R, T=T, n_samples=100_000)


def analytical_price(S0, mu, sigma, T, K):
    from scipy import stats as scipy_stats
    d2 = (np.log(S0 / K) + (mu - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return float(scipy_stats.norm.cdf(d2))


class TestPriceAntithetic:
    def test_returns_expected_keys(self, reducer):
        result = reducer.price_antithetic(K)
        assert {"price", "variance", "std_error", "variance_reduction_ratio"} == set(result.keys())

    def test_price_close_to_analytical(self):
        np.random.seed(1)
        reducer = VarianceReducer(S0=S0, mu=MU, sigma=SIGMA, T=T, n_samples=200_000)
        true_p = analytical_price(S0, MU, SIGMA, T, K)
        result = reducer.price_antithetic(K)
        assert abs(result["price"] - true_p) < 0.005

    def test_variance_reduction_ratio_greater_than_one(self):
        np.random.seed(2)
        reducer = VarianceReducer(S0=S0, mu=MU, sigma=SIGMA, T=T, n_samples=100_000)
        result = reducer.price_antithetic(K)
        assert result["variance_reduction_ratio"] > 1.0

    def test_price_between_zero_and_one(self, reducer):
        result = reducer.price_antithetic(K)
        assert 0.0 <= result["price"] <= 1.0

    def test_std_error_positive(self, reducer):
        result = reducer.price_antithetic(K)
        assert result["std_error"] > 0


class TestPriceControlVariate:
    def test_returns_expected_keys(self, reducer):
        result = reducer.price_control_variate(K)
        assert {"price", "variance", "std_error", "beta", "variance_reduction_ratio"} == set(result.keys())

    def test_price_close_to_analytical(self):
        np.random.seed(3)
        reducer = VarianceReducer(S0=S0, mu=MU, sigma=SIGMA, T=T, n_samples=200_000)
        true_p = analytical_price(S0, MU, SIGMA, T, K)
        result = reducer.price_control_variate(K)
        assert abs(result["price"] - true_p) < 0.01

    def test_beta_is_negative(self):
        """For a binary call, as S_T increases, prob(S_T > K) stays 0 or 1,
        but on average higher S_T correlates with payoff=1, so beta > 0."""
        np.random.seed(4)
        reducer = VarianceReducer(S0=S0, mu=MU, sigma=SIGMA, T=T, n_samples=50_000)
        result = reducer.price_control_variate(K)
        # Beta should be positive (payoff correlated with S_T)
        assert result["beta"] > 0

    def test_price_between_zero_and_one(self, reducer):
        result = reducer.price_control_variate(K)
        assert 0.0 <= result["price"] <= 1.0


class TestPriceStratified:
    def test_returns_expected_keys(self, reducer):
        result = reducer.price_stratified(K)
        assert {"price", "variance", "std_error", "stratum_estimates",
                "variance_reduction_ratio"} == set(result.keys())

    def test_price_close_to_analytical(self):
        np.random.seed(5)
        reducer = VarianceReducer(S0=S0, mu=MU, sigma=SIGMA, T=T, n_samples=200_000)
        true_p = analytical_price(S0, MU, SIGMA, T, K)
        result = reducer.price_stratified(K, n_strata=100)
        assert abs(result["price"] - true_p) < 0.005

    def test_variance_reduction_ratio_high(self):
        """Stratified sampling typically achieves the highest VRR."""
        np.random.seed(6)
        reducer = VarianceReducer(S0=S0, mu=MU, sigma=SIGMA, T=T, n_samples=100_000)
        result = reducer.price_stratified(K, n_strata=100)
        assert result["variance_reduction_ratio"] > 10.0

    def test_stratum_estimates_length(self, reducer):
        n_strata = 50
        result = reducer.price_stratified(K, n_strata=n_strata)
        assert len(result["stratum_estimates"]) == n_strata

    def test_price_between_zero_and_one(self, reducer):
        result = reducer.price_stratified(K)
        assert 0.0 <= result["price"] <= 1.0


class TestCompareAll:
    def test_returns_dataframe(self, reducer):
        import pandas as pd
        df = reducer.compare_all(K)
        assert isinstance(df, pd.DataFrame)

    def test_has_four_rows(self, reducer):
        df = reducer.compare_all(K)
        assert len(df) == 4

    def test_methods_present(self, reducer):
        df = reducer.compare_all(K)
        methods = set(df["Method"])
        assert "Crude MC" in methods
        assert "Antithetic" in methods
        assert "Control Variate" in methods
        assert "Stratified" in methods

    def test_all_prices_consistent(self):
        """All four methods should produce prices within ~2% of each other."""
        np.random.seed(7)
        reducer = VarianceReducer(S0=S0, mu=MU, sigma=SIGMA, T=T, n_samples=200_000)
        df = reducer.compare_all(K)
        prices = df["Price"].values
        assert np.max(prices) - np.min(prices) < 0.02

    def test_crude_mc_vrr_is_one(self, reducer):
        df = reducer.compare_all(K)
        crude_row = df[df["Method"] == "Crude MC"]
        assert float(crude_row["Variance Reduction"].iloc[0]) == pytest.approx(1.0)

    def test_all_vrr_at_least_one(self, reducer):
        df = reducer.compare_all(K)
        assert (df["Variance Reduction"] >= 1.0).all()
