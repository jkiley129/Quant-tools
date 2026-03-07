"""Tests for quant_sim.factors — PCA and factor regression."""

import numpy as np
import pandas as pd
import pytest
from quant_sim.factors.analysis import pca_returns, factor_regression


def make_return_df(n_assets=6, n_days=500, seed=0):
    """Synthetic multi-asset returns driven by 2 latent factors."""
    np.random.seed(seed)
    T = n_days
    F = np.random.randn(T, 2) * 0.01
    B = np.random.randn(n_assets, 2) * 0.5
    eps = np.random.randn(T, n_assets) * 0.008
    data = F @ B.T + eps
    dates = pd.date_range("2020-01-01", periods=T, freq="B")
    return pd.DataFrame(data, index=dates,
                        columns=[f"A{i}" for i in range(n_assets)])


# ── pca_returns ───────────────────────────────────────────────────────────────

class TestPCAReturns:
    def setup_method(self):
        self.returns = make_return_df()

    def test_components_shape(self):
        result = pca_returns(self.returns, n_components=3)
        assert result["components"].shape == (3, self.returns.shape[1])

    def test_explained_var_sums_leq_one(self):
        result = pca_returns(self.returns, n_components=3)
        assert sum(result["explained_var"]) <= 1.0 + 1e-8

    def test_cumulative_var_monotone(self):
        result = pca_returns(self.returns, n_components=4)
        cum = result["cumulative_var"]
        for i in range(len(cum) - 1):
            assert cum[i] <= cum[i + 1] + 1e-10

    def test_factor_scores_shape(self):
        result = pca_returns(self.returns, n_components=3)
        assert result["factor_scores"].shape == (len(self.returns), 3)

    def test_factor_scores_index_matches_input(self):
        result = pca_returns(self.returns, n_components=2)
        pd.testing.assert_index_equal(result["factor_scores"].index, self.returns.index)

    def test_eigenvalues_descending(self):
        result = pca_returns(self.returns, n_components=4)
        ev = result["eigenvalues"]
        assert all(ev[i] >= ev[i + 1] for i in range(len(ev) - 1))

    def test_n_components_auto_from_threshold(self):
        """With variance_threshold=0.80, should select enough components."""
        result = pca_returns(self.returns, n_components=None, variance_threshold=0.80)
        assert result["n_components"] >= 1
        assert result["cumulative_var"][result["n_components"] - 1] >= 0.80 - 1e-6

    def test_returns_asset_names(self):
        result = pca_returns(self.returns, n_components=2)
        assert result["asset_names"] == list(self.returns.columns)

    def test_two_latent_factors_captured_by_two_pcs(self):
        """Data driven by 2 factors: first 2 PCs should explain most variance."""
        result = pca_returns(self.returns, n_components=2)
        assert result["cumulative_var"][1] > 0.40


# ── factor_regression ─────────────────────────────────────────────────────────

class TestFactorRegression:
    def setup_method(self):
        np.random.seed(42)
        T = 600
        dates = pd.date_range("2020-01-01", periods=T, freq="B")
        # Two factors
        f1 = pd.Series(np.random.normal(0, 0.01, T), index=dates, name="F1")
        f2 = pd.Series(np.random.normal(0, 0.01, T), index=dates, name="F2")
        # Portfolio: 0.8·F1 + 0.3·F2 + alpha + noise
        alpha_daily = 0.0005
        noise = np.random.normal(0, 0.005, T)
        port = pd.Series(
            alpha_daily + 0.8 * f1.values + 0.3 * f2.values + noise,
            index=dates,
            name="Portfolio",
        )
        self.portfolio = port
        self.factors = pd.DataFrame({"F1": f1, "F2": f2})

    def test_returns_expected_keys(self):
        result = factor_regression(self.portfolio, self.factors)
        expected = {
            "alpha", "alpha_annualised", "alpha_tstat", "alpha_pvalue",
            "betas", "beta_tstats", "beta_pvalues",
            "r_squared", "n_obs", "summary_str",
        }
        assert expected == set(result.keys())

    def test_alpha_near_true_value(self):
        result = factor_regression(self.portfolio, self.factors)
        # True daily alpha = 0.0005; annualised ≈ 0.126
        assert abs(result["alpha"] - 0.0005) < 0.001

    def test_beta_f1_near_true(self):
        result = factor_regression(self.portfolio, self.factors)
        assert abs(result["betas"]["F1"] - 0.8) < 0.15

    def test_beta_f2_near_true(self):
        result = factor_regression(self.portfolio, self.factors)
        assert abs(result["betas"]["F2"] - 0.3) < 0.15

    def test_r_squared_in_range(self):
        result = factor_regression(self.portfolio, self.factors)
        assert 0.0 <= result["r_squared"] <= 1.0

    def test_n_obs_correct(self):
        result = factor_regression(self.portfolio, self.factors)
        assert result["n_obs"] == len(self.portfolio)

    def test_p_values_in_range(self):
        result = factor_regression(self.portfolio, self.factors)
        assert 0.0 <= result["alpha_pvalue"] <= 1.0
        for p in result["beta_pvalues"].values():
            assert 0.0 <= p <= 1.0

    def test_alpha_significant_when_true_alpha_nonzero(self):
        """With 600 obs and daily alpha=0.0005, should detect significance."""
        result = factor_regression(self.portfolio, self.factors)
        # Expect t-stat > 2 (roughly)
        assert abs(result["alpha_tstat"]) > 1.5

    def test_single_factor_regression(self):
        """Regression with one factor should still return all keys."""
        single = self.factors[["F1"]]
        result = factor_regression(self.portfolio, single)
        assert "alpha" in result
        assert "F1" in result["betas"]
