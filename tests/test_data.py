"""
Tests for quant_sim/data/calibrator.py
Uses synthetic price data only — no network calls required.
"""

import numpy as np
import pandas as pd
import pytest
from quant_sim.data.calibrator import calibrate_gbm, calibrate_correlation


def make_synthetic_prices(S0=100.0, mu=0.10, sigma=0.25, n_days=504, seed=0):
    """Generate a synthetic GBM price series for testing."""
    np.random.seed(seed)
    dt = 1 / 252
    log_returns = np.random.normal((mu - 0.5 * sigma**2) * dt, sigma * np.sqrt(dt), n_days)
    prices = S0 * np.exp(np.cumsum(log_returns))
    return pd.Series(prices, name="Close")


class TestCalibrateGbm:
    def test_returns_expected_keys(self):
        prices = make_synthetic_prices()
        result = calibrate_gbm(prices)
        assert set(result.keys()) == {"S0", "mu", "sigma"}

    def test_s0_is_last_price(self):
        prices = make_synthetic_prices()
        result = calibrate_gbm(prices)
        assert result["S0"] == pytest.approx(float(prices.iloc[-1]))

    def test_sigma_close_to_true_value(self):
        """With 2 years of daily data, estimated sigma should be within 3% of truth."""
        true_sigma = 0.25
        prices = make_synthetic_prices(sigma=true_sigma, n_days=504, seed=1)
        result = calibrate_gbm(prices)
        assert abs(result["sigma"] - true_sigma) < 0.03

    def test_sigma_positive(self):
        prices = make_synthetic_prices()
        result = calibrate_gbm(prices)
        assert result["sigma"] > 0

    def test_mu_approximately_recovered(self):
        """Drift is noisier than sigma — accept a wider band."""
        true_mu = 0.10
        prices = make_synthetic_prices(mu=true_mu, n_days=504, seed=2)
        result = calibrate_gbm(prices)
        # Drift estimation has high variance; allow ±10%
        assert abs(result["mu"] - true_mu) < 0.10

    def test_higher_sigma_yields_higher_estimate(self):
        """More volatile series should produce a higher sigma estimate."""
        low_vol = make_synthetic_prices(sigma=0.10, n_days=504, seed=3)
        high_vol = make_synthetic_prices(sigma=0.40, n_days=504, seed=4)
        assert calibrate_gbm(high_vol)["sigma"] > calibrate_gbm(low_vol)["sigma"]

    def test_ito_correction_applied(self):
        """mu estimate should satisfy: mu ≈ mean_log_return * 252 + 0.5*sigma^2.
        If Ito correction were missing, mu would be underestimated.
        """
        prices = make_synthetic_prices(mu=0.10, sigma=0.25, n_days=504, seed=5)
        result = calibrate_gbm(prices)
        log_returns = np.log(prices / prices.shift(1)).dropna()
        raw_mu = float(np.mean(log_returns) * 252)
        # Corrected mu should be larger than raw mu (Ito adds 0.5*sigma^2 > 0)
        assert result["mu"] > raw_mu

    def test_works_with_short_series(self):
        prices = make_synthetic_prices(n_days=30)
        result = calibrate_gbm(prices)
        assert result["sigma"] > 0
        assert result["S0"] > 0


class TestCalibrateCorrelation:
    def test_returns_square_matrix(self):
        prices = pd.DataFrame({
            "A": make_synthetic_prices(seed=0),
            "B": make_synthetic_prices(seed=1),
        })
        corr = calibrate_correlation(prices)
        assert corr.shape == (2, 2)

    def test_diagonal_is_one(self):
        prices = pd.DataFrame({
            "A": make_synthetic_prices(seed=0),
            "B": make_synthetic_prices(seed=1),
        })
        corr = calibrate_correlation(prices)
        np.testing.assert_array_almost_equal(np.diag(corr), np.ones(2))

    def test_symmetric(self):
        prices = pd.DataFrame({
            "A": make_synthetic_prices(seed=0),
            "B": make_synthetic_prices(seed=1),
            "C": make_synthetic_prices(seed=2),
        })
        corr = calibrate_correlation(prices)
        np.testing.assert_array_almost_equal(corr, corr.T)

    def test_perfectly_correlated_series_gives_one(self):
        base = make_synthetic_prices(seed=0)
        prices = pd.DataFrame({"A": base, "B": base * 2.0})
        corr = calibrate_correlation(prices)
        assert abs(corr[0, 1] - 1.0) < 1e-6

    def test_uncorrelated_series_gives_near_zero(self):
        """Independent GBM paths should have low correlation."""
        prices = pd.DataFrame({
            "A": make_synthetic_prices(seed=10),
            "B": make_synthetic_prices(seed=20),
        })
        corr = calibrate_correlation(prices)
        assert abs(corr[0, 1]) < 0.20  # loose bound due to finite sample

    def test_high_correlation_recovered(self):
        """Artificially correlated series should give high correlation estimate."""
        np.random.seed(30)
        n = 504
        dt = 1 / 252
        base = np.random.normal(0, 0.015, n)
        # Series B is 95% base + 5% independent noise
        noise = np.random.normal(0, 0.015, n)
        series_a = 100 * np.exp(np.cumsum(base))
        series_b = 100 * np.exp(np.cumsum(0.95 * base + 0.05 * noise))
        prices = pd.DataFrame({"A": series_a, "B": series_b})
        corr = calibrate_correlation(prices)
        assert corr[0, 1] > 0.80

    def test_three_assets(self):
        prices = pd.DataFrame({
            "A": make_synthetic_prices(seed=0),
            "B": make_synthetic_prices(seed=1),
            "C": make_synthetic_prices(seed=2),
        })
        corr = calibrate_correlation(prices)
        assert corr.shape == (3, 3)
        assert np.all(np.abs(corr) <= 1.0 + 1e-10)
