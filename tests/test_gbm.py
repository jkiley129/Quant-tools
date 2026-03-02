"""
Tests for quant_sim/monte_carlo/gbm.py
"""

import numpy as np
import pytest
from scipy import stats as scipy_stats
from quant_sim.monte_carlo.gbm import GBMSimulator


S0, MU, SIGMA, R, T = 100.0, 0.10, 0.25, 0.05, 1.0
N = 200_000  # large enough for tight statistical bounds


@pytest.fixture
def sim():
    return GBMSimulator(S0=S0, mu=MU, sigma=SIGMA, r=R, T=T, n_paths=N, seed=0)


class TestSimulateTerminal:
    def test_output_shape(self, sim):
        S_T = sim.simulate_terminal()
        assert S_T.shape == (N,)

    def test_all_positive(self, sim):
        S_T = sim.simulate_terminal()
        assert np.all(S_T > 0)

    def test_mean_close_to_analytical(self, sim):
        # E[S_T] = S0 * exp(mu * T)
        S_T = sim.simulate_terminal()
        expected_mean = S0 * np.exp(MU * T)
        assert abs(np.mean(S_T) / expected_mean - 1.0) < 0.01  # within 1%

    def test_log_returns_normally_distributed(self, sim):
        """log(S_T/S0) ~ N((mu - 0.5*sigma^2)*T, sigma^2*T)"""
        S_T = sim.simulate_terminal()
        log_ret = np.log(S_T / S0)
        expected_mean = (MU - 0.5 * SIGMA**2) * T
        expected_std = SIGMA * np.sqrt(T)
        assert abs(np.mean(log_ret) - expected_mean) < 0.005
        assert abs(np.std(log_ret) - expected_std) < 0.005


class TestSimulatePaths:
    def test_output_shape(self, sim):
        n_steps = 252
        paths = sim.simulate_paths(n_steps)
        assert paths.shape == (n_steps + 1, N)

    def test_first_row_is_s0(self, sim):
        paths = sim.simulate_paths(50)
        np.testing.assert_array_equal(paths[0], S0)

    def test_all_paths_positive(self, sim):
        paths = sim.simulate_paths(50)
        assert np.all(paths > 0)

    def test_terminal_distribution_matches_closed_form(self, sim):
        """Terminal prices from full paths should match direct terminal draws."""
        n_steps = 252
        paths = sim.simulate_paths(n_steps)
        # Mean of terminal prices
        path_mean = np.mean(paths[-1])
        direct_mean = np.mean(sim.simulate_terminal())
        expected = S0 * np.exp(MU * T)
        assert abs(path_mean / expected - 1.0) < 0.02


class TestPriceBinaryCall:
    def test_returns_expected_keys(self, sim):
        result = sim.price_binary_call(K=110.0)
        assert set(result.keys()) == {"mc_price", "std_error", "ci_95", "analytical_price"}

    def test_mc_matches_analytical(self, sim):
        """MC price should be within 3 std errors of analytical."""
        for K in [90, 100, 110, 120]:
            result = sim.price_binary_call(K=float(K))
            diff = abs(result["mc_price"] - result["analytical_price"])
            assert diff < 3 * result["std_error"], (
                f"K={K}: MC={result['mc_price']:.4f} vs "
                f"analytical={result['analytical_price']:.4f}, 3*se={3*result['std_error']:.4f}"
            )

    def test_price_between_zero_and_one(self, sim):
        for K in [50, 100, 150, 200]:
            result = sim.price_binary_call(K=float(K))
            assert 0.0 <= result["mc_price"] <= 1.0

    def test_price_decreases_with_strike(self, sim):
        """Binary call price must be monotonically decreasing in K."""
        strikes = [80.0, 90.0, 100.0, 110.0, 120.0]
        prices = [sim.price_binary_call(K)["mc_price"] for K in strikes]
        assert all(prices[i] >= prices[i + 1] for i in range(len(prices) - 1))

    def test_deep_itm_price_near_one(self):
        sim = GBMSimulator(S0=100, mu=0.10, sigma=0.25, T=1.0, n_paths=50_000, seed=1)
        result = sim.price_binary_call(K=1.0)  # K way below S0
        assert result["mc_price"] > 0.99

    def test_deep_otm_price_near_zero(self):
        sim = GBMSimulator(S0=100, mu=0.10, sigma=0.25, T=1.0, n_paths=50_000, seed=2)
        result = sim.price_binary_call(K=10_000.0)
        assert result["mc_price"] < 0.001

    def test_ci_contains_mc_price(self, sim):
        result = sim.price_binary_call(K=100.0)
        lo, hi = result["ci_95"]
        assert lo <= result["mc_price"] <= hi


class TestPriceBinaryPut:
    def test_call_plus_put_equals_one(self):
        """P(S>K) + P(S<K) = 1 (ignoring P(S=K) which is zero for continuous)."""
        sim = GBMSimulator(S0=100, mu=0.10, sigma=0.25, T=1.0, n_paths=100_000, seed=5)
        K = 105.0
        call = sim.price_binary_call(K)["mc_price"]
        # Re-seed for consistent comparison
        sim2 = GBMSimulator(S0=100, mu=0.10, sigma=0.25, T=1.0, n_paths=100_000, seed=5)
        put = sim2.price_binary_put(K)["mc_price"]
        assert abs(call + put - 1.0) < 0.01


class TestExceedanceProbability:
    def test_matches_binary_call(self):
        sim = GBMSimulator(S0=100, mu=0.10, sigma=0.25, T=1.0, n_paths=100_000, seed=7)
        K = 110.0
        p1 = sim.get_exceedance_probability(K)
        sim2 = GBMSimulator(S0=100, mu=0.10, sigma=0.25, T=1.0, n_paths=100_000, seed=7)
        p2 = sim2.price_binary_call(K)["mc_price"]
        assert abs(p1 - p2) < 0.005
