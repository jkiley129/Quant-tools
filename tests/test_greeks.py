"""Tests for quant_sim.greeks — Black-Scholes pricing and Greeks."""

import numpy as np
import pytest
from quant_sim.greeks.pricing import black_scholes, greeks, mc_price


# ── Black-Scholes pricing ──────────────────────────────────────────────────────

class TestBlackScholes:
    def test_call_positive(self):
        price = black_scholes(100, 100, 1.0, 0.05, 0.20, "call")
        assert price > 0

    def test_put_positive(self):
        price = black_scholes(100, 100, 1.0, 0.05, 0.20, "put")
        assert price > 0

    def test_put_call_parity(self):
        """C - P = S - K·e^{-rT}  (put-call parity)."""
        S, K, T, r, sigma = 100.0, 105.0, 1.0, 0.05, 0.25
        C = black_scholes(S, K, T, r, sigma, "call")
        P = black_scholes(S, K, T, r, sigma, "put")
        parity_lhs = C - P
        parity_rhs = S - K * np.exp(-r * T)
        assert abs(parity_lhs - parity_rhs) < 1e-8

    def test_call_deep_itm_approaches_forward(self):
        """Deep ITM call → S - K·e^{-rT}."""
        S, K, T, r, sigma = 200.0, 50.0, 1.0, 0.05, 0.20
        C = black_scholes(S, K, T, r, sigma, "call")
        forward = S - K * np.exp(-r * T)
        assert abs(C - forward) < 0.50

    def test_call_deep_otm_near_zero(self):
        """Deep OTM call → ~0."""
        C = black_scholes(50.0, 200.0, 1.0, 0.05, 0.20, "call")
        assert C < 0.001

    def test_put_deep_itm_approaches_discounted_strike(self):
        """Deep ITM put → K·e^{-rT} - S."""
        S, K, T, r, sigma = 50.0, 200.0, 1.0, 0.05, 0.20
        P = black_scholes(S, K, T, r, sigma, "put")
        intrinsic = K * np.exp(-r * T) - S
        assert abs(P - intrinsic) < 1.0

    def test_call_increases_with_vol(self):
        C_low  = black_scholes(100, 100, 1.0, 0.05, 0.10, "call")
        C_high = black_scholes(100, 100, 1.0, 0.05, 0.40, "call")
        assert C_high > C_low

    def test_known_value(self):
        """Spot-check against a well-known BS calculation."""
        # S=100, K=100, T=1, r=0.05, sigma=0.2 → call ≈ 10.4506
        C = black_scholes(100, 100, 1.0, 0.05, 0.20, "call")
        assert abs(C - 10.4506) < 0.01

    @pytest.mark.parametrize("option_type", ["call", "put"])
    def test_price_is_float(self, option_type):
        price = black_scholes(100, 105, 0.5, 0.05, 0.25, option_type)
        assert isinstance(price, float)


# ── Greeks ─────────────────────────────────────────────────────────────────────

class TestGreeks:
    def setup_method(self):
        self.S, self.K, self.T, self.r, self.sigma = 100.0, 100.0, 1.0, 0.05, 0.20

    def test_returns_all_keys(self):
        g = greeks(self.S, self.K, self.T, self.r, self.sigma)
        expected = {"delta_call", "delta_put", "gamma", "theta_daily", "vega_1pct", "rho_1pct"}
        assert expected == set(g.keys())

    def test_delta_call_in_range(self):
        g = greeks(self.S, self.K, self.T, self.r, self.sigma)
        assert 0.0 < g["delta_call"] < 1.0

    def test_delta_put_in_range(self):
        g = greeks(self.S, self.K, self.T, self.r, self.sigma)
        assert -1.0 < g["delta_put"] < 0.0

    def test_delta_put_call_relationship(self):
        """delta_put = delta_call - 1."""
        g = greeks(self.S, self.K, self.T, self.r, self.sigma)
        assert abs(g["delta_put"] - (g["delta_call"] - 1.0)) < 1e-10

    def test_atm_delta_call_near_half(self):
        """ATM call delta ≈ 0.5 for short-maturity options."""
        g = greeks(100.0, 100.0, 0.01, 0.0, 0.20)
        assert abs(g["delta_call"] - 0.5) < 0.05

    def test_gamma_positive(self):
        g = greeks(self.S, self.K, self.T, self.r, self.sigma)
        assert g["gamma"] > 0.0

    def test_theta_negative(self):
        """Theta should be negative for long options (time decay costs)."""
        g = greeks(self.S, self.K, self.T, self.r, self.sigma)
        assert g["theta_daily"] < 0.0

    def test_vega_positive(self):
        """Higher vol → higher price → positive vega."""
        g = greeks(self.S, self.K, self.T, self.r, self.sigma)
        assert g["vega_1pct"] > 0.0

    def test_rho_positive_for_call(self):
        """Higher rates → higher call price → positive rho."""
        g = greeks(self.S, self.K, self.T, self.r, self.sigma)
        assert g["rho_1pct"] > 0.0

    def test_gamma_finite_diff(self):
        """Gamma ≈ (C(S+ε) - 2C(S) + C(S-ε)) / ε²."""
        eps = 0.01
        C_up  = black_scholes(self.S + eps, self.K, self.T, self.r, self.sigma)
        C_mid = black_scholes(self.S,       self.K, self.T, self.r, self.sigma)
        C_dn  = black_scholes(self.S - eps, self.K, self.T, self.r, self.sigma)
        gamma_fd = (C_up - 2 * C_mid + C_dn) / eps**2
        g = greeks(self.S, self.K, self.T, self.r, self.sigma)
        assert abs(g["gamma"] - gamma_fd) < 1e-4


# ── Monte Carlo cross-check ────────────────────────────────────────────────────

class TestMCPrice:
    def test_call_converges_to_bs(self):
        S, K, T, r, sigma = 100.0, 105.0, 1.0, 0.05, 0.20
        bs = black_scholes(S, K, T, r, sigma, "call")
        mc = mc_price(S, K, T, r, sigma, n_sims=500_000, option_type="call", seed=42)
        assert abs(mc["price"] - bs) < 0.05   # within 5 cents

    def test_put_converges_to_bs(self):
        S, K, T, r, sigma = 100.0, 95.0, 0.5, 0.05, 0.25
        bs = black_scholes(S, K, T, r, sigma, "put")
        mc = mc_price(S, K, T, r, sigma, n_sims=500_000, option_type="put", seed=42)
        assert abs(mc["price"] - bs) < 0.05

    def test_mc_returns_dict_with_expected_keys(self):
        result = mc_price(100, 100, 1.0, 0.05, 0.20, n_sims=10_000, seed=0)
        assert {"price", "std_error", "ci_95"} == set(result.keys())

    def test_ci_contains_bs_price(self):
        S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
        bs = black_scholes(S, K, T, r, sigma, "call")
        mc = mc_price(S, K, T, r, sigma, n_sims=200_000, seed=7)
        lo, hi = mc["ci_95"]
        assert lo < bs < hi

    def test_std_error_decreases_with_more_paths(self):
        kwargs = dict(S=100, K=100, T=1.0, r=0.05, sigma=0.20, seed=0)
        mc_small = mc_price(**kwargs, n_sims=1_000)
        mc_large = mc_price(**kwargs, n_sims=100_000)
        assert mc_large["std_error"] < mc_small["std_error"]
