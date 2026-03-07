"""Tests for quant_sim.portfolio — Markowitz optimization and Kelly criterion."""

import numpy as np
import pytest
from quant_sim.portfolio.optimize import markowitz_optimize, efficient_frontier, kelly_weights


def make_problem(n=4, seed=0):
    np.random.seed(seed)
    mu = np.array([0.08, 0.10, 0.12, 0.07])[:n]
    A = np.random.randn(n, n) * 0.1
    cov = A @ A.T + np.diag(np.full(n, 0.04))
    return mu, cov


# ── markowitz_optimize ────────────────────────────────────────────────────────

class TestMarkowitzOptimize:
    def test_weights_sum_to_one(self):
        mu, cov = make_problem()
        result = markowitz_optimize(mu, cov, target_return=0.09)
        assert result["weights"] is not None
        assert abs(np.sum(result["weights"]) - 1.0) < 1e-4

    def test_weights_nonnegative_long_only(self):
        mu, cov = make_problem()
        result = markowitz_optimize(mu, cov, target_return=0.09, allow_short=False)
        assert np.all(result["weights"] >= -1e-4)

    def test_achieved_return_meets_target(self):
        mu, cov = make_problem()
        target = 0.09
        result = markowitz_optimize(mu, cov, target_return=target)
        assert result["portfolio_return"] >= target - 1e-4

    def test_returns_expected_keys(self):
        mu, cov = make_problem()
        result = markowitz_optimize(mu, cov, target_return=0.09)
        for key in ("weights", "portfolio_return", "portfolio_vol", "sharpe", "status"):
            assert key in result

    def test_vol_positive(self):
        mu, cov = make_problem()
        result = markowitz_optimize(mu, cov, target_return=0.09)
        assert result["portfolio_vol"] > 0.0

    def test_lower_target_lower_vol(self):
        """Minimum-variance frontier: lower return target → lower vol."""
        mu, cov = make_problem()
        r1 = markowitz_optimize(mu, cov, target_return=0.08)
        r2 = markowitz_optimize(mu, cov, target_return=0.11)
        if r1["weights"] is not None and r2["weights"] is not None:
            assert r1["portfolio_vol"] <= r2["portfolio_vol"] + 1e-4


# ── efficient_frontier ────────────────────────────────────────────────────────

class TestEfficientFrontier:
    def test_returns_expected_keys(self):
        mu, cov = make_problem()
        frontier = efficient_frontier(mu, cov, n_points=10)
        for key in ("returns", "vols", "sharpes", "weights"):
            assert key in frontier

    def test_nonempty(self):
        mu, cov = make_problem()
        frontier = efficient_frontier(mu, cov, n_points=20)
        assert len(frontier["returns"]) > 0

    def test_returns_monotone(self):
        """Returns along the frontier should be non-decreasing."""
        mu, cov = make_problem()
        frontier = efficient_frontier(mu, cov, n_points=30)
        rets = frontier["returns"]
        for i in range(len(rets) - 1):
            assert rets[i] <= rets[i + 1] + 1e-6

    def test_vols_positive(self):
        mu, cov = make_problem()
        frontier = efficient_frontier(mu, cov, n_points=10)
        for v in frontier["vols"]:
            assert v > 0.0

    def test_weights_sum_to_one(self):
        mu, cov = make_problem()
        frontier = efficient_frontier(mu, cov, n_points=10)
        for w in frontier["weights"]:
            assert abs(np.sum(w) - 1.0) < 1e-3


# ── kelly_weights ─────────────────────────────────────────────────────────────

class TestKellyWeights:
    def test_returns_expected_keys(self):
        mu, cov = make_problem()
        result = kelly_weights(mu, cov, r=0.05)
        for key in ("weights_raw", "weights_normalised", "portfolio_return",
                    "portfolio_vol", "sharpe"):
            assert key in result

    def test_normalised_weights_sum_to_one_in_abs(self):
        """Normalised Kelly weights: sum of |w| = 1."""
        mu, cov = make_problem()
        result = kelly_weights(mu, cov, r=0.05)
        w = result["weights_normalised"]
        assert abs(np.sum(np.abs(w)) - 1.0) < 1e-6

    def test_half_kelly_has_smaller_raw_weights(self):
        """Half-Kelly raw weights are exactly 0.5x full-Kelly."""
        mu, cov = make_problem()
        full = kelly_weights(mu, cov, r=0.05, fraction=1.0)
        half = kelly_weights(mu, cov, r=0.05, fraction=0.5)
        np.testing.assert_allclose(half["weights_raw"], 0.5 * full["weights_raw"], rtol=1e-8)

    def test_portfolio_vol_positive(self):
        mu, cov = make_problem()
        result = kelly_weights(mu, cov, r=0.05)
        assert result["portfolio_vol"] >= 0.0

    def test_higher_excess_return_higher_kelly(self):
        """Asset with higher excess return should get larger Kelly weight."""
        mu = np.array([0.06, 0.20])
        cov = np.diag([0.04, 0.04])
        result = kelly_weights(mu, cov, r=0.05, fraction=1.0)
        w = result["weights_raw"]
        # Asset 1 (mu=0.20) has much higher excess return → larger Kelly weight
        assert abs(w[1]) > abs(w[0])
