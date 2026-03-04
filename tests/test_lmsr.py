"""Tests for quant_sim.lmsr — Logarithmic Market Scoring Rule."""

import numpy as np
import pytest
from quant_sim.lmsr.market import LMSRMarket


class TestLMSRMarket:

    # ── Construction ──────────────────────────────────────────────────────────

    def test_default_prices_are_uniform(self):
        """With q=0, all outcomes have equal probability."""
        market = LMSRMarket(n_outcomes=2, b=100.0)
        prices = market.prices()
        np.testing.assert_allclose(prices, [0.5, 0.5], atol=1e-10)

    def test_prices_sum_to_one_initial(self):
        market = LMSRMarket(n_outcomes=3, b=50.0)
        assert abs(market.prices().sum() - 1.0) < 1e-10

    def test_max_loss_formula(self):
        """max_loss = b · ln(n)."""
        b, n = 100.0, 3
        market = LMSRMarket(n_outcomes=n, b=b)
        expected = b * np.log(n)
        assert abs(market.max_loss - expected) < 1e-10

    # ── Prices = softmax ──────────────────────────────────────────────────────

    def test_prices_are_softmax(self):
        """Verify p = softmax(q/b) after a trade."""
        market = LMSRMarket(n_outcomes=2, b=50.0)
        market.trade(0, 10.0)  # buy 10 shares of outcome 0
        q = market.q
        expected = np.exp(q / 50.0) / np.exp(q / 50.0).sum()
        np.testing.assert_allclose(market.prices(), expected, atol=1e-10)

    def test_prices_sum_to_one_after_trades(self):
        market = LMSRMarket(n_outcomes=3, b=100.0)
        market.trade(0, 20.0)
        market.trade(1, -5.0)
        market.trade(2, 10.0)
        assert abs(market.prices().sum() - 1.0) < 1e-8

    def test_buying_outcome_increases_its_price(self):
        market = LMSRMarket(n_outcomes=2, b=100.0)
        p_before = market.prices()[0]
        market.trade(0, 15.0)
        p_after = market.prices()[0]
        assert p_after > p_before

    def test_selling_outcome_decreases_its_price(self):
        market = LMSRMarket(n_outcomes=2, b=100.0)
        market.trade(0, 20.0)  # first buy so there are shares to sell
        p_before = market.prices()[0]
        market.trade(0, -10.0)
        p_after = market.prices()[0]
        assert p_after < p_before

    def test_prices_in_open_interval(self):
        """LMSR guarantees 0 < p_i < 1 always."""
        market = LMSRMarket(n_outcomes=4, b=50.0)
        market.trade(0, 1000.0)   # massive buy — price pushed toward 1
        prices = market.prices()
        assert all(0 < p < 1 for p in prices)

    # ── Cost function ─────────────────────────────────────────────────────────

    def test_cost_positive_for_buying(self):
        market = LMSRMarket(n_outcomes=2, b=100.0)
        cost = market.trade_cost(0, 5.0)
        assert cost > 0.0

    def test_cost_negative_for_selling(self):
        market = LMSRMarket(n_outcomes=2, b=100.0)
        market.trade(0, 20.0)    # buy first
        cost = market.trade_cost(0, -10.0)
        assert cost < 0.0

    def test_cost_is_convex_in_quantity(self):
        """Buying 20 shares should cost more than 2x the cost of 10 shares."""
        market = LMSRMarket(n_outcomes=2, b=100.0)
        cost_10 = market.trade_cost(0, 10.0)
        cost_20 = market.trade_cost(0, 20.0)
        assert cost_20 > 2 * cost_10

    def test_trade_cost_path_independence(self):
        """Cost of reaching state q depends only on q, not the path."""
        market1 = LMSRMarket(n_outcomes=2, b=100.0)
        market1.trade(0, 30.0)
        total1 = sum(market1.cost_history)

        market2 = LMSRMarket(n_outcomes=2, b=100.0)
        market2.trade(0, 10.0)
        market2.trade(0, 10.0)
        market2.trade(0, 10.0)
        total2 = sum(market2.cost_history)

        assert abs(total1 - total2) < 1e-8

    # ── Trade execution ───────────────────────────────────────────────────────

    def test_trade_returns_expected_keys(self):
        market = LMSRMarket(n_outcomes=2, b=100.0)
        result = market.trade(0, 5.0)
        assert {"cost", "prices_before", "prices_after"} == set(result.keys())

    def test_trade_updates_shares(self):
        market = LMSRMarket(n_outcomes=2, b=100.0)
        market.trade(0, 7.5)
        assert abs(market.q[0] - 7.5) < 1e-10
        assert abs(market.q[1] - 0.0) < 1e-10

    def test_price_history_grows_with_trades(self):
        market = LMSRMarket(n_outcomes=2, b=50.0)
        assert len(market.price_history) == 1   # initial state
        market.trade(0, 5.0)
        assert len(market.price_history) == 2
        market.trade(1, 3.0)
        assert len(market.price_history) == 3

    def test_trade_log_records_all_trades(self):
        market = LMSRMarket(n_outcomes=2, b=50.0)
        market.trade(0, 5.0, trader_id="alice")
        market.trade(1, 3.0, trader_id="bob")
        assert len(market.trade_log) == 2
        assert market.trade_log[0]["trader"] == "alice"
        assert market.trade_log[1]["trader"] == "bob"

    # ── Summary ───────────────────────────────────────────────────────────────

    def test_summary_structure(self):
        market = LMSRMarket(n_outcomes=2, b=100.0, outcome_names=["YES", "NO"])
        market.trade(0, 10.0)
        s = market.summary()
        assert "prices" in s
        assert "YES" in s["prices"]
        assert s["n_trades"] == 1

    # ── Numerical stability ───────────────────────────────────────────────────

    def test_large_share_imbalance_stable(self):
        """Extreme share counts should not produce NaN or inf."""
        market = LMSRMarket(n_outcomes=2, b=1.0)
        market.trade(0, 1000.0)   # very large relative to b
        prices = market.prices()
        assert np.all(np.isfinite(prices))
        assert abs(prices.sum() - 1.0) < 1e-8

    def test_many_outcomes_stable(self):
        market = LMSRMarket(n_outcomes=100, b=200.0)
        prices = market.prices()
        assert np.all(np.isfinite(prices))
        assert abs(prices.sum() - 1.0) < 1e-8
