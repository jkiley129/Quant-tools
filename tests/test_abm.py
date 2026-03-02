"""
Tests for quant_sim/abm/ agents and market
"""

import numpy as np
import pytest
import pandas as pd
from quant_sim.abm.agents import InformedAgent, NoiseAgent, MarketMakerAgent
from quant_sim.abm.market import PredictionMarket


# ── Market state helper ────────────────────────────────────────────────────────

def market_state(bid=0.48, ask=0.52, mid=0.50, true_prob=0.55):
    return {
        "best_bid": bid,
        "best_ask": ask,
        "mid_price": mid,
        "true_prob": true_prob,
        "time_step": 0,
        "price_history": [mid],
    }


# ── InformedAgent ──────────────────────────────────────────────────────────────

class TestInformedAgent:
    def test_buys_when_underpriced(self):
        """If true_prob >> mid_price, agent should buy."""
        np.random.seed(0)
        agent = InformedAgent("INF_0", cash=10000, true_value=0.70, epsilon=0.01)
        state = market_state(bid=0.48, ask=0.52, mid=0.50, true_prob=0.70)
        orders = agent.decide(state)
        assert len(orders) > 0
        assert orders[0]["action"] == "BUY"

    def test_sells_when_overpriced(self):
        """If true_prob << mid_price, agent should sell."""
        np.random.seed(1)
        agent = InformedAgent("INF_0", cash=10000, true_value=0.30, epsilon=0.01)
        state = market_state(bid=0.48, ask=0.52, mid=0.50, true_prob=0.30)
        orders = agent.decide(state)
        assert len(orders) > 0
        assert orders[0]["action"] == "SELL"

    def test_passes_when_fairly_priced(self):
        """If true_prob ≈ mid_price, agent should pass."""
        np.random.seed(2)
        agent = InformedAgent("INF_0", cash=10000, true_value=0.50, epsilon=0.05)
        state = market_state(bid=0.48, ask=0.52, mid=0.50, true_prob=0.50)
        orders = agent.decide(state)
        assert len(orders) == 0

    def test_respects_position_limit(self):
        """Agent should not exceed position_limit."""
        np.random.seed(3)
        agent = InformedAgent("INF_0", cash=10000, true_value=0.80, epsilon=0.005,
                               position_limit=50)
        agent.inventory = 50  # already at limit
        state = market_state(mid=0.50, true_prob=0.80)
        orders = agent.decide(state)
        assert len(orders) == 0

    def test_order_price_in_valid_range(self):
        np.random.seed(4)
        agent = InformedAgent("INF_0", cash=10000, true_value=0.70)
        state = market_state(mid=0.50, true_prob=0.70)
        orders = agent.decide(state)
        if orders:
            assert 0.001 <= orders[0]["price"] <= 0.999

    def test_order_quantity_positive(self):
        np.random.seed(5)
        agent = InformedAgent("INF_0", cash=10000, true_value=0.70)
        state = market_state(mid=0.50, true_prob=0.70)
        orders = agent.decide(state)
        if orders:
            assert orders[0]["quantity"] >= 1


class TestNoiseAgent:
    def test_returns_list(self):
        np.random.seed(0)
        agent = NoiseAgent("NOISE_0", cash=10000, trade_prob=1.0)
        orders = agent.decide(market_state())
        assert isinstance(orders, list)

    def test_trades_with_prob_one_always(self):
        """With trade_prob=1.0, agent should always return an order."""
        np.random.seed(1)
        agent = NoiseAgent("NOISE_0", cash=10000, trade_prob=1.0)
        for _ in range(20):
            orders = agent.decide(market_state())
            assert len(orders) == 1

    def test_trades_with_prob_zero_never(self):
        agent = NoiseAgent("NOISE_0", cash=10000, trade_prob=0.0)
        for _ in range(20):
            orders = agent.decide(market_state())
            assert len(orders) == 0

    def test_order_is_buy_or_sell(self):
        np.random.seed(2)
        agent = NoiseAgent("NOISE_0", cash=10000, trade_prob=1.0)
        for _ in range(10):
            orders = agent.decide(market_state())
            if orders:
                assert orders[0]["action"] in ("BUY", "SELL")

    def test_order_price_near_mid(self):
        np.random.seed(3)
        agent = NoiseAgent("NOISE_0", cash=10000, trade_prob=1.0, price_std=0.01)
        mid = 0.5
        prices = []
        for _ in range(100):
            orders = agent.decide(market_state(mid=mid))
            if orders:
                prices.append(orders[0]["price"])
        assert abs(np.mean(prices) - mid) < 0.05


class TestMarketMakerAgent:
    def test_posts_both_bid_and_ask(self):
        agent = MarketMakerAgent("MM_0", cash=50000, spread=0.015)
        orders = agent.decide(market_state(mid=0.50))
        assert len(orders) == 2
        sides = {o["action"] for o in orders}
        assert "BUY" in sides
        assert "SELL" in sides

    def test_ask_above_bid(self):
        agent = MarketMakerAgent("MM_0", cash=50000, spread=0.015)
        orders = agent.decide(market_state(mid=0.50))
        bid_price = next(o["price"] for o in orders if o["action"] == "BUY")
        ask_price = next(o["price"] for o in orders if o["action"] == "SELL")
        assert ask_price > bid_price

    def test_spread_approximately_correct(self):
        agent = MarketMakerAgent("MM_0", cash=50000, spread=0.020)
        orders = agent.decide(market_state(mid=0.50))
        bid_price = next(o["price"] for o in orders if o["action"] == "BUY")
        ask_price = next(o["price"] for o in orders if o["action"] == "SELL")
        # Spread should be approximately 2 * half-spread
        assert abs((ask_price - bid_price) - 0.040) < 0.005

    def test_skews_ask_when_long(self):
        """Long inventory → skew ask down to attract sellers."""
        agent = MarketMakerAgent("MM_0", cash=50000, spread=0.015, inventory_limit=100)
        agent.inventory = 80  # long
        orders_long = agent.decide(market_state(mid=0.50))

        agent2 = MarketMakerAgent("MM_1", cash=50000, spread=0.015, inventory_limit=100)
        orders_neutral = agent2.decide(market_state(mid=0.50))

        ask_long = next(o["price"] for o in orders_long if o["action"] == "SELL")
        ask_neutral = next(o["price"] for o in orders_neutral if o["action"] == "SELL")
        assert ask_long < ask_neutral

    def test_prices_in_valid_range(self):
        agent = MarketMakerAgent("MM_0", cash=50000)
        for mid in [0.10, 0.30, 0.50, 0.70, 0.90]:
            orders = agent.decide(market_state(mid=mid))
            for o in orders:
                assert 0.001 <= o["price"] <= 0.999


# ── PredictionMarket ───────────────────────────────────────────────────────────

@pytest.fixture
def true_prob_path():
    np.random.seed(42)
    n = 100
    path = np.clip(0.5 + np.cumsum(np.random.normal(0, 0.01, n)), 0.1, 0.9)
    return path


class TestPredictionMarket:
    def test_run_returns_dataframe(self, true_prob_path):
        market = PredictionMarket(
            true_prob_path=true_prob_path,
            n_informed=2, n_noise=5, n_market_makers=1,
            seed=0,
        )
        result = market.run()
        assert isinstance(result, pd.DataFrame)

    def test_some_trades_occur(self, true_prob_path):
        market = PredictionMarket(
            true_prob_path=true_prob_path,
            n_informed=3, n_noise=8, n_market_makers=2,
            seed=1,
        )
        result = market.run()
        assert len(result) > 0

    def test_trade_prices_in_valid_range(self, true_prob_path):
        market = PredictionMarket(
            true_prob_path=true_prob_path,
            n_informed=2, n_noise=5, n_market_makers=1,
            seed=2,
        )
        trades = market.run()
        if len(trades) > 0:
            assert trades["price"].between(0.001, 0.999).all()

    def test_trade_quantities_positive(self, true_prob_path):
        market = PredictionMarket(
            true_prob_path=true_prob_path,
            n_informed=2, n_noise=5, n_market_makers=1,
            seed=3,
        )
        trades = market.run()
        if len(trades) > 0:
            assert (trades["quantity"] > 0).all()

    def test_price_history_length(self, true_prob_path):
        n = len(true_prob_path)
        market = PredictionMarket(true_prob_path=true_prob_path, seed=4)
        market.run()
        assert len(market.price_history) == n + 1  # initial + one per step

    def test_get_metrics_returns_dict(self, true_prob_path):
        market = PredictionMarket(true_prob_path=true_prob_path, seed=5)
        market.run()
        metrics = market.get_price_discovery_metrics()
        assert isinstance(metrics, dict)
        assert "price_error_rmse" in metrics
        assert "bid_ask_spread_mean" in metrics

    def test_rmse_finite_and_positive(self, true_prob_path):
        market = PredictionMarket(true_prob_path=true_prob_path, seed=6)
        market.run()
        metrics = market.get_price_discovery_metrics()
        assert np.isfinite(metrics["price_error_rmse"])
        assert metrics["price_error_rmse"] >= 0

    def test_agent_pnl_present(self, true_prob_path):
        market = PredictionMarket(
            true_prob_path=true_prob_path,
            n_informed=2, n_noise=3, n_market_makers=1,
            seed=7,
        )
        market.run()
        metrics = market.get_price_discovery_metrics()
        assert "agent_pnl_by_type" in metrics
        assert "InformedAgent" in metrics["agent_pnl_by_type"]

    def test_correct_number_of_agents_created(self, true_prob_path):
        market = PredictionMarket(
            true_prob_path=true_prob_path,
            n_informed=3, n_noise=7, n_market_makers=2,
            seed=8,
        )
        from quant_sim.abm.agents import InformedAgent, NoiseAgent, MarketMakerAgent
        n_inf = sum(1 for a in market.agents if isinstance(a, InformedAgent))
        n_noi = sum(1 for a in market.agents if isinstance(a, NoiseAgent))
        n_mm = sum(1 for a in market.agents if isinstance(a, MarketMakerAgent))
        assert n_inf == 3
        assert n_noi == 7
        assert n_mm == 2
