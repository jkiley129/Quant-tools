"""
Tests for quant_sim/abm/order_book.py
"""

import numpy as np
import pytest
from quant_sim.abm.order_book import LimitOrderBook, Order


def make_order(agent_id, side, price, quantity, timestamp, order_id=None):
    if order_id is None:
        import uuid
        order_id = str(uuid.uuid4())[:8]
    return Order(
        order_id=order_id,
        agent_id=agent_id,
        side=side,
        price=price,
        quantity=quantity,
        timestamp=timestamp,
    )


@pytest.fixture
def book():
    return LimitOrderBook(tick_size=0.001)


class TestEmptyBook:
    def test_best_bid_none_on_empty(self, book):
        assert book.best_bid() is None

    def test_best_ask_none_on_empty(self, book):
        assert book.best_ask() is None

    def test_mid_price_none_on_empty(self, book):
        assert book.mid_price() is None

    def test_spread_none_on_empty(self, book):
        assert book.spread() is None


class TestAddLimitOrder:
    def test_add_bid_sets_best_bid(self, book):
        order = make_order("A", "BID", 0.500, 10, 1)
        book.add_limit_order(order)
        assert abs(book.best_bid() - 0.500) < 0.001

    def test_add_ask_sets_best_ask(self, book):
        order = make_order("A", "ASK", 0.510, 10, 1)
        book.add_limit_order(order)
        assert abs(book.best_ask() - 0.510) < 0.001

    def test_mid_price_computed_correctly(self, book):
        book.add_limit_order(make_order("A", "BID", 0.490, 10, 1))
        book.add_limit_order(make_order("B", "ASK", 0.510, 10, 2))
        assert abs(book.mid_price() - 0.500) < 0.001

    def test_spread_computed_correctly(self, book):
        book.add_limit_order(make_order("A", "BID", 0.490, 10, 1))
        book.add_limit_order(make_order("B", "ASK", 0.510, 10, 2))
        assert abs(book.spread() - 0.020) < 0.002

    def test_best_bid_is_highest_bid(self, book):
        for price in [0.480, 0.490, 0.470]:
            book.add_limit_order(make_order("A", "BID", price, 5, 1))
        assert abs(book.best_bid() - 0.490) < 0.001

    def test_best_ask_is_lowest_ask(self, book):
        for price in [0.520, 0.510, 0.530]:
            book.add_limit_order(make_order("A", "ASK", price, 5, 1))
        assert abs(book.best_ask() - 0.510) < 0.001


class TestMatching:
    def test_crossing_orders_generate_trade(self, book):
        book.add_limit_order(make_order("A", "BID", 0.510, 10, 1))
        trades = book.add_limit_order(make_order("B", "ASK", 0.505, 10, 2))
        assert len(trades) == 1

    def test_trade_price_is_resting_order_price(self, book):
        """Trade executes at the resting (passive) order's price."""
        book.add_limit_order(make_order("A", "BID", 0.510, 10, 1))
        trades = book.add_limit_order(make_order("B", "ASK", 0.505, 10, 2))
        assert abs(trades[0]["price"] - 0.510) < 0.001

    def test_trade_quantity(self, book):
        book.add_limit_order(make_order("A", "BID", 0.510, 5, 1))
        trades = book.add_limit_order(make_order("B", "ASK", 0.505, 5, 2))
        assert trades[0]["quantity"] == 5

    def test_partial_fill_leaves_remainder(self, book):
        book.add_limit_order(make_order("A", "BID", 0.510, 10, 1))
        book.add_limit_order(make_order("B", "ASK", 0.505, 6, 2))
        # 6 filled, 4 remain on bid side
        assert book.best_bid() is not None
        # The remaining bid qty should be 4
        assert book._bids[0].quantity == 4

    def test_full_fill_removes_order_from_book(self, book):
        book.add_limit_order(make_order("A", "BID", 0.510, 5, 1))
        book.add_limit_order(make_order("B", "ASK", 0.505, 5, 2))
        # Book should be empty after full fill
        assert book.best_bid() is None

    def test_non_crossing_orders_dont_trade(self, book):
        book.add_limit_order(make_order("A", "BID", 0.490, 10, 1))
        trades = book.add_limit_order(make_order("B", "ASK", 0.510, 10, 2))
        assert len(trades) == 0

    def test_buyer_seller_recorded_correctly(self, book):
        book.add_limit_order(make_order("BUYER", "BID", 0.510, 10, 1))
        trades = book.add_limit_order(make_order("SELLER", "ASK", 0.505, 10, 2))
        assert trades[0]["buyer_id"] == "BUYER"
        assert trades[0]["seller_id"] == "SELLER"

    def test_trade_recorded_in_history(self, book):
        book.add_limit_order(make_order("A", "BID", 0.510, 5, 1))
        book.add_limit_order(make_order("B", "ASK", 0.505, 5, 2))
        assert len(book.trade_history) == 1

    def test_multiple_fills_sweep_book(self, book):
        """A large aggressive order should fill multiple resting levels."""
        for i, price in enumerate([0.510, 0.505, 0.500]):
            book.add_limit_order(make_order(f"MM{i}", "BID", price, 5, i))
        # Aggressive seller sweeps all three levels
        trades = book.add_limit_order(make_order("SELLER", "ASK", 0.490, 15, 10))
        assert len(trades) == 3
        assert sum(t["quantity"] for t in trades) == 15


class TestPriceTimePriority:
    def test_earlier_order_filled_first_at_same_price(self, book):
        """FIFO: order at t=1 should be filled before t=2 at same price."""
        book.add_limit_order(make_order("FIRST", "BID", 0.500, 5, 1, "id1"))
        book.add_limit_order(make_order("SECOND", "BID", 0.500, 5, 2, "id2"))
        # Aggressive ask that only fills 5 contracts
        trades = book.add_limit_order(make_order("SELLER", "ASK", 0.495, 5, 3))
        assert len(trades) == 1
        assert trades[0]["buyer_id"] == "FIRST"

    def test_better_price_filled_before_worse_price(self, book):
        """Higher bid filled before lower bid."""
        book.add_limit_order(make_order("LOW", "BID", 0.490, 5, 1))
        book.add_limit_order(make_order("HIGH", "BID", 0.510, 5, 2))
        trades = book.add_limit_order(make_order("SELLER", "ASK", 0.480, 5, 3))
        assert trades[0]["buyer_id"] == "HIGH"


class TestCancelOrder:
    def test_cancel_existing_order_returns_true(self, book):
        order = make_order("A", "BID", 0.500, 10, 1, "ORDER1")
        book.add_limit_order(order)
        assert book.cancel_order("ORDER1") is True

    def test_cancel_nonexistent_order_returns_false(self, book):
        assert book.cancel_order("NONEXISTENT") is False

    def test_cancelled_order_not_in_book(self, book):
        order = make_order("A", "BID", 0.500, 10, 1, "ORDER1")
        book.add_limit_order(order)
        book.cancel_order("ORDER1")
        assert book.best_bid() is None


class TestMarketOrder:
    def test_market_buy_fills_against_asks(self, book):
        book.add_limit_order(make_order("MM", "ASK", 0.510, 10, 1))
        trades = book.add_market_order("BUYER", "BID", 5, 2)
        assert len(trades) == 1
        assert trades[0]["quantity"] == 5

    def test_market_sell_fills_against_bids(self, book):
        book.add_limit_order(make_order("MM", "BID", 0.490, 10, 1))
        trades = book.add_market_order("SELLER", "ASK", 5, 2)
        assert len(trades) == 1
        assert trades[0]["quantity"] == 5

    def test_market_order_empty_book_no_fills(self, book):
        trades = book.add_market_order("BUYER", "BID", 5, 1)
        assert len(trades) == 0


class TestGetTradeHistory:
    def test_returns_dataframe(self, book):
        import pandas as pd
        df = book.get_trade_history()
        assert isinstance(df, pd.DataFrame)

    def test_dataframe_has_required_columns(self, book):
        book.add_limit_order(make_order("A", "BID", 0.510, 5, 1))
        book.add_limit_order(make_order("B", "ASK", 0.505, 5, 2))
        df = book.get_trade_history()
        for col in ["price", "quantity", "buyer_id", "seller_id", "timestamp"]:
            assert col in df.columns

    def test_empty_book_returns_empty_dataframe(self, book):
        df = book.get_trade_history()
        assert len(df) == 0
