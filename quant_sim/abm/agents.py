"""
Agent types for the prediction market ABM.

Three heterogeneous agent types interact via a limit order book:
  - InformedAgent:     has access to the "true" event probability; trades on mispricing
  - NoiseAgent:        trades randomly; provides liquidity; has no information
  - MarketMakerAgent:  continuously quotes bid/ask spread; manages inventory
"""

import numpy as np
import uuid


class Agent:
    """Base agent class."""

    def __init__(self, agent_id: str, cash: float, inventory: int = 0):
        self.agent_id = agent_id
        self.cash = cash
        self.inventory = inventory
        self.trades_executed = 0
        self.realized_pnl = 0.0

    def decide(self, market_state: dict) -> list:
        """Return a list of order dicts (may be empty or have multiple orders).

        Each order dict: {'action': 'BUY'|'SELL'|'PASS', 'price': float,
                          'quantity': int, 'order_type': 'LIMIT'|'MARKET'}
        """
        raise NotImplementedError

    def update_pnl(self, trade: dict, role: str) -> None:
        """Update cash and inventory after a fill.

        Parameters
        ----------
        trade : dict
            Trade record with 'price' and 'quantity'.
        role : str
            'buyer' or 'seller'.
        """
        qty = trade["quantity"]
        price = trade["price"]
        if role == "buyer":
            self.cash -= price * qty
            self.inventory += qty
        else:
            self.cash += price * qty
            self.inventory -= qty
        self.trades_executed += 1


class InformedAgent(Agent):
    """Trades on privileged knowledge of the true event probability.

    Buys when market underprices (mid < true_prob - epsilon),
    sells when market overprices (mid > true_prob + epsilon).
    Order size is proportional to the mispricing magnitude.

    Parameters
    ----------
    true_value : float
        Current estimate of true event probability.
    epsilon : float
        Minimum mispricing to act on.
    position_limit : int
        Maximum absolute net inventory.
    """

    def __init__(
        self,
        agent_id: str,
        cash: float,
        true_value: float,
        epsilon: float = 0.005,
        position_limit: int = 50,
        noise_std: float = 0.01,
    ):
        super().__init__(agent_id, cash)
        self.true_value = true_value
        self.epsilon = epsilon
        self.position_limit = position_limit
        self.noise_std = noise_std

    def decide(self, market_state: dict) -> list:
        mid = market_state.get("mid_price", 0.5)
        best_bid = market_state.get("best_bid", mid - 0.01)
        best_ask = market_state.get("best_ask", mid + 0.01)
        if best_bid is None:
            best_bid = mid - 0.01
        if best_ask is None:
            best_ask = mid + 0.01

        # Add small noise to simulate private signal uncertainty
        perceived_value = self.true_value + np.random.normal(0, self.noise_std)
        perceived_value = float(np.clip(perceived_value, 0.01, 0.99))

        mispricing = perceived_value - mid

        if mispricing > self.epsilon and self.inventory < self.position_limit:
            # Market underpriced: buy
            qty = max(1, int(self.position_limit * abs(mispricing)))
            qty = min(qty, self.position_limit - self.inventory)
            if qty <= 0:
                return []
            price = float(np.clip(perceived_value * (1 - 0.002), 0.001, 0.999))
            return [{"action": "BUY", "price": price, "quantity": qty, "order_type": "LIMIT"}]

        elif mispricing < -self.epsilon and self.inventory > -self.position_limit:
            # Market overpriced: sell
            qty = max(1, int(self.position_limit * abs(mispricing)))
            qty = min(qty, self.position_limit + self.inventory)
            if qty <= 0:
                return []
            price = float(np.clip(perceived_value * (1 + 0.002), 0.001, 0.999))
            return [{"action": "SELL", "price": price, "quantity": qty, "order_type": "LIMIT"}]

        return []


class NoiseAgent(Agent):
    """Trades randomly with no information content.

    Provides market liquidity and prevents the book from being
    purely informationally efficient.

    Parameters
    ----------
    trade_prob : float
        Probability of submitting an order each time step.
    price_std : float
        Std of price noise around mid-price.
    max_qty : int
        Maximum order size.
    """

    def __init__(
        self,
        agent_id: str,
        cash: float,
        trade_prob: float = 0.3,
        price_std: float = 0.02,
        max_qty: int = 5,
    ):
        super().__init__(agent_id, cash)
        self.trade_prob = trade_prob
        self.price_std = price_std
        self.max_qty = max_qty

    def decide(self, market_state: dict) -> list:
        if np.random.random() > self.trade_prob:
            return []

        mid = market_state.get("mid_price", 0.5)
        if mid is None:
            mid = 0.5

        action = "BUY" if np.random.random() < 0.5 else "SELL"
        price = float(np.clip(mid + np.random.normal(0, self.price_std), 0.001, 0.999))
        qty = np.random.randint(1, self.max_qty + 1)
        return [{"action": action, "price": price, "quantity": qty, "order_type": "LIMIT"}]


class MarketMakerAgent(Agent):
    """Continuously posts bid/ask quotes with a fixed spread.

    Skews quotes to reduce inventory when approaching the inventory limit.

    Parameters
    ----------
    spread : float
        Half-spread on each side (total spread = 2 * spread).
    quote_size : int
        Number of contracts quoted per side.
    inventory_limit : int
        Maximum absolute inventory; triggers quote skewing at 80%.
    """

    def __init__(
        self,
        agent_id: str,
        cash: float,
        spread: float = 0.015,
        quote_size: int = 10,
        inventory_limit: int = 100,
    ):
        super().__init__(agent_id, cash)
        self.spread = spread
        self.quote_size = quote_size
        self.inventory_limit = inventory_limit
        self._pending_order_ids = []

    def decide(self, market_state: dict) -> list:
        mid = market_state.get("mid_price", 0.5)
        if mid is None:
            mid = 0.5

        # Inventory skew: push quotes to reduce position
        skew = 0.0
        inv_ratio = self.inventory / self.inventory_limit if self.inventory_limit > 0 else 0
        if abs(inv_ratio) > 0.6:
            skew = inv_ratio * self.spread * 0.5

        bid_price = float(np.clip(mid - self.spread - skew, 0.001, 0.998))
        ask_price = float(np.clip(mid + self.spread - skew, bid_price + 0.001, 0.999))

        return [
            {"action": "BUY", "price": bid_price, "quantity": self.quote_size, "order_type": "LIMIT"},
            {"action": "SELL", "price": ask_price, "quantity": self.quote_size, "order_type": "LIMIT"},
        ]
