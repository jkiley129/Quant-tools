"""
Prediction market orchestrator: runs the agent-based market simulation loop.

The PredictionMarket manages:
  - Heterogeneous agent populations (informed, noise, market maker)
  - A central limit order book
  - A pre-computed "true probability" path (from GBM simulation)
  - Per-step agent decision → order submission → LOB matching → trade recording
"""

import uuid
import numpy as np
import pandas as pd

from quant_sim.abm.agents import InformedAgent, NoiseAgent, MarketMakerAgent
from quant_sim.abm.order_book import LimitOrderBook, Order


def _new_id() -> str:
    return str(uuid.uuid4())[:8]


class PredictionMarket:
    """Agent-Based Model of a binary prediction market.

    Parameters
    ----------
    true_prob_path : np.ndarray
        Pre-computed "true" probability series of shape (n_steps,).
        Represents the latent event probability that informed agents observe.
    n_informed : int
        Number of InformedAgent instances.
    n_noise : int
        Number of NoiseAgent instances.
    n_market_makers : int
        Number of MarketMakerAgent instances.
    initial_price : float
        Starting mid-price of the contract (binary: resolves to 0 or 1).
    seed : int
        Random seed.
    """

    def __init__(
        self,
        true_prob_path: np.ndarray,
        n_informed: int = 3,
        n_noise: int = 10,
        n_market_makers: int = 2,
        initial_price: float = 0.5,
        seed: int = 42,
    ):
        np.random.seed(seed)
        self.true_prob_path = np.asarray(true_prob_path, dtype=float)
        self.n_steps = len(true_prob_path)
        self.initial_price = initial_price

        self.book = LimitOrderBook(tick_size=0.001)
        self.agents = []
        self._initialize_agents(n_informed, n_noise, n_market_makers)
        self._initialize_book()

        self.price_history = [initial_price]
        self.spread_history = []
        self.step_records = []

    def _initialize_agents(self, n_informed: int, n_noise: int, n_mm: int) -> None:
        for i in range(n_informed):
            self.agents.append(InformedAgent(
                agent_id=f"INF_{i}",
                cash=10_000.0,
                true_value=float(self.true_prob_path[0]),
                epsilon=0.005,
                position_limit=50,
            ))
        for i in range(n_noise):
            trade_prob = np.random.uniform(0.2, 0.5)
            self.agents.append(NoiseAgent(
                agent_id=f"NOISE_{i}",
                cash=10_000.0,
                trade_prob=trade_prob,
                price_std=0.025,
            ))
        for i in range(n_mm):
            self.agents.append(MarketMakerAgent(
                agent_id=f"MM_{i}",
                cash=50_000.0,
                spread=0.015,
                quote_size=10,
                inventory_limit=100,
            ))

    def _initialize_book(self) -> None:
        """Seed the book with initial market maker quotes."""
        p = self.initial_price
        for i in range(2):
            bid = Order(
                order_id=_new_id(), agent_id="SEED",
                side="BID", price=p - 0.02 * (i + 1), quantity=20, timestamp=0,
            )
            ask = Order(
                order_id=_new_id(), agent_id="SEED",
                side="ASK", price=p + 0.02 * (i + 1), quantity=20, timestamp=0,
            )
            self.book.add_limit_order(bid)
            self.book.add_limit_order(ask)

    def _build_market_state(self, t: int) -> dict:
        mid = self.book.mid_price()
        if mid is None:
            mid = self.price_history[-1]
        return {
            "best_bid": self.book.best_bid(),
            "best_ask": self.book.best_ask(),
            "mid_price": mid,
            "true_prob": float(self.true_prob_path[t]),
            "time_step": t,
            "price_history": self.price_history[-20:],
        }

    def _update_informed_true_values(self, t: int) -> None:
        true_prob = float(self.true_prob_path[t])
        for agent in self.agents:
            if isinstance(agent, InformedAgent):
                agent.true_value = true_prob

    def run(self) -> pd.DataFrame:
        """Run the full simulation.

        Returns
        -------
        pd.DataFrame
            Trade-by-trade history: ['timestamp', 'price', 'quantity', 'buyer_id', 'seller_id']
        """
        for t in range(self.n_steps):
            self._update_informed_true_values(t)
            market_state = self._build_market_state(t)

            # Shuffle agent order to prevent systematic bias
            agent_order = list(range(len(self.agents)))
            np.random.shuffle(agent_order)

            step_trades = []
            for idx in agent_order:
                agent = self.agents[idx]
                orders = agent.decide(market_state)

                for order_dict in orders:
                    if order_dict["action"] == "PASS":
                        continue

                    side = "BID" if order_dict["action"] == "BUY" else "ASK"
                    order = Order(
                        order_id=_new_id(),
                        agent_id=agent.agent_id,
                        side=side,
                        price=float(order_dict["price"]),
                        quantity=int(order_dict["quantity"]),
                        timestamp=t,
                    )

                    if order_dict["order_type"] == "MARKET":
                        trades = self.book.add_market_order(
                            agent.agent_id, side, order.quantity, t
                        )
                    else:
                        trades = self.book.add_limit_order(order)

                    for trade in trades:
                        step_trades.append(trade)
                        # Update agent P&L
                        for a in self.agents:
                            if a.agent_id == trade["buyer_id"]:
                                a.update_pnl(trade, "buyer")
                            elif a.agent_id == trade["seller_id"]:
                                a.update_pnl(trade, "seller")

            # Record state
            mid = self.book.mid_price()
            if mid is None:
                mid = self.price_history[-1]
            self.price_history.append(mid)

            spread = self.book.spread()
            self.spread_history.append(spread if spread is not None else 0.0)

            self.step_records.append({
                "t": t,
                "mid_price": mid,
                "true_prob": float(self.true_prob_path[t]),
                "n_trades": len(step_trades),
                "spread": spread,
            })

        return self.book.get_trade_history()

    def get_price_discovery_metrics(self) -> dict:
        """Analyze how well the market price tracked the true probability."""
        if not self.step_records:
            return {}

        records_df = pd.DataFrame(self.step_records)
        price_err = records_df["mid_price"] - records_df["true_prob"]
        rmse = float(np.sqrt(np.mean(price_err**2)))

        # Convergence half-life: time for |error| to decay below half of initial error
        init_err = abs(float(price_err.iloc[0]))
        if init_err > 0:
            half_target = init_err / 2
            half_life = None
            for i, e in enumerate(abs(price_err)):
                if e <= half_target:
                    half_life = i
                    break
        else:
            half_life = 0

        # Agent P&L summary
        agent_pnl = {}
        for agent in self.agents:
            final_price = self.price_history[-1]
            mark_to_market = agent.inventory * final_price
            total_pnl = (agent.cash - (10_000.0 if not isinstance(agent, MarketMakerAgent) else 50_000.0)) + mark_to_market
            agent_type = type(agent).__name__
            if agent_type not in agent_pnl:
                agent_pnl[agent_type] = []
            agent_pnl[agent_type].append(total_pnl)

        mean_spread = float(np.nanmean([s for s in self.spread_history if s is not None]))

        return {
            "price_error_rmse": rmse,
            "convergence_half_life": half_life,
            "bid_ask_spread_mean": mean_spread,
            "agent_pnl_by_type": {k: float(np.mean(v)) for k, v in agent_pnl.items()},
        }


def run_abm_demo(
    S0: float = 100.0,
    mu: float = 0.08,
    sigma: float = 0.20,
    T: float = 0.5,
    n_steps: int = 300,
    n_informed: int = 3,
    n_noise: int = 10,
    n_market_makers: int = 2,
    seed: int = 42,
    no_plots: bool = False,
    save_plots: str = None,
):
    """Demo: Run ABM prediction market and analyze price discovery."""
    from quant_sim.monte_carlo.gbm import GBMSimulator
    from quant_sim.utils.plotting import plot_abm_price_discovery

    np.random.seed(seed)

    # Generate true probability path from GBM
    sim = GBMSimulator(S0=S0, mu=mu, sigma=sigma, r=0.05, T=T, n_paths=10_000, seed=seed)
    paths = sim.simulate_paths(n_steps=n_steps)

    # True prob = fraction of paths above S0 * 1.05 at each step (rolling estimate)
    K = S0 * 1.05
    true_prob_path = np.mean(paths > K, axis=1)[1:]  # shape (n_steps,)

    print("=" * 65)
    print("  Agent-Based Market — Price Discovery Simulation")
    print("=" * 65)
    print(f"  Agents: {n_informed} informed + {n_noise} noise + {n_market_makers} market makers")
    print(f"  Steps: {n_steps}  |  True prob range: [{true_prob_path.min():.3f}, {true_prob_path.max():.3f}]")
    print()

    market = PredictionMarket(
        true_prob_path=true_prob_path,
        n_informed=n_informed,
        n_noise=n_noise,
        n_market_makers=n_market_makers,
        initial_price=float(true_prob_path[0]),
        seed=seed,
    )

    trades_df = market.run()
    metrics = market.get_price_discovery_metrics()

    print(f"  Total trades executed: {len(trades_df)}")
    print(f"  Price error RMSE:      {metrics['price_error_rmse']:.4f}")
    print(f"  Convergence half-life: {metrics['convergence_half_life']} steps")
    print(f"  Mean bid-ask spread:   {metrics['bid_ask_spread_mean']:.4f}")
    print()
    print("  Agent P&L by type (mean across agents):")
    for agent_type, pnl in metrics["agent_pnl_by_type"].items():
        print(f"    {agent_type:<20}: {pnl:>+10.2f}")
    print()

    if not no_plots:
        steps = np.arange(n_steps)
        market_prices = np.array([r["mid_price"] for r in market.step_records])
        sp = f"{save_plots}/abm_price_discovery.png" if save_plots else None
        plot_abm_price_discovery(
            steps,
            market_prices,
            true_prob_path,
            title="ABM: Market Price Discovery vs True Probability",
            save_path=sp,
        )
