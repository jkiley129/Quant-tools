"""
Logarithmic Market Scoring Rule (LMSR) — Robin Hanson (2003).

LMSR is the automated market maker powering Polymarket and other
prediction markets. Unlike a limit order book, it provides infinite
liquidity at all times: you can always buy or sell any outcome.

Mathematics
-----------
Cost function for n outcomes with outstanding shares q = (q₁, …, qₙ):

    C(q) = b · ln( Σᵢ exp(qᵢ / b) )

Price (probability) of outcome i:

    pᵢ = ∂C/∂qᵢ = exp(qᵢ/b) / Σⱼ exp(qⱼ/b)   ← softmax(q/b)

Prices always sum to 1 and lie in (0, 1). The market maker's worst-case
loss is bounded: max_loss = b · ln(n).

Cost to buy Δq shares of outcome i:

    cost = C(q + Δq·eᵢ) - C(q)

Positive cost = paying; negative cost = receiving (selling).
"""

import numpy as np


class LMSRMarket:
    """Automated market maker using the Logarithmic Market Scoring Rule.

    Parameters
    ----------
    n_outcomes : int
        Number of mutually exclusive outcomes (e.g. 2 for a binary market).
    b : float
        Liquidity parameter. Higher b → flatter prices, higher max loss.
        Typical choice: b = n * initial_subsidy.
    outcome_names : list[str], optional
        Human-readable outcome labels.
    """

    def __init__(
        self,
        n_outcomes: int = 2,
        b: float = 100.0,
        outcome_names: list = None,
    ):
        self.n = n_outcomes
        self.b = b
        self.q = np.zeros(n_outcomes)   # outstanding shares per outcome
        self.outcome_names = outcome_names or [f"Outcome {i}" for i in range(n_outcomes)]

        # History tracking
        self.price_history = [self.prices().tolist()]
        self.cost_history: list[float] = []
        self.trade_log: list[dict] = []

    # ── Core LMSR mathematics ─────────────────────────────────────────────────

    def _log_sum_exp(self, q: np.ndarray) -> float:
        """Numerically stable log-sum-exp."""
        max_q = np.max(q / self.b)
        return self.b * (max_q + np.log(np.sum(np.exp(q / self.b - max_q))))

    def cost_function(self, q: np.ndarray = None) -> float:
        """C(q) = b · ln(Σᵢ exp(qᵢ/b))."""
        if q is None:
            q = self.q
        return self._log_sum_exp(q)

    def prices(self, q: np.ndarray = None) -> np.ndarray:
        """Current prices (probabilities) via softmax(q/b).

        Returns
        -------
        np.ndarray, shape (n,), sums to 1.
        """
        if q is None:
            q = self.q
        shifted = q / self.b - np.max(q / self.b)
        exp_q = np.exp(shifted)
        return exp_q / exp_q.sum()

    def trade_cost(self, outcome: int, delta_shares: float) -> float:
        """Cost to buy (delta_shares > 0) or sell (delta_shares < 0) shares.

        Parameters
        ----------
        outcome : int
            Index of the outcome to trade.
        delta_shares : float
            Number of shares. Positive = buy, negative = sell.

        Returns
        -------
        float
            Cost in dollars. Positive = trader pays; negative = trader receives.
        """
        q_new = self.q.copy()
        q_new[outcome] += delta_shares
        return self.cost_function(q_new) - self.cost_function(self.q)

    def trade(self, outcome: int, delta_shares: float, trader_id: str = "anon") -> dict:
        """Execute a trade and update market state.

        Parameters
        ----------
        outcome : int
        delta_shares : float
            Positive = buy, negative = sell (short-selling allowed).
        trader_id : str
            Optional label for logging.

        Returns
        -------
        dict with cost, prices_before, prices_after
        """
        prices_before = self.prices().copy()
        cost = self.trade_cost(outcome, delta_shares)
        self.q[outcome] += delta_shares
        prices_after = self.prices().copy()

        self.price_history.append(prices_after.tolist())
        self.cost_history.append(cost)
        self.trade_log.append({
            "trader": trader_id,
            "outcome": outcome,
            "outcome_name": self.outcome_names[outcome],
            "delta_shares": delta_shares,
            "cost": cost,
            "price_before": float(prices_before[outcome]),
            "price_after": float(prices_after[outcome]),
        })

        return {
            "cost": cost,
            "prices_before": prices_before,
            "prices_after": prices_after,
        }

    @property
    def max_loss(self) -> float:
        """Market maker's worst-case loss = b · ln(n)."""
        return self.b * np.log(self.n)

    def summary(self) -> dict:
        """Current market state summary."""
        p = self.prices()
        return {
            "prices": {self.outcome_names[i]: float(p[i]) for i in range(self.n)},
            "outstanding_shares": {self.outcome_names[i]: float(self.q[i]) for i in range(self.n)},
            "n_trades": len(self.trade_log),
            "max_loss": self.max_loss,
            "total_cost_collected": float(sum(self.cost_history)),
        }


def run_lmsr_demo(
    true_prob: float = 0.65,
    n_traders: int = 200,
    b: float = 50.0,
    seed: int = 42,
    no_plots: bool = False,
    save_plots: str = None,
):
    """Demo: LMSR binary prediction market.

    Simulates informed traders who know the true outcome probability,
    plus noise traders, and shows how prices converge to the truth.

    Parameters
    ----------
    true_prob : float
        True probability of outcome 0 (e.g. "Yes").
    n_traders : int
        Number of traders in the simulation.
    b : float
        LMSR liquidity parameter.
    """
    np.random.seed(seed)

    print("=" * 65)
    print("  LMSR Prediction Market (Logarithmic Market Scoring Rule)")
    print("=" * 65)
    print(f"  True probability of YES = {true_prob:.2%}")
    print(f"  Liquidity parameter b   = {b}")
    print(f"  Max market-maker loss   = b·ln(n) = {b * np.log(2):.2f}")
    print()

    market = LMSRMarket(n_outcomes=2, b=b, outcome_names=["YES", "NO"])

    # Simulate traders: 60% informed, 40% noise
    for t in range(n_traders):
        is_informed = np.random.random() < 0.60
        if is_informed:
            # Informed: bet on whichever outcome is mispriced
            p = market.prices()
            mispricing_yes = true_prob - p[0]
            if abs(mispricing_yes) < 0.01:
                continue  # price is fair, no trade
            outcome = 0 if mispricing_yes > 0 else 1
            size = np.random.uniform(1.0, 5.0) * abs(mispricing_yes) * 10
            market.trade(outcome, size, trader_id=f"informed_{t}")
        else:
            # Noise trader: random direction, small size
            outcome = np.random.randint(0, 2)
            size = np.random.uniform(0.5, 2.0) * np.random.choice([-1, 1])
            market.trade(outcome, size, trader_id=f"noise_{t}")

    # Final state
    s = market.summary()
    final_p = s["prices"]["YES"]
    print(f"  After {s['n_trades']} trades:")
    print(f"    YES price (market):  {final_p:.4f}")
    print(f"    YES prob  (truth):   {true_prob:.4f}")
    print(f"    Error:               {abs(final_p - true_prob):.4f}")
    print(f"    Cost collected:      ${s['total_cost_collected']:.2f}")
    print()

    # LMSR math verification: softmax property
    p = market.prices()
    print(f"  Softmax verification: prices sum to {p.sum():.8f} (must be 1.0)")
    print(f"  Max market-maker loss: ${market.max_loss:.2f}")
    print()

    # Price history
    history = np.array(market.price_history)  # (n_trades+1, 2)
    n_hist = len(history)
    print(f"  Price convergence: started at {history[0,0]:.3f} → ended at {history[-1,0]:.3f}"
          f"  (truth={true_prob:.3f})")
    print()

    if not no_plots:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(13, 4))
        fig.suptitle(f"LMSR Prediction Market  (true p={true_prob:.2%})")

        # Price path
        trades = range(n_hist)
        axes[0].plot(trades, history[:, 0], label="P(YES)", color="steelblue")
        axes[0].axhline(true_prob, color="red", linestyle="--", label=f"Truth = {true_prob:.2%}")
        axes[0].axhline(0.5, color="gray", linestyle=":", linewidth=0.8, label="50%")
        axes[0].set_xlabel("Trade Number")
        axes[0].set_ylabel("Market Price")
        axes[0].set_title("Price Discovery")
        axes[0].legend()
        axes[0].set_ylim(0, 1)

        # Cost of buying function (for information)
        delta_range = np.linspace(0.1, 20, 200)
        costs_yes = [market.trade_cost(0, d) for d in delta_range]
        costs_no  = [market.trade_cost(1, d) for d in delta_range]
        axes[1].plot(delta_range, costs_yes, label="Cost to buy YES", color="steelblue")
        axes[1].plot(delta_range, costs_no,  label="Cost to buy NO",  color="darkorange")
        axes[1].set_xlabel("Shares Purchased")
        axes[1].set_ylabel("Cost ($)")
        axes[1].set_title("LMSR Trade Cost Function (convex)")
        axes[1].legend()

        plt.tight_layout()
        if save_plots:
            plt.savefig(f"{save_plots}/lmsr_market.png", dpi=150)
        else:
            plt.show()
        plt.close()
