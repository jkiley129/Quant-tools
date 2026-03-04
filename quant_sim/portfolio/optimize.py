"""
Markowitz Mean-Variance Portfolio Optimization + Kelly Criterion.

Two approaches to portfolio construction:

1. Markowitz (1952) — minimize variance for a target return:
      min  w' Σ w
      s.t. μ' w ≥ target_return
           1' w  = 1
           w ≥ lb   (long-only by default)

   Sweeping target_return traces the efficient frontier.

2. Kelly Criterion (continuous-time, multi-asset):
      w* = Σ^{-1} (μ - r)

   Maximizes expected log-wealth. Unconstrained and highly sensitive
   to estimation error — fractional Kelly (½ Kelly) is the practical choice.

Requires: cvxpy   (pip install cvxpy)
"""

import numpy as np


def markowitz_optimize(
    mu: np.ndarray,
    cov: np.ndarray,
    target_return: float,
    allow_short: bool = False,
    max_weight: float = 1.0,
) -> dict:
    """Single-point Markowitz mean-variance optimization.

    Parameters
    ----------
    mu : np.ndarray
        Expected annual returns, shape (n,).
    cov : np.ndarray
        Annual covariance matrix, shape (n, n).
    target_return : float
        Minimum required portfolio expected return.
    allow_short : bool
        If False (default), weights are constrained to [0, max_weight].
    max_weight : float
        Maximum weight per asset (default 1.0 = no upper limit beyond long-only).

    Returns
    -------
    dict with keys: weights, portfolio_return, portfolio_vol, sharpe, status
    """
    try:
        import cvxpy as cp
    except ImportError as e:
        raise ImportError("cvxpy is required: pip install cvxpy") from e

    n = len(mu)
    w = cp.Variable(n)

    objective = cp.Minimize(cp.quad_form(w, cov))
    constraints = [cp.sum(w) == 1.0, mu @ w >= target_return]
    if not allow_short:
        constraints += [w >= 0.0, w <= max_weight]
    else:
        constraints += [w >= -0.3, w <= max_weight]

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.CLARABEL, warm_start=True)

    if prob.status not in ("optimal", "optimal_inaccurate") or w.value is None:
        return {"status": prob.status, "weights": None}

    weights = np.array(w.value)
    port_return = float(mu @ weights)
    port_vol = float(np.sqrt(weights @ cov @ weights))
    sharpe = (port_return - 0.0) / port_vol if port_vol > 1e-10 else 0.0

    return {
        "weights": weights,
        "portfolio_return": port_return,
        "portfolio_vol": port_vol,
        "sharpe": sharpe,
        "status": prob.status,
    }


def efficient_frontier(
    mu: np.ndarray,
    cov: np.ndarray,
    n_points: int = 50,
    allow_short: bool = False,
) -> dict:
    """Trace the efficient frontier by sweeping target returns.

    Returns
    -------
    dict with keys: returns, vols, sharpes, weights  (all lists of length n_points)
    """
    ret_min = float(np.min(mu))
    ret_max = float(np.max(mu))
    targets = np.linspace(ret_min * 1.01, ret_max * 0.99, n_points)

    frontier_returns, frontier_vols, frontier_sharpes, frontier_weights = [], [], [], []

    for target in targets:
        result = markowitz_optimize(mu, cov, target, allow_short=allow_short)
        if result.get("weights") is not None:
            frontier_returns.append(result["portfolio_return"])
            frontier_vols.append(result["portfolio_vol"])
            frontier_sharpes.append(result["sharpe"])
            frontier_weights.append(result["weights"])

    return {
        "returns": frontier_returns,
        "vols": frontier_vols,
        "sharpes": frontier_sharpes,
        "weights": frontier_weights,
    }


def kelly_weights(
    mu: np.ndarray,
    cov: np.ndarray,
    r: float = 0.05,
    fraction: float = 1.0,
) -> dict:
    """Multi-asset Kelly criterion: maximise expected log-wealth.

    Full Kelly: w* = Σ^{-1} (μ - r)
    Fractional Kelly: w = fraction * w*

    Parameters
    ----------
    mu : np.ndarray
        Expected annual returns, shape (n,).
    cov : np.ndarray
        Annual covariance matrix, shape (n, n).
    r : float
        Risk-free rate (default 0.05).
    fraction : float
        Kelly fraction (0 < fraction ≤ 1). 0.5 = half-Kelly.

    Returns
    -------
    dict with keys: weights_raw (unnormalised), weights_normalised,
                    portfolio_return, portfolio_vol, sharpe
    """
    excess = mu - r
    cov_inv = np.linalg.pinv(cov)  # pseudo-inverse: handles near-singular cov
    w_raw = fraction * cov_inv @ excess

    # Normalise to sum-to-1 for comparability (optional; raw Kelly is leveraged)
    w_norm = w_raw / np.sum(np.abs(w_raw)) if np.sum(np.abs(w_raw)) > 1e-10 else w_raw

    port_return = float(mu @ w_norm)
    port_vol = float(np.sqrt(np.maximum(w_norm @ cov @ w_norm, 0.0)))
    sharpe = (port_return - r) / port_vol if port_vol > 1e-10 else 0.0

    return {
        "weights_raw": w_raw,
        "weights_normalised": w_norm,
        "portfolio_return": port_return,
        "portfolio_vol": port_vol,
        "sharpe": sharpe,
    }


def run_portfolio_demo(
    tickers: list = None,
    period: str = "3y",
    r: float = 0.05,
    allow_short: bool = False,
    seed: int = 42,
    no_plots: bool = False,
    save_plots: str = None,
):
    """Demo: Markowitz efficient frontier + Kelly weights.

    Uses real tickers via yfinance if provided, else synthetic data.
    """
    np.random.seed(seed)

    print("=" * 65)
    print("  Markowitz Portfolio Optimization + Kelly Criterion")
    print("=" * 65)

    # ── Data ─────────────────────────────────────────────────────────────────
    if tickers and len(tickers) >= 2:
        from quant_sim.data.fetcher import fetch_multi
        from quant_sim.data.calibrator import calibrate_correlation
        print(f"  [data] Fetching {tickers} ({period})...")
        prices = fetch_multi(tickers, period=period)
        log_ret = np.log(prices).diff().dropna()
        mu = log_ret.mean().values * 252
        cov = log_ret.cov().values * 252
        asset_names = tickers
    else:
        print("  [data] Using synthetic 5-asset example...")
        np.random.seed(seed)
        n = 5
        mu = np.array([0.08, 0.10, 0.12, 0.07, 0.15])
        A = np.random.randn(n, n) * 0.1
        cov = A @ A.T + np.diag([0.04, 0.06, 0.09, 0.03, 0.12])
        asset_names = [f"Asset {i+1}" for i in range(n)]

    n = len(mu)
    print(f"  {n} assets: {asset_names}")
    print()

    # ── Efficient frontier ────────────────────────────────────────────────────
    frontier = efficient_frontier(mu, cov, n_points=60, allow_short=allow_short)

    # Max-Sharpe point
    best_idx = int(np.argmax(frontier["sharpes"]))
    best_r  = frontier["returns"][best_idx]
    best_v  = frontier["vols"][best_idx]
    best_sr = frontier["sharpes"][best_idx]
    best_w  = frontier["weights"][best_idx]

    print("  Efficient Frontier:")
    print(f"    Max-Sharpe point:  Return={best_r:.2%}  Vol={best_v:.2%}  Sharpe={best_sr:.3f}")
    print()
    print(f"  Max-Sharpe portfolio weights:")
    for name, w in zip(asset_names, best_w):
        print(f"    {name:<12}  {w:+.4f}  {'▓' * max(0, int(abs(w) * 30))}")
    print()

    # ── Kelly criterion ───────────────────────────────────────────────────────
    for label, frac in [("Full Kelly (f=1.0)", 1.0), ("Half Kelly (f=0.5)", 0.5)]:
        kw = kelly_weights(mu, cov, r=r, fraction=frac)
        print(f"  {label}:")
        print(f"    Return={kw['portfolio_return']:.2%}  Vol={kw['portfolio_vol']:.2%}"
              f"  Sharpe={kw['sharpe']:.3f}")
        for name, w in zip(asset_names, kw["weights_normalised"]):
            print(f"    {name:<12}  {w:+.4f}")
        print()

    # ── Plots ─────────────────────────────────────────────────────────────────
    if not no_plots:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle("Markowitz Efficient Frontier")

        vols = np.array(frontier["vols"])
        rets = np.array(frontier["returns"])
        srs  = np.array(frontier["sharpes"])

        sc = axes[0].scatter(vols * 100, rets * 100, c=srs, cmap="viridis", s=20)
        axes[0].scatter([best_v * 100], [best_r * 100], marker="*", s=300,
                        color="red", zorder=5, label=f"Max Sharpe ({best_sr:.2f})")
        plt.colorbar(sc, ax=axes[0], label="Sharpe Ratio")
        axes[0].set_xlabel("Annualised Volatility (%)")
        axes[0].set_ylabel("Annualised Return (%)")
        axes[0].set_title("Efficient Frontier")
        axes[0].legend()

        axes[1].barh(asset_names, best_w, color=["steelblue" if w >= 0 else "salmon" for w in best_w])
        axes[1].axvline(0, color="black", lw=0.8)
        axes[1].set_xlabel("Weight")
        axes[1].set_title("Max-Sharpe Portfolio Weights")

        plt.tight_layout()
        if save_plots:
            plt.savefig(f"{save_plots}/portfolio_frontier.png", dpi=150)
        else:
            plt.show()
        plt.close()
