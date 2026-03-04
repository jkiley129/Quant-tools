"""
Vanilla Black-Scholes pricing and option Greeks.

Derives from the Black-Scholes PDE via Itô's lemma and delta-hedging.
The drift μ cancels: options are priced under the risk-neutral measure
where the stock grows at the risk-free rate r, not μ.

Closed-form call price:
    C = S·N(d1) - K·e^{-rT}·N(d2)

where
    d1 = [ln(S/K) + (r + σ²/2)·T] / (σ·√T)
    d2 = d1 - σ·√T

Greeks:
    Δ (delta)  = N(d1)                            — hedge ratio
    Γ (gamma)  = N'(d1) / (S·σ·√T)               — convexity
    Θ (theta)  = -[S·N'(d1)·σ/(2√T)] - r·K·e^{-rT}·N(d2)
    ν (vega)   = S·√T·N'(d1)
    ρ (rho)    = K·T·e^{-rT}·N(d2)
"""

import numpy as np
from scipy.stats import norm


def black_scholes(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
) -> float:
    """Vanilla European call or put price via Black-Scholes.

    Parameters
    ----------
    S : float
        Current spot price.
    K : float
        Strike price.
    T : float
        Time to expiry in years.
    r : float
        Continuously compounded risk-free rate.
    sigma : float
        Annualised implied volatility.
    option_type : {"call", "put"}

    Returns
    -------
    float
        Option price.
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        return float(S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))
    else:
        return float(K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))


def greeks(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
) -> dict:
    """Compute all five Black-Scholes Greeks analytically.

    Returns theta per calendar day and vega per 1-vol-point (1%).

    Parameters
    ----------
    S, K, T, r, sigma : floats — same as black_scholes()

    Returns
    -------
    dict with keys:
        delta_call, delta_put, gamma, theta_daily, vega_1pct, rho_1pct
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    pdf_d1 = norm.pdf(d1)
    cdf_d1 = norm.cdf(d1)
    cdf_d2 = norm.cdf(d2)

    delta_call = float(cdf_d1)
    delta_put = float(cdf_d1 - 1.0)
    gamma = float(pdf_d1 / (S * sigma * np.sqrt(T)))

    # Raw theta (annualised, for a call)
    theta_annual = float(
        -(S * pdf_d1 * sigma) / (2.0 * np.sqrt(T))
        - r * K * np.exp(-r * T) * cdf_d2
    )
    theta_daily = theta_annual / 365.0

    vega_raw = float(S * np.sqrt(T) * pdf_d1)
    vega_1pct = vega_raw / 100.0   # price change per 1% move in implied vol

    rho_raw = float(K * T * np.exp(-r * T) * cdf_d2)
    rho_1pct = rho_raw / 100.0     # price change per 1% move in rate

    return {
        "delta_call": delta_call,
        "delta_put": delta_put,
        "gamma": gamma,
        "theta_daily": theta_daily,
        "vega_1pct": vega_1pct,
        "rho_1pct": rho_1pct,
    }


def mc_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    n_sims: int = 500_000,
    option_type: str = "call",
    seed: int = None,
) -> dict:
    """Price a European option via risk-neutral Monte Carlo.

    Uses drift = r (risk-neutral measure), not the physical drift μ.

    Returns
    -------
    dict with keys: price, std_error, ci_95
    """
    if seed is not None:
        np.random.seed(seed)
    Z = np.random.standard_normal(n_sims)
    S_T = S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    if option_type == "call":
        payoffs = np.maximum(S_T - K, 0.0)
    else:
        payoffs = np.maximum(K - S_T, 0.0)
    discount = np.exp(-r * T)
    price = float(discount * np.mean(payoffs))
    stderr = float(discount * np.std(payoffs, ddof=1) / np.sqrt(n_sims))
    return {
        "price": price,
        "std_error": stderr,
        "ci_95": (price - 1.96 * stderr, price + 1.96 * stderr),
    }


def run_greeks_demo(
    S0: float = 100.0,
    r: float = 0.05,
    sigma: float = 0.20,
    T: float = 1.0,
    n_sims: int = 500_000,
    seed: int = 42,
    no_plots: bool = False,
    save_plots: str = None,
):
    """Demo: price calls and puts at multiple strikes, print Greeks table,
    verify Monte Carlo converges to Black-Scholes."""

    strikes = [K for K in [80, 90, 95, 100, 105, 110, 120]]

    print("=" * 70)
    print("  Vanilla Black-Scholes Pricing + Greeks")
    print("=" * 70)
    print(f"  S0={S0}  r={r:.2%}  σ={sigma:.2%}  T={T:.1f}y  N={n_sims:,}")
    print()

    # Pricing table: BS vs MC
    print(f"  {'Strike':>6}  {'Type':>4}  {'BS Price':>9}  {'MC Price':>9}  {'Std Err':>8}  {'Diff':>8}")
    print("  " + "-" * 56)
    for K in strikes:
        for opt_type in ("call", "put"):
            bs = black_scholes(S0, K, T, r, sigma, opt_type)
            mc = mc_price(S0, K, T, r, sigma, n_sims=n_sims, option_type=opt_type, seed=seed)
            diff = abs(bs - mc["price"])
            print(
                f"  {K:>6}  {opt_type:>4}  {bs:>9.4f}  {mc['price']:>9.4f}"
                f"  {mc['std_error']:>8.5f}  {diff:>8.5f}"
            )
    print()

    # Greeks table at-the-money
    K_atm = S0
    g = greeks(S0, K_atm, T, r, sigma)
    print(f"  Greeks at-the-money (K={K_atm}):")
    print(f"    Delta (call)  = {g['delta_call']:+.4f}   (put) = {g['delta_put']:+.4f}")
    print(f"    Gamma         = {g['gamma']:.6f}  (same for call and put)")
    print(f"    Theta/day     = {g['theta_daily']:+.4f}   (cost of holding one day)")
    print(f"    Vega/1%%      = {g['vega_1pct']:+.4f}   (price move per 1%% vol shift)")
    print(f"    Rho/1%%       = {g['rho_1pct']:+.4f}   (price move per 1%% rate shift)")
    print()

    # Plot: delta smile across strikes
    if not no_plots:
        import matplotlib.pyplot as plt
        Ks = np.linspace(70, 130, 200)
        calls = [black_scholes(S0, K, T, r, sigma, "call") for K in Ks]
        puts  = [black_scholes(S0, K, T, r, sigma, "put")  for K in Ks]
        deltas_c = [greeks(S0, K, T, r, sigma)["delta_call"] for K in Ks]
        gammas   = [greeks(S0, K, T, r, sigma)["gamma"]      for K in Ks]

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        fig.suptitle(f"Black-Scholes  S0={S0}  σ={sigma:.0%}  T={T:.1f}y  r={r:.0%}")

        axes[0].plot(Ks, calls, label="Call")
        axes[0].plot(Ks, puts, label="Put")
        axes[0].axvline(S0, color="gray", linestyle="--", linewidth=0.8)
        axes[0].set_title("Option Price vs Strike")
        axes[0].set_xlabel("Strike K")
        axes[0].legend()

        axes[1].plot(Ks, deltas_c, color="steelblue")
        axes[1].axvline(S0, color="gray", linestyle="--", linewidth=0.8)
        axes[1].axhline(0.5, color="gray", linestyle=":", linewidth=0.8)
        axes[1].set_title("Call Delta vs Strike")
        axes[1].set_xlabel("Strike K")
        axes[1].set_ylabel("Δ")

        axes[2].plot(Ks, gammas, color="darkorange")
        axes[2].axvline(S0, color="gray", linestyle="--", linewidth=0.8)
        axes[2].set_title("Gamma vs Strike")
        axes[2].set_xlabel("Strike K")
        axes[2].set_ylabel("Γ")

        plt.tight_layout()
        if save_plots:
            plt.savefig(f"{save_plots}/greeks_smile.png", dpi=150)
        else:
            plt.show()
        plt.close()
