"""
Geometric Brownian Motion simulator for binary (digital) contract pricing.

The core formula is the closed-form GBM terminal price:
    S_T = S0 * exp((mu - 0.5*sigma^2)*T + sigma*sqrt(T)*Z),  Z ~ N(0,1)

The -0.5*sigma^2 Ito correction is critical and must not be omitted.
"""

import numpy as np
from scipy import stats as scipy_stats


class GBMSimulator:
    """Monte Carlo engine for GBM-based binary contract pricing.

    Parameters
    ----------
    S0 : float
        Initial asset price.
    mu : float
        Annualized drift (physical measure). Under risk-neutral measure use r - q.
    sigma : float
        Annualized volatility.
    r : float
        Risk-free rate (used for discounting and analytical cross-check).
    T : float
        Time to expiry in years.
    n_paths : int
        Number of Monte Carlo paths.
    seed : int, optional
        Random seed for reproducibility.
    """

    def __init__(
        self,
        S0: float,
        mu: float,
        sigma: float,
        r: float = 0.05,
        T: float = 1.0,
        n_paths: int = 100_000,
        seed: int = None,
    ):
        self.S0 = S0
        self.mu = mu
        self.sigma = sigma
        self.r = r
        self.T = T
        self.n_paths = n_paths
        if seed is not None:
            np.random.seed(seed)

    def simulate_terminal(self) -> np.ndarray:
        """Draw terminal prices directly using the closed-form GBM solution.

        No path discretization — exact for log-normal GBM.

        Returns
        -------
        np.ndarray
            Shape (n_paths,). Terminal prices S_T.
        """
        Z = np.random.standard_normal(self.n_paths)
        drift = (self.mu - 0.5 * self.sigma**2) * self.T
        diffusion = self.sigma * np.sqrt(self.T) * Z
        return self.S0 * np.exp(drift + diffusion)

    def simulate_paths(self, n_steps: int) -> np.ndarray:
        """Simulate full price paths via Euler-Maruyama discretization.

        Parameters
        ----------
        n_steps : int
            Number of time steps (dt = T / n_steps).

        Returns
        -------
        np.ndarray
            Shape (n_steps+1, n_paths). Row 0 is S0.
        """
        dt = self.T / n_steps
        Z = np.random.standard_normal((n_steps, self.n_paths))
        log_returns = (self.mu - 0.5 * self.sigma**2) * dt + self.sigma * np.sqrt(dt) * Z
        log_paths = np.vstack([np.zeros(self.n_paths), log_returns])
        log_paths = np.cumsum(log_paths, axis=0)
        return self.S0 * np.exp(log_paths)

    def price_binary_call(self, K: float, use_terminal: bool = True) -> dict:
        """Price a binary (digital) call option: pays $1 if S_T > K.

        Cross-checks Monte Carlo against the analytical Black-Scholes formula.

        Parameters
        ----------
        K : float
            Strike price.
        use_terminal : bool
            If True (default), use fast closed-form terminal draw.

        Returns
        -------
        dict
            {
              'mc_price': float,
              'std_error': float,
              'ci_95': (float, float),
              'analytical_price': float,
            }
        """
        if use_terminal:
            S_T = self.simulate_terminal()
        else:
            paths = self.simulate_paths(n_steps=252)
            S_T = paths[-1]

        payoffs = (S_T > K).astype(float)
        mc_price = float(np.mean(payoffs))
        std_error = float(np.std(payoffs, ddof=1) / np.sqrt(self.n_paths))
        ci_low = mc_price - 1.96 * std_error
        ci_high = mc_price + 1.96 * std_error

        # Analytical: physical probability P(S_T > K) = N(d2) under drift mu
        d2 = (np.log(self.S0 / K) + (self.mu - 0.5 * self.sigma**2) * self.T) / (
            self.sigma * np.sqrt(self.T)
        )
        analytical_price = float(scipy_stats.norm.cdf(d2))

        return {
            "mc_price": mc_price,
            "std_error": std_error,
            "ci_95": (ci_low, ci_high),
            "analytical_price": analytical_price,
        }

    def price_binary_put(self, K: float) -> dict:
        """Price a binary put: pays $1 if S_T < K.

        Returns
        -------
        dict
            Same structure as price_binary_call.
        """
        S_T = self.simulate_terminal()
        payoffs = (S_T < K).astype(float)
        mc_price = float(np.mean(payoffs))
        std_error = float(np.std(payoffs, ddof=1) / np.sqrt(self.n_paths))
        ci_low = mc_price - 1.96 * std_error
        ci_high = mc_price + 1.96 * std_error

        d2 = (np.log(self.S0 / K) + (self.mu - 0.5 * self.sigma**2) * self.T) / (
            self.sigma * np.sqrt(self.T)
        )
        analytical_price = float(scipy_stats.norm.cdf(-d2))

        return {
            "mc_price": mc_price,
            "std_error": std_error,
            "ci_95": (ci_low, ci_high),
            "analytical_price": analytical_price,
        }

    def get_exceedance_probability(self, threshold: float) -> float:
        """Estimate P(S_T > threshold) from simulation.

        Returns
        -------
        float
        """
        S_T = self.simulate_terminal()
        return float(np.mean(S_T > threshold))


def run_gbm_demo(
    S0: float = 182.0,
    mu: float = 0.12,
    sigma: float = 0.28,
    r: float = 0.05,
    T: float = 0.5,
    n_paths: int = 100_000,
    strikes: list = None,
    seed: int = 42,
    no_plots: bool = False,
    save_plots: str = None,
):
    """Demo: Price binary calls at multiple strikes for an AAPL-like stock."""
    from quant_sim.utils.plotting import plot_price_paths, plot_histogram

    if strikes is None:
        strikes = [175, 182, 190, 200, 210]

    print("=" * 65)
    print("  Monte Carlo GBM — Binary Contract Pricing")
    print("=" * 65)
    print(f"  S0={S0}  mu={mu:.2%}  sigma={sigma:.2%}  r={r:.2%}  T={T:.2f}y  N={n_paths:,}")
    print()

    sim = GBMSimulator(S0=S0, mu=mu, sigma=sigma, r=r, T=T, n_paths=n_paths, seed=seed)

    # Print pricing table
    header = f"{'Strike':>8}  {'MC Price':>9}  {'Analytical':>10}  {'Std Error':>9}  {'95% CI':>22}"
    print(header)
    print("-" * 65)
    for K in strikes:
        result = sim.price_binary_call(K)
        ci = result["ci_95"]
        print(
            f"  {K:>6}  {result['mc_price']:>9.4f}  {result['analytical_price']:>10.4f}"
            f"  {result['std_error']:>9.6f}  [{ci[0]:.4f}, {ci[1]:.4f}]"
        )
    print()

    # Plots (regenerate terminal prices for plotting)
    sim2 = GBMSimulator(S0=S0, mu=mu, sigma=sigma, r=r, T=T, n_paths=n_paths, seed=seed)
    paths = sim2.simulate_paths(n_steps=126)  # ~half year of daily steps
    terminal = paths[-1]

    if not no_plots:
        sp_paths = f"{save_plots}/gbm_paths.png" if save_plots else None
        sp_hist = f"{save_plots}/gbm_histogram.png" if save_plots else None
        plot_price_paths(paths, title=f"GBM Price Paths (S0={S0}, σ={sigma:.0%}, T={T}y)", save_path=sp_paths)
        plot_histogram(terminal, bins=100, title="Terminal Price Distribution", xlabel="S_T", save_path=sp_hist)
