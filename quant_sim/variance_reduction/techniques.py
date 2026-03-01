"""
Variance reduction methods for binary GBM contract pricing.

Three techniques compared against crude Monte Carlo:
  1. Antithetic Variates  — exploit negative correlation of Z and -Z pairs
  2. Control Variates     — subtract out a known-mean correction term
  3. Stratified Sampling  — divide [0,1] into strata, sample uniformly within each
"""

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


class VarianceReducer:
    """Variance reduction for binary call pricing under GBM.

    Parameters
    ----------
    S0, mu, sigma, r, T : float
        GBM parameters (same as GBMSimulator).
    n_samples : int
        Total simulation draws.
    """

    def __init__(
        self,
        S0: float,
        mu: float,
        sigma: float,
        r: float = 0.05,
        T: float = 1.0,
        n_samples: int = 100_000,
    ):
        self.S0 = S0
        self.mu = mu
        self.sigma = sigma
        self.r = r
        self.T = T
        self.n_samples = n_samples
        self._drift = (mu - 0.5 * sigma**2) * T
        self._vol_sqrt_T = sigma * np.sqrt(T)

    def _terminal(self, Z: np.ndarray) -> np.ndarray:
        return self.S0 * np.exp(self._drift + self._vol_sqrt_T * Z)

    def _payoff(self, S_T: np.ndarray, K: float) -> np.ndarray:
        return (S_T > K).astype(float)

    def _crude(self, K: float) -> dict:
        Z = np.random.standard_normal(self.n_samples)
        S_T = self._terminal(Z)
        f = self._payoff(S_T, K)
        price = float(np.mean(f))
        var = float(np.var(f, ddof=1))
        return {"price": price, "variance": var, "std_error": np.sqrt(var / self.n_samples)}

    def price_antithetic(self, K: float) -> dict:
        """Antithetic variates: use Z and -Z pairs.

        Each pair yields Y_i = 0.5*(f(S^+) + f(S^-)), reducing variance
        by exploiting the negative correlation Cov(f(S^+), f(S^-)) < 0.
        """
        n_half = self.n_samples // 2
        Z = np.random.standard_normal(n_half)
        S_pos = self._terminal(Z)
        S_neg = self._terminal(-Z)
        Y = 0.5 * (self._payoff(S_pos, K) + self._payoff(S_neg, K))
        price = float(np.mean(Y))
        var = float(np.var(Y, ddof=1))
        std_error = float(np.sqrt(var / n_half))

        crude = self._crude(K)
        vrr = crude["variance"] / var if var > 0 else float("inf")

        return {
            "price": price,
            "variance": var,
            "std_error": std_error,
            "variance_reduction_ratio": float(vrr),
        }

    def price_control_variate(self, K: float) -> dict:
        """Control variates using S_T as the control (E[S_T] = S0*exp(mu*T) known).

        Uses a pilot sample of n/10 to estimate the OLS beta, then applies
        correction to the remaining 9n/10 draws to avoid bias.
        """
        n_pilot = max(1000, self.n_samples // 10)
        n_main = self.n_samples - n_pilot

        E_Y = self.S0 * np.exp(self.mu * self.T)  # known analytic expectation of S_T

        # Pilot: estimate beta
        Z_p = np.random.standard_normal(n_pilot)
        S_T_p = self._terminal(Z_p)
        f_p = self._payoff(S_T_p, K)
        beta = float(np.cov(f_p, S_T_p)[0, 1] / np.var(S_T_p, ddof=1))

        # Main: apply correction
        Z_m = np.random.standard_normal(n_main)
        S_T_m = self._terminal(Z_m)
        f_m = self._payoff(S_T_m, K)
        f_adj = f_m - beta * (S_T_m - E_Y)

        price = float(np.mean(f_adj))
        var = float(np.var(f_adj, ddof=1))
        std_error = float(np.sqrt(var / n_main))

        crude = self._crude(K)
        vrr = crude["variance"] / var if var > 0 else float("inf")

        return {
            "price": price,
            "variance": var,
            "std_error": std_error,
            "beta": float(beta),
            "variance_reduction_ratio": float(vrr),
        }

    def price_stratified(self, K: float, n_strata: int = 100) -> dict:
        """Stratified sampling: divide [0,1] into n_strata equal bins.

        Within stratum k covering [(k-1)/m, k/m]:
          U_k ~ Uniform([(k-1)/m, k/m])
          Z_k = norm.ppf(U_k)  (inverse CDF transform)
        Guarantees uniform coverage of the normal distribution.
        """
        n_per_stratum = self.n_samples // n_strata
        all_payoffs = []

        for k in range(n_strata):
            lo = k / n_strata
            hi = (k + 1) / n_strata
            U = np.random.uniform(lo, hi, n_per_stratum)
            Z = scipy_stats.norm.ppf(U)
            S_T = self._terminal(Z)
            all_payoffs.append(self._payoff(S_T, K))

        stratum_means = np.array([p.mean() for p in all_payoffs])
        price = float(np.mean(stratum_means))

        # Variance of the stratified estimator
        stratum_vars = np.array([np.var(p, ddof=1) / len(p) for p in all_payoffs])
        var_estimate = float(np.mean(stratum_vars) / n_strata)
        std_error = float(np.sqrt(var_estimate))

        crude = self._crude(K)
        # Compare variances of the mean estimators (both scaled to n_samples)
        crude_var_of_mean = crude["variance"] / self.n_samples
        vrr = crude_var_of_mean / var_estimate if var_estimate > 0 else float("inf")

        return {
            "price": price,
            "variance": float(np.mean([np.var(p, ddof=1) for p in all_payoffs])),
            "std_error": std_error,
            "stratum_estimates": stratum_means,
            "variance_reduction_ratio": float(vrr),
        }

    def compare_all(self, K: float) -> pd.DataFrame:
        """Run all four methods and return a comparison DataFrame."""
        crude = self._crude(K)
        antithetic = self.price_antithetic(K)
        cv = self.price_control_variate(K)
        stratified = self.price_stratified(K)

        rows = [
            {"Method": "Crude MC", "Price": crude["price"], "Variance": crude["variance"],
             "Std Error": crude["std_error"], "Variance Reduction": 1.0},
            {"Method": "Antithetic", "Price": antithetic["price"], "Variance": antithetic["variance"],
             "Std Error": antithetic["std_error"], "Variance Reduction": antithetic["variance_reduction_ratio"]},
            {"Method": "Control Variate", "Price": cv["price"], "Variance": cv["variance"],
             "Std Error": cv["std_error"], "Variance Reduction": cv["variance_reduction_ratio"]},
            {"Method": "Stratified", "Price": stratified["price"], "Variance": stratified["variance"],
             "Std Error": stratified["std_error"], "Variance Reduction": stratified["variance_reduction_ratio"]},
        ]
        return pd.DataFrame(rows)


def run_variance_reduction_demo(
    S0: float = 100.0,
    mu: float = 0.10,
    sigma: float = 0.25,
    r: float = 0.05,
    T: float = 0.5,
    strike_pct: float = 1.05,
    n_samples: int = 100_000,
    seed: int = 42,
    no_plots: bool = False,
    save_plots: str = None,
):
    """Demo: Compare four pricing methods for a binary call at K = 1.05*S0."""
    np.random.seed(seed)
    K = strike_pct * S0

    print("=" * 70)
    print("  Variance Reduction — Binary Call Pricing Comparison")
    print("=" * 70)
    print(f"  S0={S0}  mu={mu:.2%}  sigma={sigma:.2%}  T={T:.2f}y  K={K:.1f}  N={n_samples:,}")
    print()

    reducer = VarianceReducer(S0=S0, mu=mu, sigma=sigma, r=r, T=T, n_samples=n_samples)
    df = reducer.compare_all(K)

    print(df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print()
