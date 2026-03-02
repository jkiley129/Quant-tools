"""
Importance Sampling for rare/tail-risk events in GBM-based binary contracts.

Exponential tilting shifts the GBM drift toward the target tail, drastically
reducing the number of samples needed to estimate rare event probabilities
while keeping the estimator unbiased via likelihood-ratio weighting.

Key numerical trick: compute importance weights in log-space, then apply
the log-sum-exp stabilization before normalizing, to prevent underflow.
"""

import time

import numpy as np
from scipy import stats as scipy_stats


def _log_lognormal_density(S_T: np.ndarray, S0: float, mu: float, sigma: float, T: float) -> np.ndarray:
    """Log-density of a lognormal distribution for S_T under drift mu."""
    log_s = np.log(S_T / S0)
    mean_log = (mu - 0.5 * sigma**2) * T
    std_log = sigma * np.sqrt(T)
    return (
        -0.5 * np.log(2 * np.pi)
        - np.log(std_log * S_T)
        - 0.5 * ((log_s - mean_log) / std_log) ** 2
    )


class ImportanceSampler:
    """Importance sampler for rare events in a GBM framework.

    Parameters
    ----------
    S0 : float
        Initial price.
    mu : float
        Annualized drift under the physical measure.
    sigma : float
        Annualized volatility.
    r : float
        Risk-free rate (used for analytical baseline).
    T : float
        Time horizon in years.
    """

    def __init__(self, S0: float, mu: float, sigma: float, r: float = 0.05, T: float = 1.0):
        self.S0 = S0
        self.mu = mu
        self.sigma = sigma
        self.r = r
        self.T = T

    def _sample_from_proposal(self, n: int, mu_q: float) -> np.ndarray:
        """Sample terminal prices from GBM with proposal drift mu_q."""
        Z = np.random.standard_normal(n)
        return self.S0 * np.exp((mu_q - 0.5 * self.sigma**2) * self.T + self.sigma * np.sqrt(self.T) * Z)

    def estimate_tail_probability(
        self,
        threshold: float,
        n_samples: int = 100_000,
        shift_factor: float = 0.5,
    ) -> dict:
        """Estimate P(S_T > threshold) using exponential tilting.

        Proposal drift: mu_q = mu + shift_factor * sigma (toward the right tail).
        Importance weights in log-space for numerical stability.

        Parameters
        ----------
        threshold : float
            Target threshold (e.g., 1.5 * S0 for a 50% gain event).
        n_samples : int
            Number of IS samples.
        shift_factor : float
            How many sigma-units to shift the drift toward the tail.

        Returns
        -------
        dict
            {
              'is_estimate': float,
              'is_std_error': float,
              'naive_estimate': float,
              'analytical': float,
              'variance_reduction_ratio': float,
              'ess': float,
              'n_samples': int,
            }
        """
        mu_q = self.mu + shift_factor * self.sigma

        # --- IS estimation ---
        S_T_q = self._sample_from_proposal(n_samples, mu_q)

        log_p = _log_lognormal_density(S_T_q, self.S0, self.mu, self.sigma, self.T)
        log_q = _log_lognormal_density(S_T_q, self.S0, mu_q, self.sigma, self.T)
        log_w = log_p - log_q

        # Stabilize: subtract max before exp
        log_w_stable = log_w - np.max(log_w)
        raw_weights = np.exp(log_w_stable)

        indicator = (S_T_q > threshold).astype(float)

        # Unnormalized IS estimate (unbiased) — denominator is normalizing constant estimate
        is_num = np.mean(indicator * raw_weights)
        is_den = np.mean(raw_weights)
        is_estimate = float(is_num / is_den)

        # IS std error via delta method on normalized weights
        w_norm = raw_weights / raw_weights.sum()
        ess = float(1.0 / np.sum(w_norm**2))
        is_variance = np.sum(w_norm**2 * (indicator - is_estimate) ** 2)
        is_std_error = float(np.sqrt(is_variance / n_samples))

        # --- Naive Monte Carlo baseline ---
        S_T_naive = self._sample_from_proposal(n_samples, self.mu)
        naive_payoffs = (S_T_naive > threshold).astype(float)
        naive_estimate = float(np.mean(naive_payoffs))
        naive_variance = float(np.var(naive_payoffs, ddof=1))

        # Variance reduction ratio
        is_var_approx = float(is_std_error**2 * n_samples)
        vrr = float(naive_variance / is_var_approx) if is_var_approx > 0 else float("inf")

        # Analytical (Black-Scholes survival probability under physical measure)
        d2 = (np.log(self.S0 / threshold) + (self.mu - 0.5 * self.sigma**2) * self.T) / (
            self.sigma * np.sqrt(self.T)
        )
        analytical = float(scipy_stats.norm.cdf(d2))

        return {
            "is_estimate": is_estimate,
            "is_std_error": is_std_error,
            "naive_estimate": naive_estimate,
            "analytical": analytical,
            "variance_reduction_ratio": vrr,
            "ess": ess,
            "n_samples": n_samples,
        }

    def estimate_cvar(self, alpha: float = 0.05, n_samples: int = 100_000) -> dict:
        """Estimate CVaR (Expected Shortfall) at level alpha using IS.

        Uses a shifted distribution to oversample the lower alpha-tail.

        Returns
        -------
        dict
            {'cvar': float, 'var': float, 'alpha': float}
        """
        # Shift drift downward to oversample losses
        shift_factor = 0.5
        mu_q = self.mu - shift_factor * self.sigma

        S_T_q = self._sample_from_proposal(n_samples, mu_q)
        log_p = _log_lognormal_density(S_T_q, self.S0, self.mu, self.sigma, self.T)
        log_q = _log_lognormal_density(S_T_q, self.S0, mu_q, self.sigma, self.T)
        log_w = log_p - log_q
        log_w -= np.max(log_w)
        w = np.exp(log_w)
        w_norm = w / w.sum()

        # Weighted quantile for VaR
        sorted_idx = np.argsort(S_T_q)
        S_sorted = S_T_q[sorted_idx]
        w_sorted = w_norm[sorted_idx]
        cum_w = np.cumsum(w_sorted)
        var_idx = np.searchsorted(cum_w, alpha)
        var = float(S_sorted[min(var_idx, n_samples - 1)])

        # CVaR: weighted mean of losses below VaR
        tail_mask = S_T_q <= var
        if np.sum(tail_mask) == 0:
            cvar = var
        else:
            cvar = float(np.sum(w_norm[tail_mask] * S_T_q[tail_mask]) / np.sum(w_norm[tail_mask]))

        return {"cvar": cvar, "var": var, "alpha": alpha}

    def compare_naive_vs_is(self, threshold: float, n_samples: int = 100_000) -> dict:
        """Run both naive MC and IS; return side-by-side comparison."""
        # Naive
        t0 = time.perf_counter()
        S_T_naive = self._sample_from_proposal(n_samples, self.mu)
        naive_payoffs = (S_T_naive > threshold).astype(float)
        naive_estimate = float(np.mean(naive_payoffs))
        naive_std = float(np.std(naive_payoffs, ddof=1) / np.sqrt(n_samples))
        naive_time = time.perf_counter() - t0

        # IS
        t0 = time.perf_counter()
        result = self.estimate_tail_probability(threshold, n_samples=n_samples)
        is_time = time.perf_counter() - t0

        return {
            "naive_estimate": naive_estimate,
            "naive_std_error": naive_std,
            "naive_ci_95": (naive_estimate - 1.96 * naive_std, naive_estimate + 1.96 * naive_std),
            "naive_time_s": naive_time,
            "is_estimate": result["is_estimate"],
            "is_std_error": result["is_std_error"],
            "is_ci_95": (result["is_estimate"] - 1.96 * result["is_std_error"],
                         result["is_estimate"] + 1.96 * result["is_std_error"]),
            "is_ess": result["ess"],
            "is_time_s": is_time,
            "analytical": result["analytical"],
            "variance_reduction_ratio": result["variance_reduction_ratio"],
        }


def run_importance_sampling_demo(
    S0: float = 100.0,
    mu: float = 0.10,
    sigma: float = 0.25,
    r: float = 0.05,
    T: float = 0.5,
    threshold_multiple: float = 1.5,
    n_samples: int = 100_000,
    no_plots: bool = False,
    save_plots: str = None,
):
    """Demo: Estimate P(S_T > 1.5*S0) using IS vs naive Monte Carlo."""
    threshold = threshold_multiple * S0
    sampler = ImportanceSampler(S0=S0, mu=mu, sigma=sigma, r=r, T=T)

    print("=" * 65)
    print("  Importance Sampling — Tail Risk Estimation")
    print("=" * 65)
    print(f"  S0={S0}  mu={mu:.2%}  sigma={sigma:.2%}  T={T:.2f}y  Threshold={threshold:.1f} ({threshold_multiple:.0%}·S0)")
    print()

    comp = sampler.compare_naive_vs_is(threshold, n_samples=n_samples)

    print(f"  {'Method':<14}  {'Estimate':>10}  {'Std Error':>10}  {'95% CI Width':>14}  {'Time(s)':>8}")
    print("  " + "-" * 60)
    naive_ci_w = comp["naive_ci_95"][1] - comp["naive_ci_95"][0]
    is_ci_w = comp["is_ci_95"][1] - comp["is_ci_95"][0]
    print(f"  {'Naive MC':<14}  {comp['naive_estimate']:>10.6f}  {comp['naive_std_error']:>10.6f}  {naive_ci_w:>14.6f}  {comp['naive_time_s']:>8.3f}")
    print(f"  {'Import. Samp.':<14}  {comp['is_estimate']:>10.6f}  {comp['is_std_error']:>10.6f}  {is_ci_w:>14.6f}  {comp['is_time_s']:>8.3f}")
    print(f"  {'Analytical':<14}  {comp['analytical']:>10.6f}")
    print()
    print(f"  Variance reduction ratio: {comp['variance_reduction_ratio']:.1f}x")
    print(f"  Effective Sample Size:    {comp['is_ess']:.0f} / {n_samples:,}")
    print()

    # CVaR
    cvar_result = sampler.estimate_cvar(alpha=0.05, n_samples=n_samples)
    print(f"  CVaR(5%): ${cvar_result['cvar']:.2f}  |  VaR(5%): ${cvar_result['var']:.2f}")
    print()
