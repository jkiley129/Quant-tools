"""
Shared statistical primitives used across all quant_sim modules.
"""

import numpy as np
from scipy import stats as scipy_stats


def normal_cdf(x: np.ndarray) -> np.ndarray:
    """Standard normal CDF: Phi(x)."""
    return scipy_stats.norm.cdf(x)


def normal_ppf(u: np.ndarray) -> np.ndarray:
    """Inverse standard normal CDF (percent-point function): Phi^{-1}(u)."""
    return scipy_stats.norm.ppf(u)


def student_t_ppf(u: np.ndarray, df: float) -> np.ndarray:
    """Inverse Student-t CDF with `df` degrees of freedom."""
    return scipy_stats.t.ppf(u, df=df)


def log_likelihood_ratio(log_p: np.ndarray, log_q: np.ndarray) -> np.ndarray:
    """Compute log importance weight: log w = log p(x) - log q(x).

    Operates entirely in log space to avoid numerical underflow/overflow.

    Parameters
    ----------
    log_p : np.ndarray
        Log-density under the target distribution P.
    log_q : np.ndarray
        Log-density under the proposal distribution Q.

    Returns
    -------
    np.ndarray
        Log importance weights (unnormalized).
    """
    return log_p - log_q


def effective_sample_size(weights: np.ndarray) -> float:
    """Compute the Effective Sample Size (ESS) of a particle population.

    ESS = (sum(w))^2 / sum(w^2)

    For normalized weights that sum to 1, this simplifies to 1 / sum(w^2).

    Parameters
    ----------
    weights : np.ndarray
        Non-negative importance weights (need not sum to 1).

    Returns
    -------
    float
        ESS in the range [1, N].
    """
    w = np.asarray(weights, dtype=float)
    w_sum = np.sum(w)
    if w_sum == 0:
        return 0.0
    w_norm = w / w_sum
    return float(1.0 / np.sum(w_norm**2))


def systematic_resample(weights: np.ndarray, n: int) -> np.ndarray:
    """O(N) systematic resampling for particle filters.

    Draws one uniform u ~ U[0, 1/N] and takes positions
    u, u+1/N, u+2/N, ... against the cumulative weight CDF.

    This has lower variance than multinomial resampling.

    Parameters
    ----------
    weights : np.ndarray
        Non-negative weights (need not be normalized).
    n : int
        Number of particles to resample.

    Returns
    -------
    np.ndarray
        Integer index array of length n with selected particle indices.
    """
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    cumulative = np.cumsum(w)
    positions = (np.arange(n) + np.random.uniform(0, 1)) / n
    indices = np.searchsorted(cumulative, positions)
    return indices.astype(int)
