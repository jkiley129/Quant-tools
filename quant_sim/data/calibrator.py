"""
GBM parameter calibration from historical price data.
"""

import numpy as np
import pandas as pd

from quant_sim.data.fetcher import fetch_multi


def calibrate_gbm(prices: pd.Series, trading_days: int = 252) -> dict:
    """Estimate GBM drift (mu) and volatility (sigma) from a historical price series.

    Uses daily log-returns to estimate annualized parameters.  The drift is
    Ito-corrected so that E[S_T] = S0 * exp(mu * T) under the physical measure.

    Parameters
    ----------
    prices : pd.Series
        Time-ordered close prices (at least 30 observations recommended).
    trading_days : int
        Number of trading days per year (default 252).

    Returns
    -------
    dict
        {'S0': float, 'mu': float, 'sigma': float}
        S0  — most recent price
        mu  — annualized drift (Ito-corrected)
        sigma — annualized volatility
    """
    log_returns = np.log(prices / prices.shift(1)).dropna().values

    sigma_daily = np.std(log_returns, ddof=1)
    mu_daily = np.mean(log_returns)

    sigma_annual = sigma_daily * np.sqrt(trading_days)
    # Ito correction: E[log(S_T/S_0)] = (mu - 0.5*sigma^2)*T
    # => mu = mean_log_return * T + 0.5 * sigma^2
    mu_annual = mu_daily * trading_days + 0.5 * sigma_annual**2

    return {
        "S0": float(prices.iloc[-1]),
        "mu": float(mu_annual),
        "sigma": float(sigma_annual),
    }


def calibrate_correlation(prices_df: pd.DataFrame) -> np.ndarray:
    """Compute the log-return correlation matrix across multiple assets.

    Parameters
    ----------
    prices_df : pd.DataFrame
        DataFrame where each column is a Close price series (co-aligned dates).

    Returns
    -------
    np.ndarray
        (d, d) Pearson correlation matrix of log-returns.
    """
    log_returns = np.log(prices_df / prices_df.shift(1)).dropna()
    return log_returns.corr().values


def calibrate_multi(tickers: list, period: str = "1y") -> dict:
    """Fetch and calibrate GBM parameters for multiple tickers simultaneously.

    Parameters
    ----------
    tickers : list[str]
        Stock symbols, e.g. ['NVDA', 'AMD', 'SMCI']
    period : str
        Lookback period for yfinance (e.g. '1y', '2y')

    Returns
    -------
    dict
        {
            'tickers': list[str],
            'params': list[dict],        # per-ticker {'S0', 'mu', 'sigma'}
            'correlation_matrix': np.ndarray,  # (d, d)
        }
    """
    prices_df = fetch_multi(tickers, period=period)

    params = []
    for ticker in tickers:
        params.append(calibrate_gbm(prices_df[ticker]))

    corr_matrix = calibrate_correlation(prices_df)

    return {
        "tickers": tickers,
        "params": params,
        "correlation_matrix": corr_matrix,
    }
