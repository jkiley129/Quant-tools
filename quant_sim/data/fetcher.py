"""
yfinance wrapper for downloading historical OHLCV price data.
"""

import pandas as pd


def fetch_prices(ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    """Download historical OHLCV data for a single ticker.

    Parameters
    ----------
    ticker : str
        Stock symbol, e.g. 'NVDA'
    period : str
        Lookback period accepted by yfinance: '1mo', '3mo', '6mo', '1y', '2y', '5y', 'max'
    interval : str
        Bar interval: '1d', '1wk', '1mo'

    Returns
    -------
    pd.DataFrame
        DataFrame with DatetimeIndex and columns ['Open', 'High', 'Low', 'Close', 'Volume']
    """
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError(
            "yfinance is required for data fetching. Install it with: pip install yfinance"
        )

    data = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
    if data.empty:
        raise ValueError(f"No data returned for ticker '{ticker}'. Check the symbol and period.")

    # Flatten multi-level columns if present (yfinance >= 0.2.x sometimes returns them)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    return data[["Open", "High", "Low", "Close", "Volume"]].copy()


def fetch_multi(tickers: list, period: str = "1y") -> pd.DataFrame:
    """Download aligned Close prices for multiple tickers.

    Parameters
    ----------
    tickers : list[str]
        List of stock symbols, e.g. ['NVDA', 'AMD', 'SMCI']
    period : str
        Lookback period (same options as fetch_prices)

    Returns
    -------
    pd.DataFrame
        DataFrame with DatetimeIndex and one column per ticker (Close prices).
        Rows with any NaN are dropped so all series are co-aligned.
    """
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError(
            "yfinance is required for data fetching. Install it with: pip install yfinance"
        )

    if len(tickers) == 1:
        df = fetch_prices(tickers[0], period=period)
        return df[["Close"]].rename(columns={"Close": tickers[0]})

    data = yf.download(tickers, period=period, progress=False, auto_adjust=True)

    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"]
    else:
        close = data[["Close"]]

    close = close.dropna()

    if close.empty:
        raise ValueError(f"No overlapping data for tickers: {tickers}")

    return close
