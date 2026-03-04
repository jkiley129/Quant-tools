"""
Factor analysis: PCA on stock returns + Fama-French-style factor regression.

Key concepts from the post
--------------------------
PCA: "The first 5 eigenvectors explain 70% of all variance in a 500-stock universe."
     Eigendecomposition of the covariance matrix reveals latent risk factors.

Regression: "Regress strategy returns against known risk factors.
             The intercept α is your alpha — the return that cannot be explained
             by known factors."  Use Newey-West standard errors because financial
             data has autocorrelation and heteroskedasticity (OLS SEs are wrong).

Requires: scikit-learn>=1.4, statsmodels>=0.14
"""

import numpy as np
import pandas as pd


def pca_returns(
    returns: pd.DataFrame,
    n_components: int = None,
    variance_threshold: float = 0.90,
) -> dict:
    """PCA decomposition of multi-asset return data.

    Fits PCA on demeaned returns. Returns eigenvectors (factors),
    explained variance ratios, and factor scores (time series of loadings).

    Parameters
    ----------
    returns : pd.DataFrame
        T × N matrix of log returns (rows = dates, cols = assets).
    n_components : int, optional
        Number of principal components. If None, uses variance_threshold.
    variance_threshold : float
        Cumulative variance fraction to retain (used when n_components is None).

    Returns
    -------
    dict with keys:
        components       — shape (n_components, N): eigenvectors
        explained_var    — shape (n_components,): fraction of variance per PC
        cumulative_var   — shape (n_components,): cumulative fraction
        factor_scores    — shape (T, n_components): returns projected onto PCs
        n_components     — int: number of PCs selected
        eigenvalues      — shape (N,): all eigenvalues (descending)
    """
    try:
        from sklearn.decomposition import PCA
    except ImportError as e:
        raise ImportError("scikit-learn is required: pip install scikit-learn") from e

    n_assets = returns.shape[1]
    max_k = min(n_assets, len(returns) - 1)

    pca_full = PCA(n_components=max_k)
    pca_full.fit(returns.values)
    eigenvalues = pca_full.explained_variance_

    if n_components is None:
        cum_var = np.cumsum(pca_full.explained_variance_ratio_)
        n_components = int(np.searchsorted(cum_var, variance_threshold) + 1)
        n_components = min(n_components, max_k)

    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(returns.values)

    return {
        "components": pca.components_,                  # (n_components, N)
        "explained_var": pca.explained_variance_ratio_, # (n_components,)
        "cumulative_var": np.cumsum(pca.explained_variance_ratio_),
        "factor_scores": pd.DataFrame(
            scores,
            index=returns.index,
            columns=[f"PC{i+1}" for i in range(n_components)],
        ),
        "n_components": n_components,
        "eigenvalues": eigenvalues,
        "asset_names": list(returns.columns),
    }


def factor_regression(
    portfolio_returns: pd.Series,
    factor_returns: pd.DataFrame,
    max_lags: int = 5,
) -> dict:
    """OLS factor regression with Newey-West heteroskedasticity- and
    autocorrelation-consistent (HAC) standard errors.

    Model:
        r_t = α + β₁·F₁_t + β₂·F₂_t + … + ε_t

    α (alpha) = return unexplained by factors.
    βᵢ        = factor loading (sensitivity to factor i).

    Newey-West SEs correct for autocorrelation and heteroskedasticity,
    which are ubiquitous in financial time series.

    Parameters
    ----------
    portfolio_returns : pd.Series
        Strategy/portfolio daily log returns.
    factor_returns : pd.DataFrame
        Factor daily returns (same index), one column per factor.
    max_lags : int
        Maximum lag for Newey-West HAC correction (default 5 = one week).

    Returns
    -------
    dict with keys:
        alpha, alpha_tstat, alpha_pvalue
        betas           — dict {factor_name: beta}
        beta_tstats     — dict {factor_name: t-stat}
        beta_pvalues    — dict {factor_name: p-value}
        r_squared
        summary_str     — human-readable statsmodels summary
    """
    try:
        import statsmodels.api as sm
    except ImportError as e:
        raise ImportError("statsmodels is required: pip install statsmodels") from e

    aligned = pd.concat([portfolio_returns, factor_returns], axis=1).dropna()
    Y = aligned.iloc[:, 0].values
    X = sm.add_constant(aligned.iloc[:, 1:].values)
    factor_names = list(factor_returns.columns)

    model = sm.OLS(Y, X)
    # cov_type='HAC' = Newey-West; use_t=True for t-distribution p-values
    result = model.fit(cov_type="HAC", cov_kwds={"maxlags": max_lags}, use_t=True)

    params     = result.params
    tstats     = result.tvalues
    pvalues    = result.pvalues

    return {
        "alpha": float(params[0]),
        "alpha_annualised": float(params[0] * 252),
        "alpha_tstat": float(tstats[0]),
        "alpha_pvalue": float(pvalues[0]),
        "betas": {name: float(params[i + 1]) for i, name in enumerate(factor_names)},
        "beta_tstats": {name: float(tstats[i + 1]) for i, name in enumerate(factor_names)},
        "beta_pvalues": {name: float(pvalues[i + 1]) for i, name in enumerate(factor_names)},
        "r_squared": float(result.rsquared),
        "n_obs": int(result.nobs),
        "summary_str": str(result.summary()),
    }


def run_factors_demo(
    tickers: list = None,
    period: str = "3y",
    n_pca_components: int = 3,
    seed: int = 42,
    no_plots: bool = False,
    save_plots: str = None,
):
    """Demo: PCA on multi-asset returns + factor regression on a test portfolio."""
    np.random.seed(seed)

    print("=" * 65)
    print("  PCA Factor Analysis + Fama-French Style Regression")
    print("=" * 65)

    # ── Data ─────────────────────────────────────────────────────────────────
    if tickers and len(tickers) >= 4:
        from quant_sim.data.fetcher import fetch_multi
        print(f"  [data] Fetching {tickers} ({period})...")
        prices = fetch_multi(tickers, period=period)
        log_ret = np.log(prices).diff().dropna()
        print(f"  {len(log_ret):,} days  ×  {len(tickers)} assets")
    else:
        print("  [data] Simulating 10-asset correlated returns...")
        n, T = 10, 756  # 3 years of daily returns
        # 3 latent factors drive the returns
        F = np.random.randn(T, 3) * 0.01
        B = np.random.randn(n, 3) * 0.5
        eps = np.random.randn(T, n) * 0.01
        raw = F @ B.T + eps
        log_ret = pd.DataFrame(
            raw,
            columns=[f"Stock{i+1}" for i in range(n)],
            index=pd.date_range("2021-01-01", periods=T, freq="B"),
        )
        tickers = list(log_ret.columns)

    print()

    # ── PCA ───────────────────────────────────────────────────────────────────
    pca_result = pca_returns(log_ret, n_components=n_pca_components)
    n_pc = pca_result["n_components"]
    print(f"  PCA: {n_pc} principal components")
    print(f"  {'PC':<6}  {'Explained':>10}  {'Cumulative':>11}")
    print("  " + "-" * 32)
    for i in range(n_pc):
        ev  = pca_result["explained_var"][i]
        cum = pca_result["cumulative_var"][i]
        bar = "▓" * int(ev * 200)
        print(f"  PC{i+1:<4}  {ev:>9.2%}  {cum:>10.2%}  {bar}")
    print()

    # Top loadings for PC1
    pc1 = pca_result["components"][0]
    top_idx = np.argsort(np.abs(pc1))[::-1][:5]
    print(f"  PC1 top loadings (interpreted as 'market factor'):")
    for idx in top_idx:
        name = pca_result["asset_names"][idx]
        print(f"    {name:<12}  {pc1[idx]:+.4f}")
    print()

    # ── Factor regression ─────────────────────────────────────────────────────
    # Use first asset as "portfolio", PCA factors as regressors
    portfolio = log_ret.iloc[:, 0]
    factor_scores = pca_result["factor_scores"]

    reg = factor_regression(portfolio, factor_scores)
    print(f"  Factor regression: {tickers[0]} ~ α + β₁·PC1 + β₂·PC2 + β₃·PC3")
    print(f"  (Newey-West HAC standard errors)")
    print()
    print(f"    α (daily)        = {reg['alpha']:+.6f}"
          f"  (annualised: {reg['alpha_annualised']:+.2%})")
    print(f"    α t-stat         = {reg['alpha_tstat']:+.3f}"
          f"  p={reg['alpha_pvalue']:.4f}"
          f"  {'SIGNIFICANT' if reg['alpha_pvalue'] < 0.05 else 'not significant'}")
    for fname in list(factor_scores.columns):
        b  = reg["betas"][fname]
        t  = reg["beta_tstats"][fname]
        p  = reg["beta_pvalues"][fname]
        print(f"    β({fname:<4})         = {b:+.4f}  t={t:+.3f}  p={p:.4f}")
    print(f"    R²               = {reg['r_squared']:.4f}")
    print()

    # ── Plots ─────────────────────────────────────────────────────────────────
    if not no_plots:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle("PCA Factor Analysis")

        # Eigenvalue scree plot
        eigenvalues = pca_result["eigenvalues"]
        axes[0].bar(range(1, len(eigenvalues) + 1), eigenvalues, color="steelblue")
        axes[0].axvline(n_pc + 0.5, color="red", linestyle="--",
                        label=f"Keep {n_pc} PCs")
        axes[0].set_xlabel("Principal Component")
        axes[0].set_ylabel("Eigenvalue")
        axes[0].set_title("Scree Plot")
        axes[0].legend()

        # Cumulative explained variance
        cum = np.cumsum(eigenvalues / eigenvalues.sum())
        axes[1].plot(range(1, len(cum) + 1), cum * 100, marker="o", markersize=4)
        axes[1].axhline(90, color="gray", linestyle="--", label="90% threshold")
        axes[1].axvline(n_pc, color="red", linestyle="--", label=f"{n_pc} PCs selected")
        axes[1].set_xlabel("Number of Components")
        axes[1].set_ylabel("Cumulative Explained Variance (%)")
        axes[1].set_title("Cumulative Variance Explained")
        axes[1].legend()

        plt.tight_layout()
        if save_plots:
            plt.savefig(f"{save_plots}/factors_pca.png", dpi=150)
        else:
            plt.show()
        plt.close()
