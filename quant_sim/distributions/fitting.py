"""
Maximum Likelihood Estimation for fat-tailed return distributions,
normality testing, and permutation-based strategy significance tests.

Key insight from the post:
  "Most of what looks like signal is noise."
  Every beginner overestimates how much edge they've found.

Workflow
--------
1. Download real returns (yfinance)
2. Test normality — they fail (fat tails, excess kurtosis)
3. Fit Student-t via MLE — degrees of freedom quantifies fat-tail severity
4. Use permutation test to check if a strategy's Sharpe is above chance
"""

import numpy as np
from scipy import stats, optimize


def fit_normal(returns: np.ndarray) -> dict:
    """Fit a Gaussian via MLE (closed form: sample mean and std).

    Returns
    -------
    dict with keys: loc, scale, log_likelihood, aic
    """
    loc = float(np.mean(returns))
    scale = float(np.std(returns, ddof=1))
    ll = float(np.sum(stats.norm.logpdf(returns, loc=loc, scale=scale)))
    aic = 2 * 2 - 2 * ll  # 2 parameters
    return {"loc": loc, "scale": scale, "log_likelihood": ll, "aic": aic}


def fit_student_t(returns: np.ndarray) -> dict:
    """Fit a Student-t distribution via numerical MLE.

    The Student-t has heavier tails than a Gaussian. Low degrees of freedom
    (df < 5) indicate severe tail risk — common in equity returns.

    Parameters
    ----------
    returns : np.ndarray
        Array of log returns.

    Returns
    -------
    dict with keys: df, loc, scale, log_likelihood, aic, success
    """
    def neg_log_likelihood(params):
        df, loc, scale = params
        if df <= 2.0 or scale <= 0.0:
            return 1e10
        return -float(np.sum(stats.t.logpdf(returns, df=df, loc=loc, scale=scale)))

    x0 = [5.0, float(np.mean(returns)), float(np.std(returns, ddof=1))]
    result = optimize.minimize(neg_log_likelihood, x0, method="Nelder-Mead",
                               options={"maxiter": 10_000, "xatol": 1e-8})
    df, loc, scale = result.x
    ll = -result.fun
    aic = 2 * 3 - 2 * ll  # 3 parameters
    return {
        "df": float(df),
        "loc": float(loc),
        "scale": float(scale),
        "log_likelihood": float(ll),
        "aic": float(aic),
        "success": bool(result.success),
    }


def normality_tests(returns: np.ndarray) -> dict:
    """Run three normality tests on a return series.

    Tests
    -----
    - D'Agostino-Pearson (combined skew + kurtosis)
    - Shapiro-Wilk (sensitive to sample size; capped at 5000)
    - Jarque-Bera (asymptotic chi-squared, standard in finance)

    Returns
    -------
    dict with test statistics and p-values for each test, plus
    excess_kurtosis and skewness of the sample.
    """
    n = len(returns)

    # D'Agostino-Pearson
    dp_stat, dp_p = stats.normaltest(returns)

    # Shapiro-Wilk (accurate up to ~5000 obs)
    sw_sample = returns if n <= 5000 else returns[np.random.choice(n, 5000, replace=False)]
    sw_stat, sw_p = stats.shapiro(sw_sample)

    # Jarque-Bera
    jb_stat, jb_p = stats.jarque_bera(returns)

    return {
        "n": n,
        "skewness": float(stats.skew(returns)),
        "excess_kurtosis": float(stats.kurtosis(returns)),   # Fisher (normal=0)
        "dagostino_pearson": {"stat": float(dp_stat), "p_value": float(dp_p)},
        "shapiro_wilk":      {"stat": float(sw_stat), "p_value": float(sw_p)},
        "jarque_bera":       {"stat": float(jb_stat), "p_value": float(jb_p)},
        "reject_normality":  bool(dp_p < 0.05 or jb_p < 0.05),
    }


def permutation_test(
    signals: np.ndarray,
    returns: np.ndarray,
    n_perms: int = 10_000,
    seed: int = None,
) -> dict:
    """Test whether a trading signal beats random coin-flips.

    The null hypothesis is: the signal has no predictive power (random
    assignment of longs/shorts achieves the same Sharpe ratio).

    Method
    ------
    1. Compute the observed annualised Sharpe of signal * returns.
    2. Randomly permute the signal 10,000 times, keeping returns fixed.
    3. p-value = fraction of permutations that beat the observed Sharpe.

    Parameters
    ----------
    signals : np.ndarray
        +1 / -1 array. Positive = long, negative = short.
    returns : np.ndarray
        Asset log-returns aligned with signals.
    n_perms : int
        Number of permutation trials (default 10,000).
    seed : int, optional

    Returns
    -------
    dict with observed_sharpe, p_value, null_sharpe_mean, null_sharpe_std
    """
    if seed is not None:
        np.random.seed(seed)

    strat_returns = signals * returns
    observed_sharpe = float(
        np.mean(strat_returns) / (np.std(strat_returns, ddof=1) + 1e-12) * np.sqrt(252)
    )

    null_sharpes = np.empty(n_perms)
    for i in range(n_perms):
        perm = np.random.permutation(signals) * returns
        null_sharpes[i] = np.mean(perm) / (np.std(perm, ddof=1) + 1e-12) * np.sqrt(252)

    p_value = float(np.mean(null_sharpes >= observed_sharpe))

    return {
        "observed_sharpe": observed_sharpe,
        "p_value": p_value,
        "null_sharpe_mean": float(np.mean(null_sharpes)),
        "null_sharpe_std": float(np.std(null_sharpes)),
        "n_perms": n_perms,
        "significant_at_5pct": bool(p_value < 0.05),
    }


def run_distributions_demo(
    ticker: str = None,
    period: str = "5y",
    seed: int = 42,
    no_plots: bool = False,
    save_plots: str = None,
):
    """Demo: fetch real returns (or simulate), fit Normal vs Student-t via MLE,
    run normality tests, and run a permutation test on a momentum signal."""
    np.random.seed(seed)

    print("=" * 65)
    print("  MLE Distribution Fitting & Permutation Testing")
    print("=" * 65)

    # ── Fetch or simulate returns ─────────────────────────────────────────────
    if ticker:
        from quant_sim.data.fetcher import fetch_prices
        print(f"  [data] Fetching {ticker} ({period})...")
        prices = fetch_prices(ticker, period=period)
        log_returns = np.log(prices["Close"]).diff().dropna().values
        label = ticker
    else:
        print("  [data] Simulating Student-t returns (df=4)...")
        true_df = 4.0
        log_returns = stats.t.rvs(df=true_df, loc=0.0005, scale=0.015, size=1000,
                                  random_state=seed)
        label = "Synthetic t(df=4)"

    print(f"  {len(log_returns):,} observations")
    print()

    # ── Normality tests ───────────────────────────────────────────────────────
    nt = normality_tests(log_returns)
    print(f"  Normality tests on {label}:")
    print(f"    Skewness:          {nt['skewness']:+.4f}")
    print(f"    Excess kurtosis:   {nt['excess_kurtosis']:+.4f}  (normal=0, fat tails>0)")
    print(f"    D'Agostino-Pearson p={nt['dagostino_pearson']['p_value']:.2e}")
    print(f"    Shapiro-Wilk       p={nt['shapiro_wilk']['p_value']:.2e}")
    print(f"    Jarque-Bera        p={nt['jarque_bera']['p_value']:.2e}")
    print(f"    Reject normality?  {'YES — fat tails confirmed' if nt['reject_normality'] else 'NO'}")
    print()

    # ── MLE fitting ──────────────────────────────────────────────────────────
    norm_fit = fit_normal(log_returns)
    t_fit = fit_student_t(log_returns)
    print("  MLE fit comparison:")
    print(f"    Normal:    μ={norm_fit['loc']:.6f}  σ={norm_fit['scale']:.6f}"
          f"  AIC={norm_fit['aic']:.1f}")
    print(f"    Student-t: df={t_fit['df']:.2f}  μ={t_fit['loc']:.6f}"
          f"  σ={t_fit['scale']:.6f}  AIC={t_fit['aic']:.1f}")
    delta_aic = norm_fit["aic"] - t_fit["aic"]
    print(f"    ΔAIC (Normal - t) = {delta_aic:+.1f}"
          f"  → {'Student-t fits better' if delta_aic > 0 else 'Normal fits better'}")
    print()

    # ── Permutation test on a simple momentum signal ──────────────────────────
    signals = np.sign(np.convolve(log_returns, np.ones(5) / 5, mode="same"))
    signals[signals == 0] = 1.0

    perm = permutation_test(signals, log_returns, n_perms=5_000, seed=seed)
    print("  Permutation test — 5-day momentum signal:")
    print(f"    Observed Sharpe:   {perm['observed_sharpe']:+.3f}")
    print(f"    Null Sharpe mean:  {perm['null_sharpe_mean']:+.3f}  ± {perm['null_sharpe_std']:.3f}")
    print(f"    p-value:           {perm['p_value']:.4f}")
    print(f"    Significant?       {'YES' if perm['significant_at_5pct'] else 'NO — likely noise'}")
    print()

    # ── Plots ─────────────────────────────────────────────────────────────────
    if not no_plots:
        import matplotlib.pyplot as plt
        x = np.linspace(log_returns.min(), log_returns.max(), 500)

        fig, axes = plt.subplots(1, 2, figsize=(13, 4))
        fig.suptitle(f"Return Distribution Fitting — {label}")

        # Histogram + fitted densities
        axes[0].hist(log_returns, bins=60, density=True, alpha=0.5, color="steelblue",
                     label="Returns")
        axes[0].plot(x, stats.norm.pdf(x, norm_fit["loc"], norm_fit["scale"]),
                     "r-", lw=2, label=f"Normal (AIC={norm_fit['aic']:.0f})")
        axes[0].plot(x, stats.t.pdf(x, t_fit["df"], t_fit["loc"], t_fit["scale"]),
                     "g-", lw=2, label=f"Student-t df={t_fit['df']:.1f} (AIC={t_fit['aic']:.0f})")
        axes[0].set_xlabel("Log Return")
        axes[0].set_ylabel("Density")
        axes[0].set_title("MLE Distribution Fit")
        axes[0].legend(fontsize=8)

        # Permutation null distribution
        # Recompute null for plotting (small n_perms for speed)
        null_sharpes_plot = []
        for _ in range(2000):
            perm_r = np.random.permutation(signals) * log_returns
            null_sharpes_plot.append(
                np.mean(perm_r) / (np.std(perm_r, ddof=1) + 1e-12) * np.sqrt(252)
            )
        axes[1].hist(null_sharpes_plot, bins=50, density=True, alpha=0.6,
                     color="gray", label="Null Sharpe distribution")
        axes[1].axvline(perm["observed_sharpe"], color="red", lw=2,
                        label=f"Observed Sharpe = {perm['observed_sharpe']:.3f}")
        axes[1].set_xlabel("Annualised Sharpe")
        axes[1].set_title(f"Permutation Test  (p={perm['p_value']:.3f})")
        axes[1].legend(fontsize=8)

        plt.tight_layout()
        if save_plots:
            plt.savefig(f"{save_plots}/distributions.png", dpi=150)
        else:
            plt.show()
        plt.close()
