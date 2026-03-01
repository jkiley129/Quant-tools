"""
Copula models for correlated multi-asset terminal price simulation.

Three copulas implemented from scratch using numpy + scipy only:
  1. GaussianCopula  — linear correlation, no tail dependence
  2. StudentTCopula  — symmetric upper + lower tail dependence
  3. ClaytonCopula   — lower tail dependence only (assets crash together)

All share simulate_asset_prices() which applies lognormal marginals.
"""

import numpy as np
from scipy import stats as scipy_stats
from scipy import linalg as scipy_linalg


class GaussianCopula:
    """Gaussian (Normal) copula.

    Captures linear correlation only; upper and lower tail events are
    asymptotically independent (tail dependence coefficient = 0).

    Parameters
    ----------
    rho_matrix : np.ndarray
        (d, d) correlation matrix.
    """

    def __init__(self, rho_matrix: np.ndarray):
        self.rho_matrix = np.asarray(rho_matrix, dtype=float)
        self.d = self.rho_matrix.shape[0]

    def sample_uniforms(self, n: int) -> np.ndarray:
        """Sample (n, d) uniform marginals from the Gaussian copula.

        Algorithm:
          1. Draw Z ~ MVN(0, rho_matrix)
          2. Transform to uniforms: U = Phi(Z)
        """
        Z = np.random.multivariate_normal(np.zeros(self.d), self.rho_matrix, size=n)
        return scipy_stats.norm.cdf(Z)

    def fit(self, data: np.ndarray) -> None:
        """Estimate rho_matrix from data using Gaussian rank correlation.

        Parameters
        ----------
        data : np.ndarray
            Shape (n, d). Raw data (will be rank-transformed).
        """
        n = data.shape[0]
        # Convert each margin to pseudo-uniforms via empirical CDF
        ranks = np.argsort(np.argsort(data, axis=0), axis=0)
        pseudo_u = (ranks + 0.5) / n
        # Back-transform to Gaussian scores
        Z = scipy_stats.norm.ppf(pseudo_u)
        self.rho_matrix = np.corrcoef(Z.T)

    def simulate_asset_prices(
        self,
        n: int,
        S0_list: list,
        mu_list: list,
        sigma_list: list,
        r: float,
        T: float,
    ) -> np.ndarray:
        """Simulate correlated terminal asset prices under lognormal marginals.

        Parameters
        ----------
        n : int
            Number of scenarios.
        S0_list, mu_list, sigma_list : list[float]
            Per-asset GBM parameters (length d).
        r : float
            Risk-free rate.
        T : float
            Time horizon.

        Returns
        -------
        np.ndarray
            Shape (n, d). Terminal prices for each asset.
        """
        U = self.sample_uniforms(n)
        S0 = np.array(S0_list)
        mu = np.array(mu_list)
        sigma = np.array(sigma_list)
        drift = (mu - 0.5 * sigma**2) * T
        vol_sqrt_T = sigma * np.sqrt(T)
        Z = scipy_stats.norm.ppf(U)
        return S0 * np.exp(drift + vol_sqrt_T * Z)


class StudentTCopula:
    """Student-t copula with symmetric tail dependence.

    Both upper and lower tails exhibit positive dependence (assets tend to
    move together in both extreme up and down scenarios).

    Parameters
    ----------
    rho_matrix : np.ndarray
        (d, d) correlation matrix.
    df : float
        Degrees of freedom. Lower df → heavier tails.
    """

    def __init__(self, rho_matrix: np.ndarray, df: float = 4.0):
        self.rho_matrix = np.asarray(rho_matrix, dtype=float)
        self.df = float(df)
        self.d = self.rho_matrix.shape[0]

    def sample_uniforms(self, n: int) -> np.ndarray:
        """Sample (n, d) uniform marginals from the Student-t copula.

        Algorithm (McNeil et al., 2005, Algorithm 5.10):
          1. Cholesky decompose: L = chol(rho_matrix)
          2. Draw Z ~ N(0, I_d), compute correlated Y = L @ Z^T
          3. Draw chi-squared W ~ Chi2(df); chi_rv = sqrt(df / W)
          4. Multivariate-t: X = chi_rv * Y^T
          5. Transform to uniforms: U = t.cdf(X, df)
        """
        L = scipy_linalg.cholesky(self.rho_matrix, lower=True)
        Z = np.random.standard_normal((self.d, n))
        Y = (L @ Z).T  # shape (n, d)

        W = np.random.chisquare(df=self.df, size=n)
        chi_rv = np.sqrt(self.df / W)
        X = chi_rv[:, np.newaxis] * Y  # shape (n, d)

        return scipy_stats.t.cdf(X, df=self.df)

    def simulate_asset_prices(
        self,
        n: int,
        S0_list: list,
        mu_list: list,
        sigma_list: list,
        r: float,
        T: float,
    ) -> np.ndarray:
        """Simulate correlated terminal prices under Student-t copula."""
        U = self.sample_uniforms(n)
        S0 = np.array(S0_list)
        mu = np.array(mu_list)
        sigma = np.array(sigma_list)
        drift = (mu - 0.5 * sigma**2) * T
        vol_sqrt_T = sigma * np.sqrt(T)
        Z = scipy_stats.norm.ppf(np.clip(U, 1e-8, 1 - 1e-8))
        return S0 * np.exp(drift + vol_sqrt_T * Z)


class ClaytonCopula:
    """Clayton copula with lower tail dependence only.

    Assets tend to crash together (correlated losses) but recover
    independently (tail dependence coefficient = 0 in the upper tail).

    Parameters
    ----------
    theta : float
        Dependence parameter, theta > 0. Higher = stronger lower-tail dependence.
        As theta -> 0: independence. As theta -> inf: complete dependence.
    d : int
        Dimension (number of assets).
    """

    def __init__(self, theta: float, d: int = 2):
        if theta <= 0:
            raise ValueError("Clayton copula requires theta > 0.")
        self.theta = float(theta)
        self.d = d

    def sample_uniforms(self, n: int) -> np.ndarray:
        """Sample (n, d) uniform marginals via the Marshall-Olkin algorithm.

        Algorithm (Marshall & Olkin, 1988):
          1. Draw V ~ Gamma(1/theta, 1)   — shared frailty variable
          2. For each dimension j: draw E_j ~ Exponential(1) independently
          3. U_j = (1 + E_j / V)^(-1/theta)  — uniform marginals
        """
        # Shape (n,): shared frailty
        V = np.random.gamma(shape=1.0 / self.theta, scale=1.0, size=n)
        # Shape (n, d): independent exponentials
        E = np.random.exponential(scale=1.0, size=(n, self.d))
        U = (1.0 + E / V[:, np.newaxis]) ** (-1.0 / self.theta)
        return np.clip(U, 1e-8, 1 - 1e-8)

    def simulate_asset_prices(
        self,
        n: int,
        S0_list: list,
        mu_list: list,
        sigma_list: list,
        r: float,
        T: float,
    ) -> np.ndarray:
        """Simulate correlated terminal prices under Clayton copula."""
        U = self.sample_uniforms(n)
        S0 = np.array(S0_list)
        mu = np.array(mu_list)
        sigma = np.array(sigma_list)
        drift = (mu - 0.5 * sigma**2) * T
        vol_sqrt_T = sigma * np.sqrt(T)
        Z = scipy_stats.norm.ppf(U)
        return S0 * np.exp(drift + vol_sqrt_T * Z)


def run_copula_demo(
    tickers: list = None,
    period: str = "1y",
    n_scenarios: int = 100_000,
    T: float = 0.5,
    r: float = 0.05,
    rho: float = 0.75,
    drop_threshold: float = 0.10,
    gain_threshold: float = 0.20,
    student_t_df: float = 4.0,
    clayton_theta: float = 2.0,
    seed: int = 42,
    no_plots: bool = False,
    save_plots: str = None,
):
    """Demo: Compare Gaussian, Student-t, and Clayton copulas.

    If `tickers` is provided, fetches real data and calibrates parameters.
    Otherwise uses synthetic 2-asset SPY/QQQ-like defaults.
    """
    from quant_sim.utils.plotting import plot_copula_scatter

    np.random.seed(seed)

    if tickers is not None:
        from quant_sim.data.calibrator import calibrate_multi
        print(f"  Fetching {tickers} data ({period})...")
        calib = calibrate_multi(tickers, period=period)
        S0_list = [p["S0"] for p in calib["params"]]
        mu_list = [p["mu"] for p in calib["params"]]
        sigma_list = [p["sigma"] for p in calib["params"]]
        corr_matrix = calib["correlation_matrix"]
        labels = tickers
        d = len(tickers)
    else:
        # Default: 2-asset SPY/QQQ-like
        d = 2
        S0_list = [450.0, 380.0]
        mu_list = [0.12, 0.15]
        sigma_list = [0.18, 0.22]
        corr_matrix = np.array([[1.0, rho], [rho, 1.0]])
        labels = ["Asset 1 (SPY-like)", "Asset 2 (QQQ-like)"]

    print("=" * 70)
    print("  Copula Models — Correlated Asset Simulation")
    print("=" * 70)
    for i, lbl in enumerate(labels):
        print(f"  {lbl}: S0={S0_list[i]:.2f}  mu={mu_list[i]:.2%}  sigma={sigma_list[i]:.2%}")
    print(f"  Scenarios={n_scenarios:,}  T={T}y  Correlation matrix:")
    print(np.round(corr_matrix, 3))
    print()

    copulas = [
        ("Gaussian", GaussianCopula(corr_matrix)),
        (f"Student-t (df={student_t_df})", StudentTCopula(corr_matrix, df=student_t_df)),
        (f"Clayton (θ={clayton_theta})", ClaytonCopula(theta=clayton_theta, d=d)),
    ]

    print(f"  {'Copula':<22}  {'Emp. Corr':>10}  {'P(all drop>{:.0%})':>{16}}  {'P(all gain>{:.0%})':>{16}}".format(drop_threshold, gain_threshold))
    print("  " + "-" * 70)

    for name, cop in copulas:
        prices = cop.simulate_asset_prices(n_scenarios, S0_list, mu_list, sigma_list, r, T)
        # Compute empirical correlation of terminal prices
        log_ret = np.log(prices / np.array(S0_list))
        if d == 2:
            emp_corr = float(np.corrcoef(log_ret.T)[0, 1])
        else:
            emp_corr = float(np.mean([np.corrcoef(log_ret.T)[i, j]
                                       for i in range(d) for j in range(i+1, d)]))

        # Joint drawdown probability
        drops = np.all(prices < np.array(S0_list) * (1 - drop_threshold), axis=1)
        p_drop = float(np.mean(drops))

        # Joint gain probability
        gains = np.all(prices > np.array(S0_list) * (1 + gain_threshold), axis=1)
        p_gain = float(np.mean(gains))

        print(f"  {name:<22}  {emp_corr:>10.4f}  {p_drop:>16.4f}  {p_gain:>16.4f}")

        if not no_plots and d == 2:
            U = cop.sample_uniforms(min(n_scenarios, 20_000))
            sp = f"{save_plots}/copula_{name.split()[0].lower()}.png" if save_plots else None
            plot_copula_scatter(U[:, 0], U[:, 1], title=f"{name} Copula — Uniform Marginals", save_path=sp)

    print()
