"""
Tests for quant_sim/copulas/models.py
"""

import numpy as np
import pytest
from scipy import stats as scipy_stats
from quant_sim.copulas.models import GaussianCopula, StudentTCopula, ClaytonCopula


RHO_2D = np.array([[1.0, 0.7], [0.7, 1.0]])
S0_LIST = [100.0, 80.0]
MU_LIST = [0.10, 0.12]
SIGMA_LIST = [0.20, 0.25]
R, T = 0.05, 1.0
N = 50_000


class TestGaussianCopula:
    def test_uniforms_shape(self):
        cop = GaussianCopula(RHO_2D)
        U = cop.sample_uniforms(N)
        assert U.shape == (N, 2)

    def test_uniforms_in_unit_interval(self):
        np.random.seed(0)
        cop = GaussianCopula(RHO_2D)
        U = cop.sample_uniforms(N)
        assert np.all(U >= 0) and np.all(U <= 1)

    def test_marginals_approximately_uniform(self):
        """KS test: each marginal should not reject H0 of Uniform(0,1)."""
        np.random.seed(1)
        cop = GaussianCopula(RHO_2D)
        U = cop.sample_uniforms(N)
        for j in range(2):
            stat, p = scipy_stats.kstest(U[:, j], "uniform")
            assert p > 0.01, f"Marginal {j} failed KS test: p={p:.4f}"

    def test_empirical_correlation_close_to_rho(self):
        """Pearson correlation of Gaussian scores should ≈ rho."""
        np.random.seed(2)
        cop = GaussianCopula(RHO_2D)
        U = cop.sample_uniforms(N)
        Z = scipy_stats.norm.ppf(U)
        emp_corr = np.corrcoef(Z.T)[0, 1]
        assert abs(emp_corr - 0.7) < 0.02

    def test_simulate_asset_prices_shape(self):
        np.random.seed(3)
        cop = GaussianCopula(RHO_2D)
        prices = cop.simulate_asset_prices(N, S0_LIST, MU_LIST, SIGMA_LIST, R, T)
        assert prices.shape == (N, 2)

    def test_asset_prices_positive(self):
        np.random.seed(4)
        cop = GaussianCopula(RHO_2D)
        prices = cop.simulate_asset_prices(N, S0_LIST, MU_LIST, SIGMA_LIST, R, T)
        assert np.all(prices > 0)

    def test_asset_price_means_close_to_expected(self):
        """E[S_T] = S0 * exp(mu * T)."""
        np.random.seed(5)
        cop = GaussianCopula(RHO_2D)
        prices = cop.simulate_asset_prices(N, S0_LIST, MU_LIST, SIGMA_LIST, R, T)
        for j in range(2):
            expected = S0_LIST[j] * np.exp(MU_LIST[j] * T)
            assert abs(np.mean(prices[:, j]) / expected - 1.0) < 0.02

    def test_fit_recovers_correlation(self):
        """fit() from data should recover the input correlation."""
        np.random.seed(6)
        cop = GaussianCopula(RHO_2D)
        data = cop.sample_uniforms(10_000)
        cop2 = GaussianCopula(np.eye(2))
        cop2.fit(data)
        assert abs(cop2.rho_matrix[0, 1] - 0.7) < 0.05

    def test_identity_rho_gives_independent_marginals(self):
        np.random.seed(7)
        cop = GaussianCopula(np.eye(2))
        U = cop.sample_uniforms(N)
        corr = np.corrcoef(U.T)[0, 1]
        assert abs(corr) < 0.02


class TestStudentTCopula:
    def test_uniforms_shape(self):
        cop = StudentTCopula(RHO_2D, df=4.0)
        U = cop.sample_uniforms(N)
        assert U.shape == (N, 2)

    def test_uniforms_in_unit_interval(self):
        np.random.seed(10)
        cop = StudentTCopula(RHO_2D, df=4.0)
        U = cop.sample_uniforms(N)
        assert np.all(U > 0) and np.all(U < 1)

    def test_marginals_approximately_uniform(self):
        np.random.seed(11)
        cop = StudentTCopula(RHO_2D, df=4.0)
        U = cop.sample_uniforms(N)
        for j in range(2):
            stat, p = scipy_stats.kstest(U[:, j], "uniform")
            assert p > 0.01, f"Marginal {j} failed KS test: p={p:.4f}"

    def test_higher_df_approaches_gaussian(self):
        """With very high df, Student-t copula ≈ Gaussian copula."""
        np.random.seed(12)
        cop_t = StudentTCopula(RHO_2D, df=1000)
        cop_g = GaussianCopula(RHO_2D)
        U_t = cop_t.sample_uniforms(N)
        U_g = cop_g.sample_uniforms(N)
        # Both should have similar correlation structure
        corr_t = np.corrcoef(scipy_stats.norm.ppf(U_t).T)[0, 1]
        corr_g = np.corrcoef(scipy_stats.norm.ppf(U_g).T)[0, 1]
        assert abs(corr_t - corr_g) < 0.03

    def test_asset_prices_positive(self):
        np.random.seed(13)
        cop = StudentTCopula(RHO_2D, df=4.0)
        prices = cop.simulate_asset_prices(N, S0_LIST, MU_LIST, SIGMA_LIST, R, T)
        assert np.all(prices > 0)

    def test_symmetric_tail_dependence(self):
        """Student-t: joint lower tail P(both<q) ≈ joint upper tail P(both>1-q)."""
        np.random.seed(14)
        cop = StudentTCopula(RHO_2D, df=4.0)
        U = cop.sample_uniforms(100_000)
        q = 0.05
        lower = np.mean(np.all(U < q, axis=1))
        upper = np.mean(np.all(U > 1 - q, axis=1))
        assert abs(lower - upper) / max(lower, upper) < 0.15  # within 15% of each other


class TestClaytonCopula:
    def test_uniforms_shape(self):
        cop = ClaytonCopula(theta=2.0, d=2)
        U = cop.sample_uniforms(N)
        assert U.shape == (N, 2)

    def test_uniforms_in_unit_interval(self):
        np.random.seed(20)
        cop = ClaytonCopula(theta=2.0, d=2)
        U = cop.sample_uniforms(N)
        assert np.all(U > 0) and np.all(U < 1)

    def test_marginals_approximately_uniform(self):
        np.random.seed(21)
        cop = ClaytonCopula(theta=2.0, d=2)
        U = cop.sample_uniforms(N)
        for j in range(2):
            stat, p = scipy_stats.kstest(U[:, j], "uniform")
            assert p > 0.01, f"Clayton marginal {j} failed KS test: p={p:.4f}"

    def test_invalid_theta_raises(self):
        with pytest.raises(ValueError):
            ClaytonCopula(theta=0.0, d=2)
        with pytest.raises(ValueError):
            ClaytonCopula(theta=-1.0, d=2)

    def test_lower_tail_dependence_greater_than_gaussian(self):
        """Clayton should show stronger lower tail co-movement than Gaussian."""
        np.random.seed(22)
        q = 0.05
        # Clayton
        cop_c = ClaytonCopula(theta=2.0, d=2)
        U_c = cop_c.sample_uniforms(100_000)
        p_lower_c = np.mean(np.all(U_c < q, axis=1))

        # Gaussian with moderate correlation
        cop_g = GaussianCopula(RHO_2D)
        U_g = cop_g.sample_uniforms(100_000)
        p_lower_g = np.mean(np.all(U_g < q, axis=1))

        assert p_lower_c > p_lower_g

    def test_upper_tail_independence_vs_lower(self):
        """Clayton: lower tail P(both < q) > upper tail P(both > 1-q)."""
        np.random.seed(23)
        cop = ClaytonCopula(theta=2.0, d=2)
        U = cop.sample_uniforms(100_000)
        q = 0.05
        lower = np.mean(np.all(U < q, axis=1))
        upper = np.mean(np.all(U > 1 - q, axis=1))
        assert lower > upper

    def test_higher_theta_stronger_dependence(self):
        """Higher theta → stronger lower tail co-movement."""
        np.random.seed(24)
        q = 0.05
        n = 100_000
        cop_weak = ClaytonCopula(theta=0.5, d=2)
        cop_strong = ClaytonCopula(theta=5.0, d=2)
        U_weak = cop_weak.sample_uniforms(n)
        U_strong = cop_strong.sample_uniforms(n)
        p_weak = np.mean(np.all(U_weak < q, axis=1))
        p_strong = np.mean(np.all(U_strong < q, axis=1))
        assert p_strong > p_weak

    def test_three_dimensional(self):
        np.random.seed(25)
        cop = ClaytonCopula(theta=2.0, d=3)
        U = cop.sample_uniforms(N)
        assert U.shape == (N, 3)
        assert np.all(U > 0) and np.all(U < 1)

    def test_asset_prices_positive(self):
        np.random.seed(26)
        cop = ClaytonCopula(theta=2.0, d=2)
        prices = cop.simulate_asset_prices(N, S0_LIST, MU_LIST, SIGMA_LIST, R, T)
        assert np.all(prices > 0)


class TestCopulaJointTailComparison:
    """Verify the key insight: Gaussian < Student-t < Clayton for lower tail,
    and Gaussian > Clayton for upper tail."""

    def test_lower_tail_ordering(self):
        np.random.seed(30)
        n = 200_000
        q = 0.05

        cop_g = GaussianCopula(RHO_2D)
        cop_t = StudentTCopula(RHO_2D, df=4.0)
        cop_c = ClaytonCopula(theta=2.0, d=2)

        p_g = np.mean(np.all(cop_g.sample_uniforms(n) < q, axis=1))
        p_t = np.mean(np.all(cop_t.sample_uniforms(n) < q, axis=1))
        p_c = np.mean(np.all(cop_c.sample_uniforms(n) < q, axis=1))

        # Clayton should show the most lower-tail dependence
        assert p_c > p_g
        # Student-t should show more than Gaussian (symmetric tail dependence)
        assert p_t > p_g
