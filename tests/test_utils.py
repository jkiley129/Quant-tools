"""
Tests for quant_sim/utils/stats.py
"""

import numpy as np
import pytest
from quant_sim.utils.stats import (
    normal_cdf,
    normal_ppf,
    student_t_ppf,
    log_likelihood_ratio,
    effective_sample_size,
    systematic_resample,
)


class TestNormalCdfPpf:
    def test_cdf_at_zero(self):
        assert abs(normal_cdf(0.0) - 0.5) < 1e-10

    def test_cdf_symmetry(self):
        assert abs(normal_cdf(1.0) + normal_cdf(-1.0) - 1.0) < 1e-10

    def test_cdf_ppf_are_inverses(self):
        for u in [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]:
            assert abs(normal_cdf(normal_ppf(u)) - u) < 1e-10

    def test_ppf_cdf_are_inverses(self):
        for x in [-2.0, -1.0, 0.0, 1.0, 2.0]:
            assert abs(normal_ppf(normal_cdf(x)) - x) < 1e-10

    def test_cdf_monotone(self):
        xs = np.linspace(-3, 3, 100)
        cdfs = normal_cdf(xs)
        assert np.all(np.diff(cdfs) > 0)

    def test_student_t_ppf_df_inf_approaches_normal(self):
        # Very high df should be close to normal quantiles
        for u in [0.05, 0.25, 0.75, 0.95]:
            t_q = student_t_ppf(u, df=10_000)
            n_q = normal_ppf(u)
            assert abs(t_q - n_q) < 0.01

    def test_student_t_ppf_symmetry(self):
        for df in [3, 5, 10]:
            assert abs(student_t_ppf(0.5, df=df)) < 1e-10


class TestLogLikelihoodRatio:
    def test_equal_densities_gives_zero(self):
        log_p = np.array([1.0, 2.0, 3.0])
        log_q = np.array([1.0, 2.0, 3.0])
        result = log_likelihood_ratio(log_p, log_q)
        np.testing.assert_array_almost_equal(result, np.zeros(3))

    def test_ratio_is_difference(self):
        log_p = np.array([1.0, 2.0])
        log_q = np.array([0.5, 1.5])
        result = log_likelihood_ratio(log_p, log_q)
        np.testing.assert_array_almost_equal(result, np.array([0.5, 0.5]))

    def test_shapes_preserved(self):
        log_p = np.random.randn(1000)
        log_q = np.random.randn(1000)
        result = log_likelihood_ratio(log_p, log_q)
        assert result.shape == (1000,)


class TestEffectiveSampleSize:
    def test_uniform_weights_ess_equals_n(self):
        n = 1000
        weights = np.ones(n) / n
        ess = effective_sample_size(weights)
        assert abs(ess - n) < 1.0

    def test_single_nonzero_weight_ess_is_one(self):
        weights = np.zeros(100)
        weights[0] = 1.0
        ess = effective_sample_size(weights)
        assert abs(ess - 1.0) < 1e-6

    def test_ess_bounded_by_n(self):
        n = 500
        weights = np.random.dirichlet(np.ones(n))
        ess = effective_sample_size(weights)
        assert 1.0 <= ess <= n + 1e-6

    def test_zero_weights_returns_zero(self):
        weights = np.zeros(10)
        ess = effective_sample_size(weights)
        assert ess == 0.0

    def test_ess_decreases_with_concentration(self):
        n = 100
        uniform_ess = effective_sample_size(np.ones(n) / n)
        concentrated = np.zeros(n)
        concentrated[:5] = 1.0 / 5
        concentrated_ess = effective_sample_size(concentrated)
        assert uniform_ess > concentrated_ess


class TestSystematicResample:
    def test_output_length(self):
        weights = np.ones(100) / 100
        idx = systematic_resample(weights, 100)
        assert len(idx) == 100

    def test_all_indices_in_range(self):
        n = 200
        weights = np.random.dirichlet(np.ones(n))
        idx = systematic_resample(weights, n)
        assert np.all(idx >= 0)
        assert np.all(idx < n)

    def test_uniform_weights_all_particles_represented(self):
        # With perfectly uniform weights, every particle should appear
        # approximately once
        n = 100
        weights = np.ones(n) / n
        idx = systematic_resample(weights, n)
        unique = np.unique(idx)
        # All indices should appear (systematic resampling guarantees this for uniform)
        assert len(unique) == n

    def test_highly_concentrated_weight(self):
        n = 100
        weights = np.zeros(n)
        weights[42] = 1.0  # all weight on particle 42
        idx = systematic_resample(weights, n)
        assert np.all(idx == 42)

    def test_unnormalized_weights_work(self):
        weights = np.array([10.0, 20.0, 30.0, 40.0])
        idx = systematic_resample(weights, 1000)
        # Particle 3 (weight 40%) should appear ~40% of the time
        frac_3 = np.mean(idx == 3)
        assert abs(frac_3 - 0.40) < 0.05
