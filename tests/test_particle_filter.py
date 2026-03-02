"""
Tests for quant_sim/particle_filter/smc.py
"""

import numpy as np
import pytest
import pandas as pd
from quant_sim.particle_filter.smc import ParticleFilter, GBMStateSpaceModel


@pytest.fixture
def model():
    return GBMStateSpaceModel(mu=0.10, sigma=0.25, dt=1 / 252, obs_noise_std=0.003)


@pytest.fixture
def pf(model):
    pf = ParticleFilter(model=model, n_particles=500, resample_threshold=0.5)
    pf.initialize()
    return pf


class TestGBMStateSpaceModel:
    def test_transition_output_shape(self, model):
        particles = np.zeros(100)
        new_particles = model.transition(particles)
        assert new_particles.shape == (100,)

    def test_transition_mean_close_to_model_mean(self, model):
        """Mean of transition draws should be close to (mu - 0.5*sigma^2)*dt."""
        n = 100_000
        particles = np.zeros(n)
        new_p = model.transition(particles)
        expected_mean = model._mean_return
        assert abs(np.mean(new_p) - expected_mean) < 0.001

    def test_transition_std_close_to_model_std(self, model):
        n = 100_000
        particles = np.zeros(n)
        new_p = model.transition(particles)
        assert abs(np.std(new_p) - model._std_return) < 0.001

    def test_likelihood_all_positive(self, model):
        particles = np.random.normal(0, 0.01, 100)
        obs = 0.001
        likelihoods = model.likelihood(particles, obs)
        assert np.all(likelihoods > 0)

    def test_likelihood_peaks_near_observation(self, model):
        obs = 0.005
        particles_near = np.array([obs, obs + 0.001, obs - 0.001])
        particles_far = np.array([obs + 0.10, obs - 0.10])
        l_near = model.likelihood(particles_near, obs)
        l_far = model.likelihood(particles_far, obs)
        assert np.mean(l_near) > np.mean(l_far)

    def test_likelihood_output_shape(self, model):
        particles = np.zeros(200)
        liks = model.likelihood(particles, 0.0)
        assert liks.shape == (200,)


class TestParticleFilterInitialize:
    def test_particle_count(self, model):
        pf = ParticleFilter(model=model, n_particles=300)
        pf.initialize()
        assert len(pf.particles) == 300

    def test_weights_sum_to_one(self, model):
        pf = ParticleFilter(model=model, n_particles=200)
        pf.initialize()
        assert abs(pf.weights.sum() - 1.0) < 1e-10

    def test_weights_uniform_at_init(self, model):
        n = 150
        pf = ParticleFilter(model=model, n_particles=n)
        pf.initialize()
        np.testing.assert_array_almost_equal(pf.weights, np.ones(n) / n)

    def test_cumulative_log_price_zero_at_init(self, model):
        pf = ParticleFilter(model=model, n_particles=100)
        pf.initialize()
        assert pf.cumulative_log_price == 0.0


class TestParticleFilterUpdate:
    def test_update_returns_dict(self, pf):
        result = pf.update(0.001)
        assert isinstance(result, dict)
        assert "mean_estimate" in result
        assert "std_estimate" in result
        assert "ess" in result
        assert "resampled" in result

    def test_weights_still_sum_to_one_after_update(self, pf):
        pf.update(0.002)
        assert abs(pf.weights.sum() - 1.0) < 1e-8

    def test_ess_positive(self, pf):
        result = pf.update(0.001)
        assert result["ess"] > 0

    def test_history_grows_with_each_update(self, pf):
        for i in range(5):
            pf.update(0.001)
        assert len(pf.history) == 5

    def test_resampling_triggered_on_degenerate_weights(self):
        """When weights are already concentrated, ESS < threshold → resample on next update."""
        model = GBMStateSpaceModel(mu=0.10, sigma=0.25, dt=1/252, obs_noise_std=0.003)
        n = 100
        pf = ParticleFilter(model=model, n_particles=n, resample_threshold=0.9)
        pf.initialize()
        # Manually concentrate all weight on one particle (ESS = 1 << 0.9*100)
        pf.weights = np.zeros(n)
        pf.weights[0] = 1.0
        # Next update: likelihood step will keep most weight on particle 0,
        # ESS will remain << threshold, triggering resample
        result = pf.update(model._mean_return)  # observation near prior mean
        assert result["resampled"] is True

    def test_cumulative_log_price_updates(self, pf):
        initial = pf.cumulative_log_price
        pf.update(0.001)
        assert pf.cumulative_log_price != initial

    def test_particles_change_after_update(self, pf):
        before = pf.particles.copy()
        pf.update(0.001)
        # Particles should be different after transition
        assert not np.array_equal(before, pf.particles)


class TestParticleFilterRunSequence:
    def test_returns_dataframe(self, pf):
        obs = np.random.normal(0, 0.015, 50)
        result = pf.run_sequence(obs)
        assert isinstance(result, pd.DataFrame)

    def test_dataframe_row_count(self, pf):
        obs = np.random.normal(0, 0.015, 30)
        result = pf.run_sequence(obs)
        assert len(result) == 30

    def test_dataframe_columns(self, pf):
        obs = np.random.normal(0, 0.015, 10)
        result = pf.run_sequence(obs)
        assert "t" in result.columns
        assert "mean_estimate" in result.columns
        assert "ess" in result.columns

    def test_t_column_is_sequential(self, pf):
        obs = np.random.normal(0, 0.015, 20)
        result = pf.run_sequence(obs)
        np.testing.assert_array_equal(result["t"].values, np.arange(20))

    def test_event_probability_column_present_when_s0_k_given(self):
        model = GBMStateSpaceModel(mu=0.10, sigma=0.25, dt=1/252)
        pf = ParticleFilter(model=model, n_particles=200)
        pf.initialize()
        obs = np.random.normal(0, 0.015, 15)
        result = pf.run_sequence(obs, S0=100.0, K=110.0)
        assert "event_probability" in result.columns
        assert np.all((result["event_probability"] >= 0) & (result["event_probability"] <= 1))


class TestParticleFilterTracking:
    def test_filter_tracks_signal_direction(self):
        """Filter mean log-return should be correlated with true log-returns."""
        np.random.seed(99)
        dt = 1 / 252
        mu, sigma = 0.10, 0.20
        T = 100
        obs_noise = 0.002

        true_log_ret = np.random.normal((mu - 0.5*sigma**2)*dt, sigma*np.sqrt(dt), T)
        obs = true_log_ret + np.random.normal(0, obs_noise, T)

        model = GBMStateSpaceModel(mu=mu, sigma=sigma, dt=dt, obs_noise_std=obs_noise)
        pf = ParticleFilter(model=model, n_particles=1000)
        pf.initialize()

        means = []
        for o in obs:
            result = pf.update(o)
            means.append(result["mean_estimate"])

        # Filter estimates should be positively correlated with true log-returns
        corr = np.corrcoef(means, true_log_ret)[0, 1]
        assert corr > 0.5
