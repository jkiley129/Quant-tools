"""
Sequential Monte Carlo (SMC) / Particle Filter for real-time Bayesian updating.

The particle filter tracks a hidden state (log-returns or asset prices) by
maintaining a population of N weighted particles. As each new price observation
arrives, it:
  1. Predicts: propagates each particle through the state transition model
  2. Weights:  multiplies particle weights by the likelihood of the observation
  3. Resamples: if ESS drops below a threshold, draws a new particle set

This enables real-time updating of binary event probabilities:
  P(S_T > K | observations so far) updated at each time step.
"""

import numpy as np
import pandas as pd

from quant_sim.utils.stats import effective_sample_size, systematic_resample


class GBMStateSpaceModel:
    """State-space model for a GBM-driven hidden log-return process.

    State:       x_t = log(S_t / S_{t-1})  — daily log-return (hidden)
    Observation: y_t = x_t + epsilon_t     where epsilon_t ~ N(0, obs_noise_std^2)

    Parameters
    ----------
    mu : float
        Annualized drift.
    sigma : float
        Annualized volatility.
    dt : float
        Time step in years (default: 1/252 for daily).
    obs_noise_std : float
        Standard deviation of observation noise.
    """

    def __init__(self, mu: float, sigma: float, dt: float = 1 / 252, obs_noise_std: float = 0.005):
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.obs_noise_std = obs_noise_std

        self._mean_return = (mu - 0.5 * sigma**2) * dt
        self._std_return = sigma * np.sqrt(dt)

    def transition(self, particles: np.ndarray) -> np.ndarray:
        """Sample x_t | x_{t-1} for each particle.

        Each particle carries the current log-return; the transition draws a
        new independent log-return from the GBM model (Markov process).
        """
        noise = np.random.normal(0, self._std_return, size=len(particles))
        return self._mean_return + noise

    def likelihood(self, particles: np.ndarray, observation: float) -> np.ndarray:
        """Compute p(y_t | x_t) for each particle.

        Gaussian likelihood: y_t = x_t + epsilon,  epsilon ~ N(0, obs_noise_std^2)
        """
        diff = observation - particles
        return np.exp(-0.5 * (diff / self.obs_noise_std) ** 2) / (
            self.obs_noise_std * np.sqrt(2 * np.pi)
        )

    def log_likelihood(self, particles: np.ndarray, observation: float) -> np.ndarray:
        """Compute log p(y_t | x_t) for each particle (numerically stable).

        Avoids underflow for extreme observations where exp() would return 0.
        """
        diff = observation - particles
        return -0.5 * (diff / self.obs_noise_std) ** 2 - np.log(
            self.obs_noise_std * np.sqrt(2 * np.pi)
        )


class ParticleFilter:
    """Bootstrap Particle Filter (Sequential Importance Resampling).

    Parameters
    ----------
    model : GBMStateSpaceModel
        The state-space model defining transition and likelihood.
    n_particles : int
        Number of particles.
    resample_threshold : float
        Resample when ESS/N drops below this fraction (default: 0.5).
    """

    def __init__(self, model: GBMStateSpaceModel, n_particles: int = 1000, resample_threshold: float = 0.5):
        self.model = model
        self.n_particles = n_particles
        self.resample_threshold = resample_threshold

        self.particles = None
        self.weights = None
        self.cumulative_log_price = 0.0  # tracks cumulative log-return for price estimate
        self.history = []

    def initialize(self, x0_mean: float = 0.0, x0_std: float = None) -> None:
        """Sample initial particles from a Gaussian prior over log-returns.

        Parameters
        ----------
        x0_mean : float
            Prior mean log-return (typically 0).
        x0_std : float
            Prior std; defaults to the model's daily return std.
        """
        if x0_std is None:
            x0_std = self.model._std_return
        self.particles = np.random.normal(x0_mean, x0_std, self.n_particles)
        self.weights = np.ones(self.n_particles) / self.n_particles
        self.history = []

    def update(self, observation: float) -> dict:
        """Perform one Bootstrap Particle Filter step.

        Algorithm:
          1. Predict:   x_t^i ~ p(x_t | x_{t-1}^i)
          2. Weight:    w^i  = w^i_{t-1} * p(y_t | x_t^i)
          3. Normalize: w^i  /= sum(w)
          4. ESS check: if ESS < threshold * N, resample
          5. Resample:  systematic resampling if triggered

        Parameters
        ----------
        observation : float
            Observed log-return at this time step: log(P_t / P_{t-1}).

        Returns
        -------
        dict
            {'mean_estimate', 'std_estimate', 'ess', 'resampled'}
        """
        # 1. Predict
        self.particles = self.model.transition(self.particles)

        # 2. Weight update in log-space to avoid underflow for extreme observations.
        #    Particles with weight=0 get log(1e-300) ≈ -690, ensuring they remain at
        #    near-zero weight without causing -inf propagation through log(0).
        log_likelihoods = self.model.log_likelihood(self.particles, observation)
        log_weights = np.log(np.where(self.weights > 0, self.weights, 1e-300)) + log_likelihoods
        log_weights -= np.max(log_weights)  # shift for numerical stability before exp
        self.weights = np.exp(log_weights)
        w_sum = self.weights.sum()
        if not np.isfinite(w_sum) or w_sum == 0:
            self.weights = np.ones(self.n_particles) / self.n_particles
        else:
            self.weights /= w_sum

        # 3. ESS
        ess = effective_sample_size(self.weights)

        # 4. Resample if needed
        resampled = False
        if ess < self.resample_threshold * self.n_particles:
            idx = systematic_resample(self.weights, self.n_particles)
            self.particles = self.particles[idx]
            self.weights = np.ones(self.n_particles) / self.n_particles
            resampled = True

        # Update cumulative log-price using weighted mean
        mean_log_return = float(np.sum(self.weights * self.particles))
        self.cumulative_log_price += mean_log_return

        mean_est = float(np.sum(self.weights * self.particles))
        std_est = float(np.sqrt(np.sum(self.weights * (self.particles - mean_est) ** 2)))

        step_result = {
            "mean_estimate": mean_est,
            "std_estimate": std_est,
            "ess": ess,
            "resampled": resampled,
        }
        self.history.append(step_result)
        return step_result

    def estimate_event_probability(self, S0: float, K: float, direction: str = "above") -> float:
        """Estimate P(S_T_accumulated > K) from current particle population.

        Converts each particle's accumulated log-return to a price level,
        then computes the weighted fraction satisfying the threshold condition.

        Parameters
        ----------
        S0 : float
            Initial asset price.
        K : float
            Target price level.
        direction : str
            'above' for P(S > K), 'below' for P(S < K).
        """
        # Convert cumulative log-return particles to price estimates
        # particles are log-returns; sum over history gives cumulative log-return
        # Here we use cumulative_log_price as the filter's price estimate
        # For a single particle's probability, convert current log-return to price move
        price_particles = S0 * np.exp(self.cumulative_log_price + self.particles)
        if direction == "above":
            indicator = (price_particles > K).astype(float)
        else:
            indicator = (price_particles < K).astype(float)
        return float(np.sum(self.weights * indicator))

    def run_sequence(self, observations: np.ndarray, S0: float = None, K: float = None) -> pd.DataFrame:
        """Run the filter over a full sequence of observations.

        Parameters
        ----------
        observations : np.ndarray
            Array of observed log-returns, shape (T,).
        S0 : float, optional
            Initial price for event probability estimation.
        K : float, optional
            Target price threshold.

        Returns
        -------
        pd.DataFrame
            Columns: ['t', 'mean_estimate', 'std_estimate', 'ess', 'resampled', 'event_probability']
        """
        records = []
        for t, obs in enumerate(observations):
            result = self.update(obs)
            row = {"t": t, **result}
            if S0 is not None and K is not None:
                row["event_probability"] = self.estimate_event_probability(S0, K)
            records.append(row)
        return pd.DataFrame(records)


def run_particle_filter_demo(
    S0: float = 100.0,
    mu: float = 0.10,
    sigma: float = 0.25,
    T_days: int = 252,
    n_particles: int = 1000,
    obs_noise: float = 0.003,
    seed: int = 42,
    no_plots: bool = False,
    save_plots: str = None,
):
    """Demo: Particle filter tracking the current price and forward-looking P(S_T_end > K).

    Shows two outputs:
    1. Filter's running price estimate vs the true (but hidden) price path
    2. Forward-looking P(S_end > K) as remaining time shrinks
    """
    from quant_sim.utils.plotting import plot_particle_evolution
    from scipy import stats as scipy_stats

    np.random.seed(seed)
    dt = 1 / 252
    K = 1.1 * S0

    # Generate synthetic true price path (hidden from filter — observed only with noise)
    true_log_returns = np.random.normal((mu - 0.5 * sigma**2) * dt, sigma * np.sqrt(dt), T_days)
    observed_log_returns = true_log_returns + np.random.normal(0, obs_noise, T_days)
    true_prices = S0 * np.exp(np.cumsum(true_log_returns))

    # Run particle filter
    model = GBMStateSpaceModel(mu=mu, sigma=sigma, dt=dt, obs_noise_std=obs_noise)
    pf = ParticleFilter(model=model, n_particles=n_particles, resample_threshold=0.5)
    pf.initialize()

    print("=" * 70)
    print("  Particle Filter — Real-Time Price Tracking & Probability Updating")
    print("=" * 70)
    print(f"  S0={S0}  mu={mu:.2%}  sigma={sigma:.2%}  K={K:.0f} (+10%)  N_particles={n_particles}")
    print()
    print(f"  {'Day':>5}  {'Filter Price':>13}  {'True Price':>11}  {'Error':>8}  {'ESS':>7}  {'P(end>K)':>10}")
    print("  " + "-" * 62)

    inferred_prices = []
    forward_probs = []

    for t, obs in enumerate(observed_log_returns):
        result = pf.update(obs)

        # Current price estimate: S0 * exp(cumulative weighted mean log-return)
        inferred_price = S0 * np.exp(pf.cumulative_log_price)
        inferred_prices.append(inferred_price)

        # Forward-looking P(S_end > K | current price estimate, remaining time)
        remaining_T = (T_days - t) / 252.0
        if remaining_T > 0:
            d2 = (np.log(inferred_price / K) + (mu - 0.5 * sigma**2) * remaining_T) / (
                sigma * np.sqrt(remaining_T)
            )
            fwd_prob = float(scipy_stats.norm.cdf(d2))
        else:
            fwd_prob = 1.0 if inferred_price > K else 0.0
        forward_probs.append(fwd_prob)

        if t % 50 == 0 or t == T_days - 1:
            err = inferred_price - true_prices[t]
            print(
                f"  {t:>5}  {inferred_price:>13.2f}  {true_prices[t]:>11.2f}"
                f"  {err:>+8.2f}  {result['ess']:>7.0f}  {fwd_prob:>10.4f}"
            )

    rmse = float(np.sqrt(np.mean((np.array(inferred_prices) - true_prices)**2)))
    print()
    print(f"  Price tracking RMSE:  {rmse:.4f}")
    print(f"  Final filter price:   {inferred_prices[-1]:.2f}  |  True: {true_prices[-1]:.2f}")
    print(f"  Outcome (K={K:.0f}):      S_end {'>' if true_prices[-1] > K else '<='} K → P converged to {forward_probs[-1]:.4f}")
    print()

    if not no_plots:
        steps = np.arange(T_days)
        sp = f"{save_plots}/particle_filter.png" if save_plots else None
        plot_particle_evolution(
            steps,
            np.array(forward_probs),
            true_values=None,
            title=f"Particle Filter: Forward P(S_end > {K:.0f}) over {T_days} Days",
            save_path=sp,
        )
