"""
Brier Score calibration scoring and Murphy decomposition for probability forecasts.

The Brier Score measures the accuracy of probabilistic binary predictions:
    BS = (1/N) * sum((f_i - o_i)^2)
where f_i is the forecast probability and o_i in {0, 1} is the outcome.

Murphy (1973) decomposition: BS = Uncertainty - Resolution + Reliability
"""

import numpy as np


class BrierScorer:
    """Brier Score scorer and calibration analyzer for binary probability forecasts."""

    def score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute the Brier Score.

        Parameters
        ----------
        y_true : np.ndarray
            Binary outcomes in {0, 1}.
        y_pred : np.ndarray
            Forecast probabilities in [0, 1].

        Returns
        -------
        float
            Brier Score (lower is better; 0 = perfect).
        """
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        return float(np.mean((y_pred - y_true) ** 2))

    def decompose(self, y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 10) -> dict:
        """Murphy (1973) decomposition: BS = Uncertainty - Resolution + Reliability.

        Parameters
        ----------
        y_true : np.ndarray
            Binary outcomes in {0, 1}.
        y_pred : np.ndarray
            Forecast probabilities in [0, 1].
        n_bins : int
            Number of equal-width bins over [0, 1].

        Returns
        -------
        dict
            {
              'brier_score': float,
              'uncertainty': float,   # irreducible: o_bar*(1-o_bar)
              'resolution': float,    # good: spread of outcomes across bins
              'reliability': float,   # bad: calibration error
              'bin_centers': np.ndarray,
              'empirical_freqs': np.ndarray,
              'predicted_means': np.ndarray,
            }
        """
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        N = len(y_true)

        o_bar = np.mean(y_true)
        uncertainty = o_bar * (1.0 - o_bar)

        bin_edges = np.linspace(0.0, 1.0 + 1e-10, n_bins + 1)
        bin_indices = np.digitize(y_pred, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)

        bin_centers = []
        empirical_freqs = []
        predicted_means = []
        reliability = 0.0
        resolution = 0.0

        for k in range(n_bins):
            mask = bin_indices == k
            n_k = np.sum(mask)
            if n_k == 0:
                continue
            o_k = np.mean(y_true[mask])
            p_k = np.mean(y_pred[mask])
            reliability += n_k * (p_k - o_k) ** 2
            resolution += n_k * (o_k - o_bar) ** 2
            bin_centers.append(0.5 * (bin_edges[k] + bin_edges[k + 1]))
            empirical_freqs.append(o_k)
            predicted_means.append(p_k)

        reliability /= N
        resolution /= N

        # The Murphy (1973) decomposition: BS = Uncertainty - Resolution + Reliability
        # holds exactly when BS is computed from the same bin-mean predictions used for
        # Reliability and Resolution (the "bin-approximated" Brier score).
        # Individual-level BS differs by the within-bin prediction variance.
        bs = float(uncertainty - resolution + reliability)

        return {
            "brier_score": bs,
            "uncertainty": float(uncertainty),
            "resolution": float(resolution),
            "reliability": float(reliability),
            "bin_centers": np.array(bin_centers),
            "empirical_freqs": np.array(empirical_freqs),
            "predicted_means": np.array(predicted_means),
        }

    def skill_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Brier Skill Score vs climatological baseline.

        BSS = 1 - BS / BS_climate
        where BS_climate = o_bar * (1 - o_bar) (predicting the base rate always).

        Returns
        -------
        float
            BSS in (-inf, 1]. Positive = better than climatology.
        """
        y_true = np.asarray(y_true, dtype=float)
        o_bar = np.mean(y_true)
        bs_climate = o_bar * (1.0 - o_bar)
        if bs_climate == 0:
            return 0.0
        return float(1.0 - self.score(y_true, y_pred) / bs_climate)

    def calibration_curve(self, y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 10):
        """Return (bin_centers, empirical_freqs) for a reliability diagram."""
        result = self.decompose(y_true, y_pred, n_bins=n_bins)
        return result["bin_centers"], result["empirical_freqs"]


def run_brier_demo(n_samples: int = 2000, n_bins: int = 10, no_plots: bool = False, save_plots: str = None):
    """Demo: Compare well-calibrated vs overconfident forecaster using Brier decomposition."""
    from quant_sim.utils.plotting import plot_reliability_diagram

    rng = np.random.default_rng(42)
    true_probs = rng.uniform(0.05, 0.95, n_samples)
    y_true = (rng.uniform(size=n_samples) < true_probs).astype(float)

    # Well-calibrated: slight Gaussian noise around true probability
    y_pred_calibrated = np.clip(true_probs + rng.normal(0, 0.05, n_samples), 0.01, 0.99)

    # Overconfident: pushes predictions toward extremes
    y_pred_overconfident = np.clip(
        0.5 + 1.8 * (true_probs - 0.5) + rng.normal(0, 0.03, n_samples), 0.01, 0.99
    )

    scorer = BrierScorer()
    print("=" * 60)
    print("  Brier Score Calibration Analysis")
    print("=" * 60)

    for label, y_pred in [("Well-calibrated", y_pred_calibrated), ("Overconfident", y_pred_overconfident)]:
        result = scorer.decompose(y_true, y_pred, n_bins=n_bins)
        bss = scorer.skill_score(y_true, y_pred)
        print(f"\n  {label}")
        print(f"    Brier Score:   {result['brier_score']:.4f}")
        print(f"    Uncertainty:   {result['uncertainty']:.4f}  (irreducible)")
        print(f"    Resolution:    {result['resolution']:.4f}  (higher = better)")
        print(f"    Reliability:   {result['reliability']:.4f}  (lower = better)")
        print(f"    Skill Score:   {bss:+.4f}")

        if not no_plots:
            sp = f"{save_plots}/brier_{label.lower().replace(' ', '_')}.png" if save_plots else None
            plot_reliability_diagram(
                result["bin_centers"],
                result["empirical_freqs"],
                result["predicted_means"],
                title=f"Reliability Diagram — {label}",
                save_path=sp,
            )
    print()
