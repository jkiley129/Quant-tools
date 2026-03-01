"""
Shared matplotlib helpers with a consistent style.

All functions accept an optional `save_path` kwarg:
  - If None, the plot is displayed interactively (plt.show()).
  - If a file path string, the figure is saved as PNG and closed.
  - If no_show=True, the figure object is returned without displaying (for testing/CI).
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def _finish(fig, save_path, no_show=False):
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    elif no_show:
        return fig
    else:
        plt.tight_layout()
        plt.show()
        plt.close(fig)
    return fig


def plot_price_paths(
    paths: np.ndarray,
    title: str = "Simulated Price Paths",
    xlabel: str = "Time Step",
    ylabel: str = "Price",
    max_paths: int = 50,
    save_path: str = None,
    no_show: bool = False,
):
    """Line chart for GBM simulation paths.

    Parameters
    ----------
    paths : np.ndarray
        Shape (n_steps+1, n_paths). First row is S0.
    max_paths : int
        Maximum paths to plot (avoids overdrawing).
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    n_plot = min(max_paths, paths.shape[1])
    ax.plot(paths[:, :n_plot], alpha=0.4, linewidth=0.7)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    return _finish(fig, save_path, no_show)


def plot_histogram(
    data: np.ndarray,
    bins: int = 100,
    title: str = "Distribution",
    xlabel: str = "Value",
    ylabel: str = "Frequency",
    save_path: str = None,
    no_show: bool = False,
):
    """Histogram of simulated terminal prices or payoffs."""
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(data, bins=bins, edgecolor="none", alpha=0.75, color="steelblue")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3, axis="y")
    return _finish(fig, save_path, no_show)


def plot_reliability_diagram(
    bin_centers: np.ndarray,
    empirical_freqs: np.ndarray,
    predicted_means: np.ndarray = None,
    title: str = "Reliability Diagram",
    save_path: str = None,
    no_show: bool = False,
):
    """Calibration reliability diagram.

    Perfect calibration appears as the diagonal line y = x.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration", linewidth=1)
    ax.plot(bin_centers, empirical_freqs, "o-", label="Observed frequency", color="steelblue")
    if predicted_means is not None:
        ax.plot(bin_centers, predicted_means, "s--", label="Mean prediction", color="orange", alpha=0.7)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Forecast Probability")
    ax.set_ylabel("Observed Frequency")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    return _finish(fig, save_path, no_show)


def plot_particle_evolution(
    time_steps: np.ndarray,
    estimates: np.ndarray,
    lower_band: np.ndarray = None,
    upper_band: np.ndarray = None,
    true_values: np.ndarray = None,
    title: str = "Particle Filter: Probability Estimate",
    save_path: str = None,
    no_show: bool = False,
):
    """Tracks filter probability estimates over time with optional confidence band."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(time_steps, estimates, label="Filter estimate", color="steelblue", linewidth=1.5)
    if lower_band is not None and upper_band is not None:
        ax.fill_between(time_steps, lower_band, upper_band, alpha=0.25, color="steelblue", label="95% CI")
    if true_values is not None:
        ax.plot(time_steps, true_values, "r--", label="True probability", linewidth=1, alpha=0.8)
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Event Probability")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    return _finish(fig, save_path, no_show)


def plot_copula_scatter(
    u1: np.ndarray,
    u2: np.ndarray,
    title: str = "Copula Uniform Marginals",
    xlabel: str = "Asset 1 (uniform)",
    ylabel: str = "Asset 2 (uniform)",
    save_path: str = None,
    no_show: bool = False,
):
    """2D scatter of copula uniform marginals to visualize tail dependence."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(u1, u2, alpha=0.15, s=3, color="steelblue")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return _finish(fig, save_path, no_show)


def plot_order_book_snapshot(
    bids: list,
    asks: list,
    title: str = "Order Book Snapshot",
    save_path: str = None,
    no_show: bool = False,
):
    """Horizontal bar chart of LOB state.

    Parameters
    ----------
    bids : list of (price, qty) tuples, sorted descending by price
    asks : list of (price, qty) tuples, sorted ascending by price
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    bid_prices = [b[0] for b in bids[:10]]
    bid_qtys = [b[1] for b in bids[:10]]
    ask_prices = [a[0] for a in asks[:10]]
    ask_qtys = [a[1] for a in asks[:10]]

    ax.barh(bid_prices, bid_qtys, height=0.003, color="green", alpha=0.7, label="Bids")
    ax.barh(ask_prices, ask_qtys, height=0.003, color="red", alpha=0.7, label="Asks")
    ax.set_xlabel("Quantity")
    ax.set_ylabel("Price")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="x")
    return _finish(fig, save_path, no_show)


def plot_abm_price_discovery(
    time_steps: np.ndarray,
    market_prices: np.ndarray,
    true_probs: np.ndarray,
    title: str = "ABM: Price Discovery",
    save_path: str = None,
    no_show: bool = False,
):
    """Compare market mid-price trajectory against true probability path."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(time_steps, market_prices, label="Market mid-price", color="steelblue", linewidth=1)
    ax.plot(time_steps, true_probs, "r--", label="True probability", linewidth=1.5, alpha=0.8)
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Probability / Price")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    return _finish(fig, save_path, no_show)
