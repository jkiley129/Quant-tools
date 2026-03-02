#!/usr/bin/env python3
"""
Quant-Tools Simulation Engine — CLI Entry Point

Usage:
  python main.py <command> [options]

Commands:
  gbm           Monte Carlo GBM binary contract pricing
  brier         Brier Score calibration analysis
  importance    Importance Sampling for tail risk events
  particle      Particle Filter real-time probability updating
  variance      Variance reduction technique comparison
  copula        Copula model correlated asset simulation
  abm           Agent-Based Market simulation
  all           Run all simulations sequentially
  list          List all available simulations

Global Options:
  -s, --seed SEED      Random seed (default: 42)
  -n, --n-paths N      Simulation paths/samples (default: 100000)
  --no-plots           Suppress matplotlib display
  --save-plots DIR     Save plots to directory instead of displaying

Examples:
  python main.py gbm --ticker NVDA --strike 1000 --maturity 0.5
  python main.py copula --tickers NVDA AMD SMCI --period 2y
  python main.py importance --ticker NVDA --threshold 1.5
  python main.py all --seed 42 --no-plots
"""

import argparse
import sys
import time
import os


# ─── Demo registry ────────────────────────────────────────────────────────────

COMMANDS = {
    "gbm": "Monte Carlo GBM binary contract pricing",
    "brier": "Brier Score calibration analysis",
    "importance": "Importance Sampling for tail risk events",
    "particle": "Particle Filter real-time probability updating",
    "variance": "Variance reduction technique comparison",
    "copula": "Copula model correlated asset simulation",
    "abm": "Agent-Based Market simulation",
    "all": "Run all simulations sequentially",
    "list": "List all available simulations",
}


# ─── Argument parsing ─────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="quant_sim",
        description="Quant-Tools Simulation Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Global options
    parser.add_argument("-s", "--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("-n", "--n-paths", type=int, default=100_000,
                        help="Number of simulation paths/samples (default: 100000)")
    parser.add_argument("--no-plots", action="store_true", help="Suppress matplotlib display")
    parser.add_argument("--save-plots", metavar="DIR", default=None,
                        help="Save plots to this directory as PNG files")

    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")

    # ── gbm ──────────────────────────────────────────────────────────────────
    p_gbm = subparsers.add_parser("gbm", help=COMMANDS["gbm"])
    p_gbm.add_argument("--ticker", metavar="SYMBOL", help="Fetch and calibrate from yfinance (e.g. NVDA)")
    p_gbm.add_argument("--period", default="1y", help="yfinance period (default: 1y)")
    p_gbm.add_argument("--spot", type=float, default=None, help="Override S0")
    p_gbm.add_argument("--drift", type=float, default=None, help="Override annualized drift mu")
    p_gbm.add_argument("--vol", type=float, default=None, help="Override annualized sigma")
    p_gbm.add_argument("--rate", type=float, default=0.05, help="Risk-free rate (default: 0.05)")
    p_gbm.add_argument("--maturity", type=float, default=0.5, help="Time to expiry in years (default: 0.5)")
    p_gbm.add_argument("--strike", type=float, nargs="+", default=None, help="Strike(s) to price")

    # ── brier ─────────────────────────────────────────────────────────────────
    p_brier = subparsers.add_parser("brier", help=COMMANDS["brier"])
    p_brier.add_argument("--n-samples", type=int, default=2000, help="Number of forecast samples (default: 2000)")
    p_brier.add_argument("--n-bins", type=int, default=10, help="Reliability diagram bins (default: 10)")

    # ── importance ────────────────────────────────────────────────────────────
    p_imp = subparsers.add_parser("importance", help=COMMANDS["importance"])
    p_imp.add_argument("--ticker", metavar="SYMBOL", help="Fetch and calibrate from yfinance")
    p_imp.add_argument("--period", default="1y")
    p_imp.add_argument("--spot", type=float, default=None)
    p_imp.add_argument("--drift", type=float, default=None)
    p_imp.add_argument("--vol", type=float, default=None)
    p_imp.add_argument("--maturity", type=float, default=0.5)
    p_imp.add_argument("--threshold", type=float, default=1.5,
                       help="Threshold as multiple of S0 (default: 1.5, i.e. 50% gain)")
    p_imp.add_argument("--shift", type=float, default=0.5, help="IS drift shift factor (default: 0.5)")

    # ── particle ──────────────────────────────────────────────────────────────
    p_pf = subparsers.add_parser("particle", help=COMMANDS["particle"])
    p_pf.add_argument("--ticker", metavar="SYMBOL")
    p_pf.add_argument("--period", default="1y")
    p_pf.add_argument("--drift", type=float, default=None)
    p_pf.add_argument("--vol", type=float, default=None)
    p_pf.add_argument("--n-particles", type=int, default=1000, help="Number of particles (default: 1000)")
    p_pf.add_argument("--obs-noise", type=float, default=0.003, help="Observation noise std (default: 0.003)")
    p_pf.add_argument("--n-steps", type=int, default=252, help="Simulation days (default: 252)")

    # ── variance ──────────────────────────────────────────────────────────────
    p_var = subparsers.add_parser("variance", help=COMMANDS["variance"])
    p_var.add_argument("--ticker", metavar="SYMBOL")
    p_var.add_argument("--period", default="1y")
    p_var.add_argument("--spot", type=float, default=None)
    p_var.add_argument("--drift", type=float, default=None)
    p_var.add_argument("--vol", type=float, default=None)
    p_var.add_argument("--maturity", type=float, default=0.5)
    p_var.add_argument("--strike", type=float, default=None, help="Strike price (default: 1.05*S0)")
    p_var.add_argument("--n-strata", type=int, default=100, help="Strata for stratified sampling (default: 100)")

    # ── copula ────────────────────────────────────────────────────────────────
    p_cop = subparsers.add_parser("copula", help=COMMANDS["copula"])
    p_cop.add_argument("--tickers", nargs="+", metavar="SYMBOL", help="Multiple tickers for real-data mode")
    p_cop.add_argument("--period", default="1y")
    p_cop.add_argument("--correlation", type=float, default=0.75, help="Pairwise correlation for synthetic mode")
    p_cop.add_argument("--df", type=float, default=4.0, help="Student-t degrees of freedom (default: 4)")
    p_cop.add_argument("--theta", type=float, default=2.0, help="Clayton theta (default: 2.0)")
    p_cop.add_argument("--drop-threshold", type=float, default=0.10, help="Joint drop threshold (default: 0.10)")
    p_cop.add_argument("--gain-threshold", type=float, default=0.20, help="Joint gain threshold (default: 0.20)")
    p_cop.add_argument("--maturity", type=float, default=0.5)

    # ── abm ───────────────────────────────────────────────────────────────────
    p_abm = subparsers.add_parser("abm", help=COMMANDS["abm"])
    p_abm.add_argument("--ticker", metavar="SYMBOL")
    p_abm.add_argument("--period", default="1y")
    p_abm.add_argument("--spot", type=float, default=None)
    p_abm.add_argument("--drift", type=float, default=None)
    p_abm.add_argument("--vol", type=float, default=None)
    p_abm.add_argument("--maturity", type=float, default=0.5)
    p_abm.add_argument("--n-steps", type=int, default=300, help="Simulation time steps (default: 300)")
    p_abm.add_argument("--n-informed", type=int, default=3, help="Number of informed agents (default: 3)")
    p_abm.add_argument("--n-noise", type=int, default=10, help="Number of noise agents (default: 10)")
    p_abm.add_argument("--n-mm", type=int, default=2, help="Number of market makers (default: 2)")

    # ── all ───────────────────────────────────────────────────────────────────
    subparsers.add_parser("all", help=COMMANDS["all"])

    # ── list ──────────────────────────────────────────────────────────────────
    subparsers.add_parser("list", help=COMMANDS["list"])

    return parser


# ─── Ticker calibration helper ────────────────────────────────────────────────

def calibrate_from_ticker(ticker: str, period: str) -> dict:
    from quant_sim.data.calibrator import calibrate_gbm
    from quant_sim.data.fetcher import fetch_prices
    print(f"  [data] Fetching {ticker} ({period})...")
    prices = fetch_prices(ticker, period=period)
    params = calibrate_gbm(prices["Close"])
    print(f"  [data] Calibrated: S0={params['S0']:.2f}  mu={params['mu']:.2%}  sigma={params['sigma']:.2%}")
    return params


def apply_overrides(params: dict, args, fields=("spot", "drift", "vol")) -> dict:
    """Let CLI manual args override calibrated values."""
    if hasattr(args, "spot") and args.spot is not None:
        params["S0"] = args.spot
    if hasattr(args, "drift") and args.drift is not None:
        params["mu"] = args.drift
    if hasattr(args, "vol") and args.vol is not None:
        params["sigma"] = args.vol
    return params


# ─── Command handlers ─────────────────────────────────────────────────────────

def cmd_gbm(args):
    from quant_sim.monte_carlo.gbm import run_gbm_demo

    if args.ticker:
        params = calibrate_from_ticker(args.ticker, args.period)
    else:
        params = {"S0": args.spot or 182.0, "mu": args.drift or 0.12, "sigma": args.vol or 0.28}
    params = apply_overrides(params, args)

    strikes = args.strike
    if strikes is None:
        S0 = params["S0"]
        strikes = [round(S0 * f, 2) for f in [0.90, 0.95, 1.00, 1.05, 1.10, 1.20]]

    run_gbm_demo(
        S0=params["S0"], mu=params["mu"], sigma=params["sigma"],
        r=args.rate, T=args.maturity, n_paths=args.n_paths,
        strikes=strikes, seed=args.seed,
        no_plots=args.no_plots, save_plots=args.save_plots,
    )


def cmd_brier(args):
    from quant_sim.calibration.brier import run_brier_demo
    run_brier_demo(
        n_samples=args.n_samples, n_bins=args.n_bins,
        no_plots=args.no_plots, save_plots=args.save_plots,
    )


def cmd_importance(args):
    from quant_sim.importance_sampling.tail_risk import run_importance_sampling_demo

    if args.ticker:
        params = calibrate_from_ticker(args.ticker, args.period)
    else:
        params = {"S0": args.spot or 100.0, "mu": args.drift or 0.10, "sigma": args.vol or 0.25}
    params = apply_overrides(params, args)

    run_importance_sampling_demo(
        S0=params["S0"], mu=params["mu"], sigma=params["sigma"],
        T=args.maturity, threshold_multiple=args.threshold,
        n_samples=args.n_paths, no_plots=args.no_plots, save_plots=args.save_plots,
    )


def cmd_particle(args):
    from quant_sim.particle_filter.smc import run_particle_filter_demo

    if args.ticker:
        params = calibrate_from_ticker(args.ticker, args.period)
    else:
        params = {"S0": 100.0, "mu": args.drift or 0.10, "sigma": args.vol or 0.25}
    params = apply_overrides(params, args)

    run_particle_filter_demo(
        S0=params.get("S0", 100.0), mu=params["mu"], sigma=params["sigma"],
        T_days=args.n_steps, n_particles=args.n_particles, obs_noise=args.obs_noise,
        seed=args.seed, no_plots=args.no_plots, save_plots=args.save_plots,
    )


def cmd_variance(args):
    from quant_sim.variance_reduction.techniques import run_variance_reduction_demo

    if args.ticker:
        params = calibrate_from_ticker(args.ticker, args.period)
    else:
        params = {"S0": args.spot or 100.0, "mu": args.drift or 0.10, "sigma": args.vol or 0.25}
    params = apply_overrides(params, args)

    S0 = params["S0"]
    strike_pct = (args.strike / S0) if args.strike else 1.05

    run_variance_reduction_demo(
        S0=S0, mu=params["mu"], sigma=params["sigma"],
        T=args.maturity, strike_pct=strike_pct, n_samples=args.n_paths,
        seed=args.seed, no_plots=args.no_plots, save_plots=args.save_plots,
    )


def cmd_copula(args):
    from quant_sim.copulas.models import run_copula_demo
    run_copula_demo(
        tickers=args.tickers,
        period=args.period,
        n_scenarios=args.n_paths,
        T=args.maturity,
        rho=args.correlation,
        drop_threshold=args.drop_threshold,
        gain_threshold=args.gain_threshold,
        student_t_df=args.df,
        clayton_theta=args.theta,
        seed=args.seed,
        no_plots=args.no_plots,
        save_plots=args.save_plots,
    )


def cmd_abm(args):
    from quant_sim.abm.market import run_abm_demo

    if args.ticker:
        params = calibrate_from_ticker(args.ticker, args.period)
    else:
        params = {"S0": args.spot or 100.0, "mu": args.drift or 0.08, "sigma": args.vol or 0.20}
    params = apply_overrides(params, args)

    run_abm_demo(
        S0=params["S0"], mu=params["mu"], sigma=params["sigma"],
        T=args.maturity, n_steps=args.n_steps,
        n_informed=args.n_informed, n_noise=args.n_noise, n_market_makers=args.n_mm,
        seed=args.seed, no_plots=args.no_plots, save_plots=args.save_plots,
    )


def cmd_all(args):
    handlers = [
        ("GBM", lambda: cmd_gbm(args)),
        ("Brier Score", lambda: cmd_brier(args)),
        ("Importance Sampling", lambda: cmd_importance(args)),
        ("Particle Filter", lambda: cmd_particle(args)),
        ("Variance Reduction", lambda: cmd_variance(args)),
        ("Copula Models", lambda: cmd_copula(args)),
        ("Agent-Based Market", lambda: cmd_abm(args)),
    ]

    timings = []
    for name, fn in handlers:
        t0 = time.perf_counter()
        fn()
        elapsed = time.perf_counter() - t0
        timings.append((name, elapsed))

    print("\n" + "=" * 50)
    print("  All simulations complete. Timings:")
    print("=" * 50)
    for name, elapsed in timings:
        print(f"  {name:<25}  {elapsed:>6.1f}s")
    print(f"  {'TOTAL':<25}  {sum(t for _, t in timings):>6.1f}s")
    print()


def cmd_list(_args):
    print("\n  Available commands:\n")
    for cmd, desc in COMMANDS.items():
        if cmd in ("all", "list"):
            continue
        print(f"    {cmd:<14}  {desc}")
    print()
    print("  Run any command with --help for per-command options.")
    print("  Add --ticker SYMBOL to auto-fetch and calibrate from yfinance.\n")


# ─── Dispatch ─────────────────────────────────────────────────────────────────

DISPATCH = {
    "gbm": cmd_gbm,
    "brier": cmd_brier,
    "importance": cmd_importance,
    "particle": cmd_particle,
    "variance": cmd_variance,
    "copula": cmd_copula,
    "abm": cmd_abm,
    "all": cmd_all,
    "list": cmd_list,
}


def main():
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend; individual commands re-enable if needed

    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    # Create save-plots directory if needed
    if args.save_plots:
        os.makedirs(args.save_plots, exist_ok=True)

    # Re-enable interactive backend if plots are requested to display
    if not args.no_plots and not args.save_plots:
        try:
            matplotlib.use("TkAgg")
        except Exception:
            try:
                matplotlib.use("Qt5Agg")
            except Exception:
                pass  # fall back to Agg (headless); plots will silently skip

    np.random.seed(args.seed)

    try:
        DISPATCH[args.command](args)
    except KeyboardInterrupt:
        print("\n  Interrupted.")
        sys.exit(1)
    except ImportError as e:
        print(f"\n  Error: {e}")
        print("  Run: pip install -r requirements.txt")
        sys.exit(1)


if __name__ == "__main__":
    main()
