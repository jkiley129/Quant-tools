#!/usr/bin/env python3
"""
Quant-Tools Simulation Engine — CLI Entry Point

Usage:
  python main.py <command> [options]

Commands:
  gbm           Monte Carlo GBM binary contract pricing
  greeks        Vanilla Black-Scholes pricing + all option Greeks
  distributions MLE fat-tail fitting + permutation significance test
  portfolio     Markowitz efficient frontier + Kelly criterion
  factors       PCA factor analysis + Fama-French style regression
  lmsr          LMSR automated prediction market (Polymarket math)
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
  python main.py greeks --spot 100 --vol 0.20 --maturity 1.0
  python main.py distributions --ticker SPY --period 5y
  python main.py portfolio --tickers AAPL MSFT NVDA AMZN --period 3y
  python main.py factors --tickers AAPL MSFT NVDA AMZN GOOG --period 3y
  python main.py lmsr --true-prob 0.65 --n-traders 300
  python main.py gbm --ticker NVDA --strike 1000 --maturity 0.5
  python main.py copula --tickers NVDA AMD SMCI --period 2y
  python main.py all --seed 42 --no-plots
"""

import argparse
import sys
import time
import os


# ─── Demo registry ────────────────────────────────────────────────────────────

COMMANDS = {
    "gbm":           "Monte Carlo GBM binary contract pricing",
    "greeks":        "Vanilla Black-Scholes pricing + all option Greeks (Δ Γ Θ ν ρ)",
    "distributions": "MLE fat-tail fitting (Student-t) + permutation significance test",
    "portfolio":     "Markowitz efficient frontier + Kelly criterion portfolio sizing",
    "factors":       "PCA factor analysis + Fama-French style regression (Newey-West SEs)",
    "lmsr":          "LMSR automated prediction market — the math behind Polymarket",
    "brier":         "Brier Score calibration analysis",
    "importance":    "Importance Sampling for tail risk events",
    "particle":      "Particle Filter real-time probability updating",
    "variance":      "Variance reduction technique comparison",
    "copula":        "Copula model correlated asset simulation",
    "abm":           "Agent-Based Market simulation (LOB + informed/noise/MM agents)",
    "all":           "Run all simulations sequentially",
    "list":          "List all available simulations",
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

    # ── greeks ────────────────────────────────────────────────────────────────
    p_greeks = subparsers.add_parser("greeks", help=COMMANDS["greeks"])
    p_greeks.add_argument("--spot",    type=float, default=100.0, help="Spot price S0 (default: 100)")
    p_greeks.add_argument("--rate",    type=float, default=0.05,  help="Risk-free rate (default: 0.05)")
    p_greeks.add_argument("--vol",     type=float, default=0.20,  help="Implied volatility (default: 0.20)")
    p_greeks.add_argument("--maturity",type=float, default=1.0,   help="Time to expiry in years (default: 1.0)")

    # ── distributions ─────────────────────────────────────────────────────────
    p_dist = subparsers.add_parser("distributions", help=COMMANDS["distributions"])
    p_dist.add_argument("--ticker", metavar="SYMBOL", help="Fetch real returns from yfinance")
    p_dist.add_argument("--period", default="5y",     help="yfinance period (default: 5y)")

    # ── portfolio ─────────────────────────────────────────────────────────────
    p_port = subparsers.add_parser("portfolio", help=COMMANDS["portfolio"])
    p_port.add_argument("--tickers", nargs="+", metavar="SYMBOL",
                        help="Tickers to include in the portfolio")
    p_port.add_argument("--period",  default="3y",  help="yfinance period (default: 3y)")
    p_port.add_argument("--rate",    type=float, default=0.05, help="Risk-free rate (default: 0.05)")
    p_port.add_argument("--allow-short", action="store_true", help="Allow short positions")

    # ── factors ───────────────────────────────────────────────────────────────
    p_fac = subparsers.add_parser("factors", help=COMMANDS["factors"])
    p_fac.add_argument("--tickers", nargs="+", metavar="SYMBOL",
                       help="At least 4 tickers for meaningful PCA")
    p_fac.add_argument("--period",       default="3y", help="yfinance period (default: 3y)")
    p_fac.add_argument("--n-components", type=int, default=3,
                       help="Number of PCA components (default: 3)")

    # ── lmsr ──────────────────────────────────────────────────────────────────
    p_lmsr = subparsers.add_parser("lmsr", help=COMMANDS["lmsr"])
    p_lmsr.add_argument("--true-prob",  type=float, default=0.65,
                        help="True probability of YES outcome (default: 0.65)")
    p_lmsr.add_argument("--n-traders",  type=int,   default=200,
                        help="Number of traders in simulation (default: 200)")
    p_lmsr.add_argument("--b",          type=float, default=50.0,
                        help="LMSR liquidity parameter b (default: 50)")

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


def cmd_greeks(args):
    from quant_sim.greeks.pricing import run_greeks_demo
    run_greeks_demo(
        S0=args.spot, r=args.rate, sigma=args.vol, T=args.maturity,
        n_sims=args.n_paths, seed=args.seed,
        no_plots=args.no_plots, save_plots=args.save_plots,
    )


def cmd_distributions(args):
    from quant_sim.distributions.fitting import run_distributions_demo
    run_distributions_demo(
        ticker=getattr(args, "ticker", None),
        period=getattr(args, "period", "5y"),
        seed=args.seed,
        no_plots=args.no_plots, save_plots=args.save_plots,
    )


def cmd_portfolio(args):
    from quant_sim.portfolio.optimize import run_portfolio_demo
    run_portfolio_demo(
        tickers=getattr(args, "tickers", None),
        period=getattr(args, "period", "3y"),
        r=getattr(args, "rate", 0.05),
        allow_short=getattr(args, "allow_short", False),
        seed=args.seed,
        no_plots=args.no_plots, save_plots=args.save_plots,
    )


def cmd_factors(args):
    from quant_sim.factors.analysis import run_factors_demo
    run_factors_demo(
        tickers=getattr(args, "tickers", None),
        period=getattr(args, "period", "3y"),
        n_pca_components=getattr(args, "n_components", 3),
        seed=args.seed,
        no_plots=args.no_plots, save_plots=args.save_plots,
    )


def cmd_lmsr(args):
    from quant_sim.lmsr.market import run_lmsr_demo
    run_lmsr_demo(
        true_prob=args.true_prob,
        n_traders=args.n_traders,
        b=args.b,
        seed=args.seed,
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
        ("GBM",                lambda: cmd_gbm(args)),
        ("Greeks",             lambda: cmd_greeks(args)),
        ("Distributions",      lambda: cmd_distributions(args)),
        ("Portfolio",          lambda: cmd_portfolio(args)),
        ("Factors / PCA",      lambda: cmd_factors(args)),
        ("LMSR Market",        lambda: cmd_lmsr(args)),
        ("Brier Score",        lambda: cmd_brier(args)),
        ("Importance Sampling",lambda: cmd_importance(args)),
        ("Particle Filter",    lambda: cmd_particle(args)),
        ("Variance Reduction", lambda: cmd_variance(args)),
        ("Copula Models",      lambda: cmd_copula(args)),
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
    "gbm":           cmd_gbm,
    "greeks":        cmd_greeks,
    "distributions": cmd_distributions,
    "portfolio":     cmd_portfolio,
    "factors":       cmd_factors,
    "lmsr":          cmd_lmsr,
    "brier":         cmd_brier,
    "importance":    cmd_importance,
    "particle":      cmd_particle,
    "variance":      cmd_variance,
    "copula":        cmd_copula,
    "abm":           cmd_abm,
    "all":           cmd_all,
    "list":          cmd_list,
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
