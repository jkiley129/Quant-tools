"""
Quant-Tools Streamlit UI

Run with:
    streamlit run app.py
"""

import io
import os
import sys
import contextlib
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Quant-Tools",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

TOOLS = {
    "GBM — Binary Contract Pricing":        "gbm",
    "Greeks — Black-Scholes":               "greeks",
    "Distributions — Fat-tail Fitting":     "distributions",
    "Portfolio — Efficient Frontier":       "portfolio",
    "Factors — PCA & Fama-French":          "factors",
    "LMSR — Prediction Market":             "lmsr",
    "Brier Score — Calibration":            "brier",
    "Importance Sampling — Tail Risk":      "importance",
    "Particle Filter — SMC":               "particle",
    "Variance Reduction":                   "variance",
    "Copula — Correlated Assets":           "copula",
    "ABM — Agent-Based Market":             "abm",
}

TOOL_DESCRIPTIONS = {
    "gbm":           "Monte Carlo GBM binary contract pricing. Simulates many asset-price paths under Geometric Brownian Motion and prices digital contracts at multiple strikes.",
    "greeks":        "Vanilla Black-Scholes option pricing with all five Greeks (Δ, Γ, Θ, ν, ρ), plus a Monte Carlo cross-check and volatility-smile plots.",
    "distributions": "Fits a Student-t distribution to stock returns via MLE, measures fat-tail significance with a permutation test, and compares tail probabilities to a Normal.",
    "portfolio":     "Markowitz mean-variance efficient frontier with the Sharpe-optimal and minimum-variance portfolios, plus Kelly criterion sizing.",
    "factors":       "PCA factor extraction and Fama-French style multi-factor regression with Newey-West standard errors.",
    "lmsr":          "Logarithmic Market Scoring Rule automated market maker — the core math behind Polymarket. Simulates informed and noise traders converging on the true probability.",
    "brier":         "Brier Score decomposition (reliability + resolution + uncertainty) and reliability-diagram calibration analysis.",
    "importance":    "Importance Sampling via exponential tilting to efficiently estimate rare tail-risk probabilities (e.g. P[S_T > 1.5·S_0]).",
    "particle":      "Sequential Monte Carlo particle filter for real-time drift and volatility estimation from a noisy price stream.",
    "variance":      "Side-by-side comparison of Antithetic Variates, Control Variates, and Stratified Sampling variance-reduction techniques.",
    "copula":        "Simulates correlated multi-asset scenarios with Gaussian, Student-t, and Clayton copulas. Compares joint tail probabilities.",
    "abm":           "Agent-Based Market with a limit-order book populated by informed traders, noise traders, and market makers.",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

@contextlib.contextmanager
def capture_output():
    """Redirect stdout/stderr to StringIO buffers."""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        yield out, err


def run_and_display(fn, *args, **kwargs):
    """Run fn, capture printed output and matplotlib figures, display in Streamlit."""
    plt.close("all")

    with tempfile.TemporaryDirectory() as tmpdir:
        kwargs["save_plots"] = tmpdir
        kwargs["no_plots"] = False

        with capture_output() as (out, err):
            try:
                fn(*args, **kwargs)
                success = True
                error_msg = ""
            except Exception as exc:
                success = False
                error_msg = str(exc)

        stdout_text = out.getvalue()
        stderr_text = err.getvalue()

        # ── Text output ───────────────────────────────────────────────────────
        if stdout_text.strip():
            st.subheader("Output")
            st.code(stdout_text, language=None)

        if not success:
            st.error(f"Simulation error: {error_msg}")
            if stderr_text.strip():
                st.code(stderr_text, language=None)
            return

        # ── Plots ─────────────────────────────────────────────────────────────
        pngs = sorted(
            [f for f in os.listdir(tmpdir) if f.endswith(".png")],
            key=lambda f: os.path.getmtime(os.path.join(tmpdir, f)),
        )
        if pngs:
            st.subheader("Plots")
            for fname in pngs:
                st.image(os.path.join(tmpdir, fname), use_container_width=True)

    plt.close("all")


# ── Sidebar — tool selection ───────────────────────────────────────────────────

st.sidebar.title("Quant-Tools")
st.sidebar.caption("Quantitative Finance Simulation Engine")

tool_label = st.sidebar.radio(
    "Select a simulation",
    list(TOOLS.keys()),
    label_visibility="collapsed",
)
tool = TOOLS[tool_label]

st.sidebar.divider()
st.sidebar.caption("Global options")
seed    = st.sidebar.number_input("Random seed", value=42, step=1)
n_paths = st.sidebar.number_input("Simulation paths / samples", value=10_000, step=1_000, min_value=100)

# ── Main panel ────────────────────────────────────────────────────────────────

st.title(tool_label)
st.caption(TOOL_DESCRIPTIONS[tool])
st.divider()


# ════════════════════════════════════════════════════════════════════════════════
# Per-tool parameter forms
# ════════════════════════════════════════════════════════════════════════════════

if tool == "gbm":
    st.subheader("Parameters")
    c1, c2 = st.columns(2)
    with c1:
        ticker   = st.text_input("Ticker (optional, e.g. NVDA)", value="")
        period   = st.selectbox("Period", ["1y", "2y", "3y", "5y"], index=0)
        spot     = st.number_input("Spot price S₀ (ignored if ticker set)", value=182.0)
        drift    = st.number_input("Annual drift μ (ignored if ticker set)", value=0.12, format="%.4f")
    with c2:
        vol      = st.number_input("Volatility σ", value=0.28, format="%.4f")
        rate     = st.number_input("Risk-free rate r", value=0.05, format="%.4f")
        maturity = st.number_input("Maturity T (years)", value=0.5, format="%.4f")
        strikes_raw = st.text_input("Strikes (comma-separated, blank = auto)", value="")

    if st.button("Run GBM Simulation", type="primary"):
        from quant_sim.monte_carlo.gbm import run_gbm_demo
        from main import calibrate_from_ticker, apply_overrides

        import types
        fake_args = types.SimpleNamespace(
            ticker=ticker or None, period=period,
            spot=spot, drift=drift, vol=vol,
        )

        if ticker:
            with st.spinner(f"Fetching {ticker}..."):
                params = calibrate_from_ticker(ticker, period)
            params = apply_overrides(params, fake_args)
        else:
            params = {"S0": spot, "mu": drift, "sigma": vol}

        if strikes_raw.strip():
            strikes = [float(x.strip()) for x in strikes_raw.split(",")]
        else:
            S0 = params["S0"]
            strikes = [round(S0 * f, 2) for f in [0.90, 0.95, 1.00, 1.05, 1.10, 1.20]]

        with st.spinner("Running simulation..."):
            run_and_display(
                run_gbm_demo,
                S0=params["S0"], mu=params["mu"], sigma=params["sigma"],
                r=rate, T=maturity, n_paths=int(n_paths),
                strikes=strikes, seed=int(seed),
            )


elif tool == "greeks":
    st.subheader("Parameters")
    c1, c2 = st.columns(2)
    with c1:
        spot     = st.number_input("Spot price S₀", value=100.0)
        vol      = st.number_input("Implied volatility σ", value=0.20, format="%.4f")
    with c2:
        rate     = st.number_input("Risk-free rate r", value=0.05, format="%.4f")
        maturity = st.number_input("Maturity T (years)", value=1.0, format="%.4f")

    if st.button("Run Greeks Analysis", type="primary"):
        from quant_sim.greeks.pricing import run_greeks_demo
        with st.spinner("Computing Greeks..."):
            run_and_display(
                run_greeks_demo,
                S0=spot, r=rate, sigma=vol, T=maturity,
                n_sims=int(n_paths), seed=int(seed),
            )


elif tool == "distributions":
    st.subheader("Parameters")
    ticker = st.text_input("Ticker (optional, e.g. SPY — leave blank for synthetic data)", value="")
    period = st.selectbox("Period", ["1y", "2y", "3y", "5y"], index=3)

    if st.button("Run Distribution Fitting", type="primary"):
        from quant_sim.distributions.fitting import run_distributions_demo
        with st.spinner("Fitting distributions..."):
            run_and_display(
                run_distributions_demo,
                ticker=ticker or None,
                period=period,
                seed=int(seed),
            )


elif tool == "portfolio":
    st.subheader("Parameters")
    tickers_raw = st.text_input(
        "Tickers (space or comma separated, leave blank for synthetic)",
        value="AAPL MSFT NVDA AMZN",
    )
    c1, c2 = st.columns(2)
    with c1:
        period      = st.selectbox("Period", ["1y", "2y", "3y", "5y"], index=2)
        rate        = st.number_input("Risk-free rate r", value=0.05, format="%.4f")
    with c2:
        allow_short = st.checkbox("Allow short positions", value=False)

    if st.button("Optimize Portfolio", type="primary"):
        from quant_sim.portfolio.optimize import run_portfolio_demo
        tickers = [t.strip() for t in tickers_raw.replace(",", " ").split() if t.strip()] or None
        with st.spinner("Running optimization..."):
            run_and_display(
                run_portfolio_demo,
                tickers=tickers,
                period=period,
                r=rate,
                allow_short=allow_short,
                seed=int(seed),
            )


elif tool == "factors":
    st.subheader("Parameters")
    tickers_raw = st.text_input(
        "Tickers (space or comma separated, at least 4)",
        value="AAPL MSFT NVDA AMZN GOOG",
    )
    c1, c2 = st.columns(2)
    with c1:
        period       = st.selectbox("Period", ["1y", "2y", "3y", "5y"], index=2)
    with c2:
        n_components = st.number_input("PCA components", value=3, min_value=1, max_value=10, step=1)

    if st.button("Run Factor Analysis", type="primary"):
        from quant_sim.factors.analysis import run_factors_demo
        tickers = [t.strip() for t in tickers_raw.replace(",", " ").split() if t.strip()] or None
        with st.spinner("Running factor analysis..."):
            run_and_display(
                run_factors_demo,
                tickers=tickers,
                period=period,
                n_pca_components=int(n_components),
                seed=int(seed),
            )


elif tool == "lmsr":
    st.subheader("Parameters")
    c1, c2, c3 = st.columns(3)
    with c1:
        true_prob = st.number_input("True probability of YES", value=0.65, min_value=0.01, max_value=0.99, format="%.4f")
    with c2:
        n_traders = st.number_input("Number of traders", value=200, min_value=10, step=10)
    with c3:
        b = st.number_input("Liquidity parameter b", value=50.0, min_value=1.0, format="%.1f")

    if st.button("Run LMSR Market", type="primary"):
        from quant_sim.lmsr.market import run_lmsr_demo
        with st.spinner("Simulating prediction market..."):
            run_and_display(
                run_lmsr_demo,
                true_prob=true_prob,
                n_traders=int(n_traders),
                b=b,
                seed=int(seed),
            )


elif tool == "brier":
    st.subheader("Parameters")
    c1, c2 = st.columns(2)
    with c1:
        n_samples = st.number_input("Number of forecast samples", value=2000, min_value=100, step=100)
    with c2:
        n_bins = st.number_input("Reliability diagram bins", value=10, min_value=2, max_value=50, step=1)

    if st.button("Run Brier Score Analysis", type="primary"):
        from quant_sim.calibration.brier import run_brier_demo
        with st.spinner("Computing calibration..."):
            run_and_display(
                run_brier_demo,
                n_samples=int(n_samples),
                n_bins=int(n_bins),
            )


elif tool == "importance":
    st.subheader("Parameters")
    c1, c2 = st.columns(2)
    with c1:
        ticker    = st.text_input("Ticker (optional)", value="")
        period    = st.selectbox("Period", ["1y", "2y", "3y", "5y"], index=0)
        spot      = st.number_input("Spot price S₀ (if no ticker)", value=100.0)
        drift     = st.number_input("Annual drift μ (if no ticker)", value=0.10, format="%.4f")
    with c2:
        vol       = st.number_input("Volatility σ (if no ticker)", value=0.25, format="%.4f")
        maturity  = st.number_input("Maturity T (years)", value=0.5, format="%.4f")
        threshold = st.number_input("Threshold (multiple of S₀, e.g. 1.5 = +50%)", value=1.5, format="%.4f")
        shift     = st.number_input("IS drift shift factor", value=0.5, format="%.4f")

    if st.button("Run Importance Sampling", type="primary"):
        from quant_sim.importance_sampling.tail_risk import run_importance_sampling_demo
        from main import calibrate_from_ticker, apply_overrides
        import types

        fake_args = types.SimpleNamespace(spot=spot, drift=drift, vol=vol)
        if ticker:
            with st.spinner(f"Fetching {ticker}..."):
                params = calibrate_from_ticker(ticker, period)
            params = apply_overrides(params, fake_args)
        else:
            params = {"S0": spot, "mu": drift, "sigma": vol}

        with st.spinner("Running simulation..."):
            run_and_display(
                run_importance_sampling_demo,
                S0=params["S0"], mu=params["mu"], sigma=params["sigma"],
                T=maturity, threshold_multiple=threshold,
                n_samples=int(n_paths),
            )


elif tool == "particle":
    st.subheader("Parameters")
    c1, c2 = st.columns(2)
    with c1:
        ticker      = st.text_input("Ticker (optional)", value="")
        period      = st.selectbox("Period", ["1y", "2y", "3y", "5y"], index=0)
        drift       = st.number_input("Annual drift μ (if no ticker)", value=0.10, format="%.4f")
        vol         = st.number_input("Volatility σ (if no ticker)", value=0.25, format="%.4f")
    with c2:
        n_particles = st.number_input("Number of particles", value=1000, min_value=100, step=100)
        obs_noise   = st.number_input("Observation noise std", value=0.003, format="%.5f")
        n_steps     = st.number_input("Simulation days", value=252, min_value=10, step=10)

    if st.button("Run Particle Filter", type="primary"):
        from quant_sim.particle_filter.smc import run_particle_filter_demo
        from main import calibrate_from_ticker, apply_overrides
        import types

        fake_args = types.SimpleNamespace(drift=drift, vol=vol, spot=None)
        if ticker:
            with st.spinner(f"Fetching {ticker}..."):
                params = calibrate_from_ticker(ticker, period)
            params = apply_overrides(params, fake_args)
        else:
            params = {"S0": 100.0, "mu": drift, "sigma": vol}

        with st.spinner("Running particle filter..."):
            run_and_display(
                run_particle_filter_demo,
                S0=params.get("S0", 100.0), mu=params["mu"], sigma=params["sigma"],
                T_days=int(n_steps), n_particles=int(n_particles), obs_noise=obs_noise,
                seed=int(seed),
            )


elif tool == "variance":
    st.subheader("Parameters")
    c1, c2 = st.columns(2)
    with c1:
        ticker   = st.text_input("Ticker (optional)", value="")
        period   = st.selectbox("Period", ["1y", "2y", "3y", "5y"], index=0)
        spot     = st.number_input("Spot price S₀ (if no ticker)", value=100.0)
        drift    = st.number_input("Annual drift μ (if no ticker)", value=0.10, format="%.4f")
    with c2:
        vol      = st.number_input("Volatility σ (if no ticker)", value=0.25, format="%.4f")
        maturity = st.number_input("Maturity T (years)", value=0.5, format="%.4f")
        strike   = st.number_input("Strike price (0 = 1.05×S₀)", value=0.0)
        n_strata = st.number_input("Strata for stratified sampling", value=100, min_value=10, step=10)

    if st.button("Run Variance Reduction Comparison", type="primary"):
        from quant_sim.variance_reduction.techniques import run_variance_reduction_demo
        from main import calibrate_from_ticker, apply_overrides
        import types

        fake_args = types.SimpleNamespace(spot=spot, drift=drift, vol=vol)
        if ticker:
            with st.spinner(f"Fetching {ticker}..."):
                params = calibrate_from_ticker(ticker, period)
            params = apply_overrides(params, fake_args)
        else:
            params = {"S0": spot, "mu": drift, "sigma": vol}

        S0 = params["S0"]
        strike_pct = (strike / S0) if strike > 0 else 1.05

        with st.spinner("Running comparison..."):
            run_and_display(
                run_variance_reduction_demo,
                S0=S0, mu=params["mu"], sigma=params["sigma"],
                T=maturity, strike_pct=strike_pct, n_samples=int(n_paths),
                seed=int(seed),
            )


elif tool == "copula":
    st.subheader("Parameters")
    c1, c2 = st.columns(2)
    with c1:
        tickers_raw    = st.text_input("Tickers (optional, e.g. NVDA AMD SMCI)", value="")
        period         = st.selectbox("Period", ["1y", "2y", "3y", "5y"], index=0)
        correlation    = st.number_input("Pairwise correlation (synthetic mode)", value=0.75, min_value=-1.0, max_value=1.0, format="%.4f")
        df             = st.number_input("Student-t degrees of freedom", value=4.0, min_value=2.0, format="%.1f")
    with c2:
        theta          = st.number_input("Clayton theta", value=2.0, min_value=0.1, format="%.2f")
        drop_threshold = st.number_input("Joint drop threshold (e.g. 0.10 = 10%)", value=0.10, format="%.4f")
        gain_threshold = st.number_input("Joint gain threshold (e.g. 0.20 = 20%)", value=0.20, format="%.4f")
        maturity       = st.number_input("Maturity T (years)", value=0.5, format="%.4f")

    if st.button("Run Copula Simulation", type="primary"):
        from quant_sim.copulas.models import run_copula_demo
        tickers = [t.strip() for t in tickers_raw.replace(",", " ").split() if t.strip()] or None
        with st.spinner("Simulating correlated scenarios..."):
            run_and_display(
                run_copula_demo,
                tickers=tickers,
                period=period,
                n_scenarios=int(n_paths),
                T=maturity,
                rho=correlation,
                drop_threshold=drop_threshold,
                gain_threshold=gain_threshold,
                student_t_df=df,
                clayton_theta=theta,
                seed=int(seed),
            )


elif tool == "abm":
    st.subheader("Parameters")
    c1, c2 = st.columns(2)
    with c1:
        ticker     = st.text_input("Ticker (optional)", value="")
        period     = st.selectbox("Period", ["1y", "2y", "3y", "5y"], index=0)
        spot       = st.number_input("Spot price S₀ (if no ticker)", value=100.0)
        drift      = st.number_input("Annual drift μ (if no ticker)", value=0.08, format="%.4f")
        vol        = st.number_input("Volatility σ (if no ticker)", value=0.20, format="%.4f")
    with c2:
        maturity   = st.number_input("Maturity T (years)", value=0.5, format="%.4f")
        n_steps    = st.number_input("Time steps", value=300, min_value=10, step=10)
        n_informed = st.number_input("Informed agents", value=3, min_value=1, step=1)
        n_noise    = st.number_input("Noise agents", value=10, min_value=1, step=1)
        n_mm       = st.number_input("Market makers", value=2, min_value=1, step=1)

    if st.button("Run ABM Simulation", type="primary"):
        from quant_sim.abm.market import run_abm_demo
        from main import calibrate_from_ticker, apply_overrides
        import types

        fake_args = types.SimpleNamespace(spot=spot, drift=drift, vol=vol)
        if ticker:
            with st.spinner(f"Fetching {ticker}..."):
                params = calibrate_from_ticker(ticker, period)
            params = apply_overrides(params, fake_args)
        else:
            params = {"S0": spot, "mu": drift, "sigma": vol}

        with st.spinner("Running agent-based simulation..."):
            run_and_display(
                run_abm_demo,
                S0=params["S0"], mu=params["mu"], sigma=params["sigma"],
                T=maturity, n_steps=int(n_steps),
                n_informed=int(n_informed), n_noise=int(n_noise), n_market_makers=int(n_mm),
                seed=int(seed),
            )
