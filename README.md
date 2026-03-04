# Quant-Tools Simulation Engine

A quantitative finance toolkit covering the full stack described in *"How I'd Become a Quant If I Had to Start Over Tomorrow"* — from stochastic calculus through portfolio optimization, factor models, and prediction markets.

---

## Setup Guide

### Prerequisites

- **Python 3.10 or higher** — check with `python --version`
- **pip** — check with `pip --version`

If you don't have Python 3.10+, install it from [python.org](https://www.python.org/downloads/) or via your OS package manager.

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd Quant-tools
```

### 2. (Recommended) Create a virtual environment

```bash
python -m venv .venv

# Activate it:
# macOS / Linux:
source .venv/bin/activate

# Windows (PowerShell):
.venv\Scripts\Activate.ps1

# Windows (cmd.exe):
.venv\Scripts\activate.bat
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

This installs:

| Package | What it's for |
|---|---|
| `numpy`, `scipy` | Core numerics, MLE, statistics |
| `pandas` | Time series, data alignment |
| `matplotlib` | All plots |
| `yfinance` | Fetch real market data |
| `cvxpy` | Convex portfolio optimization (Markowitz) |
| `scikit-learn` | PCA decomposition |
| `statsmodels` | Factor regression with Newey-West SEs |

### 4. Verify the install

```bash
python main.py list
```

You should see all 12 available commands.

### 5. Run a quick smoke test (no internet required)

```bash
python main.py --no-plots all
```

This runs every simulation sequentially and prints results. Takes ~30–60 seconds.

### 6. Run the full test suite

```bash
pip install pytest
pytest tests/ -v
```

Expected: all tests pass.

---

## Commands

```
python main.py <command> [options]
```

### Global options (work with every command)

| Flag | Default | Description |
|---|---|---|
| `-s, --seed` | `42` | Random seed for reproducibility |
| `-n, --n-paths` | `100000` | Monte Carlo paths / samples |
| `--no-plots` | off | Suppress matplotlib windows |
| `--save-plots DIR` | off | Save plots as PNG files to DIR |

---

## Module Reference & Examples

### `greeks` — Vanilla Black-Scholes + option Greeks

Prices European calls and puts. Computes all five Greeks analytically:
- **Δ (delta)** — hedge ratio; how much the option moves per $1 stock move
- **Γ (gamma)** — convexity; how fast delta changes
- **Θ (theta)** — time decay per day
- **ν (vega)** — sensitivity to a 1% move in implied vol
- **ρ (rho)** — sensitivity to a 1% move in the risk-free rate

Verifies Monte Carlo converges to the closed-form price.

```bash
# Default synthetic example (S0=100, sigma=20%, T=1yr)
python main.py greeks

# Custom parameters
python main.py greeks --spot 150 --vol 0.30 --maturity 0.5 --rate 0.04

# Save plots instead of displaying
python main.py greeks --save-plots ./charts
```

---

### `distributions` — MLE fat-tail fitting + permutation testing

Demonstrates the post's core statistical insight: *"Most of what looks like signal is noise."*

1. Fetches real returns (or simulates synthetic fat-tailed data)
2. Tests normality with D'Agostino-Pearson, Shapiro-Wilk, and Jarque-Bera
3. Fits Normal vs Student-t via MLE — compares AIC scores
4. Runs a permutation test on a momentum signal to check if it beats a random coin-flip

```bash
# Synthetic Student-t(df=4) example (no internet needed)
python main.py distributions

# Real returns from yfinance
python main.py distributions --ticker SPY --period 5y
python main.py distributions --ticker NVDA --period 3y
```

---

### `portfolio` — Markowitz efficient frontier + Kelly criterion

Implements two complementary approaches to portfolio construction:

**Markowitz (1952):** minimise variance for a target return, sweeping the efficient frontier.

**Kelly criterion (multi-asset):** maximise expected log-wealth:
`w* = inv(Sigma) * (mu - r)`. Fractional Kelly (half Kelly) is the practical choice — full Kelly is theoretically optimal but highly sensitive to estimation error.

```bash
# Synthetic 5-asset example (no internet needed)
python main.py portfolio

# Real tickers — long-only
python main.py portfolio --tickers AAPL MSFT NVDA AMZN GOOG --period 3y

# Allow short positions (up to -30% per asset)
python main.py portfolio --tickers AAPL MSFT NVDA AMZN --allow-short
```

---

### `factors` — PCA + Fama-French style regression

Two tools for decomposing what drives portfolio returns:

**PCA:** Extracts latent risk factors. In a 500-stock universe, the first 5 eigenvectors typically explain 70% of variance — the rest is idiosyncratic noise.

**Factor regression (Newey-West):** Regresses portfolio returns on factor returns:
`r_t = alpha + beta_1*F1 + ... + epsilon_t`

The **alpha** intercept is the return unexplained by known factors. Uses Newey-West HAC standard errors — standard OLS errors are wrong for financial time series (autocorrelation + heteroskedasticity).

```bash
# Synthetic 10-asset example (no internet needed)
python main.py factors

# Real tickers — needs at least 4 for meaningful PCA
python main.py factors --tickers AAPL MSFT NVDA AMZN GOOG META TSLA NFLX --period 3y

# Control number of PCA components
python main.py factors --tickers AAPL MSFT NVDA AMZN GOOG --n-components 2
```

---

### `lmsr` — LMSR automated prediction market

Implements the Logarithmic Market Scoring Rule (Robin Hanson, 2003) — the automated market maker powering Polymarket and other prediction markets.

**Cost function:** `C(q) = b * ln(sum_i exp(q_i / b))`

**Prices = softmax:** `p_i = exp(q_i/b) / sum_j exp(q_j/b)` — the same function that powers every neural network classifier.

**Key properties:**
- Prices always sum to 1 and lie in (0, 1) — infinite liquidity guaranteed
- Market maker's worst-case loss is bounded: `b * ln(n)`
- Informed traders push prices toward the true probability

```bash
# Default: binary market, true P(YES)=65%, 200 traders
python main.py lmsr

# Higher true probability
python main.py lmsr --true-prob 0.80 --n-traders 500

# More liquid market (higher b)
python main.py lmsr --b 200 --true-prob 0.55
```

---

### `gbm` — Monte Carlo GBM binary contract pricing

Prices binary (digital) options: pays $1 if S_T > K. Cross-checks Monte Carlo against the analytical Black-Scholes formula. Includes the Ito correction (`-sigma^2/2`) and Euler-Maruyama path simulation.

```bash
python main.py gbm

# Real ticker calibration
python main.py gbm --ticker NVDA --maturity 0.5
python main.py gbm --ticker AAPL --strike 200 210 220 --maturity 1.0
```

---

### `importance` — Importance Sampling for tail risk

Estimates rare-event probabilities (e.g. P(S_T > 1.5*S0)) using exponential tilting. Far more efficient than crude Monte Carlo for extreme tails.

```bash
python main.py importance

# 50% gain threshold
python main.py importance --ticker NVDA --threshold 1.5

# Custom threshold
python main.py importance --threshold 2.0 --maturity 1.0
```

---

### `variance` — Variance reduction techniques

Benchmarks four variance reduction methods against crude Monte Carlo:
- **Antithetic variates** (Z and -Z pairs)
- **Control variates** (OLS correction using known E[S_T])
- **Stratified sampling** (uniform quantile coverage)

```bash
python main.py variance
python main.py variance --ticker AAPL --maturity 1.0
```

---

### `copula` — Multi-asset tail dependence

Models correlations between assets during crashes vs normal times:
- **Gaussian copula** — linear correlation, no tail dependence
- **Student-t copula** — symmetric upper + lower tail dependence
- **Clayton copula** — lower tail dependence only (assets crash together, recover independently)

```bash
python main.py copula

# Real tickers
python main.py copula --tickers NVDA AMD SMCI --period 2y

# Custom parameters
python main.py copula --correlation 0.8 --df 3 --drop-threshold 0.15
```

---

### `particle` — Sequential Monte Carlo particle filter

Bayesian real-time updating of P(S_T > K | observations). Uses log-space weight updates to prevent underflow. Systematic resampling triggered by ESS threshold.

```bash
python main.py particle
python main.py particle --ticker SPY --n-particles 2000
```

---

### `brier` — Brier score calibration

Forecast calibration analysis with Murphy (1973) decomposition:
`BS = Uncertainty - Resolution + Reliability`

```bash
python main.py brier
python main.py brier --n-samples 5000 --n-bins 15
```

---

### `abm` — Agent-based prediction market

Limit order book simulation with three agent archetypes:
- **InformedAgent** — trades on true probability with mispricing threshold
- **NoiseAgent** — random trades (liquidity provision)
- **MarketMakerAgent** — continuous bid/ask quotes with inventory skewing

```bash
python main.py abm
python main.py abm --n-informed 5 --n-noise 20 --n-mm 3 --n-steps 500
```

---

## Running all modules at once

```bash
# All modules, no plots (fastest, ~60s)
python main.py --no-plots all

# All modules, save every plot
python main.py --save-plots ./output all

# All modules with a fixed seed
python main.py --seed 123 --no-plots all
```

---

## Learning path (from the post)

| Level | Topic | Module(s) |
|---|---|---|
| 1 | Probability & simulation | `gbm`, `particle` |
| 2 | Statistics & MLE | `distributions`, `brier` |
| 3 | Linear algebra & PCA | `factors`, `copula` |
| 4 | Convex optimization | `portfolio` |
| 5 | Stochastic calculus & Black-Scholes | `greeks`, `variance`, `importance` |
| + | Market microstructure | `abm`, `lmsr` |
