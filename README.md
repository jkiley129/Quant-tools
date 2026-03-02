# Quant-Tools

A self-contained Python toolkit for institutional-grade quantitative finance simulation. Run Monte Carlo pricing, importance sampling, particle filters, variance reduction, copula models, and agent-based markets — all from a single CLI, with optional real-ticker calibration via yfinance.

---

## Quickstart

```bash
pip install -r requirements.txt

# Run any simulation
python main.py list                                          # show all commands
python main.py --no-plots all                               # run every demo sequentially

# Synthetic demos (no internet required)
python main.py --no-plots gbm
python main.py --no-plots copula
python main.py --no-plots abm

# Real-ticker analysis (requires internet)
python main.py gbm --ticker NVDA --strike 1000 --maturity 0.5
python main.py copula --tickers NVDA AMD SMCI --period 2y
python main.py importance --ticker NVDA --threshold 1.5
```

---

## What's Inside

| Module | Class | What it answers |
|---|---|---|
| `monte_carlo/gbm.py` | `GBMSimulator` | What is P(NVDA > $1000 in 6 months)? |
| `calibration/brier.py` | `BrierScorer` | Is my probability forecast well-calibrated? |
| `importance_sampling/tail_risk.py` | `ImportanceSampler` | How do I efficiently estimate rare 50%-gain events? |
| `variance_reduction/techniques.py` | `VarianceReducer` | Which sampling technique gives the tightest confidence interval? |
| `particle_filter/smc.py` | `ParticleFilter` | How does the probability of hitting a target update as prices arrive? |
| `copulas/models.py` | `GaussianCopula`, `StudentTCopula`, `ClaytonCopula` | Do NVDA, AMD, and SMCI crash together more than Gaussian assumes? |
| `abm/market.py` | `PredictionMarket` | How quickly does a market with informed and noise traders converge to fair value? |

---

## CLI Reference

### Global options (must come before the subcommand)

```
python main.py [GLOBAL OPTIONS] COMMAND [COMMAND OPTIONS]

  -s, --seed SEED      Random seed for reproducibility (default: 42)
  -n, --n-paths N      Simulation paths or samples (default: 100000)
  --no-plots           Suppress matplotlib display (useful in CI/headless)
  --save-plots DIR     Save all plots to DIR as PNG files instead of displaying
```

### Commands

#### `gbm` — Monte Carlo binary contract pricing

Prices digital (binary) options under Geometric Brownian Motion and cross-checks against the analytical lognormal formula.

```bash
# Synthetic: AAPL-like parameters
python main.py --no-plots gbm

# Real ticker: auto-fetch price and calibrate sigma/mu from 1y history
python main.py gbm --ticker NVDA --strike 1000 --maturity 0.5

# Custom parameters
python main.py --no-plots gbm --spot 100 --vol 0.30 --drift 0.12 --strike 115 --maturity 1.0
```

Options: `--ticker`, `--period`, `--spot`, `--drift`, `--vol`, `--rate`, `--maturity`, `--strike`

Sample output:
```
  Strike   MC Price  Analytical  Std Error                  95% CI
-----------------------------------------------------------------
  163.8      0.7690      0.7692   0.001333  [0.7663, 0.7716]
  182.0      0.5780      0.5808   0.001562  [0.5749, 0.5810]
  200.2      0.3878      0.3908   0.001541  [0.3848, 0.3909]
```

---

#### `brier` — Brier Score calibration analysis

Decomposes forecast accuracy into **Uncertainty** (irreducible), **Resolution** (discriminative power), and **Reliability** (calibration error), comparing a well-calibrated vs overconfident forecaster.

```bash
python main.py --no-plots brier
python main.py --no-plots brier --n-samples 5000 --n-bins 15
```

Sample output:
```
  Well-calibrated
    Brier Score:   0.1891
    Uncertainty:   0.2498  (irreducible)
    Resolution:    0.0624  (higher = better)
    Reliability:   0.0010  (lower = better)
    Skill Score:   +0.2429

  Overconfident
    Reliability:   0.0185  (lower = better)
    Skill Score:   +0.1776
```

---

#### `importance` — Importance Sampling for tail risk

Estimates P(S_T > threshold) using exponential tilting — shifting the GBM drift toward the rare event. Achieves orders-of-magnitude variance reduction for tail probabilities that naive Monte Carlo struggles to estimate.

```bash
# Estimate P(S_T > 1.5*S0): a 50% gain event over 6 months
python main.py --no-plots importance

# Real ticker
python main.py importance --ticker NVDA --threshold 1.5 --maturity 0.5
```

Sample output:
```
  Method            Estimate   Std Error    95% CI Width   Time(s)
  Naive MC          0.017910    0.000419        0.001644     0.006
  Import. Samp.     0.018069    0.000001        0.000004     0.016
  Analytical        0.017900

  Variance reduction ratio: 212047x
  CVaR(5%): $72.02  |  VaR(5%): $77.44
```

---

#### `variance` — Variance reduction comparison

Prices a binary call using four methods and reports variance reduction ratios relative to crude Monte Carlo.

| Method | Mechanism | Typical VRR |
|---|---|---|
| Crude MC | Baseline | 1x |
| Antithetic variates | Use Z and −Z pairs | ~17x |
| Control variates | OLS correction using known E[S_T] | ~3x |
| Stratified sampling | Force uniform quantile coverage | ~130x |

```bash
python main.py --no-plots variance
python main.py variance --ticker NVDA --strike 1000 --maturity 0.5
```

---

#### `particle` — Particle Filter real-time probability updating

Tracks the current price estimate from noisy observations and computes the forward-looking probability P(S_end > K) at each time step using sequential importance resampling.

```bash
python main.py --no-plots particle
python main.py particle --n-particles 2000 --obs-noise 0.003 --n-steps 252
```

Sample output — probability evolves as price path becomes clearer:
```
    Day   Filter Price   True Price     Error      ESS    P(end>K)
      0         101.39       100.81     +0.58      189      0.4797
     50          85.65        85.34     +0.31      245      0.1917
    251         106.87       105.53     +1.35      145      0.0349
```

---

#### `copula` — Correlated multi-asset simulation

Simulates joint terminal prices under three copula structures and quantifies tail dependence differences.

| Copula | Tail behavior | Use case |
|---|---|---|
| Gaussian | No tail dependence | Standard linear correlation |
| Student-t | Symmetric upper + lower tail | Crash AND rally clustering |
| Clayton | Lower tail only | Assets crash together, recover independently |

```bash
# Synthetic 2-asset (SPY/QQQ-like) comparison
python main.py --no-plots copula

# Real 3-asset semiconductor sector analysis
python main.py copula --tickers NVDA AMD SMCI --period 2y
```

Sample output:
```
  Copula                   Emp. Corr  P(all drop>10%)  P(all gain>20%)
  Gaussian                    0.7516            0.0671            0.1058
  Student-t (df=4.0)          0.7398            0.0697            0.1088
  Clayton (θ=2.0)             0.6880            0.0857            0.0741
```
Clayton's higher `P(all drop>10%)` and lower `P(all gain>20%)` confirms lower-tail dependence — the sector is more likely to crash together than rally together.

---

#### `abm` — Agent-Based prediction market

Simulates a limit order book with three agent types competing in a binary prediction market:

- **InformedAgent** — knows the true event probability; trades on mispricing
- **NoiseAgent** — trades randomly; provides liquidity
- **MarketMakerAgent** — continuously quotes bid/ask spread; manages inventory

```bash
python main.py --no-plots abm
python main.py --no-plots abm --n-steps 500 --n-informed 5 --n-noise 15 --n-mm 3
```

Sample output:
```
  Total trades executed: 722
  Price error RMSE:      0.3273
  Convergence half-life: 5 steps
  Mean bid-ask spread:   0.0080

  Agent P&L by type:
    InformedAgent       :      +0.12
    NoiseAgent          :      -0.57
    MarketMakerAgent    :      +2.78
```

---

#### `all` — Run everything

```bash
python main.py --no-plots --seed 42 all
python main.py --save-plots ./plots all      # save every plot to ./plots/
```

---

## Using as a Python Library

All simulation engines are plain Python classes — import them directly for notebooks, scripts, or a backend API.

```python
from quant_sim.monte_carlo.gbm import GBMSimulator
from quant_sim.copulas.models import ClaytonCopula, StudentTCopula
from quant_sim.data.calibrator import calibrate_multi
import numpy as np

# Price a binary call on NVDA
sim = GBMSimulator(S0=875, mu=0.25, sigma=0.48, r=0.05, T=0.5, n_paths=100_000)
result = sim.price_binary_call(K=1000)
print(f"P(NVDA > $1000 in 6mo): {result['mc_price']:.4f} ± {result['std_error']:.4f}")

# Calibrate from real data and run copula analysis
calib = calibrate_multi(["NVDA", "AMD", "SMCI"], period="2y")
rho = calib["correlation_matrix"]
params = calib["params"]

cop = StudentTCopula(rho_matrix=rho, df=4.0)
prices = cop.simulate_asset_prices(
    n=100_000,
    S0_list=[p["S0"] for p in params],
    mu_list=[p["mu"] for p in params],
    sigma_list=[p["sigma"] for p in params],
    r=0.05,
    T=0.5,
)
S0_arr = np.array([p["S0"] for p in params])
joint_crash = np.mean(np.all(prices < S0_arr * 0.80, axis=1))
print(f"P(all three drop >20%): {joint_crash:.4f}")
```

---

## Architecture

```
quant_sim/
├── data/
│   ├── fetcher.py       # yfinance wrapper: fetch_prices(), fetch_multi()
│   └── calibrator.py    # GBM calibration: calibrate_gbm(), calibrate_multi()
├── utils/
│   ├── stats.py         # ESS, systematic_resample, log_likelihood_ratio
│   └── plotting.py      # matplotlib helpers (price paths, reliability diagrams, etc.)
├── monte_carlo/
│   └── gbm.py           # GBMSimulator
├── calibration/
│   └── brier.py         # BrierScorer
├── importance_sampling/
│   └── tail_risk.py     # ImportanceSampler
├── variance_reduction/
│   └── techniques.py    # VarianceReducer
├── particle_filter/
│   └── smc.py           # ParticleFilter, GBMStateSpaceModel
├── copulas/
│   └── models.py        # GaussianCopula, StudentTCopula, ClaytonCopula
└── abm/
    ├── order_book.py    # LimitOrderBook (price-time priority FIFO)
    ├── agents.py        # InformedAgent, NoiseAgent, MarketMakerAgent
    └── market.py        # PredictionMarket
main.py                  # CLI entry point (argparse subcommands)
```

**Dependency flow:** `utils/` ← `monte_carlo/` ← `importance_sampling/`, `variance_reduction/`, `abm/`. No circular imports. All math implemented from scratch using numpy + scipy.

**Frontend-ready:** The CLI layer (`main.py`) is thin — each command calls a `run_*_demo()` function that wraps the underlying class. A FastAPI or Streamlit frontend can import `quant_sim` classes directly without touching the CLI layer.

---

## Dependencies

```
numpy>=1.26.0
scipy>=1.12.0
pandas>=2.2.0
matplotlib>=3.8.0
yfinance>=0.2.40
```

No QuantLib, PyMC, filterpy, pyvinecopulib, or other specialized quant libraries. All algorithms (copulas, particle filter, importance sampling, Brier decomposition) are implemented from scratch.

---

## Key Mathematical Notes

**GBM terminal price** — uses closed-form `S_T = S0 * exp((mu - 0.5*sigma²)*T + sigma*sqrt(T)*Z)`. The Ito correction `-0.5*sigma²` is essential; omitting it biases all probability estimates.

**Importance weights** — computed in log-space (`log_w = log_p - log_q`) with log-sum-exp stabilization before `exp()`. Prevents underflow when estimating probabilities that span many orders of magnitude.

**Particle filter resampling** — uses O(N) systematic resampling (not multinomial). Draws one uniform `u ~ U[0, 1/N]` then steps through cumulative weight CDF. Lower variance than multinomial at same cost.

**Clayton copula** — sampled via the Marshall-Olkin Gamma frailty algorithm, which generalizes to d > 2 dimensions. Produces lower tail dependence coefficient `λ_L = 2^(-1/θ) > 0`, upper tail coefficient `λ_U = 0`.

**Student-t copula** — sampled via Cholesky decomposition + chi-squared scaling (McNeil et al., 2005, Algorithm 5.10). Symmetric tail dependence: `λ_L = λ_U = 2 * t_{df+1}(-sqrt((df+1)*(1-rho)/(1+rho)))`.

**Brier decomposition** — Murphy (1973) three-way split: `BS = Uncertainty - Resolution + Reliability`. Bins with zero observations are excluded from Resolution and Reliability sums.
