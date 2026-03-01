"""
quant_sim — Quantitative Finance Simulation Engine

Modules:
  monte_carlo      — GBM-based binary contract pricing (GBMSimulator)
  calibration      — Brier Score calibration analysis (BrierScorer)
  importance_sampling — Tail risk via exponential tilting (ImportanceSampler)
  particle_filter  — Sequential Monte Carlo / particle filter (ParticleFilter)
  variance_reduction — Antithetic, control variate, stratified sampling (VarianceReducer)
  copulas          — Gaussian, Student-t, Clayton copulas
  abm              — Agent-based prediction market (PredictionMarket)
  data             — yfinance fetching + GBM calibration from historical data
"""

__version__ = "0.1.0"
__author__ = "Joe Kiley"

from quant_sim.monte_carlo.gbm import GBMSimulator
from quant_sim.calibration.brier import BrierScorer
from quant_sim.importance_sampling.tail_risk import ImportanceSampler
from quant_sim.particle_filter.smc import ParticleFilter, GBMStateSpaceModel
from quant_sim.variance_reduction.techniques import VarianceReducer
from quant_sim.copulas.models import GaussianCopula, StudentTCopula, ClaytonCopula
from quant_sim.abm.market import PredictionMarket

__all__ = [
    "GBMSimulator",
    "BrierScorer",
    "ImportanceSampler",
    "ParticleFilter",
    "GBMStateSpaceModel",
    "VarianceReducer",
    "GaussianCopula",
    "StudentTCopula",
    "ClaytonCopula",
    "PredictionMarket",
]
