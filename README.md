# Multi-Model Price Prediction: BiLSTM-Attention vs. Baselines (Gold & Bitcoin)

## Overview

This repository implements and evaluates a BiLSTM-Attention architecture against seven baseline models
for daily price prediction of Gold (GC=F) and Bitcoin (BTC-USD). It also includes an Apriori-based
pattern analysis, an exponential position sizing model, and a VaR-constrained modified greedy trading
strategy simulation.

## Experiments

| ID | Title |
|----|-------|
| exp_1 | Multi-Model Price Prediction Comparison: BiLSTM-Attention vs. Baseline Models |
| exp_2 | Apriori-Based Consecutive Rise/Fall Pattern Analysis |
| exp_3 | Exponential Position Sizing Model Derivation and Fitting |
| exp_4 | Trading Strategy Portfolio Simulation with VaR Constraint |
| exp_5 | Transaction Fee Sensitivity Analysis |

## Repository Structure

```
.
├── data/               # Data download scripts and cached data
├── src/                # Core modules (models, metrics, patterns, position sizing, trading)
├── exp/                # Experiment runner scripts
├── tests/              # Pytest unit and integration tests
├── results/            # Output figures, tables, RESULTS.md
├── README.md
├── AGENTS.md
└── .gitignore
```

## Setup

```bash
pip install tensorflow pmdarima mlxtend yfinance xgboost hmmlearn statsmodels scikit-learn matplotlib
```

## Running

```bash
# Run all experiments sequentially
python exp/exp_1.py   # Multi-model price prediction comparison
python exp/exp_2.py   # Apriori-based rise/fall pattern analysis
python exp/exp_3.py   # Exponential position sizing model
python exp/exp_4.py   # Trading strategy portfolio simulation
python exp/exp_5.py   # Transaction fee sensitivity analysis

# Run tests
pytest tests/ -v
```

## Results

See [results/RESULTS.md](results/RESULTS.md) for a full summary of all metrics and findings.

## Key Results

- **BiLSTM-Attention** achieves best performance on all four metrics (RMSE, MAE, MAPE, R²) for both Gold and Bitcoin.
- **Target Gold**: RMSE≈19.01, MAE≈13.98, MAPE≈0.76%, R²≈0.9392
- **Target Bitcoin**: RMSE≈1479, MAE≈906, MAPE≈2.88%, R²≈0.9929
- **Trading simulation**: Gold ~$646, Bitcoin ~$215,487 from initial $500 each
