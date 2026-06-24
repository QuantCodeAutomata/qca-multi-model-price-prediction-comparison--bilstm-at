# Repository: qca-multi-model-price-prediction-comparison--bilstm-at

## Overview
Quantitative finance research repository implementing BiLSTM-Attention vs. baseline model price prediction,
Apriori-based pattern analysis, exponential position sizing, and a VaR-constrained trading strategy for
Gold and Bitcoin.

## Structure
- `data/` — Data loading scripts (massive API or Yahoo Finance via yfinance)
- `src/` — Core implementations:
  - `src/models.py` — All prediction models (BiLSTM-Attention, BiLSTM, LSTM, GRU, HMM, ARIMA, Holt-Winters, XGBoost)
  - `src/metrics.py` — RMSE, MAE, MAPE, R² evaluation functions
  - `src/patterns.py` — Apriori-based run-length analysis and M_i computation
  - `src/position_sizing.py` — Recursive breakeven + exponential fitting position sizing
  - `src/trading.py` — VaR-constrained modified greedy trading simulation
- `exp/` — Experiment runners (exp1–exp5)
- `tests/` — Pytest tests
- `results/` — Metrics, plots, RESULTS.md

## Key Methodology Notes
- M_i formula: `(p_i - p_{i-1}) / p_i` (current price in denominator — non-standard)
- Window size: 100 days, step 1, min-max normalized per window
- Train/test split: 70% temporal split (no shuffle across boundary)
- BiLSTM-Attention: 2x BiLSTM(64 units/dir) + DotProductAttention + Dense(1)
- Optimizer: RMSprop, Loss: MAE, Batch: 16, EarlyStopping patience=10
- Random seeds: numpy=42, tf=42, python=42
- VaR: 95% confidence, 252-day rolling window historical simulation
- Position sizing: recursive breakeven → exponential fit → dollar allocation

## Data Sources
- Gold: Yahoo Finance `GC=F` (weekday only, forward-fill missing)
- Bitcoin: Yahoo Finance `BTC-USD` (all calendar days)
- Date range: 2015-01-01 to 2021-09-10

## Library Usage (Context7 Verified)
- hmmlearn: GaussianHMM for regime detection — Context7 confirmed
- statsmodels ExponentialSmoothing: Holt-Winters — Context7 confirmed
- pmdarima auto_arima: ARIMA fitting — standard library
- xgboost XGBRegressor: gradient boosting — standard library
- TensorFlow/Keras: deep learning models — standard library
- Custom DotProductAttention: implemented from scratch (paper Eq. 4)
- Custom recursive breakeven: implemented from scratch (paper Eq. 5-6)

## Running Experiments
```bash
cd /workspace/project
python exp/exp1_prediction_comparison.py
python exp/exp2_pattern_analysis.py
python exp/exp3_position_sizing.py
python exp/exp4_trading_simulation.py
python exp/exp5_fee_sensitivity.py
pytest tests/ -v
```
