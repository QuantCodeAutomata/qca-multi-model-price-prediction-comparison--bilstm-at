"""
VaR-Constrained Modified Greedy Trading Strategy Simulation (paper exp_4).

Combines:
  - BiLSTM-Attention 3-day forward price predictions
  - Exponential position sizing (from position_sizing.py)
  - Historical VaR constraint (95% confidence, 252-day rolling window)

Custom — Context7 found no library equivalent (paper Algorithm 1)
"""

import numpy as np
import pandas as pd
import tensorflow as tf

from src.models import DotProductAttention, WINDOW_SIZE


def predict_3day(
    model: tf.keras.Model,
    window_100: np.ndarray,
) -> list[float]:
    """
    Generate 3-day ahead price predictions via rolling single-step predictions.

    For step 1: uses actual 100-day window.
    For step 2: slides window, replacing oldest with latest actual price.
    For step 3: slides window, replacing oldest with step-1 prediction.

    Parameters
    ----------
    model : tf.keras.Model
        Trained BiLSTM-Attention model.
    window_100 : np.ndarray, shape (100,)
        The last 100 actual closing prices (not normalised).

    Returns
    -------
    list[float]  [pred_t+1, pred_t+2, pred_t+3] in original price scale.
    """
    w = window_100.copy().astype(float)
    preds: list[float] = []

    for _ in range(3):
        mn = w.min()
        mx = w.max() + 1e-8
        w_norm = (w - mn) / (mx - mn)
        inp = w_norm.reshape(1, WINDOW_SIZE, 1)
        p_norm = float(model.predict(inp, verbose=0)[0, 0])
        p_actual = p_norm * (mx - mn) + mn
        preds.append(p_actual)
        w = np.append(w[1:], p_actual)

    return preds


def compute_var_dollar(
    mi_window: np.ndarray,
    portfolio_value: float,
    confidence: float = 0.95,
) -> float:
    """
    Compute the dollar VaR at `confidence` level using historical simulation.

    VaR_pct = abs(5th percentile of mi_window)
    VaR_dollar = VaR_pct * portfolio_value

    Parameters
    ----------
    mi_window : np.ndarray
        Rolling window of M_i values (last 252 days).
    portfolio_value : float
        Current total portfolio value.
    confidence : float
        Confidence level (default 0.95).

    Returns
    -------
    float  Dollar VaR (non-negative). Returns inf if window is too short.
    """
    if len(mi_window) < 10:
        return float("inf")
    var_pct = abs(np.percentile(mi_window, (1 - confidence) * 100))
    return var_pct * portfolio_value


def simulate_trading(
    prices: pd.Series,
    model: tf.keras.Model,
    mi_all: pd.Series,
    s_norm: np.ndarray,
    u_n: int,
    fee_pct: float = 0.01,
    pro: float = 500.0,
    train_frac: float = 0.70,
    window_size: int = WINDOW_SIZE,
    var_confidence: float = 0.95,
    var_window: int = 252,
) -> tuple[pd.DataFrame, float]:
    """
    Run the full trading simulation for one asset.

    Parameters
    ----------
    prices : pd.Series
        Full price series (training + test).
    model : tf.keras.Model
        Trained BiLSTM-Attention model.
    mi_all : pd.Series
        M_i series for the full price history (for VaR computation).
    s_norm : np.ndarray
        Normalised position size proportions from exponential model (sums to 1).
    u_n : int
        Maximum number of position addition steps.
    fee_pct : float
        Transaction fee rate (e.g. 0.01 = 1%).
    pro : float
        Initial capital.
    train_frac : float
        Fraction of data used for training (70%).
    window_size : int
        Input sequence length (100).
    var_confidence : float
        VaR confidence level (95%).
    var_window : int
        Rolling window length for VaR computation (252 trading days).

    Returns
    -------
    history : pd.DataFrame
        Daily records: date, portfolio_value, price, cash, units, signal.
    final_value : float
        Final liquidated portfolio value.
    """
    prices_arr = prices.values.astype(float)
    n_total = len(prices_arr)
    split_idx = int(n_total * train_frac)

    train_prices = prices_arr[:split_idx]
    test_prices = prices_arr[split_idx:]
    test_dates = prices.index[split_idx:]
    mi_arr = mi_all.values.astype(float)

    # Initialise rolling window from end of training data
    rolling_window = train_prices[-window_size:].copy()

    cash = float(pro)
    units = 0.0
    pos_step = 0
    records: list[dict] = []

    for t_idx, (price_t, date_t) in enumerate(zip(test_prices, test_dates)):
        # Current portfolio value (before any trade)
        pv = cash + units * price_t

        # ── 3-day price prediction ────────────────────────────────────────────
        preds = predict_3day(model, rolling_window)
        avg_change = (preds[-1] - price_t) / (price_t + 1e-8)
        signal = "BUY" if avg_change > 0 else "SELL"

        # ── VaR constraint ────────────────────────────────────────────────────
        # Global index in Mi_all: Mi series starts at index 1 of prices
        # training has split_idx prices → split_idx - 1 M_i values
        # so at test step t_idx, global M_i index is (split_idx - 1 + t_idx)
        global_mi_idx = (split_idx - 1) + t_idx
        start_mi = max(0, global_mi_idx - var_window)
        ret_window = mi_arr[start_mi:global_mi_idx]
        var_dollar = compute_var_dollar(ret_window, pv, var_confidence)

        # Predicted 1-day loss on existing position
        pred_loss = max(0.0, price_t - preds[0]) / (price_t + 1e-8) * (units * price_t)
        if pred_loss > var_dollar:
            signal = "HOLD"

        # ── Execute trade ─────────────────────────────────────────────────────
        if signal == "BUY" and cash > 1.0 and pos_step < u_n:
            pos_step += 1
            # Allocate proportionally: use remaining s_norm slice
            remaining_s = s_norm[pos_step - 1 :].sum()
            alloc_frac = s_norm[pos_step - 1] / remaining_s if remaining_s > 0 else 1.0
            alloc_dollar = min(cash, alloc_frac * cash)
            units_bought = alloc_dollar * (1.0 - fee_pct) / price_t
            units += units_bought
            cash -= alloc_dollar

        elif signal == "SELL" and units > 0.0:
            proceeds = units * price_t * (1.0 - fee_pct)
            cash += proceeds
            units = 0.0
            pos_step = 0

        records.append(
            {
                "date": date_t,
                "portfolio_value": pv,
                "price": price_t,
                "cash": cash,
                "units": units,
                "signal": signal,
            }
        )

        # Update rolling window with today's actual price
        rolling_window = np.append(rolling_window[1:], price_t)

    # Final liquidated value
    final_price = test_prices[-1]
    final_value = cash + units * final_price * (1.0 - fee_pct)

    history = pd.DataFrame(records)
    return history, final_value
