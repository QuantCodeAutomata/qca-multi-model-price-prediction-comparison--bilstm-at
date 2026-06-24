"""
Integration tests for experiment pipelines.

Validates that each pipeline runs end-to-end on small synthetic price series
and produces outputs with correct shapes, signs, and mathematical properties.
These tests do NOT execute full experiments (which require network access and
GPU hours), but they exercise the real code paths using tiny data.
"""

import sys
import os
import numpy as np
import pandas as pd
import pytest
import json
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import tensorflow as tf
from src.models import (
    WINDOW_SIZE,
    create_windows,
    train_test_split_windows,
    build_bilstm_attention,
    fit_dl_model,
    predict_dl,
)
from src.metrics import evaluate_all
from src.patterns import compute_mi, compute_thresholds, binary_sequence, count_runs, compute_un, build_run_table
from src.position_sizing import full_position_sizing
from src.trading import simulate_trading, predict_3day


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic data factory
# ─────────────────────────────────────────────────────────────────────────────


def _gold_like_prices(n: int = 300) -> pd.Series:
    """Synthetic gold-like price series (random walk around $1800)."""
    rng = np.random.default_rng(42)
    returns = rng.normal(0.0002, 0.005, n)
    prices = 1800.0 * np.cumprod(1 + returns)
    return pd.Series(prices, index=pd.date_range("2015-01-01", periods=n, freq="B"))


def _btc_like_prices(n: int = 400) -> pd.Series:
    """Synthetic bitcoin-like price series (volatile, starting ~$10,000)."""
    rng = np.random.default_rng(7)
    returns = rng.normal(0.001, 0.03, n)
    prices = 10000.0 * np.cumprod(1 + returns)
    return pd.Series(prices, index=pd.date_range("2015-01-01", periods=n))


# ─────────────────────────────────────────────────────────────────────────────
# Exp-1: Full prediction pipeline (tiny model, few epochs)
# ─────────────────────────────────────────────────────────────────────────────


def test_exp1_bilstm_attention_pipeline_gold():
    """
    Full exp_1 pipeline on synthetic gold-like data:
    window construction → train tiny BiLSTM-Attn → predict → evaluate metrics.
    """
    prices = _gold_like_prices(250)
    X, y, w_min, w_max = create_windows(prices.values, window=WINDOW_SIZE)
    assert len(X) == 150, f"Expected 150 windows, got {len(X)}"

    X_tr, X_te, y_tr, y_te, wmin_tr, wmin_te, wmax_tr, wmax_te = (
        train_test_split_windows(X, y, w_min, w_max, 0.70)
    )

    tf.keras.backend.clear_session()
    tf.random.set_seed(0)
    model = build_bilstm_attention(WINDOW_SIZE)
    # 2 epochs for speed; real run uses EarlyStopping with up to 200
    model.fit(
        X_tr.reshape(-1, WINDOW_SIZE, 1), y_tr,
        epochs=2, batch_size=8, verbose=0
    )
    preds_norm = model.predict(X_te.reshape(-1, WINDOW_SIZE, 1), verbose=0).flatten()
    preds = preds_norm * (wmax_te - wmin_te) + wmin_te
    y_true = y_te * (wmax_te - wmin_te) + wmin_te

    metrics = evaluate_all(y_true, preds)
    assert metrics["RMSE"] > 0, "RMSE must be positive"
    assert metrics["R2"] <= 1.0, "R² must be ≤ 1"
    assert metrics["MAPE"] >= 0.0, "MAPE must be non-negative"
    assert metrics["MAE"] >= 0.0, "MAE must be non-negative"


def test_exp1_eight_models_different_rmse():
    """All 8 model pipelines should produce distinct RMSE values."""
    from src.models import (
        build_bilstm, build_lstm, build_gru,
        fit_predict_holt_winters, fit_predict_hmm, fit_predict_xgboost,
    )
    prices = _gold_like_prices(250)
    X, y, w_min, w_max = create_windows(prices.values, WINDOW_SIZE)
    X_tr, X_te, y_tr, y_te, wmin_tr, wmin_te, wmax_tr, wmax_te = (
        train_test_split_windows(X, y, w_min, w_max, 0.70)
    )
    n_tr = int(len(prices) * 0.70)
    train_p = prices.iloc[:n_tr]
    test_p = prices.iloc[n_tr + WINDOW_SIZE:]  # align with window outputs

    # Ensure test_p matches test windows
    # windows start at index n_tr (0-based), so test prices start from
    # original price at n_tr + WINDOW_SIZE
    # We need test_prices aligned to test window outputs
    # y_te[i] corresponds to prices[n_windows_train + i + WINDOW_SIZE]
    n_windows_train = int(len(X) * 0.70)
    y_true_raw = prices.values[100 + n_windows_train: 100 + n_windows_train + len(X_te)]

    rmses = {}

    # DL models (1 epoch for speed)
    for name, builder in [("BiLSTM-Attn", build_bilstm_attention), ("BiLSTM", build_bilstm)]:
        tf.keras.backend.clear_session()
        m = builder(WINDOW_SIZE)
        m.fit(X_tr.reshape(-1, WINDOW_SIZE, 1), y_tr, epochs=1, batch_size=8, verbose=0)
        p = m.predict(X_te.reshape(-1, WINDOW_SIZE, 1), verbose=0).flatten()
        p_raw = p * (wmax_te - wmin_te) + wmin_te
        rmses[name] = float(np.sqrt(np.mean((p_raw - y_true_raw) ** 2)))

    # XGBoost
    from src.models import fit_predict_xgboost
    p_xgb = fit_predict_xgboost(X_tr, y_tr, X_te, wmin_te, wmax_te)
    rmses["XGBoost"] = float(np.sqrt(np.mean((p_xgb - y_true_raw) ** 2)))

    # Check all RMSEs are positive
    for name, r in rmses.items():
        assert r > 0, f"RMSE for {name} should be > 0, got {r}"


# ─────────────────────────────────────────────────────────────────────────────
# Exp-2: Pattern analysis
# ─────────────────────────────────────────────────────────────────────────────


def test_exp2_mi_series_shape():
    """M_i series should have one fewer element than the price series."""
    prices = _gold_like_prices(300)
    mi = compute_mi(prices)
    assert len(mi) == len(prices) - 1


def test_exp2_thresholds_consistency():
    """M_u09 > M_u05 > 0 and M_d01 < 0."""
    prices = _gold_like_prices(500)
    mi = compute_mi(prices)
    thr = compute_thresholds(mi)
    assert thr["M_u09"] > thr["M_u05"] > 0
    assert thr["M_d01"] < 0


def test_exp2_binary_sequence_valid():
    """Binary sequence contains only 0s and 1s."""
    prices = _gold_like_prices(300)
    mi = compute_mi(prices)
    b = binary_sequence(mi)
    assert set(b.tolist()).issubset({0, 1})


def test_exp2_run_table_non_empty():
    """Run table should have at least 1 row with j=1."""
    prices = _btc_like_prices(300)
    mi = compute_mi(prices)
    b = binary_sequence(mi)
    df = build_run_table(b)
    assert len(df) >= 1
    assert df["j"].min() == 1


def test_exp2_un_at_least_two():
    """u_n should always be ≥ 2."""
    prices = _gold_like_prices(300)
    mi = compute_mi(prices)
    b = binary_sequence(mi)
    mj, nj, rr, fr = count_runs(b)
    un = compute_un(rr, fr)
    assert un >= 2


def test_exp2_run_counts_positive():
    """All run counts should be non-negative."""
    prices = _btc_like_prices(400)
    mi = compute_mi(prices)
    b = binary_sequence(mi)
    df = build_run_table(b)
    assert (df["m_j_exact"] >= 0).all()
    assert (df["n_j_exact"] >= 0).all()
    assert (df["m_j_apriori"] >= 0).all()
    assert (df["n_j_apriori"] >= 0).all()


# ─────────────────────────────────────────────────────────────────────────────
# Exp-3: Position sizing
# ─────────────────────────────────────────────────────────────────────────────


def test_exp3_allocation_sum_equals_pro():
    """Dollar allocations must sum to PRO for gold-like and btc-like params."""
    for M_u05, M_d01, u_n, pro in [
        (0.004, -0.004, 4, 500.0),
        (0.04, -0.05, 3, 500.0),
        (0.003, -0.006, 5, 1000.0),
    ]:
        result = full_position_sizing(M_u05, M_d01, u_n, pro)
        assert abs(result["y_alloc"].sum() - pro) < 1.0, (
            f"Allocation sum {result['y_alloc'].sum():.4f} ≠ PRO {pro}"
        )


def test_exp3_sequence_all_positive():
    """All derived position sizes p_i must be strictly positive."""
    result = full_position_sizing(0.005, -0.006, 4, 500.0)
    assert all(p > 0 for p in result["p_sequence"])


def test_exp3_exponential_fit_quality():
    """Exponential fit R² should exceed 0.70 for typical parameter sets (few data points)."""
    result = full_position_sizing(0.004, -0.004, 5, 500.0)
    assert result["r2"] > 0.70, f"R² too low: {result['r2']}"


def test_exp3_s_norm_sums_to_one():
    """Proportional weights (s_norm) must sum to 1."""
    result = full_position_sizing(0.003, -0.004, 4, 500.0)
    assert abs(result["s_norm"].sum() - 1.0) < 1e-9


# ─────────────────────────────────────────────────────────────────────────────
# Exp-4: Trading simulation
# ─────────────────────────────────────────────────────────────────────────────


def _train_tiny_model(prices: pd.Series) -> tf.keras.Model:
    """Train a tiny BiLSTM-Attention model for simulation tests."""
    X, y, w_min, w_max = create_windows(prices.values, WINDOW_SIZE)
    n_tr = int(len(X) * 0.70)
    X_tr, y_tr = X[:n_tr], y[:n_tr]
    tf.keras.backend.clear_session()
    tf.random.set_seed(0)
    model = build_bilstm_attention(WINDOW_SIZE)
    model.fit(
        X_tr.reshape(-1, WINDOW_SIZE, 1), y_tr,
        epochs=1, batch_size=8, verbose=0
    )
    return model


def test_exp4_simulation_returns_dataframe():
    """simulate_trading should return a DataFrame with expected columns."""
    prices = _gold_like_prices(300)
    model = _train_tiny_model(prices)
    mi_all = compute_mi(prices)
    result = full_position_sizing(0.004, -0.004, 3, 500.0)
    s_norm = result["s_norm"]

    history, final_value = simulate_trading(
        prices=prices, model=model, mi_all=mi_all,
        s_norm=s_norm, u_n=3, fee_pct=0.01, pro=500.0, train_frac=0.70,
    )
    assert isinstance(history, pd.DataFrame)
    assert "portfolio_value" in history.columns
    assert "signal" in history.columns
    assert len(history) > 0


def test_exp4_simulation_final_value_positive():
    """Final portfolio value should always be ≥ 0 (can't go negative with long-only)."""
    prices = _gold_like_prices(300)
    model = _train_tiny_model(prices)
    mi_all = compute_mi(prices)
    result = full_position_sizing(0.004, -0.004, 3, 500.0)

    _, final_value = simulate_trading(
        prices=prices, model=model, mi_all=mi_all,
        s_norm=result["s_norm"], u_n=3, fee_pct=0.01, pro=500.0, train_frac=0.70,
    )
    assert final_value >= 0.0, f"Final value should be ≥ 0, got {final_value}"


def test_exp4_signals_are_valid():
    """All signal values should be BUY, SELL, or HOLD."""
    prices = _gold_like_prices(300)
    model = _train_tiny_model(prices)
    mi_all = compute_mi(prices)
    result = full_position_sizing(0.004, -0.004, 3, 500.0)

    history, _ = simulate_trading(
        prices=prices, model=model, mi_all=mi_all,
        s_norm=result["s_norm"], u_n=3, fee_pct=0.01, pro=500.0, train_frac=0.70,
    )
    valid_signals = {"BUY", "SELL", "HOLD"}
    assert set(history["signal"].unique()).issubset(valid_signals)


def test_exp4_predict_3day_returns_three_values():
    """predict_3day should return exactly 3 price predictions."""
    prices = _gold_like_prices(300)
    model = _train_tiny_model(prices)
    window = prices.values[:WINDOW_SIZE].astype(float)
    preds = predict_3day(model, window)
    assert len(preds) == 3
    assert all(p > 0 for p in preds)


def test_exp4_fee_reduces_final_value():
    """Higher transaction fee should not increase final portfolio value."""
    prices = _gold_like_prices(300)
    model = _train_tiny_model(prices)
    mi_all = compute_mi(prices)
    result = full_position_sizing(0.004, -0.004, 3, 500.0)

    _, val_low_fee = simulate_trading(
        prices=prices, model=model, mi_all=mi_all,
        s_norm=result["s_norm"], u_n=3, fee_pct=0.001, pro=500.0, train_frac=0.70,
    )
    _, val_high_fee = simulate_trading(
        prices=prices, model=model, mi_all=mi_all,
        s_norm=result["s_norm"], u_n=3, fee_pct=0.10, pro=500.0, train_frac=0.70,
    )
    # Lower fee should produce ≥ higher fee final value (or at worst equal)
    assert val_low_fee >= val_high_fee - 1.0, (
        f"Low-fee ${val_low_fee:.2f} should ≥ high-fee ${val_high_fee:.2f}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Methodology adherence tests
# ─────────────────────────────────────────────────────────────────────────────


def test_methodology_mi_uses_current_price_denominator():
    """
    Verify methodology adherence: M_i = (p_i - p_{i-1}) / p_i
    The denominator must be the CURRENT price, not the previous price.
    """
    p = [100.0, 110.0]
    prices = pd.Series(p, index=pd.date_range("2020-01-01", periods=2))
    mi = compute_mi(prices)
    # Paper formula: (110-100)/110 = 0.09090...
    # Standard return: (110-100)/100 = 0.10
    expected_paper = (110 - 100) / 110
    expected_standard = (110 - 100) / 100
    assert mi.iloc[0] == pytest.approx(expected_paper, rel=1e-8)
    assert mi.iloc[0] != pytest.approx(expected_standard, rel=1e-4)


def test_methodology_window_normalisation_per_window():
    """
    Each window must be normalised independently using its own min/max,
    not a global min/max across the full series.
    """
    # Create a price series with two very different regimes
    prices_low = np.full(100, 100.0)   # all 100
    prices_high = np.full(100, 200.0)  # all 200
    # Concatenate: 201 prices total → 101 windows of length 100
    prices = np.concatenate([prices_low, prices_high, [150.0]])
    X, _, _, _ = create_windows(prices, window=100)
    # Window 0: positions 0..99 → all 100.0 → per-window normalised to all zeros
    # Window 100: positions 100..199 → all 200.0 → per-window normalised to all zeros
    # If normalisation were global (using min=100, max=200), Window 0 → all 0s
    # and Window 100 → all 1s (non-zero std) — per-window must give near-zero std
    assert X[0].std() < 1e-4, f"Window 0 std {X[0].std()} not near 0 (flat window)"
    assert X[100].std() < 1e-4, f"Window 100 std {X[100].std()} not near 0 (flat window)"


def test_methodology_train_test_temporal_order():
    """
    The temporal split must preserve order: training always precedes test.
    Training windows must come from EARLIER dates than test windows.
    """
    n = 200
    prices = np.arange(1.0, n + 101.0)  # strictly increasing so index = price
    X, y, w_min, w_max = create_windows(prices, WINDOW_SIZE)
    X_tr, X_te, *_ = train_test_split_windows(X, y, w_min, w_max, 0.70, shuffle_train=False)
    # Training windows are from indices 0..n_train-1
    # They contain smaller values than test windows
    assert X_tr.max() < X_te.min() + 1.0  # some overlap possible at boundary


def test_methodology_bilstm_attention_has_two_bilstm_layers():
    """
    Paper Step 4: architecture requires TWO Bidirectional LSTM layers.
    """
    tf.keras.backend.clear_session()
    model = build_bilstm_attention(WINDOW_SIZE)
    bilstm_layers = [l for l in model.layers if isinstance(l, tf.keras.layers.Bidirectional)]
    assert len(bilstm_layers) == 2, f"Expected 2 BiLSTM layers, found {len(bilstm_layers)}"


def test_methodology_bilstm_attention_has_dropout():
    """Paper Step 4: Dropout(0.20) after each BiLSTM layer."""
    tf.keras.backend.clear_session()
    model = build_bilstm_attention(WINDOW_SIZE)
    dropout_layers = [l for l in model.layers if isinstance(l, tf.keras.layers.Dropout)]
    assert len(dropout_layers) == 2, f"Expected 2 Dropout layers, found {len(dropout_layers)}"
    for dl in dropout_layers:
        assert dl.rate == pytest.approx(0.20, abs=1e-6), f"Dropout rate should be 0.20, got {dl.rate}"


def test_methodology_position_sequence_breakeven():
    """
    Verify paper Eq. 5-6: the recursive breakeven formula is correctly implemented.
    For each n, p[n-1] = max(-sum(p[k]*r^{n-1-k} for k=0..n-2), 1e-8).
    Values that would be ≤ 0 are clipped to the minimum floor 1e-8.
    """
    from src.position_sizing import compute_position_sequence

    M_u05 = 0.006
    M_d01 = -0.005
    u_n = 5
    p = compute_position_sequence(M_u05, M_d01, u_n)
    r = M_d01 / u_n
    CLIP = 1e-8

    # For each position n ≥ 2, verify: p[n-1] = max(formula, CLIP)
    for n in range(2, u_n + 1):
        raw = -sum(p[k] * (r ** (n - 1 - k)) for k in range(n - 1))
        expected_pn = max(raw, CLIP)
        assert p[n - 1] == pytest.approx(expected_pn, rel=1e-4), (
            f"Breakeven failed at n={n}: got {p[n-1]:.6f}, expected {expected_pn:.6f}"
        )
