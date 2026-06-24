"""
Unit tests for src/models.py.

Validates sliding-window construction, inverse transform, DL model
architecture shapes, and XGBoost/HMM baseline pipelines.
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models import (
    WINDOW_SIZE,
    create_windows,
    inverse_transform,
    train_test_split_windows,
    build_bilstm_attention,
    build_bilstm,
    build_lstm,
    build_gru,
    DotProductAttention,
)

import tensorflow as tf


# ─────────────────────────────────────────────────────────────────────────────
# create_windows
# ─────────────────────────────────────────────────────────────────────────────


def test_create_windows_output_shapes():
    """Output arrays should have correct shapes."""
    prices = np.arange(1, 202, dtype=float)  # 201 prices
    X, y, w_min, w_max = create_windows(prices, window=100)
    # 201 - 100 = 101 windows
    assert X.shape == (101, 100)
    assert y.shape == (101,)
    assert w_min.shape == (101,)
    assert w_max.shape == (101,)


def test_create_windows_normalised_range():
    """Each window in X should have values in [0, 1]."""
    rng = np.random.default_rng(0)
    prices = rng.uniform(1000, 2000, 200)
    X, _, _, _ = create_windows(prices, window=100)
    assert X.min() >= 0.0 - 1e-9
    assert X.max() <= 1.0 + 1e-9


def test_create_windows_target_in_range():
    """Normalised target y should be mostly in [0, 1]; can exceed slightly."""
    rng = np.random.default_rng(1)
    prices = rng.uniform(1000, 2000, 300)
    _, y, _, _ = create_windows(prices, window=100)
    # Most targets should be positive
    assert (y > -1.0).all()


def test_create_windows_minimum_prices():
    """Works with exactly window+1 prices (1 window)."""
    prices = np.linspace(100, 200, 101)
    X, y, w_min, w_max = create_windows(prices, window=100)
    assert len(X) == 1


def test_create_windows_insufficient_prices():
    """Fewer prices than window size produces empty arrays."""
    prices = np.arange(50, dtype=float)
    X, y, w_min, w_max = create_windows(prices, window=100)
    assert len(X) == 0


# ─────────────────────────────────────────────────────────────────────────────
# inverse_transform
# ─────────────────────────────────────────────────────────────────────────────


def test_inverse_transform_round_trip():
    """Forward + inverse normalisation should recover original values."""
    rng = np.random.default_rng(42)
    prices = rng.uniform(1700, 1900, 200)
    X, y, w_min, w_max = create_windows(prices, window=100)
    y_recovered = inverse_transform(y, w_min, w_max)
    # Recovered targets should match the original price values
    expected = prices[100:]
    np.testing.assert_allclose(y_recovered, expected, rtol=1e-5)


def test_inverse_transform_zeros():
    """Normalised 0 → w_min; normalised 1 → w_max."""
    w_min = np.array([100.0, 200.0])
    w_max = np.array([200.0, 400.0])
    y_norm = np.array([0.0, 1.0])
    y_actual = inverse_transform(y_norm, w_min, w_max)
    assert y_actual[0] == pytest.approx(100.0)
    assert y_actual[1] == pytest.approx(400.0)


# ─────────────────────────────────────────────────────────────────────────────
# train_test_split_windows
# ─────────────────────────────────────────────────────────────────────────────


def test_split_sizes():
    """70/30 split should produce correct sizes."""
    n = 1000
    X = np.zeros((n, WINDOW_SIZE))
    y = np.zeros(n)
    w_min = np.zeros(n)
    w_max = np.ones(n)
    X_tr, X_te, y_tr, y_te, *_ = train_test_split_windows(X, y, w_min, w_max, 0.70)
    assert len(X_tr) == 700
    assert len(X_te) == 300


def test_split_no_data_leakage():
    """Test set must only contain indices ≥ split point (temporal integrity)."""
    n = 100
    X = np.arange(n, dtype=float).reshape(n, 1)
    y = np.arange(n, dtype=float)
    w_min = np.zeros(n)
    w_max = np.ones(n)
    X_tr, X_te, y_tr, y_te, *_ = train_test_split_windows(
        X, y, w_min, w_max, 0.70, shuffle_train=False
    )
    # Without shuffle, training values should all be < split value
    # and test values all >= split value
    split_val = int(n * 0.70)
    # Flatten to check indices
    assert (X_te.flatten() >= split_val).all()
    assert (X_tr.flatten() < split_val).all()


# ─────────────────────────────────────────────────────────────────────────────
# Model architecture tests
# ─────────────────────────────────────────────────────────────────────────────


def test_bilstm_attention_output_shape():
    """BiLSTM-Attention should produce (batch, 1) outputs."""
    tf.keras.backend.clear_session()
    model = build_bilstm_attention(seq_len=10)
    X = np.random.randn(5, 10, 1).astype("float32")
    out = model(X, training=False)
    assert out.shape == (5, 1)


def test_bilstm_output_shape():
    """Standard BiLSTM should produce (batch, 1) outputs."""
    tf.keras.backend.clear_session()
    model = build_bilstm(seq_len=10)
    X = np.random.randn(3, 10, 1).astype("float32")
    out = model(X, training=False)
    assert out.shape == (3, 1)


def test_lstm_output_shape():
    """LSTM model should produce (batch, 1) outputs."""
    tf.keras.backend.clear_session()
    model = build_lstm(seq_len=10)
    X = np.random.randn(4, 10, 1).astype("float32")
    out = model(X, training=False)
    assert out.shape == (4, 1)


def test_gru_output_shape():
    """GRU model should produce (batch, 1) outputs."""
    tf.keras.backend.clear_session()
    model = build_gru(seq_len=10)
    X = np.random.randn(4, 10, 1).astype("float32")
    out = model(X, training=False)
    assert out.shape == (4, 1)


def test_dot_product_attention_output_shape():
    """DotProductAttention should reduce time dimension."""
    tf.keras.backend.clear_session()
    # Input: (batch=2, timesteps=5, features=8)
    layer = DotProductAttention()
    x = tf.constant(np.random.randn(2, 5, 8).astype("float32"))
    out = layer(x)
    # Output: (batch=2, features=8) — time dimension removed
    assert out.shape == (2, 8)


def test_all_models_different_predictions():
    """All four DL models should produce different predictions (not identical)."""
    tf.keras.backend.clear_session()
    X = np.random.randn(1, 10, 1).astype("float32")
    preds = []
    for builder in [build_bilstm_attention, build_bilstm, build_lstm, build_gru]:
        tf.keras.backend.clear_session()
        m = builder(seq_len=10)
        preds.append(float(m(X, training=False)[0, 0]))

    # At least one pair should differ (models have different architectures)
    assert len(set(f"{p:.4f}" for p in preds)) > 1


# ─────────────────────────────────────────────────────────────────────────────
# XGBoost (light integration test)
# ─────────────────────────────────────────────────────────────────────────────


def test_xgboost_predict_shape():
    """XGBoost predictions should have the same length as test set."""
    from src.models import fit_predict_xgboost

    rng = np.random.default_rng(0)
    n_tr, n_te = 200, 50
    X_tr = rng.uniform(0, 1, (n_tr, WINDOW_SIZE)).astype("float32")
    y_tr = rng.uniform(0, 1, n_tr).astype("float32")
    X_te = rng.uniform(0, 1, (n_te, WINDOW_SIZE)).astype("float32")
    wmin = rng.uniform(1000, 1500, n_te)
    wmax = rng.uniform(1500, 2000, n_te)

    preds = fit_predict_xgboost(X_tr, y_tr, X_te, wmin, wmax)
    assert preds.shape == (n_te,)


def test_xgboost_predictions_positive():
    """XGBoost predictions on gold-like data should be positive."""
    from src.models import fit_predict_xgboost

    rng = np.random.default_rng(1)
    prices = rng.uniform(1700, 1950, 300)
    X, y, w_min, w_max = create_windows(prices, window=100)
    X_tr, X_te = X[:140], X[140:]
    y_tr = y[:140]
    wmin_te = w_min[140:]
    wmax_te = w_max[140:]

    preds = fit_predict_xgboost(X_tr, y_tr, X_te, wmin_te, wmax_te)
    assert (preds > 0).all()
