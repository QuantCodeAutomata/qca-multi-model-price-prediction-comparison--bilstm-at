"""
Unit tests for src/metrics.py.

Validates mathematical properties of RMSE, MAE, MAPE, and R².
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.metrics import rmse, mae, mape, r2, evaluate_all


# ─────────────────────────────────────────────────────────────────────────────
# Perfect prediction tests
# ─────────────────────────────────────────────────────────────────────────────


def test_rmse_perfect_prediction():
    """RMSE should be 0 when predictions equal true values."""
    y = np.array([100.0, 200.0, 300.0])
    assert rmse(y, y) == pytest.approx(0.0, abs=1e-10)


def test_mae_perfect_prediction():
    """MAE should be 0 for perfect predictions."""
    y = np.array([1800.0, 1850.0, 1900.0])
    assert mae(y, y) == pytest.approx(0.0, abs=1e-10)


def test_mape_perfect_prediction():
    """MAPE should be 0 for perfect predictions."""
    y = np.array([50000.0, 55000.0, 60000.0])
    assert mape(y, y) == pytest.approx(0.0, abs=1e-10)


def test_r2_perfect_prediction():
    """R² should be 1.0 for perfect predictions."""
    y = np.array([10.0, 20.0, 30.0, 40.0])
    assert r2(y, y) == pytest.approx(1.0, abs=1e-10)


# ─────────────────────────────────────────────────────────────────────────────
# Known-value regression tests
# ─────────────────────────────────────────────────────────────────────────────


def test_rmse_known_value():
    """RMSE of [0,0,0] vs [3,4,0] = sqrt((9+16+0)/3) = sqrt(25/3)."""
    y_true = np.array([3.0, 4.0, 0.0])
    y_pred = np.array([0.0, 0.0, 0.0])
    expected = np.sqrt((9 + 16 + 0) / 3)
    assert rmse(y_true, y_pred) == pytest.approx(expected, rel=1e-6)


def test_mae_known_value():
    """MAE of [1,2,3] vs [4,5,6] = mean([3,3,3]) = 3.0."""
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([4.0, 5.0, 6.0])
    assert mae(y_true, y_pred) == pytest.approx(3.0, rel=1e-6)


def test_mape_known_value():
    """MAPE of [100] vs [110] = 10%."""
    y_true = np.array([100.0])
    y_pred = np.array([110.0])
    assert mape(y_true, y_pred) == pytest.approx(10.0, rel=1e-6)


def test_r2_known_value():
    """R² for a constant prediction equals negative value."""
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([2.0, 2.0, 2.0])  # mean prediction
    # SS_res = 0+0+1 = 1+1 = 2? Let's compute: (1-2)^2+(2-2)^2+(3-2)^2 = 1+0+1=2
    # SS_tot = (1-2)^2+(2-2)^2+(3-2)^2 = 2
    # R² = 1 - 2/2 = 0.0
    assert r2(y_true, y_pred) == pytest.approx(0.0, abs=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# Property tests
# ─────────────────────────────────────────────────────────────────────────────


def test_rmse_non_negative():
    """RMSE is always ≥ 0."""
    rng = np.random.default_rng(0)
    y_true = rng.uniform(100, 2000, 50)
    y_pred = rng.uniform(100, 2000, 50)
    assert rmse(y_true, y_pred) >= 0.0


def test_mae_non_negative():
    """MAE is always ≥ 0."""
    rng = np.random.default_rng(1)
    y_true = rng.uniform(1000, 50000, 50)
    y_pred = rng.uniform(1000, 50000, 50)
    assert mae(y_true, y_pred) >= 0.0


def test_mape_non_negative():
    """MAPE is always ≥ 0."""
    rng = np.random.default_rng(2)
    y_true = rng.uniform(1000, 50000, 50) + 1  # avoid zero
    y_pred = rng.uniform(1000, 50000, 50)
    assert mape(y_true, y_pred) >= 0.0


def test_r2_at_most_one():
    """R² ≤ 1.0 always (but can be negative for bad models)."""
    rng = np.random.default_rng(3)
    y_true = rng.uniform(1800, 2000, 100)
    y_pred = rng.uniform(1800, 2000, 100)
    assert r2(y_true, y_pred) <= 1.0 + 1e-9


def test_evaluate_all_returns_all_keys():
    """evaluate_all returns a dict with RMSE, MAE, MAPE, R2 keys."""
    y = np.array([100.0, 200.0, 300.0])
    result = evaluate_all(y, y)
    assert set(result.keys()) == {"RMSE", "MAE", "MAPE", "R2"}


def test_evaluate_all_gold_like():
    """Sanity check: evaluate on gold-like scale returns reasonable ranges."""
    rng = np.random.default_rng(42)
    y_true = rng.uniform(1700, 1900, 200)
    y_pred = y_true + rng.normal(0, 20, 200)
    result = evaluate_all(y_true, y_pred)
    assert 0 < result["RMSE"] < 100
    assert 0 < result["MAE"] < 100
    assert 0 < result["MAPE"] < 10
    assert 0 < result["R2"] <= 1.0


def test_evaluate_all_bitcoin_like():
    """Sanity check: evaluate on bitcoin-like scale (large absolute values)."""
    rng = np.random.default_rng(42)
    y_true = rng.uniform(10000, 50000, 200)
    y_pred = y_true + rng.normal(0, 500, 200)
    result = evaluate_all(y_true, y_pred)
    assert result["RMSE"] > 0
    assert 0.9 < result["R2"] <= 1.0  # high R² expected for small noise


# ─────────────────────────────────────────────────────────────────────────────
# Edge cases
# ─────────────────────────────────────────────────────────────────────────────


def test_rmse_single_point():
    """RMSE with single data point."""
    assert rmse(np.array([100.0]), np.array([110.0])) == pytest.approx(10.0)


def test_mae_single_point():
    """MAE with single data point."""
    assert mae(np.array([100.0]), np.array([90.0])) == pytest.approx(10.0)


def test_mape_single_point():
    """MAPE with single data point: |100 - 50| / |100| * 100 = 50%."""
    assert mape(np.array([100.0]), np.array([50.0])) == pytest.approx(50.0, rel=1e-6)
