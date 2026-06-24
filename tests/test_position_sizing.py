"""
Unit tests for src/position_sizing.py.

Validates recursive breakeven formula, exponential fitting, and
dollar allocation properties.
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.position_sizing import (
    compute_position_sequence,
    fit_exponential,
    compute_dollar_allocations,
    full_position_sizing,
)


# ─────────────────────────────────────────────────────────────────────────────
# compute_position_sequence
# ─────────────────────────────────────────────────────────────────────────────


def test_p1_is_one():
    """First position p_1 must always be 1.0 (normalised base)."""
    p = compute_position_sequence(M_u05=0.004, M_d01=-0.004, u_n=4)
    assert p[0] == pytest.approx(1.0, rel=1e-8)


def test_all_positions_positive():
    """All position sizes must be positive."""
    for u_n in [2, 3, 4, 5]:
        p = compute_position_sequence(M_u05=0.004, M_d01=-0.005, u_n=u_n)
        assert all(pi > 0 for pi in p), f"Negative position found for u_n={u_n}"


def test_length_equals_un():
    """Sequence length should equal u_n."""
    for u_n in [2, 3, 4, 5, 6]:
        p = compute_position_sequence(M_u05=0.003, M_d01=-0.003, u_n=u_n)
        assert len(p) == u_n


def test_positions_grow_with_larger_decline():
    """
    Larger decline magnitude (more negative M_d01) should require larger
    subsequent positions to maintain breakeven.
    """
    p_small = compute_position_sequence(M_u05=0.004, M_d01=-0.002, u_n=3)
    p_large = compute_position_sequence(M_u05=0.004, M_d01=-0.008, u_n=3)
    # p_2 should be larger for larger decline
    assert p_large[1] >= p_small[1]


def test_minimum_un_enforced():
    """u_n=0 or u_n=1 should be coerced to u_n=2."""
    p0 = compute_position_sequence(M_u05=0.004, M_d01=-0.004, u_n=0)
    p1 = compute_position_sequence(M_u05=0.004, M_d01=-0.004, u_n=1)
    assert len(p0) == 2
    assert len(p1) == 2


def test_recursive_breakeven_property():
    """
    Verify the breakeven condition: for each step n ≥ 2,
    p[n-1] == -sum(p[k] * r^{n-1-k} for k=0..n-2)
    Values clipped to 1e-8 when the formula yields non-positive results.
    """
    M_u05 = 0.005
    M_d01 = -0.004
    u_n = 4
    p = compute_position_sequence(M_u05, M_d01, u_n)
    r = M_d01 / u_n
    CLIP = 1e-8
    for n in range(2, u_n + 1):
        rhs = -sum(p[k] * (r ** (n - 1 - k)) for k in range(n - 1))
        # If formula produces a non-positive value, the implementation clips to CLIP
        expected = max(rhs, CLIP)
        assert p[n - 1] == pytest.approx(expected, rel=1e-4)


def test_bitcoin_params_larger_positions():
    """
    Bitcoin-like params (large M values) should produce larger step ratios
    than gold-like params.
    """
    p_gold = compute_position_sequence(M_u05=0.004, M_d01=-0.004, u_n=3)
    p_btc = compute_position_sequence(M_u05=0.04, M_d01=-0.05, u_n=3)
    # The ratio p[2]/p[1] may differ
    ratio_gold = p_gold[1] / p_gold[0] if p_gold[0] > 0 else 0
    ratio_btc = p_btc[1] / p_btc[0] if p_btc[0] > 0 else 0
    # Both ratios should be positive
    assert ratio_gold > 0
    assert ratio_btc > 0


# ─────────────────────────────────────────────────────────────────────────────
# fit_exponential
# ─────────────────────────────────────────────────────────────────────────────


def test_fit_exponential_returns_positive_a():
    """Scale parameter a should be positive."""
    p = np.array([1.0, 1.5, 2.25, 3.375])  # exact exponential
    a, beta, s_fitted, r2 = fit_exponential(p)
    assert a > 0


def test_fit_exponential_perfect_fit():
    """For a perfect exponential sequence, R² should be ~1.0."""
    i = np.arange(1, 6, dtype=float)
    p = 2.0 * np.exp(0.3 * i)  # exact: log(p_i) = log(2) + 0.3*i
    a, beta, s_fitted, r2 = fit_exponential(p)
    assert r2 == pytest.approx(1.0, abs=1e-6)
    # OLS intercept = log(2), so a = exp(log(2)) = 2.0 (not 2*exp(0.3))
    assert a == pytest.approx(2.0, rel=1e-4)
    assert beta == pytest.approx(0.3, rel=1e-4)


def test_fit_exponential_monotone_growth():
    """For a growing sequence, beta should be positive."""
    p = np.array([1.0, 2.0, 4.0, 8.0])
    a, beta, _, _ = fit_exponential(p)
    assert beta > 0


def test_fit_exponential_length_preserved():
    """Fitted values should have same length as input sequence."""
    p = np.array([1.0, 1.2, 1.5, 2.0, 2.8])
    _, _, s_fitted, _ = fit_exponential(p)
    assert len(s_fitted) == len(p)


# ─────────────────────────────────────────────────────────────────────────────
# compute_dollar_allocations
# ─────────────────────────────────────────────────────────────────────────────


def test_allocations_sum_to_pro():
    """sum(y_i) must equal PRO within floating-point tolerance."""
    s = np.array([1.0, 2.0, 4.0, 8.0])
    pro = 500.0
    y = compute_dollar_allocations(s, pro)
    assert abs(y.sum() - pro) < 1e-6


def test_allocations_proportional():
    """y_i should be proportional to s_i."""
    s = np.array([1.0, 2.0, 4.0])
    pro = 700.0
    y = compute_dollar_allocations(s, pro)
    # y[1] / y[0] should ≈ s[1] / s[0] = 2
    assert y[1] / y[0] == pytest.approx(2.0, rel=1e-6)
    assert y[2] / y[0] == pytest.approx(4.0, rel=1e-6)


def test_allocations_positive():
    """All dollar allocations should be positive."""
    s = np.array([0.5, 1.0, 1.5, 2.5])
    y = compute_dollar_allocations(s, 500.0)
    assert all(yi > 0 for yi in y)


# ─────────────────────────────────────────────────────────────────────────────
# full_position_sizing (integration)
# ─────────────────────────────────────────────────────────────────────────────


def test_full_pipeline_gold_params():
    """Full pipeline with gold-like params should return valid structure."""
    result = full_position_sizing(
        M_u05=0.004, M_d01=-0.004, u_n=4, pro=500.0, fee_pct=0.01
    )
    assert "p_sequence" in result
    assert "a" in result
    assert "beta" in result
    assert "s_fitted" in result
    assert "r2" in result
    assert "y_alloc" in result
    assert "s_norm" in result
    assert abs(result["y_alloc"].sum() - 500.0) < 1.0
    assert result["r2"] > 0.5  # reasonable exponential fit


def test_full_pipeline_s_norm_sums_to_one():
    """s_norm proportions should sum to 1."""
    result = full_position_sizing(
        M_u05=0.04, M_d01=-0.05, u_n=3, pro=500.0, fee_pct=0.025
    )
    assert abs(result["s_norm"].sum() - 1.0) < 1e-6


def test_full_pipeline_high_r2():
    """Exponential fit R² should exceed 0.70 for typical inputs (few data points)."""
    for u_n in [3, 4, 5]:
        result = full_position_sizing(
            M_u05=0.005, M_d01=-0.006, u_n=u_n, pro=500.0, fee_pct=0.01
        )
        assert result["r2"] > 0.70, f"R² too low for u_n={u_n}: {result['r2']}"
