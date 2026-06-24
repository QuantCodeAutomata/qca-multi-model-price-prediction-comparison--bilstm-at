"""
Unit tests for src/patterns.py.

Validates M_i formula, threshold computation, run-length encoding,
Apriori counting, and u_n determination.
"""

import sys
import os
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.patterns import (
    compute_mi,
    compute_thresholds,
    binary_sequence,
    count_runs,
    compute_un,
    apriori_consecutive_count,
    build_run_table,
)


# ─────────────────────────────────────────────────────────────────────────────
# M_i formula
# ─────────────────────────────────────────────────────────────────────────────


def test_mi_formula_exact():
    """
    M_i = (p_i - p_{i-1}) / p_i uses current price in denominator.
    For p = [100, 110]: M_1 = (110-100)/110 = 10/110 ≈ 0.09090...
    """
    prices = pd.Series([100.0, 110.0], index=pd.date_range("2020-01-01", periods=2))
    mi = compute_mi(prices)
    expected = (110.0 - 100.0) / 110.0
    assert mi.iloc[0] == pytest.approx(expected, rel=1e-8)


def test_mi_denominator_is_current_not_previous():
    """
    Verify denominator is p_i (current), not p_{i-1} (previous).
    M_i != (p_i - p_{i-1}) / p_{i-1}  for non-trivial changes.
    """
    prices = pd.Series([200.0, 210.0], index=pd.date_range("2020-01-01", periods=2))
    mi = compute_mi(prices)
    standard_return = (210.0 - 200.0) / 200.0  # 0.05 (standard convention)
    paper_mi = (210.0 - 200.0) / 210.0         # ≈ 0.04762 (paper convention)
    assert mi.iloc[0] != pytest.approx(standard_return, abs=1e-6)
    assert mi.iloc[0] == pytest.approx(paper_mi, rel=1e-8)


def test_mi_length():
    """M_i series should have length N-1 for N prices."""
    prices = pd.Series(
        np.random.default_rng(0).uniform(100, 200, 50),
        index=pd.date_range("2020-01-01", periods=50),
    )
    mi = compute_mi(prices)
    assert len(mi) == 49


def test_mi_sign_on_rise():
    """M_i > 0 when price rises."""
    prices = pd.Series([100.0, 105.0], index=pd.date_range("2020-01-01", periods=2))
    assert compute_mi(prices).iloc[0] > 0


def test_mi_sign_on_fall():
    """M_i < 0 when price falls."""
    prices = pd.Series([105.0, 100.0], index=pd.date_range("2020-01-01", periods=2))
    assert compute_mi(prices).iloc[0] < 0


# ─────────────────────────────────────────────────────────────────────────────
# Threshold computation
# ─────────────────────────────────────────────────────────────────────────────


def test_thresholds_signs():
    """M_u09 and M_u05 should be positive; M_d01 should be negative."""
    rng = np.random.default_rng(42)
    mi = pd.Series(rng.normal(0, 0.01, 500))
    thr = compute_thresholds(mi)
    assert thr["M_u09"] > 0
    assert thr["M_u05"] > 0
    assert thr["M_d01"] < 0


def test_thresholds_ordering():
    """M_u05 < M_u09 (median rise < 90th-pct rise)."""
    rng = np.random.default_rng(0)
    mi = pd.Series(rng.normal(0, 0.005, 1000))
    thr = compute_thresholds(mi)
    assert thr["M_u05"] < thr["M_u09"]


def test_threshold_m_d01_is_extreme_decline():
    """M_d01 should be the 10th-percentile (most negative) of declines."""
    rng = np.random.default_rng(1)
    mi = pd.Series(rng.normal(0, 0.01, 500))
    declines = mi[mi < 0]
    thr = compute_thresholds(mi)
    expected = float(np.percentile(declines, 10))
    assert thr["M_d01"] == pytest.approx(expected, rel=1e-8)


# ─────────────────────────────────────────────────────────────────────────────
# Binary sequence
# ─────────────────────────────────────────────────────────────────────────────


def test_binary_sequence_values():
    """Binary sequence should only contain 0s and 1s."""
    mi = pd.Series([-0.01, 0.005, 0.0, -0.002, 0.01])
    b = binary_sequence(mi)
    assert set(b.tolist()).issubset({0, 1})


def test_binary_zero_on_negative_mi():
    """M_i <= 0 maps to B_i = 0."""
    mi = pd.Series([-0.01, 0.0])
    b = binary_sequence(mi)
    assert b[0] == 0
    assert b[1] == 0  # zero M_i is a fall


def test_binary_one_on_positive_mi():
    """M_i > 0 maps to B_i = 1."""
    mi = pd.Series([0.005])
    b = binary_sequence(mi)
    assert b[0] == 1


# ─────────────────────────────────────────────────────────────────────────────
# Run-length counting
# ─────────────────────────────────────────────────────────────────────────────


def test_count_runs_known_sequence():
    """
    Sequence [1,1,1,0,0,1,0,1,1] has:
      rise runs: [3, 1, 2]  → mj = {3:1, 1:1, 2:1}
      fall runs: [2, 1]     → nj = {2:1, 1:1}
    """
    b = np.array([1, 1, 1, 0, 0, 1, 0, 1, 1])
    mj, nj, rr, fr = count_runs(b)
    assert mj[3] == 1
    assert mj[1] == 1
    assert mj[2] == 1
    assert nj[2] == 1
    assert nj[1] == 1
    assert len(rr) == 3
    assert len(fr) == 2


def test_count_runs_all_rises():
    """All-1 sequence: one rise run of full length, no fall runs."""
    b = np.ones(10, dtype=int)
    mj, nj, rr, fr = count_runs(b)
    assert mj[10] == 1
    assert len(fr) == 0


def test_count_runs_alternating():
    """Alternating 1,0,1,0 → 4 runs each of length 1."""
    b = np.array([1, 0, 1, 0])
    mj, nj, rr, fr = count_runs(b)
    assert mj[1] == 2
    assert nj[1] == 2


# ─────────────────────────────────────────────────────────────────────────────
# u_n determination
# ─────────────────────────────────────────────────────────────────────────────


def test_un_minimum_is_two():
    """u_n should always be ≥ 2."""
    assert compute_un([], []) == 2
    assert compute_un([1], [1]) == 2


def test_un_90th_percentile():
    """
    For rise_runs=[1,1,1,1,1,1,1,1,1,2], fall_runs=[],
    all_runs=[1,1,1,1,1,1,1,1,1,2]  →  90th percentile ≈ 2.
    """
    rise_runs = [1] * 9 + [2]
    un = compute_un(rise_runs, [])
    assert un == 2


def test_un_increases_with_longer_runs():
    """u_n should be larger when runs are longer."""
    short_runs = [1, 1, 1, 2, 2]
    long_runs = [3, 4, 5, 6, 7]
    assert compute_un(long_runs, []) > compute_un(short_runs, [])


# ─────────────────────────────────────────────────────────────────────────────
# Apriori overlapping count
# ─────────────────────────────────────────────────────────────────────────────


def test_apriori_j1_equals_sum_of_ones():
    """j=1 count for value=1 should equal total number of 1s."""
    b = np.array([1, 0, 1, 1, 0, 1])
    count = apriori_consecutive_count(b, j=1, value=1)
    assert count == 4  # four 1s


def test_apriori_j3_run_of_5():
    """
    In a run of 5 consecutive 1s, there are 3 overlapping windows of length 3.
    """
    b = np.array([1, 1, 1, 1, 1])
    count = apriori_consecutive_count(b, j=3, value=1)
    assert count == 3


def test_apriori_no_pattern():
    """No 3-consecutive-rises in [1,0,1,0,1]."""
    b = np.array([1, 0, 1, 0, 1])
    assert apriori_consecutive_count(b, j=3, value=1) == 0


def test_apriori_j_exceeds_length():
    """j larger than sequence length should return 0."""
    b = np.array([1, 1, 1])
    assert apriori_consecutive_count(b, j=5, value=1) == 0


# ─────────────────────────────────────────────────────────────────────────────
# Build run table
# ─────────────────────────────────────────────────────────────────────────────


def test_build_run_table_columns():
    """Run table should have the expected columns."""
    b = np.array([1, 1, 0, 1, 0, 0, 1])
    df = build_run_table(b)
    expected_cols = {"j", "m_j_exact", "n_j_exact", "m_j_apriori", "n_j_apriori"}
    assert set(df.columns) == expected_cols


def test_build_run_table_j_starts_at_1():
    """j column should start at 1."""
    b = np.array([1, 0, 1, 1, 0])
    df = build_run_table(b)
    assert df["j"].min() == 1


def test_build_run_table_apriori_ge_exact():
    """
    Apriori (overlapping) count should be ≥ exact (non-overlapping) count.
    A run of length 3 contributes 1 exact count of length 3 but also
    an overlapping count of 1; so they're equal in this case.
    """
    b = np.array([1, 1, 1, 0, 1, 0])
    df = build_run_table(b, max_j=3)
    row = df[df["j"] == 2].iloc[0]
    # Apriori count for j=2 in [1,1,1]: 2 overlapping windows; exact: 0 runs of exactly 2
    assert row["m_j_apriori"] >= row["m_j_exact"]
