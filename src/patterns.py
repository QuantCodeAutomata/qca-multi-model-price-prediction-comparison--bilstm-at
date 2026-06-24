"""
Apriori-based consecutive rise/fall pattern analysis for price series.

Implements:
  - M_i computation: (p_i - p_{i-1}) / p_i  (current price in denominator)
  - Binary sequence construction
  - Run-length encoding and counting (m_j, n_j)
  - Statistical thresholds (M_u0.9, M_d0.1, M_u0.5)
  - u_n determination (90th percentile of empirical run-length distribution)
  - Apriori-style overlapping consecutive pattern counting
"""

from collections import Counter
from itertools import groupby
from typing import Any

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Core M_i formula
# ─────────────────────────────────────────────────────────────────────────────


def compute_mi(prices: pd.Series) -> pd.Series:
    """
    Compute M_i = (p_i - p_{i-1}) / p_i for all consecutive price pairs.

    NOTE: Denominator is p_i (current day's price), not p_{i-1}.
    This is the exact formula from the paper.

    Parameters
    ----------
    prices : pd.Series
        Daily closing price series indexed by date.

    Returns
    -------
    pd.Series
        M_i values indexed by date (one shorter than prices).
    """
    p = prices.values.astype(float)
    mi_values = (p[1:] - p[:-1]) / p[1:]
    return pd.Series(mi_values, index=prices.index[1:], name="M_i")


# ─────────────────────────────────────────────────────────────────────────────
# Statistical thresholds
# ─────────────────────────────────────────────────────────────────────────────


def compute_thresholds(mi_series: pd.Series) -> dict[str, float]:
    """
    Compute M_u0.9, M_d0.1, and M_u0.5 from an M_i series.

    Parameters
    ----------
    mi_series : pd.Series
        M_i values (can be full history or training portion).

    Returns
    -------
    dict with keys: 'M_u09', 'M_d01', 'M_u05'
        M_u09 : 90th percentile of positive M_i (strong rises)
        M_d01 : 10th percentile of negative M_i (most extreme declines, negative float)
        M_u05 : median of positive M_i (typical rise)
    """
    rises = mi_series[mi_series > 0].values
    declines = mi_series[mi_series < 0].values

    assert len(rises) > 0, "No positive M_i values found — check price data."
    assert len(declines) > 0, "No negative M_i values found — check price data."

    return {
        "M_u09": float(np.percentile(rises, 90)),
        "M_d01": float(np.percentile(declines, 10)),  # most negative 10%
        "M_u05": float(np.percentile(rises, 50)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Binary sequence and run-length analysis
# ─────────────────────────────────────────────────────────────────────────────


def binary_sequence(mi_series: pd.Series) -> np.ndarray:
    """
    Convert M_i series to binary: 1 if price rose (M_i > 0), 0 otherwise.

    Parameters
    ----------
    mi_series : pd.Series

    Returns
    -------
    np.ndarray of 0s and 1s.
    """
    return (mi_series.values > 0).astype(int)


def count_runs(binary_seq: np.ndarray) -> tuple[Counter, Counter, list, list]:
    """
    Run-length encode a binary sequence and count runs by length.

    Parameters
    ----------
    binary_seq : np.ndarray of 0s and 1s

    Returns
    -------
    mj : Counter  — {length: count} for rise runs (value=1)
    nj : Counter  — {length: count} for fall runs (value=0)
    rise_runs : list of individual rise run lengths
    fall_runs : list of individual fall run lengths
    """
    rise_runs: list[int] = []
    fall_runs: list[int] = []
    for key, grp in groupby(binary_seq.tolist()):
        length = sum(1 for _ in grp)
        if key == 1:
            rise_runs.append(length)
        else:
            fall_runs.append(length)

    mj = Counter(rise_runs)
    nj = Counter(fall_runs)
    return mj, nj, rise_runs, fall_runs


def compute_un(rise_runs: list[int], fall_runs: list[int], percentile: float = 90.0) -> int:
    """
    Compute u_n: the 90th percentile of the empirical run-length distribution.

    Creates an expanded list where each run length appears as many times as its
    count, then takes the ceiling of the specified percentile.

    Parameters
    ----------
    rise_runs : list[int]
        All individual rise run lengths.
    fall_runs : list[int]
        All individual fall run lengths.
    percentile : float
        Which percentile to use (default 90).

    Returns
    -------
    int  ≥ 2 (minimum enforced for numerical stability)
    """
    all_runs = rise_runs + fall_runs
    if len(all_runs) == 0:
        return 2
    un = int(np.ceil(np.percentile(all_runs, percentile)))
    return max(un, 2)


# ─────────────────────────────────────────────────────────────────────────────
# Apriori-style consecutive pattern counting (overlapping)
# Custom — Context7 found no library equivalent for sequential overlapping runs
# ─────────────────────────────────────────────────────────────────────────────


def apriori_consecutive_count(binary_seq: np.ndarray, j: int, value: int = 1) -> int:
    """
    Count overlapping occurrences of j consecutive `value` in binary_seq.

    A run of length k ≥ j contributes (k - j + 1) overlapping windows of length j.

    Parameters
    ----------
    binary_seq : np.ndarray
        Binary sequence (0s and 1s).
    j : int
        Length of the consecutive pattern to count.
    value : int
        Pattern element (1 = rise, 0 = fall).

    Returns
    -------
    int  count of overlapping j-length patterns.
    """
    count = 0
    arr = binary_seq.tolist()
    for i in range(len(arr) - j + 1):
        if all(arr[i + k] == value for k in range(j)):
            count += 1
    return count


def build_run_table(
    binary_seq: np.ndarray, max_j: int | None = None
) -> pd.DataFrame:
    """
    Build a summary table of run counts for all j up to max_j.

    Uses both exact run-length counts (mj, nj from count_runs) and
    overlapping Apriori counts.

    Parameters
    ----------
    binary_seq : np.ndarray
    max_j : int or None
        Maximum j to compute. Defaults to the longest observed run.

    Returns
    -------
    pd.DataFrame with columns [j, m_j_exact, n_j_exact, m_j_apriori, n_j_apriori]
    """
    mj, nj, rise_runs, fall_runs = count_runs(binary_seq)
    if max_j is None:
        max_j = max(max(rise_runs, default=1), max(fall_runs, default=1))

    records = []
    for j in range(1, max_j + 1):
        records.append(
            {
                "j": j,
                "m_j_exact": mj.get(j, 0),
                "n_j_exact": nj.get(j, 0),
                "m_j_apriori": apriori_consecutive_count(binary_seq, j, value=1),
                "n_j_apriori": apriori_consecutive_count(binary_seq, j, value=0),
            }
        )
    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# Full analysis pipeline for one asset
# ─────────────────────────────────────────────────────────────────────────────


def analyse_asset(
    prices: pd.Series,
    train_frac: float = 0.70,
) -> dict[str, Any]:
    """
    Run the full Apriori pattern analysis pipeline for one asset.

    Parameters
    ----------
    prices : pd.Series
        Daily closing price series.
    train_frac : float
        Fraction of data to use as training set for threshold computation.

    Returns
    -------
    dict with keys:
        'mi_full', 'mi_train',
        'thresholds_full', 'thresholds_train',
        'binary_full', 'binary_train',
        'mj_full', 'nj_full', 'rise_runs_full', 'fall_runs_full',
        'mj_train', 'nj_train', 'rise_runs_train', 'fall_runs_train',
        'un_full', 'un_train',
        'run_table_full'
    """
    n = len(prices)
    split = int(n * train_frac)
    train_prices = prices.iloc[:split]

    mi_full = compute_mi(prices)
    mi_train = compute_mi(train_prices)

    thresholds_full = compute_thresholds(mi_full)
    thresholds_train = compute_thresholds(mi_train)

    binary_full = binary_sequence(mi_full)
    binary_train = binary_sequence(mi_train)

    mj_f, nj_f, rr_f, fr_f = count_runs(binary_full)
    mj_t, nj_t, rr_t, fr_t = count_runs(binary_train)

    un_full = compute_un(rr_f, fr_f)
    un_train = compute_un(rr_t, fr_t)

    run_table = build_run_table(binary_full)

    return {
        "mi_full": mi_full,
        "mi_train": mi_train,
        "thresholds_full": thresholds_full,
        "thresholds_train": thresholds_train,
        "binary_full": binary_full,
        "binary_train": binary_train,
        "mj_full": mj_f,
        "nj_full": nj_f,
        "rise_runs_full": rr_f,
        "fall_runs_full": fr_f,
        "mj_train": mj_t,
        "nj_train": nj_t,
        "rise_runs_train": rr_t,
        "fall_runs_train": fr_t,
        "un_full": un_full,
        "un_train": un_train,
        "run_table_full": run_table,
    }
