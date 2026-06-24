"""
Exponential Position Sizing Model (paper Steps 2–8 from exp_3).

Derives the sequence p_1, ..., p_{u_n} from recursive breakeven equations,
fits an exponential model s_i = a * exp(beta * i) via log-linear OLS, then
computes dollar allocations y_i = (PRO / sum(s_i)) * s_i.

Custom — Context7 found no library equivalent (paper Eq. 5-6)
"""

import numpy as np
from scipy import stats


def compute_position_sequence(
    M_u05: float,
    M_d01: float,
    u_n: int,
    fee_pct: float = 0.01,
) -> np.ndarray:
    """
    Derive the relative position size sequence p_1, ..., p_{u_n} from the
    recursive breakeven equations (paper Equations 5–6).

    The condition for adding the n-th position is that a median recovery M_u05
    on all positions offsets the accumulated drawdown at rate M_d01/u_n per
    period on all prior positions.

    General recursive formula (derived from breakeven condition):
        p_n = -sum(p_k * r^{n-1-k}  for k in 0..n-2)
    where r = M_d01 / u_n  (negative number)

    Parameters
    ----------
    M_u05 : float
        Median daily rise (positive, e.g. 0.004).
    M_d01 : float
        10th-percentile daily decline (negative, e.g. -0.004).
    u_n : int
        Number of position addition steps (≥ 2).
    fee_pct : float
        Transaction fee rate (not used in the simplified recursive formula
        but included for API completeness; fee cancels in the breakeven).

    Returns
    -------
    np.ndarray, shape (u_n,)
        Relative position sizes p_1, ..., p_{u_n}. All positive.
    """
    M_u05 = max(float(M_u05), 1e-8)
    M_d01 = min(float(M_d01), -1e-8)  # ensure negative
    u_n = max(int(u_n), 2)

    r = M_d01 / u_n  # negative ratio per period

    p = [1.0]  # p_1 = 1.0 (normalised unit)
    for n in range(2, u_n + 1):
        # Breakeven: p_n = -sum(p_k * r^{n-1-k} for k=0..n-2)
        cumulative = sum(p[k] * (r ** (n - 1 - k)) for k in range(len(p)))
        p_n = -cumulative  # r < 0 so cumulative can be negative; p_n > 0
        p.append(max(float(p_n), 1e-8))

    return np.array(p, dtype=float)


def fit_exponential(p_sequence: np.ndarray) -> tuple[float, float, np.ndarray, float]:
    """
    Fit s_i = a * exp(beta * i) to the position sequence via OLS on log scale.

    log(p_i) = log(a) + beta * i  →  slope = beta, intercept = log(a)

    Parameters
    ----------
    p_sequence : np.ndarray
        Derived position sizes p_1, ..., p_{u_n}.

    Returns
    -------
    a : float
        Scale parameter (> 0).
    beta : float
        Exponential growth rate.
    s_fitted : np.ndarray
        Smooth fitted values s_i = a * exp(beta * i).
    r2 : float
        Coefficient of determination of the log-linear fit.
    """
    n = len(p_sequence)
    i_vals = np.arange(1, n + 1, dtype=float)
    log_p = np.log(np.maximum(p_sequence, 1e-12))
    slope, intercept, r_value, _, _ = stats.linregress(i_vals, log_p)
    a = float(np.exp(intercept))
    beta = float(slope)
    s_fitted = a * np.exp(beta * i_vals)
    return a, beta, s_fitted, float(r_value ** 2)


def compute_dollar_allocations(
    s_fitted: np.ndarray, pro: float
) -> np.ndarray:
    """
    Compute per-step dollar allocations (paper Equation 8).

    y_i = (PRO / sum(s_i)) * s_i

    Parameters
    ----------
    s_fitted : np.ndarray
        Fitted exponential position sizes.
    pro : float
        Total available capital (PRO).

    Returns
    -------
    np.ndarray
        Dollar allocations y_i. sum(y_i) == pro (within floating-point precision).
    """
    total = s_fitted.sum()
    y = (pro / total) * s_fitted
    assert abs(y.sum() - pro) < 1.0, f"Allocations sum {y.sum():.4f} ≠ PRO {pro}"
    return y


def full_position_sizing(
    M_u05: float,
    M_d01: float,
    u_n: int,
    pro: float,
    fee_pct: float = 0.01,
) -> dict:
    """
    Run the complete position sizing pipeline for one asset.

    Parameters
    ----------
    M_u05 : float
        Median daily rise (from training data).
    M_d01 : float
        10th-percentile daily decline (from training data).
    u_n : int
        Number of position steps.
    pro : float
        Total capital to allocate.
    fee_pct : float
        Transaction fee rate.

    Returns
    -------
    dict with keys: 'p_sequence', 'a', 'beta', 's_fitted', 'r2', 'y_alloc', 's_norm'
    """
    p_seq = compute_position_sequence(M_u05, M_d01, u_n, fee_pct)
    a, beta, s_fitted, r2 = fit_exponential(p_seq)
    y_alloc = compute_dollar_allocations(s_fitted, pro)
    s_norm = s_fitted / s_fitted.sum()  # proportional weights

    return {
        "p_sequence": p_seq,
        "a": a,
        "beta": beta,
        "s_fitted": s_fitted,
        "r2": r2,
        "y_alloc": y_alloc,
        "s_norm": s_norm,
    }
