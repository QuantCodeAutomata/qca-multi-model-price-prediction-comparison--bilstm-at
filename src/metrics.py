"""
Evaluation metrics for price prediction models.

Implements RMSE, MAE, MAPE, and R² as described in the methodology.
"""

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(mean_absolute_error(y_true, y_pred))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Absolute Percentage Error (returned as a percentage, e.g. 0.76 means 0.76%).

    MAPE = mean(|y_pred - y_true| / |y_true|) * 100
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    return float(np.mean(np.abs(y_pred[mask] - y_true[mask]) / np.abs(y_true[mask])) * 100)


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R² (coefficient of determination)."""
    return float(r2_score(y_true, y_pred))


def evaluate_all(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """
    Compute all four metrics for a given set of predictions.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth prices (original scale).
    y_pred : np.ndarray
        Predicted prices (original scale).

    Returns
    -------
    dict with keys: 'RMSE', 'MAE', 'MAPE', 'R2'
    """
    return {
        "RMSE": rmse(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
        "R2": r2(y_true, y_pred),
    }
