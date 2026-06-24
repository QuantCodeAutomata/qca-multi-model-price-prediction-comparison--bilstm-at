"""
Visualisation helpers for all experiments.

Produces:
  Figure 6  — Actual vs. predicted price overlays (exp_1)
  Figure 7  — M_i time-series (exp_2)
  Figure 8  — Separated rise / fall curves (exp_2)
  Figure 9/10 — Portfolio value curves (exp_4)
  Figure 11/12 — Fee sensitivity bar charts (exp_5)
  Position sizing curve + bar chart (exp_3)
"""

from __future__ import annotations

import os
from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Exp-1: Prediction overlay (Figure 6)
# ─────────────────────────────────────────────────────────────────────────────


def plot_predictions(
    dates: pd.DatetimeIndex | list,
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
    asset_name: str,
    save_path: str | None = None,
) -> None:
    """
    Plot actual vs. predicted prices for multiple models.

    Parameters
    ----------
    dates : DatetimeIndex or list
        Dates corresponding to test-set predictions.
    y_true : np.ndarray
        Actual prices.
    predictions : dict[str, np.ndarray]
        Model-name → predicted-price array.
    asset_name : str
        Label for the asset (e.g. 'Gold', 'Bitcoin').
    save_path : str or None
        File path to save the figure.
    """
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(dates, y_true, color="black", linewidth=2.0, label="Actual", zorder=10)

    colors = plt.cm.tab10(np.linspace(0, 1, len(predictions)))
    linestyles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 2)), (0, (1, 1)), "-"]
    for (name, preds), color, ls in zip(predictions.items(), colors, linestyles):
        ax.plot(dates, preds, color=color, linestyle=ls, linewidth=1.2, alpha=0.85, label=name)

    ax.set_title(f"{asset_name} — Actual vs. Predicted Closing Prices (Test Set)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Exp-2: M_i time-series (Figure 7)
# ─────────────────────────────────────────────────────────────────────────────


def plot_mi_series(
    mi_gold: pd.Series,
    mi_btc: pd.Series,
    save_path: str | None = None,
) -> None:
    """
    Plot M_i time-series for gold and bitcoin (Figure 7 equivalent).

    Rises coloured green, declines red, zero reference line shown.
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=False)

    for ax, mi, name in zip(axes, [mi_gold, mi_btc], ["Gold", "Bitcoin"]):
        ax.plot(mi.index, mi.values, color="grey", linewidth=0.5, alpha=0.6)
        ax.fill_between(
            mi.index, mi.values, 0,
            where=mi.values > 0, color="green", alpha=0.45, label="Rise (M_i > 0)",
        )
        ax.fill_between(
            mi.index, mi.values, 0,
            where=mi.values <= 0, color="red", alpha=0.45, label="Fall (M_i ≤ 0)",
        )
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_title(f"{name} — Daily M_i Values  (M_i = (p_i − p_{{i−1}}) / p_i)")
        ax.set_xlabel("Date")
        ax.set_ylabel("M_i")
        ax.legend(fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Exp-2: Separated rise/fall curves (Figure 8)
# ─────────────────────────────────────────────────────────────────────────────


def plot_rise_fall_curves(
    mi_gold: pd.Series,
    mi_btc: pd.Series,
    thresholds_gold: dict,
    thresholds_btc: dict,
    save_path: str | None = None,
) -> None:
    """
    Plot rise and fall M_i sub-series with threshold reference lines (Figure 8).

    Parameters
    ----------
    thresholds_gold, thresholds_btc : dict
        Output of patterns.compute_thresholds(); must have keys 'M_u09', 'M_d01'.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 8))

    for col, (mi, name, thr) in enumerate(
        [
            (mi_gold, "Gold", thresholds_gold),
            (mi_btc, "Bitcoin", thresholds_btc),
        ]
    ):
        rises = mi[mi > 0]
        declines = mi[mi < 0]

        # Rise subplot
        ax = axes[0, col]
        ax.plot(rises.index, rises.values, color="green", linewidth=0.6, alpha=0.8)
        ax.axhline(thr["M_u09"], color="darkgreen", linestyle="--", linewidth=1.2,
                   label=f"M_u0.9 = {thr['M_u09']:.4f}")
        ax.axhline(thr["M_u05"], color="lime", linestyle=":", linewidth=1.2,
                   label=f"M_u0.5 = {thr['M_u05']:.4f}")
        ax.set_title(f"{name} — Rise Days (M_i > 0)")
        ax.set_ylabel("M_i")
        ax.legend(fontsize=8)

        # Fall subplot
        ax = axes[1, col]
        ax.plot(declines.index, declines.values, color="red", linewidth=0.6, alpha=0.8)
        ax.axhline(thr["M_d01"], color="darkred", linestyle="--", linewidth=1.2,
                   label=f"M_d0.1 = {thr['M_d01']:.4f}")
        ax.set_title(f"{name} — Fall Days (M_i ≤ 0)")
        ax.set_ylabel("M_i")
        ax.legend(fontsize=8)

    for ax in axes.flat:
        ax.set_xlabel("Date")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Exp-3: Position sizing curve
# ─────────────────────────────────────────────────────────────────────────────


def plot_position_sizing(
    p_sequence: np.ndarray,
    s_fitted: np.ndarray,
    y_alloc: np.ndarray,
    a: float,
    beta: float,
    asset_name: str,
    pro: float,
    save_path: str | None = None,
) -> None:
    """Plot exponential position sizing curve and dollar allocation bar chart."""
    u_n = len(p_sequence)
    i_vals = np.arange(1, u_n + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Curve + scatter
    ax = axes[0]
    ax.scatter(i_vals, p_sequence, color="steelblue", zorder=5, label="Derived p_i")
    ax.plot(
        i_vals, s_fitted, color="red", linewidth=2,
        label=f"s_i = {a:.3f}·exp({beta:.3f}·i)",
    )
    ax.set_title(f"{asset_name} — Exponential Position Sizing")
    ax.set_xlabel("Position Step i")
    ax.set_ylabel("Relative Size")
    ax.legend()

    # Bar chart of dollar allocations
    ax = axes[1]
    ax.bar(i_vals, y_alloc, color="seagreen", alpha=0.8)
    ax.set_title(f"{asset_name} — Dollar Allocations (PRO = ${pro:,.0f})")
    ax.set_xlabel("Position Step i")
    ax.set_ylabel("Dollar Amount (USD)")

    plt.suptitle(f"{asset_name} Position Sizing (a={a:.4f}, β={beta:.4f})")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Exp-4: Portfolio value curves (Figures 9 & 10)
# ─────────────────────────────────────────────────────────────────────────────


def plot_portfolio_value(
    gold_history: pd.DataFrame,
    btc_history: pd.DataFrame,
    gold_final: float,
    btc_final: float,
    save_path: str | None = None,
) -> None:
    """Plot cumulative portfolio value for gold and bitcoin over simulation period."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    for ax, hist, name, final in zip(
        axes,
        [gold_history, btc_history],
        ["Gold", "Bitcoin"],
        [gold_final, btc_final],
    ):
        ax.plot(hist["date"], hist["portfolio_value"], linewidth=1.5, color="royalblue")
        ax.axhline(500, color="red", linestyle="--", linewidth=1.2, label="Initial $500")
        ax.set_title(f"{name} Portfolio Value  (Final: ${final:,.0f})")
        ax.set_xlabel("Date")
        ax.set_ylabel("Portfolio Value (USD)")
        ax.legend()

    plt.suptitle("Portfolio Value Over Simulation Period (exp_4)")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Exp-5: Fee sensitivity bar charts (Figures 11 & 12)
# ─────────────────────────────────────────────────────────────────────────────


def plot_fee_sensitivity(
    fees: list[float],
    final_values: list[float],
    asset_name: str,
    colormap: str = "Blues",
    save_path: str | None = None,
) -> None:
    """
    Bar chart of final portfolio value vs. transaction fee (Figures 11 & 12).

    Parameters
    ----------
    fees : list[float]
        Transaction fee levels as decimals.
    final_values : list[float]
        Corresponding final portfolio values.
    asset_name : str
        'Gold' or 'Bitcoin'.
    colormap : str
        Matplotlib colormap name for bar colouring.
    save_path : str or None
    """
    fee_labels = [f"{f * 100:.1f}%" for f in fees]
    cmap = cm.get_cmap(colormap)
    colours = [cmap(v) for v in np.linspace(0.4, 0.9, len(fees))]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(fee_labels, final_values, color=colours, edgecolor="black", linewidth=0.6)
    ax.plot(fee_labels, final_values, "o-", color="darkblue", linewidth=2, markersize=7,
            label="Final Value Trend")
    ax.axhline(500, color="green", linestyle="--", linewidth=1.5, label="Initial Capital ($500)")

    # Annotate bars
    for bar, val in zip(bars, final_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.005,
            f"${val:,.0f}",
            ha="center", va="bottom", fontsize=8,
        )

    ax.set_title(f"{asset_name} — Final Portfolio Value vs. Transaction Fee")
    ax.set_xlabel("Transaction Fee (%)")
    ax.set_ylabel("Final Portfolio Value (USD)")
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
