"""
exp_4 — Trading Strategy Portfolio Simulation.

BiLSTM-Attention + Exponential Position Sizing + VaR-Constrained Modified
Greedy Algorithm over ~last 30% of price data (ending 2021-09-10).

Initial capital: $500 per asset.
Fees: Gold 1.0%, Bitcoin 2.5% (base case).

Targets (±20% tolerance):
  Gold    ~$646   Bitcoin  ~$215,487

Outputs:
  - results/fig9_10_portfolio_value.png
  - results/exp4_simulation_gold.csv
  - results/exp4_simulation_btc.csv
  - results/exp4_summary.csv
  - data/trained_models/  (saved Keras models)
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import tensorflow as tf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.download import get_prices
from src.models import (
    WINDOW_SIZE,
    DotProductAttention,
    build_bilstm_attention,
    create_windows,
    train_test_split_windows,
    fit_dl_model,
)
from src.patterns import compute_mi, compute_thresholds, count_runs, compute_un
from src.position_sizing import full_position_sizing
from src.trading import simulate_trading
from src.visualisation import plot_portfolio_value

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
MODELS_DIR = os.path.join(DATA_DIR, "trained_models")

TRAIN_FRAC = 0.70
CONFIGS = {
    "Gold": {"fee_pct": 0.01, "pro": 500.0},
    "Bitcoin": {"fee_pct": 0.025, "pro": 500.0},
}


def train_bilstm_model(prices: pd.Series) -> tf.keras.Model:
    """Train BiLSTM-Attention on the training portion of prices."""
    X, y, w_min, w_max = create_windows(prices.values, WINDOW_SIZE)
    X_tr, X_te, y_tr, y_te, *_ = train_test_split_windows(
        X, y, w_min, w_max, TRAIN_FRAC, shuffle_train=True
    )
    tf.keras.backend.clear_session()
    tf.random.set_seed(42)
    model = build_bilstm_attention(WINDOW_SIZE)
    fit_dl_model(model, X_tr, y_tr)
    return model


def compute_pos_params(prices: pd.Series, fee_pct: float, pro: float) -> dict:
    """Derive position sizing from training data."""
    n = len(prices)
    split = int(n * TRAIN_FRAC)
    train_prices = prices.iloc[:split]
    mi_train = compute_mi(train_prices)
    thr = compute_thresholds(mi_train)
    binary = (mi_train.values > 0).astype(int)
    from itertools import groupby
    rise_runs, fall_runs = [], []
    for key, grp in groupby(binary.tolist()):
        length = sum(1 for _ in grp)
        (rise_runs if key == 1 else fall_runs).append(length)
    u_n = compute_un(rise_runs, fall_runs)
    return full_position_sizing(thr["M_u05"], thr["M_d01"], u_n, pro, fee_pct)


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    gold, btc = get_prices(cache_dir=DATA_DIR)

    histories = {}
    finals = {}
    summary_rows = []

    for asset_name, prices in [("Gold", gold), ("Bitcoin", btc)]:
        print(f"\n=== {asset_name} ===", flush=True)
        cfg = CONFIGS[asset_name]

        # ── Train model (or load if cached) ───────────────────────────────
        model_path = os.path.join(MODELS_DIR, f"bilstm_attn_{asset_name.lower()}.keras")
        if os.path.exists(model_path):
            print(f"  Loading cached model from {model_path}", flush=True)
            model = tf.keras.models.load_model(
                model_path,
                custom_objects={"DotProductAttention": DotProductAttention},
            )
        else:
            print("  Training BiLSTM-Attention model …", flush=True)
            model = train_bilstm_model(prices)
            model.save(model_path)
            print(f"  Model saved to {model_path}")

        # ── Position sizing ────────────────────────────────────────────────
        pos_params = compute_pos_params(prices, cfg["fee_pct"], cfg["pro"])
        s_norm = pos_params["s_norm"]
        u_n = int(len(s_norm))
        print(
            f"  u_n={u_n}, a={pos_params['a']:.4f}, beta={pos_params['beta']:.4f}",
            flush=True,
        )

        # ── M_i series (full history for VaR) ─────────────────────────────
        mi_all = compute_mi(prices)

        # ── Simulation ────────────────────────────────────────────────────
        print("  Running simulation …", flush=True)
        history, final_value = simulate_trading(
            prices=prices,
            model=model,
            mi_all=mi_all,
            s_norm=s_norm,
            u_n=u_n,
            fee_pct=cfg["fee_pct"],
            pro=cfg["pro"],
            train_frac=TRAIN_FRAC,
        )
        histories[asset_name] = history
        finals[asset_name] = final_value

        print(
            f"  Final portfolio value: ${final_value:,.2f}  "
            f"(initial ${cfg['pro']:.0f})",
            flush=True,
        )

        # Save per-asset simulation history
        csv_path = os.path.join(
            RESULTS_DIR, f"exp4_simulation_{asset_name.lower()}.csv"
        )
        history.to_csv(csv_path, index=False)
        print(f"  Saved {csv_path}")

        total_ret = (final_value - cfg["pro"]) / cfg["pro"] * 100
        summary_rows.append(
            {
                "Asset": asset_name,
                "Initial_Capital": cfg["pro"],
                "Final_Value": round(final_value, 2),
                "Total_Return_Pct": round(total_ret, 2),
                "Fee_Pct": cfg["fee_pct"] * 100,
            }
        )

    # ── Figures 9 & 10 ────────────────────────────────────────────────────
    fig_path = os.path.join(RESULTS_DIR, "fig9_10_portfolio_value.png")
    plot_portfolio_value(
        histories["Gold"],
        histories["Bitcoin"],
        finals["Gold"],
        finals["Bitcoin"],
        save_path=fig_path,
    )
    print(f"\nSaved {fig_path}")

    # ── Summary ───────────────────────────────────────────────────────────
    df = pd.DataFrame(summary_rows)
    df["Total"] = df["Final_Value"].sum()
    csv_path = os.path.join(RESULTS_DIR, "exp4_summary.csv")
    df.to_csv(csv_path, index=False)
    print("\n" + df.to_string(index=False))
    print(f"\nTotal portfolio value: ${df['Final_Value'].sum():,.2f}")
    print(f"(Target: ~$216,133)")


if __name__ == "__main__":
    main()
