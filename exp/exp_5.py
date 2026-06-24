"""
exp_5 — Transaction Fee Sensitivity Analysis.

Runs the complete trading simulation for gold across 6 fee levels and bitcoin
across 5 fee levels, keeping the trained model and position sizing fixed to
isolate the fee effect.

Fee ranges:
  Gold:    0.2%, 0.4%, 0.6%, 0.8%, 1.0%, 1.2%
  Bitcoin: 1.5%, 2.0%, 2.5%, 3.0%, 3.5%

Outputs:
  - results/fig11_gold_fee_sensitivity.png
  - results/fig12_btc_fee_sensitivity.png
  - results/exp5_fee_sensitivity.csv
"""

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
from src.visualisation import plot_fee_sensitivity

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
MODELS_DIR = os.path.join(DATA_DIR, "trained_models")

TRAIN_FRAC = 0.70

GOLD_FEES = [0.002, 0.004, 0.006, 0.008, 0.010, 0.012]
BTC_FEES = [0.015, 0.020, 0.025, 0.030, 0.035]
BASE_FEES = {"Gold": 0.010, "Bitcoin": 0.025}
PRO = 500.0


def load_or_train_model(prices: pd.Series, asset_name: str) -> tf.keras.Model:
    """Load cached model or train fresh."""
    model_path = os.path.join(MODELS_DIR, f"bilstm_attn_{asset_name.lower()}.keras")
    if os.path.exists(model_path):
        print(f"  Loading cached model from {model_path}", flush=True)
        return tf.keras.models.load_model(
            model_path,
            custom_objects={"DotProductAttention": DotProductAttention},
        )
    print(f"  Training BiLSTM-Attention model for {asset_name} …", flush=True)
    X, y, w_min, w_max = create_windows(prices.values, WINDOW_SIZE)
    X_tr, _, y_tr, *_ = train_test_split_windows(
        X, y, w_min, w_max, TRAIN_FRAC, shuffle_train=True
    )
    tf.keras.backend.clear_session()
    tf.random.set_seed(42)
    model = build_bilstm_attention(WINDOW_SIZE)
    fit_dl_model(model, X_tr, y_tr)
    os.makedirs(MODELS_DIR, exist_ok=True)
    model.save(model_path)
    return model


def compute_pos_params_base(prices: pd.Series, base_fee: float) -> tuple:
    """
    Compute position sizing parameters using the BASE fee level.
    Same parameters are reused across all fee sweeps (approach a — isolate fee effect).
    """
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
    pos = full_position_sizing(thr["M_u05"], thr["M_d01"], u_n, PRO, base_fee)
    return pos["s_norm"], u_n


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    gold, btc = get_prices(cache_dir=DATA_DIR)

    all_rows = []

    for asset_name, prices, fee_levels in [
        ("Gold", gold, GOLD_FEES),
        ("Bitcoin", btc, BTC_FEES),
    ]:
        print(f"\n=== {asset_name} fee sensitivity sweep ===", flush=True)

        # Train model ONCE with base fee (same model for all fee levels)
        model = load_or_train_model(prices, asset_name)

        # Compute position sizing ONCE at base fee (isolate fee effect on costs)
        base_fee = BASE_FEES[asset_name]
        s_norm, u_n = compute_pos_params_base(prices, base_fee)
        mi_all = compute_mi(prices)

        fee_vals, final_vals = [], []
        for fee in fee_levels:
            print(f"  fee={fee*100:.1f}% …", end=" ", flush=True)
            _, final_value = simulate_trading(
                prices=prices,
                model=model,
                mi_all=mi_all,
                s_norm=s_norm,
                u_n=u_n,
                fee_pct=fee,
                pro=PRO,
                train_frac=TRAIN_FRAC,
            )
            print(f"${final_value:,.2f}", flush=True)
            fee_vals.append(fee)
            final_vals.append(final_value)
            all_rows.append(
                {
                    "Asset": asset_name,
                    "Fee_Pct": round(fee * 100, 2),
                    "Final_Value": round(final_value, 2),
                    "Return_Pct": round((final_value - PRO) / PRO * 100, 2),
                }
            )

        # Figures 11 / 12
        fig_num = 11 if asset_name == "Gold" else 12
        cmap = "Blues" if asset_name == "Gold" else "Oranges"
        fig_path = os.path.join(
            RESULTS_DIR, f"fig{fig_num}_{asset_name.lower()}_fee_sensitivity.png"
        )
        plot_fee_sensitivity(fee_vals, final_vals, asset_name, cmap, save_path=fig_path)
        print(f"  Saved {fig_path}")

        # Correlation
        corr = np.corrcoef(fee_vals, final_vals)[0, 1]
        print(f"  Correlation(fee, final_value): {corr:.4f}")

    # Save combined CSV
    df = pd.DataFrame(all_rows)
    csv_path = os.path.join(RESULTS_DIR, "exp5_fee_sensitivity.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nFee sensitivity results saved to {csv_path}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
