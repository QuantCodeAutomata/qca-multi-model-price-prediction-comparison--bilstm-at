"""
exp_3 — Exponential Position Sizing Model.

Derives p_1..p_{u_n} from recursive breakeven equations, fits
s_i = a * exp(beta * i) via log-linear OLS, and computes dollar
allocations y_i for Gold and Bitcoin.

Outputs:
  - results/fig_pos_sizing_gold.png
  - results/fig_pos_sizing_btc.png
  - results/exp3_position_sizing.csv
  - data/position_sizing_params.json  (inputs for exp_4 & exp_5)
"""

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.download import get_prices
from src.patterns import compute_mi, compute_thresholds, count_runs, compute_un
from src.position_sizing import full_position_sizing
from src.visualisation import plot_position_sizing

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

CONFIGS = {
    "Gold": {"fee_pct": 0.01, "pro": 500.0},
    "Bitcoin": {"fee_pct": 0.025, "pro": 500.0},
}
TRAIN_FRAC = 0.70


def compute_training_params(prices: pd.Series) -> tuple[float, float, int]:
    """
    Compute M_u05, M_d01, and u_n from the training portion of prices.

    Returns
    -------
    M_u05 : float  (median rise)
    M_d01 : float  (10th-pct decline, negative)
    u_n   : int    (90th-pct run length, ≥ 2)
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
    u_n = compute_un(rise_runs, fall_runs, percentile=90.0)
    return float(thr["M_u05"]), float(thr["M_d01"]), u_n


def sensitivity_sweep(
    M_u05: float,
    M_d01: float,
    u_n: int,
    pro: float,
    fee_levels: list[float],
) -> pd.DataFrame:
    """Compute position sizing for a range of fee levels."""
    rows = []
    for fee in fee_levels:
        res = full_position_sizing(M_u05, M_d01, u_n, pro, fee_pct=fee)
        rows.append(
            {
                "fee_pct": fee,
                "a": res["a"],
                "beta": res["beta"],
                "r2": res["r2"],
                "sum_y_alloc": float(res["y_alloc"].sum()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    gold, btc = get_prices(cache_dir=DATA_DIR)

    params_out = {}
    summary_rows = []

    for asset_name, prices in [("Gold", gold), ("Bitcoin", btc)]:
        print(f"\n=== {asset_name} ===", flush=True)
        cfg = CONFIGS[asset_name]

        M_u05, M_d01, u_n = compute_training_params(prices)
        print(f"  M_u05={M_u05:.6f}  M_d01={M_d01:.6f}  u_n={u_n}")

        res = full_position_sizing(M_u05, M_d01, u_n, cfg["pro"], cfg["fee_pct"])
        a, beta, r2 = res["a"], res["beta"], res["r2"]
        y_alloc = res["y_alloc"]
        p_seq = res["p_sequence"]
        s_fitted = res["s_fitted"]

        print(f"  Exponential fit: a={a:.4f}, beta={beta:.4f}, R²={r2:.4f}")
        print(f"  Allocations sum: {y_alloc.sum():.4f}  (PRO={cfg['pro']})")
        print(f"  y_alloc: {np.round(y_alloc, 4)}")

        assert abs(y_alloc.sum() - cfg["pro"]) < 1.0, "Allocation sum mismatch"

        # Visualise
        fig_path = os.path.join(
            RESULTS_DIR, f"fig_pos_sizing_{asset_name.lower()}.png"
        )
        plot_position_sizing(
            p_seq, s_fitted, y_alloc, a, beta,
            asset_name=asset_name, pro=cfg["pro"], save_path=fig_path,
        )
        print(f"  Saved {fig_path}")

        # Fee sensitivity
        gold_fees = [0.002, 0.004, 0.006, 0.008, 0.010, 0.012]
        btc_fees = [0.015, 0.020, 0.025, 0.030, 0.035]
        sweep_fees = gold_fees if asset_name == "Gold" else btc_fees
        sweep_df = sensitivity_sweep(M_u05, M_d01, u_n, cfg["pro"], sweep_fees)
        print(f"  Fee sensitivity:\n{sweep_df.to_string(index=False)}")

        # Collect for output
        params_out[asset_name] = {
            "M_u05": float(M_u05),
            "M_d01": float(M_d01),
            "u_n": int(u_n),
            "a": float(a),
            "beta": float(beta),
            "r2": float(r2),
            "y_alloc": y_alloc.tolist(),
            "s_norm": res["s_norm"].tolist(),
            "fee_pct": cfg["fee_pct"],
            "pro": cfg["pro"],
        }
        summary_rows.append(
            {
                "Asset": asset_name,
                "M_u05": round(M_u05, 6),
                "M_d01": round(M_d01, 6),
                "u_n": u_n,
                "a": round(a, 4),
                "beta": round(beta, 4),
                "R2_fit": round(r2, 4),
                "sum_y_alloc": round(float(y_alloc.sum()), 4),
            }
        )

    # Save summary CSV
    df = pd.DataFrame(summary_rows)
    csv_path = os.path.join(RESULTS_DIR, "exp3_position_sizing.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSummary saved to {csv_path}")
    print(df.to_string(index=False))

    # Save JSON for downstream experiments
    ps_path = os.path.join(DATA_DIR, "position_sizing_params.json")
    with open(ps_path, "w") as f:
        json.dump(params_out, f, indent=2)
    print(f"Position sizing parameters saved to {ps_path}")


if __name__ == "__main__":
    main()
