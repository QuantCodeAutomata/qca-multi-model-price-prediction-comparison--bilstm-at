"""
exp_2 — Apriori-Based Consecutive Rise/Fall Pattern Analysis.

Computes M_i series, statistical thresholds (M_u0.9, M_d0.1, M_u0.5),
run-length distributions, and u_n for Gold and Bitcoin.

Outputs:
  - results/fig7_mi_series.png
  - results/fig8_rise_fall_curves.png
  - results/exp2_run_table.csv
  - results/exp2_thresholds.csv
  - data/pattern_params.json  (inputs for exp_3 & exp_4)
"""

import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.download import get_prices
from src.patterns import analyse_asset
from src.visualisation import plot_mi_series, plot_rise_fall_curves

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    gold, btc = get_prices(cache_dir=DATA_DIR)

    results = {}
    for asset_name, prices in [("Gold", gold), ("Bitcoin", btc)]:
        print(f"\n=== {asset_name} ===", flush=True)
        res = analyse_asset(prices, train_frac=0.70)
        results[asset_name] = res

        thr_full = res["thresholds_full"]
        thr_train = res["thresholds_train"]
        print(f"  Full-history thresholds: {thr_full}")
        print(f"  Training-only thresholds: {thr_train}")
        print(f"  u_n (full): {res['un_full']},  u_n (train): {res['un_train']}")

        # Run table summary
        rt = res["run_table_full"]
        print(f"\n  Run table (first 10 rows):\n{rt.head(10).to_string(index=False)}")

    # ── Figure 7: M_i time series ────────────────────────────────────────────
    fig7_path = os.path.join(RESULTS_DIR, "fig7_mi_series.png")
    plot_mi_series(
        results["Gold"]["mi_full"],
        results["Bitcoin"]["mi_full"],
        save_path=fig7_path,
    )
    print(f"\nSaved {fig7_path}")

    # ── Figure 8: Rise / fall curves ─────────────────────────────────────────
    fig8_path = os.path.join(RESULTS_DIR, "fig8_rise_fall_curves.png")
    plot_rise_fall_curves(
        results["Gold"]["mi_full"],
        results["Bitcoin"]["mi_full"],
        results["Gold"]["thresholds_full"],
        results["Bitcoin"]["thresholds_full"],
        save_path=fig8_path,
    )
    print(f"Saved {fig8_path}")

    # ── Combined run table ───────────────────────────────────────────────────
    rt_gold = results["Gold"]["run_table_full"].rename(
        columns={
            "m_j_exact": "m_j_gold_exact",
            "n_j_exact": "n_j_gold_exact",
            "m_j_apriori": "m_j_gold_apriori",
            "n_j_apriori": "n_j_gold_apriori",
        }
    )
    rt_btc = results["Bitcoin"]["run_table_full"].rename(
        columns={
            "m_j_exact": "m_j_btc_exact",
            "n_j_exact": "n_j_btc_exact",
            "m_j_apriori": "m_j_btc_apriori",
            "n_j_apriori": "n_j_btc_apriori",
        }
    )
    run_table = rt_gold.merge(rt_btc, on="j", how="outer").fillna(0).astype(int)
    csv_path = os.path.join(RESULTS_DIR, "exp2_run_table.csv")
    run_table.to_csv(csv_path, index=False)
    print(f"Saved {csv_path}")

    # ── Thresholds summary ───────────────────────────────────────────────────
    thr_rows = []
    for asset_name in ("Gold", "Bitcoin"):
        for scope in ("full", "train"):
            thr = results[asset_name][f"thresholds_{scope}"]
            un = results[asset_name][f"un_{scope}"]
            thr_rows.append(
                {
                    "Asset": asset_name,
                    "Scope": scope,
                    "M_u09": round(thr["M_u09"], 6),
                    "M_d01": round(thr["M_d01"], 6),
                    "M_u05": round(thr["M_u05"], 6),
                    "u_n": un,
                }
            )
    thr_df = pd.DataFrame(thr_rows)
    thr_csv = os.path.join(RESULTS_DIR, "exp2_thresholds.csv")
    thr_df.to_csv(thr_csv, index=False)
    print(f"Saved {thr_csv}")
    print(thr_df.to_string(index=False))

    # ── Save pattern parameters for exp_3 & exp_4 ───────────────────────────
    params = {}
    for asset_name in ("Gold", "Bitcoin"):
        thr_t = results[asset_name]["thresholds_train"]
        params[asset_name] = {
            "M_u09": thr_t["M_u09"],
            "M_d01": thr_t["M_d01"],
            "M_u05": thr_t["M_u05"],
            "u_n": results[asset_name]["un_train"],
        }
    params_path = os.path.join(DATA_DIR, "pattern_params.json")
    with open(params_path, "w") as f:
        json.dump(params, f, indent=2)
    print(f"Pattern parameters saved to {params_path}")
    print(json.dumps(params, indent=2))


if __name__ == "__main__":
    main()
