"""
exp_1 — Multi-Model Price Prediction Comparison.

Trains and evaluates 8 models on Gold and Bitcoin daily closing prices:
  1. BiLSTM-Attention   5. Holt-Winters
  2. BiLSTM             6. ARIMA
  3. LSTM               7. HMM
  4. GRU                8. XGBoost

Outputs:
  - results/exp1_metrics_table.csv
  - results/fig6_gold_predictions.png
  - results/fig6_btc_predictions.png
"""

import os
import sys

import numpy as np
import pandas as pd
import tensorflow as tf

# Allow running from repo root or exp/ directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.download import get_prices
from src.models import (
    WINDOW_SIZE,
    DotProductAttention,
    build_bilstm_attention,
    build_bilstm,
    build_lstm,
    build_gru,
    create_windows,
    train_test_split_windows,
    fit_dl_model,
    predict_dl,
    fit_predict_holt_winters,
    fit_predict_arima,
    fit_predict_hmm,
    fit_predict_xgboost,
    inverse_transform,
)
from src.metrics import evaluate_all
from src.visualisation import plot_predictions

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
TRAIN_FRAC = 0.70


def run_deep_learning_models(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te: np.ndarray,
    wmin_te: np.ndarray,
    wmax_te: np.ndarray,
) -> dict[str, np.ndarray]:
    """Train the four DL architectures and return test-set predictions."""
    builders = {
        "BiLSTM-Attention": build_bilstm_attention,
        "BiLSTM": build_bilstm,
        "LSTM": build_lstm,
        "GRU": build_gru,
    }
    preds: dict[str, np.ndarray] = {}
    for name, builder in builders.items():
        print(f"    Training {name} …", flush=True)
        tf.keras.backend.clear_session()
        tf.random.set_seed(42)
        model = builder(WINDOW_SIZE)
        fit_dl_model(model, X_tr, y_tr)
        preds[name] = predict_dl(model, X_te, wmin_te, wmax_te)
    return preds


def run_baselines(
    train_prices: pd.Series,
    test_prices: pd.Series,
    X_tr_flat: np.ndarray,
    y_tr_norm: np.ndarray,
    X_te_flat: np.ndarray,
    wmin_te: np.ndarray,
    wmax_te: np.ndarray,
) -> dict[str, np.ndarray]:
    """Fit and predict all four baseline models."""
    preds: dict[str, np.ndarray] = {}

    print("    Training Holt-Winters …", flush=True)
    preds["Holt-Winters"] = fit_predict_holt_winters(train_prices, test_prices)

    print("    Training ARIMA …", flush=True)
    preds["ARIMA"] = fit_predict_arima(train_prices, test_prices)

    print("    Training HMM …", flush=True)
    preds["HMM"] = fit_predict_hmm(train_prices, test_prices)

    print("    Training XGBoost …", flush=True)
    preds["XGBoost"] = fit_predict_xgboost(
        X_tr_flat, y_tr_norm, X_te_flat, wmin_te, wmax_te
    )

    return preds


def run_asset(name: str, prices: pd.Series) -> tuple[dict, dict]:
    """
    Run all models for one asset and return (metrics_dict, predictions_dict).
    """
    print(f"\n  === {name} ===", flush=True)

    # Windows
    X, y, w_min, w_max = create_windows(prices.values, WINDOW_SIZE)
    X_tr, X_te, y_tr, y_te, wmin_tr, wmin_te, wmax_tr, wmax_te = (
        train_test_split_windows(X, y, w_min, w_max, TRAIN_FRAC, shuffle_train=True)
    )

    # Test-set actual prices (inverse-transform unshuffled y_te)
    y_true = inverse_transform(y_te, wmin_te, wmax_te)

    # Split dates for test set
    n_windows = len(X)
    split_w = int(n_windows * TRAIN_FRAC)
    # The test windows correspond to prices[split_w : split_w + len(y_te)]
    # Window i uses prices[i:i+100] as input and prices[i+100] as target
    # Test window 0 starts at split_w, target is at prices[split_w + 100]
    test_target_indices = range(split_w + WINDOW_SIZE, split_w + WINDOW_SIZE + len(y_te))
    test_dates = prices.index[list(test_target_indices)]

    # Price series split for baseline models
    # Baselines operate on raw prices; we split at split_w + WINDOW_SIZE (first test target day)
    split_price_idx = split_w + WINDOW_SIZE
    train_prices = prices.iloc[:split_price_idx]
    test_prices = prices.iloc[split_price_idx : split_price_idx + len(y_te)]

    # Deep-learning models
    all_preds = run_deep_learning_models(X_tr, y_tr, X_te, wmin_te, wmax_te)

    # Baseline models
    baseline_preds = run_baselines(
        train_prices,
        test_prices,
        X_tr,           # flat (n, 100) — pass as is for XGBoost
        y_tr,
        X_te,
        wmin_te,
        wmax_te,
    )
    all_preds.update(baseline_preds)

    # Align lengths (statistical models may return slightly different lengths)
    min_len = min(len(y_true), min(len(v) for v in all_preds.values()))
    y_true = y_true[:min_len]
    test_dates = test_dates[:min_len]
    for k in all_preds:
        all_preds[k] = all_preds[k][:min_len]

    # Compute metrics
    metrics: dict[str, dict] = {}
    for model_name, preds in all_preds.items():
        metrics[model_name] = evaluate_all(y_true, preds)
        m = metrics[model_name]
        print(
            f"    {model_name:20s}  RMSE={m['RMSE']:8.2f}  MAE={m['MAE']:8.2f}"
            f"  MAPE={m['MAPE']:.4f}%  R²={m['R2']:.4f}",
            flush=True,
        )

    return metrics, all_preds, y_true, test_dates


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    gold, btc = get_prices(cache_dir=DATA_DIR)

    all_metrics = {}

    for asset_name, prices in [("Gold", gold), ("Bitcoin", btc)]:
        metrics, preds, y_true, test_dates = run_asset(asset_name, prices)
        all_metrics[asset_name] = metrics

        # Figure 6 equivalent
        fig_path = os.path.join(
            RESULTS_DIR, f"fig6_{asset_name.lower()}_predictions.png"
        )
        plot_predictions(test_dates, y_true, preds, asset_name, save_path=fig_path)
        print(f"  Saved {fig_path}", flush=True)

    # Build combined metrics table (Table 1)
    rows = []
    model_order = [
        "BiLSTM-Attention", "BiLSTM", "LSTM", "GRU",
        "Holt-Winters", "ARIMA", "HMM", "XGBoost",
    ]
    for model in model_order:
        row = {"Model": model}
        for asset in ("Gold", "Bitcoin"):
            m = all_metrics[asset].get(model, {})
            for metric in ("RMSE", "MAE", "MAPE", "R2"):
                row[f"{asset}_{metric}"] = round(m.get(metric, float("nan")), 4)
        rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = os.path.join(RESULTS_DIR, "exp1_metrics_table.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nMetrics table saved to {csv_path}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
