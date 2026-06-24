"""
Price prediction models for Gold and Bitcoin.

Models implemented:
  1. BiLSTM-Attention (dot-product attention)  — Custom, paper architecture
  2. BiLSTM (no attention)
  3. LSTM
  4. GRU
  5. Holt-Winters (statsmodels ExponentialSmoothing)
  6. ARIMA (pmdarima auto_arima)
  7. HMM (hmmlearn GaussianHMM, n_components selected by BIC)
  8. XGBoost (XGBRegressor)

All deep-learning models share the same hyperparameter configuration (paper Step 4-7).
"""

import os
import random

import numpy as np
import pandas as pd
import tensorflow as tf
from itertools import groupby

os.environ["PYTHONHASHSEED"] = "42"
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# Window construction helpers
# ─────────────────────────────────────────────────────────────────────────────

WINDOW_SIZE = 100


def create_windows(
    prices: np.ndarray | pd.Series,
    window: int = WINDOW_SIZE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create overlapping sliding windows with per-window min-max normalisation.

    Parameters
    ----------
    prices : array-like
        1-D price series.
    window : int
        Number of time steps per input window (default 100).

    Returns
    -------
    X : np.ndarray, shape (N-window, window)
        Normalised input windows.
    y : np.ndarray, shape (N-window,)
        Normalised target values (price at day window+1, normalised by its window).
    w_min : np.ndarray, shape (N-window,)
        Per-window minimum (for inverse transform).
    w_max : np.ndarray, shape (N-window,)
        Per-window maximum (for inverse transform).
    """
    p = np.asarray(prices, dtype=float)
    X_list, y_list, mins, maxs = [], [], [], []
    for i in range(len(p) - window):
        w = p[i : i + window]
        mn = w.min()
        mx = w.max() + 1e-8
        X_list.append((w - mn) / (mx - mn))
        y_list.append((p[i + window] - mn) / (mx - mn))
        mins.append(mn)
        maxs.append(mx)
    return (
        np.array(X_list, dtype=float),
        np.array(y_list, dtype=float),
        np.array(mins, dtype=float),
        np.array(maxs, dtype=float),
    )


def inverse_transform(
    y_norm: np.ndarray, w_min: np.ndarray, w_max: np.ndarray
) -> np.ndarray:
    """Inverse min-max transform: y_actual = y_norm * (w_max - w_min) + w_min."""
    return y_norm * (w_max - w_min) + w_min


def train_test_split_windows(
    X: np.ndarray,
    y: np.ndarray,
    w_min: np.ndarray,
    w_max: np.ndarray,
    train_frac: float = 0.70,
    shuffle_train: bool = True,
    seed: int = 42,
) -> tuple:
    """
    Temporal 70/30 split; optionally shuffles the training subset.

    Returns
    -------
    X_train, X_test, y_train, y_test, wmin_train, wmin_test, wmax_train, wmax_test
    """
    n = len(X)
    split_idx = int(n * train_frac)
    X_tr, X_te = X[:split_idx], X[split_idx:]
    y_tr, y_te = y[:split_idx], y[split_idx:]
    wmin_tr, wmin_te = w_min[:split_idx], w_min[split_idx:]
    wmax_tr, wmax_te = w_max[:split_idx], w_max[split_idx:]

    if shuffle_train:
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(X_tr))
        X_tr, y_tr, wmin_tr, wmax_tr = X_tr[idx], y_tr[idx], wmin_tr[idx], wmax_tr[idx]

    return X_tr, X_te, y_tr, y_te, wmin_tr, wmin_te, wmax_tr, wmax_te


# ─────────────────────────────────────────────────────────────────────────────
# Custom Keras Attention Layer
# Custom — Context7 found no library equivalent (paper dot-product attention)
# ─────────────────────────────────────────────────────────────────────────────


class DotProductAttention(tf.keras.layers.Layer):
    """
    Dot-product attention layer (paper Step 4).

    For each time step i, score S(x_i, q) = x_i^T * q where q is a learned
    weight vector. Softmax over all scores gives attention weights; the context
    vector is the weighted sum of all hidden states.
    """

    def build(self, input_shape: tf.TensorShape) -> None:
        self.W = self.add_weight(
            name="attention_weight",
            shape=(input_shape[-1], 1),
            initializer="glorot_uniform",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        # x: (batch, timesteps, features)
        scores = tf.squeeze(tf.matmul(x, self.W), axis=-1)  # (batch, timesteps)
        weights = tf.nn.softmax(scores, axis=-1)              # (batch, timesteps)
        weights = tf.expand_dims(weights, -1)                 # (batch, timesteps, 1)
        context = tf.reduce_sum(x * weights, axis=1)          # (batch, features)
        return context

    def get_config(self) -> dict:
        return super().get_config()


# ─────────────────────────────────────────────────────────────────────────────
# Deep-learning model builders (paper Steps 4–7)
# ─────────────────────────────────────────────────────────────────────────────


def _compile(model: tf.keras.Model) -> tf.keras.Model:
    """Compile with MAE loss and RMSprop optimizer (paper hyperparameters)."""
    model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001), loss="mae")
    return model


def build_bilstm_attention(seq_len: int = WINDOW_SIZE) -> tf.keras.Model:
    """
    BiLSTM with Dot-Product Attention (paper Step 4).

    Architecture:
        Input(seq_len, 1)
        → BiLSTM(64 units/dir, return_sequences=True)
        → Dropout(0.20)
        → BiLSTM(64 units/dir, return_sequences=True)
        → Dropout(0.20)
        → DotProductAttention
        → Dense(1)
    """
    inp = tf.keras.Input(shape=(seq_len, 1))
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(64, return_sequences=True)
    )(inp)
    x = tf.keras.layers.Dropout(0.20)(x)
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(64, return_sequences=True)
    )(x)
    x = tf.keras.layers.Dropout(0.20)(x)
    x = DotProductAttention()(x)
    out = tf.keras.layers.Dense(1)(x)
    return _compile(tf.keras.Model(inp, out, name="BiLSTM_Attention"))


def build_bilstm(seq_len: int = WINDOW_SIZE) -> tf.keras.Model:
    """
    Standard BiLSTM without attention (paper Step 5).
    GlobalAveragePooling1D replaces the attention layer.
    """
    inp = tf.keras.Input(shape=(seq_len, 1))
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(64, return_sequences=True)
    )(inp)
    x = tf.keras.layers.Dropout(0.20)(x)
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(64, return_sequences=True)
    )(x)
    x = tf.keras.layers.Dropout(0.20)(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    out = tf.keras.layers.Dense(1)(x)
    return _compile(tf.keras.Model(inp, out, name="BiLSTM"))


def build_lstm(seq_len: int = WINDOW_SIZE) -> tf.keras.Model:
    """
    Standard LSTM (paper Step 6).
    First layer: return_sequences=True; second: return_sequences=False.
    """
    inp = tf.keras.Input(shape=(seq_len, 1))
    x = tf.keras.layers.LSTM(64, return_sequences=True)(inp)
    x = tf.keras.layers.Dropout(0.20)(x)
    x = tf.keras.layers.LSTM(64, return_sequences=False)(x)
    x = tf.keras.layers.Dropout(0.20)(x)
    out = tf.keras.layers.Dense(1)(x)
    return _compile(tf.keras.Model(inp, out, name="LSTM"))


def build_gru(seq_len: int = WINDOW_SIZE) -> tf.keras.Model:
    """
    Standard GRU (paper Step 7).
    First layer: return_sequences=True; second: return_sequences=False.
    """
    inp = tf.keras.Input(shape=(seq_len, 1))
    x = tf.keras.layers.GRU(64, return_sequences=True)(inp)
    x = tf.keras.layers.Dropout(0.20)(x)
    x = tf.keras.layers.GRU(64, return_sequences=False)(x)
    x = tf.keras.layers.Dropout(0.20)(x)
    out = tf.keras.layers.Dense(1)(x)
    return _compile(tf.keras.Model(inp, out, name="GRU"))


def _early_stopping() -> tf.keras.callbacks.EarlyStopping:
    return tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )


def fit_dl_model(
    model: tf.keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    seq_len: int = WINDOW_SIZE,
    batch_size: int = 16,
    epochs: int = 200,
    val_split: float = 0.10,
) -> tf.keras.callbacks.History:
    """
    Train a deep-learning model with the paper's hyperparameters.

    Parameters
    ----------
    model : tf.keras.Model
        Pre-compiled Keras model (BiLSTM-Attn, BiLSTM, LSTM, or GRU).
    X_train : np.ndarray, shape (n, seq_len)
        Normalised input windows.
    y_train : np.ndarray, shape (n,)
        Normalised targets.

    Returns
    -------
    Training history object.
    """
    X_3d = X_train.reshape(-1, seq_len, 1)
    return model.fit(
        X_3d,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=val_split,
        callbacks=[_early_stopping()],
        verbose=0,
    )


def predict_dl(
    model: tf.keras.Model,
    X_test: np.ndarray,
    wmin_test: np.ndarray,
    wmax_test: np.ndarray,
    seq_len: int = WINDOW_SIZE,
) -> np.ndarray:
    """
    Generate predictions and inverse-transform to original price scale.

    Returns
    -------
    np.ndarray of predicted prices in original scale.
    """
    X_3d = X_test.reshape(-1, seq_len, 1)
    y_norm = model.predict(X_3d, verbose=0).flatten()
    return inverse_transform(y_norm, wmin_test, wmax_test)


# ─────────────────────────────────────────────────────────────────────────────
# Statistical / ML baseline models
# ─────────────────────────────────────────────────────────────────────────────


def fit_predict_holt_winters(
    train_prices: pd.Series,
    test_prices: pd.Series,
) -> np.ndarray:
    """
    Rolling one-step-ahead Holt-Winters forecast (paper Step 8).

    Uses statsmodels ExponentialSmoothing with trend='add', seasonal='add',
    seasonal_periods=7. Falls back to seasonal='mul' then no seasonal if needed.

    # Using statsmodels ExponentialSmoothing — Context7 confirmed
    """
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    all_prices = pd.concat([train_prices, test_prices])
    preds = []
    n_train = len(train_prices)

    for t in range(len(test_prices)):
        history = all_prices.iloc[: n_train + t].values
        fitted_value = None
        for seasonal in ["add", "mul", None]:
            try:
                kwargs = dict(trend="add", initialization_method="estimated")
                if seasonal is not None:
                    kwargs["seasonal"] = seasonal
                    kwargs["seasonal_periods"] = 7
                m = ExponentialSmoothing(history, **kwargs).fit(
                    optimized=True, use_brute=False
                )
                fitted_value = float(m.forecast(1)[0])
                break
            except Exception:
                continue
        if fitted_value is None:
            # Last-value fallback
            fitted_value = float(history[-1])
        preds.append(fitted_value)

    return np.array(preds)


def fit_predict_arima(
    train_prices: pd.Series,
    test_prices: pd.Series,
) -> np.ndarray:
    """
    Rolling one-step-ahead ARIMA forecast using pmdarima auto_arima (paper Step 9).

    Fits on training prices, then for each test step updates the model with the
    latest actual price before forecasting the next step.
    """
    import pmdarima as pm

    model = pm.auto_arima(
        train_prices.values,
        seasonal=False,
        max_p=5,
        max_q=5,
        information_criterion="aic",
        suppress_warnings=True,
        error_action="ignore",
    )

    preds = []
    history = train_prices.values.tolist()
    for actual in test_prices.values:
        forecast = model.predict(n_periods=1)[0]
        preds.append(float(forecast))
        model.update([actual])

    return np.array(preds)


def fit_predict_hmm(
    train_prices: pd.Series,
    test_prices: pd.Series,
) -> np.ndarray:
    """
    HMM-based price prediction (paper Step 10).

    Fits GaussianHMM on log-returns of training data; selects n_components in
    {2,3,4,5} by BIC. Maps Viterbi-decoded states to expected next-day returns
    and reconstructs predicted prices.

    # Using hmmlearn GaussianHMM — Context7 confirmed
    """
    from hmmlearn import hmm

    train_log_ret = np.diff(np.log(train_prices.values)).reshape(-1, 1)
    test_log_ret = np.diff(np.log(
        np.concatenate([[train_prices.values[-1]], test_prices.values])
    )).reshape(-1, 1)

    best_model = None
    best_bic = np.inf
    for n in [2, 3, 4, 5]:
        try:
            m = hmm.GaussianHMM(
                n_components=n,
                covariance_type="diag",
                n_iter=1000,
                random_state=42,
            )
            m.fit(train_log_ret)
            # BIC = -2*log_likelihood + n_params * log(n_samples)
            ll = m.score(train_log_ret)
            n_params = n * n + 2 * n  # transition + means + vars
            bic = -2 * ll + n_params * np.log(len(train_log_ret))
            if bic < best_bic:
                best_bic = bic
                best_model = m
        except Exception:
            continue

    if best_model is None:
        return np.full(len(test_prices), test_prices.values[0])

    # Decode test sequence
    try:
        states = best_model.predict(test_log_ret)
    except Exception:
        return np.full(len(test_prices), test_prices.values[0])

    # State → expected log-return (from emission means)
    state_means = best_model.means_.flatten()

    preds = []
    prev_price = train_prices.values[-1]
    for s in states:
        expected_log_ret = state_means[s]
        pred_price = prev_price * np.exp(expected_log_ret)
        preds.append(pred_price)
        prev_price = pred_price  # chain predictions

    return np.array(preds)


def fit_predict_xgboost(
    X_train_flat: np.ndarray,
    y_train_raw: np.ndarray,
    X_test_flat: np.ndarray,
    wmin_test: np.ndarray,
    wmax_test: np.ndarray,
) -> np.ndarray:
    """
    XGBoost regression model (paper Step 11).

    Uses flattened 100-day windows as features (100 features per sample).
    # Using xgboost XGBRegressor — standard library
    """
    import xgboost as xgb

    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        verbosity=0,
    )
    model.fit(X_train_flat, y_train_raw)
    y_norm_pred = model.predict(X_test_flat)
    return inverse_transform(y_norm_pred, wmin_test, wmax_test)
