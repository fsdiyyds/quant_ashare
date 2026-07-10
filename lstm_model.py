"""
LSTM 深度学习预测模块

使用价格、成交量、换手率等时序特征，预测未来 N 日收益率，
在 B1 候选池中排序选出最值得买入的股票。

依赖: tensorflow (Python 3.8+ 建议 tensorflow>=2.10)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import tensorflow as tf
    from tensorflow.keras import layers, models, callbacks
    HAS_TF = True
except ImportError:
    HAS_TF = False

# 每根 K 线的序列特征
SEQ_FEATURES = [
    "close_norm", "volume_norm", "turnover_norm",
    "pct_norm", "high_low_pos", "kdj_j_norm",
    "vol_ratio5_norm", "macd_dif_norm",
]


@dataclass
class LSTMResult:
    model: object
    seq_len: int
    feature_names: List[str]
    train_loss: float
    valid_loss: float
    valid_ic: float  # 预测与标签秩相关


def _norm_series(s: pd.Series) -> pd.Series:
    mu, std = s.mean(), s.std()
    if std == 0 or np.isnan(std):
        return s * 0
    return (s - mu) / std


def build_lstm_sequences(
    df: pd.DataFrame,
    seq_len: int = 30,
    forward_days: int = 5,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    为单只股票构建 LSTM 训练样本。
    返回 X (n_samples, seq_len, n_features), y (n_samples,), codes
    """
    from b1_selector import compute_b1_indicators

    g = compute_b1_indicators(df.sort_values("date"))
    if len(g) < seq_len + forward_days + 10:
        return np.array([]), np.array([]), []

    c = g["close"]
    g["close_norm"] = _norm_series(c.pct_change().fillna(0))
    g["volume_norm"] = _norm_series(np.log1p(g["volume"].fillna(0)))
    g["turnover_norm"] = _norm_series(g["turnover"].fillna(0))
    g["pct_norm"] = _norm_series(g["pct_chg"].fillna(0) / 100)
    hl = (g["high"] - g["low"]).replace(0, np.nan)
    g["high_low_pos"] = ((g["close"] - g["low"]) / hl).fillna(0.5)
    g["kdj_j_norm"] = _norm_series(g["kdj_j"].fillna(50) / 100)
    g["vol_ratio5_norm"] = _norm_series(g["vol_ratio5"].fillna(1))
    g["macd_dif_norm"] = _norm_series(g["macd_dif"].fillna(0))

    g["label"] = c.shift(-forward_days) / c - 1
    g = g.dropna(subset=SEQ_FEATURES + ["label"])

    if len(g) < seq_len + 1:
        return np.array([]), np.array([]), []

    feat = g[SEQ_FEATURES].values.astype(np.float32)
    labels = g["label"].values.astype(np.float32)

    X_list, y_list = [], []
    for i in range(seq_len, len(feat)):
        X_list.append(feat[i - seq_len:i])
        y_list.append(labels[i - 1])

    if not X_list:
        return np.array([]), np.array([]), []

    return np.array(X_list), np.array(y_list), SEQ_FEATURES


def _configure_tf_memory():
    """限制 TensorFlow 显存/内存占用，降低 Cloud OOM 概率。"""
    if not HAS_TF:
        return
    try:
        gpus = tf.config.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass
    # 限制 CPU 线程，减少峰值内存
    try:
        tf.config.threading.set_intra_op_parallelism_threads(2)
        tf.config.threading.set_inter_op_parallelism_threads(2)
    except Exception:
        pass


def build_panel_sequences(
    raw: pd.DataFrame,
    codes: Optional[List[str]] = None,
    seq_len: int = 30,
    forward_days: int = 5,
    max_codes: Optional[int] = None,
    max_samples: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """合并多只股票序列为训练集。

    max_codes / max_samples 用于 Streamlit Cloud 等低内存环境，避免 OOM。
    """
    X_all, y_all = [], []
    target_codes = list(codes or raw["code"].unique().tolist())
    if max_codes is not None and len(target_codes) > max_codes:
        # 均匀抽样，避免只取前缀代码
        idx = np.linspace(0, len(target_codes) - 1, max_codes).astype(int)
        target_codes = [target_codes[i] for i in idx]
        print(f"  LSTM序列: 股票数限制为 {max_codes}", flush=True)

    from progress_utils import ProgressBar
    bar = ProgressBar(len(target_codes), desc="  LSTM序列", unit="股")
    n_samples = 0
    for code in target_codes:
        if max_samples is not None and n_samples >= max_samples:
            break
        sub = raw[raw["code"] == code]
        X, y, _ = build_lstm_sequences(sub, seq_len, forward_days)
        if len(X) > 0:
            if max_samples is not None and n_samples + len(X) > max_samples:
                keep = max(0, max_samples - n_samples)
                if keep <= 0:
                    break
                X, y = X[-keep:], y[-keep:]
            X_all.append(X.astype(np.float32, copy=False))
            y_all.append(y.astype(np.float32, copy=False))
            n_samples += len(X)
        bar.update(1, postfix=str(code))
    bar.close(f"样本 {n_samples}")

    if not X_all:
        return np.array([]), np.array([])

    return np.concatenate(X_all), np.concatenate(y_all)


def _build_lstm_model(seq_len: int, n_features: int, units: int = 64, light: bool = False):
    if not HAS_TF:
        raise ImportError("请安装 tensorflow: pip install tensorflow")

    _configure_tf_memory()
    if light:
        units = min(units, 32)
        model = models.Sequential([
            layers.Input(shape=(seq_len, n_features)),
            layers.LSTM(units),
            layers.Dropout(0.2),
            layers.Dense(16, activation="relu"),
            layers.Dense(1),
        ])
    else:
        model = models.Sequential([
            layers.Input(shape=(seq_len, n_features)),
            layers.LSTM(units, return_sequences=True),
            layers.Dropout(0.2),
            layers.LSTM(units // 2),
            layers.Dropout(0.2),
            layers.Dense(32, activation="relu"),
            layers.Dense(1),
        ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="mse")
    return model


def train_lstm(
    X: np.ndarray,
    y: np.ndarray,
    seq_len: int = 30,
    epochs: int = 30,
    batch_size: int = 64,
    train_ratio: float = 0.8,
    min_samples: int = 100,
    light: bool = False,
) -> LSTMResult:
    if not HAS_TF:
        raise ImportError(
            "未安装 tensorflow。Streamlit Cloud 请选 Python 3.11/3.12；"
            "本地执行: pip install 'tensorflow>=2.15,<2.20'"
        )
    if len(X) < min_samples:
        raise ValueError(f"LSTM 训练样本不足: {len(X)}，至少需要 {min_samples} 条")

    n = len(X)
    cut = int(n * train_ratio)
    # 保证验证集至少有几条
    if n - cut < 5:
        cut = max(n - 5, int(n * 0.7))
    X_train, X_valid = X[:cut], X[cut:]
    y_train, y_valid = y[:cut], y[cut:]
    if len(X_train) < 10 or len(X_valid) < 3:
        raise ValueError(f"LSTM 划分后样本过少: train={len(X_train)} valid={len(X_valid)}")

    model = _build_lstm_model(seq_len, X.shape[2], light=light)
    es = callbacks.EarlyStopping(patience=5, restore_best_weights=True, monitor="val_loss")

    print(f"  LSTM train: {len(X_train)} | valid: {len(X_valid)}", flush=True)
    cb = [es]

    class _EpochPrint(callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            print(
                f"    epoch {epoch + 1}/{epochs}: loss={logs.get('loss', 0):.5f} "
                f"val_loss={logs.get('val_loss', 0):.5f}",
                flush=True,
            )

    cb.append(_EpochPrint())

    hist = model.fit(
        X_train, y_train,
        validation_data=(X_valid, y_valid),
        epochs=epochs,
        batch_size=min(batch_size, max(8, len(X_train))),
        callbacks=cb,
        verbose=0,
    )

    train_loss = float(hist.history["loss"][-1])
    valid_loss = float(hist.history["val_loss"][-1])

    pred = model.predict(X_valid, verbose=0).flatten()
    if len(pred) > 10:
        ic = float(np.corrcoef(pred, y_valid)[0, 1])
    else:
        ic = 0.0

    return LSTMResult(
        model=model,
        seq_len=seq_len,
        feature_names=SEQ_FEATURES,
        train_loss=train_loss,
        valid_loss=valid_loss,
        valid_ic=ic,
    )


def predict_latest_lstm(
    model,
    raw: pd.DataFrame,
    codes: List[str],
    seq_len: int = 30,
) -> pd.DataFrame:
    """对候选股票用最近 seq_len 日序列预测未来收益。"""
    rows = []
    from progress_utils import ProgressBar
    bar = ProgressBar(len(codes), desc="  LSTM预测", unit="股")
    for code in codes:
        sub = raw[raw["code"] == code]
        X, _, _ = build_lstm_sequences(sub, seq_len=seq_len, forward_days=5)
        if len(X) == 0:
            bar.update(1, postfix=str(code))
            continue
        pred = float(model.predict(X[-1:], verbose=0)[0, 0])
        latest = sub.sort_values("date").iloc[-1]
        rows.append({
            "code": code,
            "date": latest["date"],
            "close": latest["close"],
            "turnover": latest.get("turnover", 0),
            "lstm_pred_return": pred,
        })
        bar.update(1, postfix=str(code))
    bar.close()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["lstm_score"] = df["lstm_pred_return"].rank(pct=True)
    return df.sort_values("lstm_pred_return", ascending=False)


def _build_lstm_feature_frame(df: pd.DataFrame, forward_days: int = 5) -> pd.DataFrame:
    """Build per-row LSTM features and forward return label."""
    from b1_selector import compute_b1_indicators

    g = compute_b1_indicators(df.sort_values("date"))
    if len(g) < 30:
        return pd.DataFrame()

    c = g["close"]
    g = g.copy()
    g["close_norm"] = _norm_series(c.pct_change().fillna(0))
    g["volume_norm"] = _norm_series(np.log1p(g["volume"].fillna(0)))
    g["turnover_norm"] = _norm_series(g["turnover"].fillna(0))
    g["pct_norm"] = _norm_series(g["pct_chg"].fillna(0) / 100)
    hl = (g["high"] - g["low"]).replace(0, np.nan)
    g["high_low_pos"] = ((g["close"] - g["low"]) / hl).fillna(0.5)
    g["kdj_j_norm"] = _norm_series(g["kdj_j"].fillna(50) / 100)
    g["vol_ratio5_norm"] = _norm_series(g["vol_ratio5"].fillna(1))
    g["macd_dif_norm"] = _norm_series(g["macd_dif"].fillna(0))
    g["label"] = c.shift(-forward_days) / c - 1
    return g.dropna(subset=SEQ_FEATURES + ["label"]).reset_index(drop=True)


def predict_lstm_historical(
    model,
    df: pd.DataFrame,
    seq_len: int = 30,
    forward_days: int = 5,
    max_points: int = 120,
) -> pd.DataFrame:
    """
    Walk-forward LSTM backtest on one stock.
    Returns signal_date, target_date, pred/actual price and return, error metrics.
    """
    g = _build_lstm_feature_frame(df, forward_days=forward_days)
    if len(g) < seq_len + forward_days + 5:
        return pd.DataFrame()

    feat = g[SEQ_FEATURES].values.astype(np.float32)
    rows = []
    start_i = max(seq_len, len(g) - max_points - forward_days)

    for i in range(start_i, len(g)):
        seq_end = i - 1
        if seq_end < seq_len - 1:
            continue
        X = feat[i - seq_len:i].reshape(1, seq_len, -1)
        pred_ret = float(model.predict(X, verbose=0)[0, 0])
        actual_ret = float(g.iloc[seq_end]["label"])
        signal_row = g.iloc[seq_end]
        signal_close = float(signal_row["close"])
        actual_close = signal_close * (1 + actual_ret)
        pred_close = signal_close * (1 + pred_ret)
        if seq_end + forward_days < len(g):
            target_date = g.iloc[seq_end + forward_days]["date"]
        else:
            target_date = pd.to_datetime(signal_row["date"]) + pd.offsets.BDay(forward_days)
        err_pct = (pred_close - actual_close) / actual_close * 100 if actual_close else 0.0

        rows.append({
            "signal_date": signal_row["date"],
            "target_date": target_date,
            "signal_close": signal_close,
            "pred_close": pred_close,
            "actual_close": actual_close,
            "pred_return": pred_ret,
            "actual_return": actual_ret,
            "error_pct": err_pct,
            "direction_hit": int((pred_ret > 0) == (actual_ret > 0)),
        })

    return pd.DataFrame(rows)


def predict_lstm_panel(
    model,
    raw: pd.DataFrame,
    panel: pd.DataFrame,
    seq_len: int = 30,
    forward_days: int = 5,
) -> pd.DataFrame:
    """Generate LSTM predictions aligned with panel (date, code) for real backtest."""
    parts = []
    codes = panel["code"].unique().tolist()
    from progress_utils import ProgressBar
    bar = ProgressBar(len(codes), desc="  LSTM回测", unit="股")

    for code in codes:
        sub = raw[raw["code"] == code]
        hist = predict_lstm_historical(
            model, sub, seq_len=seq_len, forward_days=forward_days, max_points=99999,
        )
        if not hist.empty:
            hist["code"] = code
            parts.append(hist[["code", "signal_date", "pred_return"]].rename(
                columns={"signal_date": "date", "pred_return": "lstm_pred"},
            ))
        bar.update(1, postfix=str(code))
    bar.close()

    if not parts:
        return pd.DataFrame(columns=["date", "code", "lstm_pred"])

    preds = pd.concat(parts, ignore_index=True)
    preds["date"] = pd.to_datetime(preds["date"])
    out = panel[["date", "code"]].copy()
    out["date"] = pd.to_datetime(out["date"])
    merged = out.merge(preds, on=["date", "code"], how="left")
    return merged[["date", "code", "lstm_pred"]].dropna(subset=["lstm_pred"])


def lstm_backtest_metrics(hist: pd.DataFrame) -> dict:
    """Summary metrics for historical LSTM fit."""
    if hist.empty:
        return {}
    mae = float((hist["pred_close"] - hist["actual_close"]).abs().mean())
    mape = float((hist["pred_close"] - hist["actual_close"]).abs().mean() / hist["actual_close"].mean() * 100)
    dir_acc = float(hist["direction_hit"].mean())
    ret_corr = float(np.corrcoef(hist["pred_return"], hist["actual_return"])[0, 1]) if len(hist) > 5 else 0.0
    return {
        "n_points": len(hist),
        "mae_price": round(mae, 4),
        "mape_pct": round(mape, 2),
        "direction_accuracy": round(dir_acc, 4),
        "return_correlation": round(ret_corr, 4),
        "mean_error_pct": round(float(hist["error_pct"].mean()), 2),
    }
