"""
单股多模型分析：输入股票代码 → 训练多种模型 → 回测准确度 + 未来股价预测。

支持原生 LightGBM/LSTM 与 Qlib 风格表格/时序模型。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ensemble_config import MODEL_LABELS, QLIB_SEQ_KEYS, QLIB_TABULAR_KEYS
from ensemble_model import TABULAR_FEATURES
from qlib_models import (
    HAS_TF,
    QLIB_MODEL_LABELS,
    _build_seq_model,
    _fit_tabular_one,
    _predict_tabular_one,
    available_qlib_models,
)

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False


FEATURE_COLS = list(TABULAR_FEATURES)


@dataclass
class ModelBacktestResult:
    key: str
    label: str
    hist: pd.DataFrame  # signal_date, target_date, pred_close, actual_close, ...
    metrics: Dict[str, float] = field(default_factory=dict)
    future_pred_return: float = 0.0
    future_pred_close: float = 0.0
    error: str = ""


@dataclass
class StockAnalysisResult:
    code: str
    name: str
    hist: pd.DataFrame
    asof: str
    last_close: float
    forward_days: int
    models: List[ModelBacktestResult]
    summary: pd.DataFrame


def _norm_code(code: str) -> str:
    return str(code).strip().zfill(6)


def fetch_stock_hist(
    code: str,
    start_date: str = "20230101",
    force_refresh: bool = True,
) -> pd.DataFrame:
    """拉取单只股票最新行情。"""
    import os
    from data_fetcher import fetch_daily_hist

    os.environ.setdefault("QUANT_DATA_SOURCE", "sina")
    code = _norm_code(code)
    df = fetch_daily_hist(code, start_date=start_date.replace("-", ""))
    if df.empty:
        return df
    df["code"] = code
    return df.sort_values("date").reset_index(drop=True)


def _build_feature_panel(hist: pd.DataFrame, forward_days: int = 5) -> pd.DataFrame:
    """单股特征面板（含未来收益标签）。"""
    from b1_selector import compute_b1_indicators, score_b1_row
    from b2_selector import detect_b2_signal

    ind = compute_b1_indicators(hist.sort_values("date"))
    if len(ind) < 80:
        return pd.DataFrame()

    c = ind["close"]
    ind = ind.copy()
    ind["ma20"] = c.rolling(20).mean()
    ind["ret5"] = c.pct_change(5)
    ind["ret10"] = c.pct_change(10)
    ind["volatility20"] = c.pct_change().rolling(20).std()

    b1_params = {"j_threshold": 15, "j_best": 13, "price_range_pct": 1.0}
    b2_params = {
        "up_pct_threshold": 0.05, "vol_multiple": 1.5,
        "lookback_n": 10, "j_threshold": 15,
    }

    rows = []
    for i in range(60, len(ind) - forward_days):
        row = ind.iloc[i]
        b1, _ = score_b1_row(row, b1_params)
        _, b2, _ = detect_b2_signal(ind, i, b2_params)
        close = float(row["close"])
        fut = float(ind.iloc[i + forward_days]["close"])
        label = fut / close - 1
        ma20 = row.get("ma20", close)
        rows.append({
            "date": row["date"],
            "code": str(hist["code"].iloc[0]).zfill(6) if "code" in hist.columns else "",
            "close": close,
            "b1_score": b1,
            "b2_score": b2,
            "kdj_j": row.get("kdj_j", 50),
            "rsi3": row.get("rsi3", 50),
            "vol_ratio5": row.get("vol_ratio5", 1),
            "macd_dif": row.get("macd_dif", 0),
            "bias20": close / ma20 - 1 if ma20 else 0,
            "ret5": row.get("ret5", 0),
            "ret10": row.get("ret10", 0),
            "volatility20": row.get("volatility20", 0),
            "label": label,
            "target_date": ind.iloc[i + forward_days]["date"],
            "actual_close": fut,
        })
    return pd.DataFrame(rows)


def _metrics_from_hist(hist: pd.DataFrame) -> Dict[str, float]:
    if hist is None or hist.empty:
        return {}
    mae = float((hist["pred_close"] - hist["actual_close"]).abs().mean())
    mape = float(
        (hist["pred_close"] - hist["actual_close"]).abs().mean()
        / (hist["actual_close"].abs().mean() + 1e-8) * 100
    )
    dir_acc = float(hist["direction_hit"].mean())
    if len(hist) > 5:
        ret_corr = float(np.corrcoef(hist["pred_return"], hist["actual_return"])[0, 1])
        if np.isnan(ret_corr):
            ret_corr = 0.0
    else:
        ret_corr = 0.0
    mean_err = float(hist["error_pct"].mean())
    return {
        "n_points": len(hist),
        "direction_accuracy": round(dir_acc, 4),
        "mae_price": round(mae, 4),
        "mape_pct": round(mape, 2),
        "mean_error_pct": round(mean_err, 2),
        "return_correlation": round(ret_corr, 4),
    }


def _hist_from_preds(
    panel_test: pd.DataFrame,
    pred_ret: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for i, (_, r) in enumerate(panel_test.iterrows()):
        pr = float(pred_ret[i])
        ar = float(r["label"])
        sc = float(r["close"])
        pc = sc * (1 + pr)
        ac = float(r["actual_close"])
        rows.append({
            "signal_date": r["date"],
            "target_date": r["target_date"],
            "signal_close": sc,
            "pred_close": pc,
            "actual_close": ac,
            "pred_return": pr,
            "actual_return": ar,
            "error_pct": (pc - ac) / ac * 100 if ac else 0.0,
            "direction_hit": int((pr > 0) == (ar > 0)),
        })
    return pd.DataFrame(rows)


def _train_eval_tabular(
    key: str,
    panel: pd.DataFrame,
    train_ratio: float = 0.7,
) -> Tuple[Optional[Any], Optional[Any], pd.DataFrame, float]:
    """训练表格模型，返回 model, scaler, backtest_hist, future_pred_return。"""
    from sklearn.preprocessing import StandardScaler

    n = len(panel)
    split = max(int(n * train_ratio), 40)
    if n - split < 10:
        return None, None, pd.DataFrame(), 0.0

    train = panel.iloc[:split]
    test = panel.iloc[split:]
    X_tr = train[FEATURE_COLS].fillna(0).values.astype(np.float32)
    y_tr = train["label"].values.astype(np.float32)
    X_te = test[FEATURE_COLS].fillna(0).values.astype(np.float32)

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    if key == "lgb":
        if not HAS_LGB:
            return None, None, pd.DataFrame(), 0.0
        model = lgb.LGBMRegressor(
            n_estimators=120, learning_rate=0.05, max_depth=5,
            num_leaves=31, random_state=42, verbose=-1,
        )
        model.fit(X_tr_s, y_tr)
        preds = model.predict(X_te_s)
    else:
        # 用一小段 valid 给需要 eval_set 的模型
        va_n = max(len(train) // 5, 5)
        X_va, y_va = X_tr_s[-va_n:], y_tr[-va_n:]
        X_fit, y_fit = X_tr_s[:-va_n], y_tr[:-va_n]
        if len(X_fit) < 20:
            X_fit, y_fit = X_tr_s, y_tr
            X_va, y_va = X_tr_s[-va_n:], y_tr[-va_n:]
        model = _fit_tabular_one(key, X_fit, y_fit, X_va, y_va)
        if model is None:
            return None, None, pd.DataFrame(), 0.0
        preds = _predict_tabular_one(key, model, X_te_s)

    hist = _hist_from_preds(test, np.asarray(preds, dtype=float))
    # 最新截面预测未来
    X_last = scaler.transform(panel.iloc[[-1]][FEATURE_COLS].fillna(0).values.astype(np.float32))
    if key == "lgb":
        fut = float(model.predict(X_last)[0])
    else:
        fut = float(_predict_tabular_one(key, model, X_last)[0])
    return model, scaler, hist, fut


def _train_eval_seq(
    key: str,
    hist: pd.DataFrame,
    seq_len: int = 30,
    forward_days: int = 5,
    epochs: int = 12,
    train_ratio: float = 0.75,
) -> Tuple[Optional[Any], pd.DataFrame, float]:
    """训练时序模型，返回 model, backtest_hist, future_pred_return。"""
    if not HAS_TF:
        return None, pd.DataFrame(), 0.0

    from lstm_model import build_lstm_sequences, train_lstm

    X, y, _ = build_lstm_sequences(hist, seq_len=seq_len, forward_days=forward_days)
    if len(X) < 40:
        return None, pd.DataFrame(), 0.0

    split = max(int(len(X) * train_ratio), 30)
    if len(X) - split < 8:
        return None, pd.DataFrame(), 0.0

    X_tr, y_tr = X[:split], y[:split]
    X_te = X[split:]

    if key == "lstm":
        result = train_lstm(X_tr, y_tr, seq_len=seq_len, epochs=epochs, batch_size=32)
        model = result.model
    else:
        model = _build_seq_model(key, seq_len, X.shape[-1])
        if model is None:
            return None, pd.DataFrame(), 0.0
        from tensorflow.keras import callbacks
        es = callbacks.EarlyStopping(monitor="loss", patience=3, restore_best_weights=True)
        model.fit(X_tr, y_tr, epochs=epochs, batch_size=32, callbacks=[es], verbose=0)

    # 对齐测试样本的日期：序列末尾对应的信号日
    g = hist.sort_values("date").reset_index(drop=True)
    # build_lstm_sequences 从 seq_len 开始，每个样本对应 labels[i-1] 即索引 i-1 的 label
    # 简化：用收盘价序列重建
    closes = g["close"].values
    dates = pd.to_datetime(g["date"].values)
    # 测试段对应全局索引 split+seq_len ...
    rows = []
    preds = model.predict(X_te, verbose=0).ravel()
    for j, pr in enumerate(preds):
        # 全局样本索引
        gi = split + j
        # 信号日约在 hist 的 seq_len + gi - 1
        sig_i = seq_len + gi - 1
        tgt_i = sig_i + forward_days
        if sig_i < 0 or tgt_i >= len(closes):
            continue
        sc = float(closes[sig_i])
        ar = float(closes[tgt_i] / sc - 1)
        pc = sc * (1 + float(pr))
        ac = float(closes[tgt_i])
        rows.append({
            "signal_date": dates[sig_i],
            "target_date": dates[tgt_i],
            "signal_close": sc,
            "pred_close": pc,
            "actual_close": ac,
            "pred_return": float(pr),
            "actual_return": ar,
            "error_pct": (pc - ac) / ac * 100 if ac else 0.0,
            "direction_hit": int((float(pr) > 0) == (ar > 0)),
        })
    bt = pd.DataFrame(rows)
    fut = float(model.predict(X[-1:], verbose=0).ravel()[0])
    return model, bt, fut


def analyze_stock(
    code: str,
    model_keys: Optional[List[str]] = None,
    start_date: str = "20230101",
    forward_days: int = 5,
    seq_len: int = 30,
    epochs: int = 12,
    force_refresh: bool = True,
    hist: Optional[pd.DataFrame] = None,
) -> StockAnalysisResult:
    """
    对单只股票运行多种模型，返回回测与未来预测。

    model_keys 示例: ["lgb", "ridge", "xgb", "lstm", "gru"]
    """
    code = _norm_code(code)
    avail = available_qlib_models()

    default_keys = ["lgb", "ridge", "rf", "xgb", "lstm", "gru"]
    keys = model_keys or default_keys
    keys = [k.strip().lower() for k in keys if k.strip()]

    if hist is None or hist.empty:
        hist = fetch_stock_hist(code, start_date=start_date, force_refresh=force_refresh)
    if hist.empty:
        raise RuntimeError(f"无法获取 {code} 行情，请检查代码或网络")

    hist = hist.sort_values("date").reset_index(drop=True)
    hist["code"] = code
    last_close = float(hist["close"].iloc[-1])
    asof = str(pd.to_datetime(hist["date"]).max().date())

    try:
        from stock_info import get_stock_name
        name = get_stock_name(code) or code
    except Exception:
        name = code

    panel = _build_feature_panel(hist, forward_days=forward_days)
    if len(panel) < 50:
        raise RuntimeError(f"{code} 有效训练样本不足（{len(panel)}），请拉长历史区间")

    results: List[ModelBacktestResult] = []

    for key in keys:
        label = MODEL_LABELS.get(key) or QLIB_MODEL_LABELS.get(key, key)
        try:
            if key in ("lgb",) or key in QLIB_TABULAR_KEYS:
                if key in QLIB_TABULAR_KEYS and not avail.get(key, False):
                    results.append(ModelBacktestResult(
                        key=key, label=label, hist=pd.DataFrame(),
                        error="依赖未安装，已跳过",
                    ))
                    continue
                if key == "lgb" and not HAS_LGB:
                    results.append(ModelBacktestResult(
                        key=key, label=label, hist=pd.DataFrame(),
                        error="未安装 lightgbm",
                    ))
                    continue
                _, _, bt, fut = _train_eval_tabular(key, panel)
                if bt.empty:
                    results.append(ModelBacktestResult(
                        key=key, label=label, hist=pd.DataFrame(),
                        error="样本不足或训练失败",
                    ))
                    continue
                m = _metrics_from_hist(bt)
                results.append(ModelBacktestResult(
                    key=key, label=label, hist=bt, metrics=m,
                    future_pred_return=fut,
                    future_pred_close=last_close * (1 + fut),
                ))
            elif key in ("lstm",) or key in QLIB_SEQ_KEYS:
                if not HAS_TF:
                    results.append(ModelBacktestResult(
                        key=key, label=label, hist=pd.DataFrame(),
                        error="未安装 tensorflow（Streamlit 请选 Python 3.11/3.12）",
                    ))
                    continue
                _, bt, fut = _train_eval_seq(
                    key, hist, seq_len=seq_len, forward_days=forward_days, epochs=epochs,
                )
                if bt.empty:
                    results.append(ModelBacktestResult(
                        key=key, label=label, hist=pd.DataFrame(),
                        error="时序样本不足或训练失败",
                    ))
                    continue
                m = _metrics_from_hist(bt)
                results.append(ModelBacktestResult(
                    key=key, label=label, hist=bt, metrics=m,
                    future_pred_return=fut,
                    future_pred_close=last_close * (1 + fut),
                ))
            else:
                results.append(ModelBacktestResult(
                    key=key, label=label, hist=pd.DataFrame(),
                    error="单股分析暂不支持该模型（规则模型请用组合选股）",
                ))
        except Exception as e:
            results.append(ModelBacktestResult(
                key=key, label=label, hist=pd.DataFrame(), error=str(e),
            ))

    # 汇总表
    rows = []
    for r in results:
        if r.error:
            rows.append({
                "模型": r.label, "key": r.key, "状态": f"失败: {r.error}",
                "方向准确率": None, "均价误差%": None, "收益相关": None,
                f"预测{forward_days}日收益%": None, "预测目标价": None,
            })
        else:
            rows.append({
                "模型": r.label, "key": r.key, "状态": "成功",
                "方向准确率": r.metrics.get("direction_accuracy"),
                "均价误差%": r.metrics.get("mean_error_pct"),
                "收益相关": r.metrics.get("return_correlation"),
                f"预测{forward_days}日收益%": round(r.future_pred_return * 100, 2),
                "预测目标价": round(r.future_pred_close, 3),
            })
    summary = pd.DataFrame(rows)

    return StockAnalysisResult(
        code=code, name=name, hist=hist, asof=asof,
        last_close=last_close, forward_days=forward_days,
        models=results, summary=summary,
    )
