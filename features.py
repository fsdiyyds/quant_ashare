"""技术因子与标签工程（简化版 Alpha158 思路，参考 microsoft/qlib）。"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _macd(close: pd.Series) -> tuple:
    ema12 = _ema(close, 12)
    ema26 = _ema(close, 26)
    dif = ema12 - ema26
    dea = _ema(dif, 9)
    hist = 2 * (dif - dea)
    return dif, dea, hist


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """按单只股票时间序列计算因子。"""
    g = df.sort_values("date").copy()
    close = g["close"]
    high = g["high"]
    low = g["low"]
    vol = g["volume"].replace(0, np.nan)

    # 均线与乖离
    for w in (5, 10, 20, 60):
        ma = close.rolling(w).mean()
        g[f"ma{w}"] = ma
        g[f"bias{w}"] = close / ma - 1

    # 动量
    for w in (5, 10, 20):
        g[f"ret{w}"] = close.pct_change(w)

    # 波动率
    g["volatility20"] = close.pct_change().rolling(20).std()

    # RSI / MACD
    g["rsi14"] = _rsi(close, 14)
    dif, dea, hist = _macd(close)
    g["macd_dif"] = dif
    g["macd_dea"] = dea
    g["macd_hist"] = hist

    # 量价
    g["vol_ratio5"] = vol / vol.rolling(5).mean()
    g["amount_ratio5"] = g["amount"] / g["amount"].rolling(5).mean()

    # 价格位置
    g["high_low_ratio"] = (close - low) / (high - low).replace(0, np.nan)

    # KDJ 简化
    low9 = low.rolling(9).min()
    high9 = high.rolling(9).max()
    rsv = (close - low9) / (high9 - low9).replace(0, np.nan) * 100
    g["kdj_k"] = rsv.ewm(com=2, adjust=False).mean()
    g["kdj_d"] = g["kdj_k"].ewm(com=2, adjust=False).mean()
    g["kdj_j"] = 3 * g["kdj_k"] - 2 * g["kdj_d"]

    return g


def add_label(df: pd.DataFrame, forward_days: int = 5) -> pd.DataFrame:
    """未来 N 日收益率作为监督学习标签。"""
    g = df.copy()
    g["label"] = g.groupby("code")["close"].shift(-forward_days) / g["close"] - 1
    return g


def prepare_panel(
    raw: pd.DataFrame,
    forward_days: int = 5,
    exclude_limit: bool = True,
) -> tuple:
    """全市场面板：因子 + 标签。"""
    if raw.empty:
        return raw

    parts = []
    for code, sub in raw.groupby("code"):
        feat = build_features(sub)
        feat = add_label(feat, forward_days)
        parts.append(feat)

    panel = pd.concat(parts, ignore_index=True)

    feature_cols = [
        c for c in panel.columns
        if c not in ("date", "code", "open", "high", "low", "close", "volume",
                     "amount", "pct_chg", "turnover", "label")
        and panel[c].dtype in (np.float64, np.float32, np.int64, np.int32)
    ]

    panel = panel.dropna(subset=feature_cols + ["label"])

    if exclude_limit and "pct_chg" in panel.columns:
        panel = panel[panel["pct_chg"].abs() < 9.8]

    return panel, feature_cols


def latest_feature_row(panel: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """每只股票最近一个有效因子截面（用于预测）。"""
    idx = panel.groupby("code")["date"].idxmax()
    latest = panel.loc[idx].copy()
    return latest.dropna(subset=feature_cols)
