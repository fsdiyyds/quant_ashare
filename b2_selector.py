"""
B2战法（放量启动信号）检测与评分

参考 zettaranc / StockTradebyZ BigBullishVolumeSelector、B2学习笔记:
  - B1 缩量调整后出现放量长阳（涨幅>=5%）
  - 成交量 >= B1阶段最大量的 1.5 倍
  - 阳线反包前序调整阴线
  - 知行线多头
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from b1_selector import compute_b1_indicators, score_b1_row


def detect_b2_signal(
    ind: pd.DataFrame,
    idx: int,
    params: Optional[dict] = None,
) -> Tuple[bool, float, Dict[str, bool]]:
    """
    检测某日是否出现 B2 放量启动信号，返回 (是否触发, 得分0~100, 检查项)。
    """
    params = params or {}
    up_pct = params.get("up_pct_threshold", 0.05)
    vol_mult = params.get("vol_multiple", 1.5)
    lookback = params.get("lookback_n", 10)
    j_th = params.get("j_threshold", 15)

    if idx < lookback + 5:
        return False, 0.0, {}

    row = ind.iloc[idx]
    window = ind.iloc[max(0, idx - lookback): idx]
    prev = ind.iloc[idx - 1]

    checks: Dict[str, bool] = {}
    score = 0.0

    close, open_ = float(row["close"]), float(row.get("open", row["close"]))
    pct = float(row.get("pct_chg", 0)) / 100 if abs(row.get("pct_chg", 0)) > 1 else float(row.get("pct_chg", 0))
    if "pct_chg" in row and abs(row["pct_chg"]) > 1:
        pct = row["pct_chg"] / 100

    if pct == 0 and prev["close"] > 0:
        pct = close / float(prev["close"]) - 1

    vol = float(row["volume"])
    b1_vol_max = float(window["volume"].max()) if len(window) else vol

    # 核心 B2 条件
    bullish = close >= open_
    checks["bullish"] = bullish
    checks["big_up"] = pct >= up_pct
    checks["volume_surge"] = vol >= b1_vol_max * vol_mult if b1_vol_max > 0 else False

    if checks["big_up"]:
        score += 35
    if checks["volume_surge"]:
        score += 30
    if bullish:
        score += 10

    # B1 前置：lookback 内曾出现 J 低位
    j_low_in_window = (window["kdj_j"] <= j_th).any()
    checks["b1_prior"] = bool(j_low_in_window)
    if j_low_in_window:
        score += 15

    # 反包前 1~2 根阴线
    engulf = close > float(prev["open"]) and close > float(prev["close"])
    checks["engulf"] = engulf
    if engulf:
        score += 10

    # 知行多头
    if pd.notna(row.get("zx_short")) and pd.notna(row.get("zx_long")):
        checks["zhixing"] = float(row["zx_short"]) > float(row["zx_long"])
        if checks["zhixing"]:
            score += 5

    triggered = checks.get("big_up") and checks.get("volume_surge") and bullish
    return triggered, min(score, 100.0), checks


def scan_b2_history(
    df: pd.DataFrame,
    params: Optional[dict] = None,
) -> pd.DataFrame:
    """扫描单只股票历史上每日 B2 得分。"""
    params = params or {}
    ind = compute_b1_indicators(df.sort_values("date"))
    rows = []
    for i in range(len(ind)):
        triggered, score, checks = detect_b2_signal(ind, i, params)
        b1_score, _ = score_b1_row(ind.iloc[i], params)
        rows.append({
            "date": ind.iloc[i]["date"],
            "code": ind.iloc[i].get("code", df["code"].iloc[0] if "code" in df.columns else ""),
            "close": ind.iloc[i]["close"],
            "volume": ind.iloc[i]["volume"],
            "pct_chg": ind.iloc[i].get("pct_chg", 0),
            "b1_score": b1_score,
            "b2_score": score,
            "b2_signal": triggered,
            "kdj_j": ind.iloc[i].get("kdj_j"),
        })
    return pd.DataFrame(rows)


def latest_b2_score(df: pd.DataFrame, params: Optional[dict] = None) -> dict:
    """最近一个交易日的 B2 状态。"""
    hist = scan_b2_history(df, params)
    if hist.empty:
        return {"b2_score": 0, "b2_signal": False}
    last = hist.iloc[-1]
    _, score, checks = detect_b2_signal(
        compute_b1_indicators(df.sort_values("date")), len(hist) - 1, params,
    )
    return {
        "b2_score": score,
        "b2_signal": bool(last.get("b2_signal", False)),
        "b2_checks": checks,
        "date": last["date"],
    }
