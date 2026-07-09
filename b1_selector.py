"""
B1战法（少妇战法 / BBIKDJSelector）选股与评分

参考开源实现:
  - Noctis-lzy/StockTradebyZ  https://github.com/Noctis-lzy/StockTradebyZ
  - sycdirdir/StockTradebySyc   https://github.com/sycdirdir/StockTradebySyc

核心逻辑（顺大势、逆小势）:
  1. 趋势: 收盘 > BBI，MA60 走平/向上，收盘 >= MA60
  2. KDJ: J 值低位（<=13 最佳，<=15 较好）
  3. 缩量: 成交量 < 5日均量 且 < 34日EMA量
  4. MACD: DIF > 0
  5. 知行约束: 收盘 > 长期线，短期线 > 长期线
  6. 风险过滤: 排除 ST、连续跌停、MA60 下行、高位巨量阴线
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def _sma(s: pd.Series, n: int, m: int = 1) -> pd.Series:
    """通达信 SMA(X,N,M)。"""
    result = np.full(len(s), np.nan)
    arr = s.values.astype(float)
    for i in range(len(arr)):
        if np.isnan(arr[i]):
            continue
        if i == 0 or np.isnan(result[i - 1]):
            result[i] = arr[i]
        else:
            result[i] = (m * arr[i] + (n - m) * result[i - 1]) / n
    return pd.Series(result, index=s.index)


def compute_b1_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """为单只股票计算 B1 所需全部指标。"""
    g = df.sort_values("date").copy()
    c, h, l, o = g["close"], g["high"], g["low"], g["open"]
    v = g["volume"].replace(0, np.nan)

    # BBI 多空线
    g["bbi"] = (
        c.rolling(3).mean() + c.rolling(6).mean()
        + c.rolling(12).mean() + c.rolling(24).mean()
    ) / 4

    # MA60
    g["ma60"] = c.rolling(60).mean()
    g["ma60_slope"] = g["ma60"].diff(5)

    # 知行线: 短期 EMA(EMA(C,10),10), 长期 EMA(EMA(C,20),20)
    g["zx_short"] = _ema(_ema(c, 10), 10)
    g["zx_long"] = _ema(_ema(c, 20), 20)

    # KDJ (9,3,3)
    low9 = l.rolling(9).min()
    high9 = h.rolling(9).max()
    rsv = (c - low9) / (high9 - low9).replace(0, np.nan) * 100
    g["kdj_k"] = _sma(rsv.fillna(50), 3, 1)
    g["kdj_d"] = _sma(g["kdj_k"].fillna(50), 3, 1)
    g["kdj_j"] = 3 * g["kdj_k"] - 2 * g["kdj_d"]

    # RSI(3)
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(3).mean()
    loss = (-delta.clip(upper=0)).rolling(3).mean()
    rs = gain / loss.replace(0, np.nan)
    g["rsi3"] = 100 - 100 / (1 + rs)

    # MACD
    dif = _ema(c, 12) - _ema(c, 26)
    g["macd_dif"] = dif
    g["macd_dea"] = _ema(dif, 9)

    # 量能
    g["vol_ma5"] = v.rolling(5).mean()
    g["vol_ema34"] = v.ewm(span=34, adjust=False).mean()
    g["vol_ratio5"] = v / g["vol_ma5"]
    g["vol_ratio34"] = v / g["vol_ema34"]
    g["vol_min16"] = v.rolling(16).min()

    # 价格波动窗口
    g["price_range"] = c.rolling(20).max() / c.rolling(20).min() - 1

    # 阳阴量比 (14日)
    up_vol = v.where(c > o, 0).rolling(14).sum()
    dn_vol = v.where(c <= o, 0).rolling(14).sum()
    g["up_dn_vol_ratio"] = up_vol / dn_vol.replace(0, np.nan)

    # 有效上穿 MA60: 前一日 close < MA60, 当日 close >= MA60
    g["cross_ma60"] = (c >= g["ma60"]) & (c.shift(1) < g["ma60"].shift(1))

    g["pct_chg"] = g.get("pct_chg", c.pct_change() * 100)
    if "turnover" not in g.columns:
        g["turnover"] = 0.0

    return g


def score_b1_row(row: pd.Series, params: dict) -> Tuple[float, Dict[str, bool]]:
    """
    对单日截面计算 B1 符合度得分 (0~100) 及各项是否满足。
    分数越高越符合 B1 买入特征。
    """
    j_th = params.get("j_threshold", 15)
    j_best = params.get("j_best", 13)
    max_range = params.get("price_range_pct", 1.0)

    checks: Dict[str, bool] = {}
    score = 0.0

    j = row.get("kdj_j", np.nan)
    if pd.notna(j):
        checks["kdj_low"] = j <= j_th
        if j <= 0:
            score += 40
        elif j <= j_best:
            score += 35
        elif j <= j_th:
            score += 25
        elif j <= j_th + 5:
            score += 10

    vol = row.get("volume", np.nan)
    vol5 = row.get("vol_ma5", np.nan)
    vol34 = row.get("vol_ema34", np.nan)
    if pd.notna(vol) and pd.notna(vol5) and pd.notna(vol34):
        shrink = vol < vol5 and vol < vol34
        checks["vol_shrink"] = bool(shrink)
        if shrink:
            score += 20
            vol_min16 = row.get("vol_min16", np.nan)
            if pd.notna(vol_min16) and vol <= vol_min16 * 1.2:
                score += 5  # 地量加分

    close = row.get("close", np.nan)
    bbi = row.get("bbi", np.nan)
    ma60 = row.get("ma60", np.nan)
    ma60_slope = row.get("ma60_slope", np.nan)
    if pd.notna(close) and pd.notna(bbi) and pd.notna(ma60):
        above_bbi = close > bbi
        above_ma60 = close >= ma60
        ma60_up = pd.notna(ma60_slope) and ma60_slope >= 0
        checks["above_bbi"] = bool(above_bbi)
        checks["above_ma60"] = bool(above_ma60)
        checks["ma60_up"] = bool(ma60_up)
        if above_bbi:
            score += 8
        if above_ma60:
            score += 6
        if ma60_up:
            score += 6

    dif = row.get("macd_dif", np.nan)
    if pd.notna(dif):
        checks["macd_bull"] = dif > 0
        if dif > 0:
            score += 10

    zx_s = row.get("zx_short", np.nan)
    zx_l = row.get("zx_long", np.nan)
    if pd.notna(close) and pd.notna(zx_s) and pd.notna(zx_l):
        zx_ok = close > zx_l and zx_s > zx_l
        checks["zhixing"] = bool(zx_ok)
        if zx_ok:
            score += 10

    pr = row.get("price_range", np.nan)
    if pd.notna(pr) and pr <= max_range:
        checks["price_stable"] = True
        score += 5
    else:
        checks["price_stable"] = False

    rsi3 = row.get("rsi3", np.nan)
    if pd.notna(rsi3) and rsi3 <= 20:
        checks["rsi3_low"] = True
        score += 5
    else:
        checks["rsi3_low"] = False

    ud = row.get("up_dn_vol_ratio", np.nan)
    if pd.notna(ud) and ud >= 1.75:
        checks["up_vol_strong"] = True
        score += 3
    else:
        checks["up_vol_strong"] = False

    return min(score, 100.0), checks


def is_b1_risk(row: pd.Series, recent: pd.DataFrame) -> bool:
    """风险过滤：True 表示应排除。"""
    if row.get("pct_chg", 0) <= -9.8:
        return True
    if len(recent) >= 2:
        last2 = recent.tail(2)
        if (last2["pct_chg"] <= -9.8).all():
            return True
    turnover = row.get("turnover", 0) or 0
    if turnover > 15 and row.get("close", 0) < row.get("open", row.get("close", 0)):
        return True
    ma60_slope = row.get("ma60_slope", 0)
    if pd.notna(ma60_slope) and ma60_slope < -0.01:
        return True
    return False


def screen_b1_universe(
    raw: pd.DataFrame,
    top_n: int = 520,
    min_score: float = 30.0,
    params: Optional[dict] = None,
    st_codes: Optional[set] = None,
) -> pd.DataFrame:
    """
    全市场 B1 评分，返回得分最高的 top_n 只。
    """
    params = params or {}
    st_codes = st_codes or set()
    rows: List[dict] = []
    groups = [(c, sub) for c, sub in raw.groupby("code") if c not in st_codes and len(sub) >= 70]

    from progress_utils import ProgressBar
    bar = ProgressBar(len(groups), desc="  B1评分扫描", unit="股")
    for code, sub in groups:
        ind = compute_b1_indicators(sub)
        latest = ind.iloc[-1]
        recent = ind.tail(30)

        if is_b1_risk(latest, recent):
            bar.update(1, postfix=str(code))
            continue

        score, checks = score_b1_row(latest, params)
        if score < min_score:
            bar.update(1, postfix=str(code))
            continue

        rows.append({
            "code": code,
            "date": latest["date"],
            "close": latest["close"],
            "b1_score": score,
            "kdj_j": latest.get("kdj_j"),
            "rsi3": latest.get("rsi3"),
            "vol_ratio5": latest.get("vol_ratio5"),
            "turnover": latest.get("turnover", 0),
            "macd_dif": latest.get("macd_dif"),
            "above_bbi": checks.get("above_bbi", False),
            "vol_shrink": checks.get("vol_shrink", False),
            "zhixing": checks.get("zhixing", False),
        })
        bar.update(1, postfix=str(code))

    bar.close(f"命中 {len(rows)} 只")

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows).sort_values("b1_score", ascending=False)
    return result.head(top_n).reset_index(drop=True)
