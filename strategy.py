"""选股建议与简易回测。"""

from __future__ import annotations

from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd

from predict_model import direction_label


def build_recommendations(
    scores: pd.DataFrame,
    top_k: int = 10,
    min_score: float = 0.0,
    st_codes: Optional[Set[str]] = None,
    exclude_st: bool = True,
) -> pd.DataFrame:
    df = scores.copy()
    if exclude_st and st_codes:
        df = df[~df["code"].isin(st_codes)]

    df = df[df["pred_return"] >= min_score]
    df = df.head(top_k).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)
    df["direction"] = df["pred_return"].apply(direction_label)
    df["pred_return_pct"] = (df["pred_return"] * 100).round(2)
    df["score_pct"] = (df["score"] * 100).round(1)

    advice = []
    for _, row in df.iterrows():
        if row["direction"] == "看涨":
            advice.append("模型预测未来收益为正，可考虑纳入观察/轻仓试探")
        elif row["direction"] == "看跌":
            advice.append("模型预测偏弱，建议回避或减仓")
        else:
            advice.append("信号中性，建议观望")
    df["advice"] = advice
    return df


def simple_backtest(
    panel: pd.DataFrame,
    feature_cols: List[str],
    model,
    top_k: int = 10,
    initial_cash: float = 1_000_000,
    commission: float = 0.0003,
    stamp_tax: float = 0.001,
    slippage: float = 0.001,
) -> Dict:
    """
    简易 Top-K 等权回测（按验证集日期逐日调仓）。
    注意：仅供研究，未考虑涨跌停、T+1 等 A 股细节。
    """
    dates = sorted(panel["date"].unique())
    cut = int(len(dates) * 0.8)
    test_dates = dates[cut:]

    cash = initial_cash
    holdings: Dict[str, float] = {}
    equity_curve = []

    for dt in test_dates:
        day = panel[panel["date"] == dt].copy()
        if day.empty:
            continue

        X = day[feature_cols].values
        day = day.assign(pred=model.predict(X))
        picks = day.nlargest(top_k, "pred")

        # 卖出不在 picks 中的
        target_codes = set(picks["code"].tolist())
        for code in list(holdings.keys()):
            if code not in target_codes:
                price_row = day[day["code"] == code]
                if price_row.empty:
                    continue
                price = float(price_row.iloc[0]["close"]) * (1 - slippage)
                shares = holdings.pop(code)
                proceeds = shares * price * (1 - commission - stamp_tax)
                cash += proceeds

        # 等权买入
        if not picks.empty:
            per_stock = cash / len(picks)
            for _, row in picks.iterrows():
                code = row["code"]
                price = float(row["close"]) * (1 + slippage)
                if price <= 0:
                    continue
                shares = per_stock / price
                cost = shares * price * (1 + commission)
                if cost > cash:
                    continue
                cash -= cost
                holdings[code] = holdings.get(code, 0) + shares

        # 当日市值
        mv = cash
        for code, shares in holdings.items():
            pr = day[day["code"] == code]
            if not pr.empty:
                mv += shares * float(pr.iloc[0]["close"])
        equity_curve.append({"date": dt, "equity": mv})

    if not equity_curve:
        return {"total_return": 0, "max_drawdown": 0, "curve": pd.DataFrame()}

    curve = pd.DataFrame(equity_curve)
    curve["return"] = curve["equity"] / initial_cash - 1
    peak = curve["equity"].cummax()
    dd = (curve["equity"] - peak) / peak
    total_ret = curve["equity"].iloc[-1] / initial_cash - 1

    return {
        "total_return": float(total_ret),
        "max_drawdown": float(dd.min()),
        "curve": curve,
        "final_equity": float(curve["equity"].iloc[-1]),
    }
