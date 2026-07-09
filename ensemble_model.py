"""
多模型融合：B1 规则 + B2 放量 + LightGBM + LSTM
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

from b1_selector import compute_b1_indicators, score_b1_row
from b2_selector import detect_b2_signal


TABULAR_FEATURES = [
    "b1_score", "b2_score", "kdj_j", "rsi3", "vol_ratio5",
    "macd_dif", "bias20", "ret5", "ret10", "volatility20",
]


@dataclass
class EnsembleModels:
    lgb_model: Any = None
    lstm_model: Any = None
    feature_cols: List[str] = field(default_factory=list)
    weights: Dict[str, float] = field(default_factory=dict)


def build_historical_panel(
    raw: pd.DataFrame,
    b1_params: dict,
    b2_params: dict,
    forward_days: int = 5,
    min_b1_score: float = 30,
) -> pd.DataFrame:
    """构建全历史面板（含 B1/B2 得分与未来标签）。"""
    parts = []
    groups = list(raw.groupby("code"))
    from progress_utils import ProgressBar
    bar = ProgressBar(len(groups), desc="  构建训练面板", unit="股")
    for code, sub in groups:
        if len(sub) < 80:
            continue
        ind = compute_b1_indicators(sub.sort_values("date"))
        c = ind["close"]
        ind["ma20"] = c.rolling(20).mean()
        ind["ret5"] = c.pct_change(5)
        ind["ret10"] = c.pct_change(10)
        ind["volatility20"] = c.pct_change().rolling(20).std()
        rows = []
        for i in range(60, len(ind) - forward_days):
            row = ind.iloc[i]
            b1, _ = score_b1_row(row, b1_params)
            if b1 < min_b1_score:
                continue
            _, b2, _ = detect_b2_signal(ind, i, b2_params)
            close = float(row["close"])
            fut = float(ind.iloc[i + forward_days]["close"])
            label = fut / close - 1
            label_dir = 1 if label > 0 else 0

            rows.append({
                "date": row["date"],
                "code": code,
                "close": close,
                "b1_score": b1,
                "b2_score": b2,
                "kdj_j": row.get("kdj_j"),
                "rsi3": row.get("rsi3"),
                "vol_ratio5": row.get("vol_ratio5"),
                "macd_dif": row.get("macd_dif"),
                "bias20": row.get("close", close) / row.get("ma20", close) - 1 if row.get("ma20") else 0,
                "ret5": row.get("ret5", 0),
                "ret10": row.get("ret10", 0),
                "volatility20": row.get("volatility20", 0),
                "label": label,
                "label_dir": label_dir,
            })
        if rows:
            parts.append(pd.DataFrame(rows))
        bar.update(1, postfix=str(code))
    bar.close(f"共 {sum(len(p) for p in parts)} 条样本" if parts else "无样本")

    if not parts:
        return pd.DataFrame()
    panel = pd.concat(parts, ignore_index=True)
    panel["date"] = pd.to_datetime(panel["date"])
    return panel.sort_values(["date", "code"]).reset_index(drop=True)


def _split_panel(panel: pd.DataFrame, train_ratio=0.6, valid_ratio=0.15):
    dates = sorted(panel["date"].unique())
    n = len(dates)
    t1 = int(n * train_ratio)
    t2 = int(n * (train_ratio + valid_ratio))
    train_d = set(dates[:t1])
    valid_d = set(dates[t1:t2])
    test_d = set(dates[t2:])
    return (
        panel[panel["date"].isin(train_d)],
        panel[panel["date"].isin(valid_d)],
        panel[panel["date"].isin(test_d)],
    )


def train_lightgbm(panel: pd.DataFrame, feature_cols: List[str]) -> Any:
    if not HAS_LGB:
        return None
    train, valid, _ = _split_panel(panel)
    if train.empty or valid.empty:
        return None
    model = lgb.LGBMRegressor(
        n_estimators=200, learning_rate=0.05, max_depth=6,
        num_leaves=31, random_state=42, verbose=-1,
    )
    model.fit(
        train[feature_cols], train["label"],
        eval_set=[(valid[feature_cols], valid["label"])],
    )
    return model


def predict_lgb(model, df: pd.DataFrame, feature_cols: List[str]) -> np.ndarray:
    if model is None:
        return np.zeros(len(df))
    return model.predict(df[feature_cols].fillna(0))


def ensemble_score(
    df: pd.DataFrame,
    lgb_pred: Optional[np.ndarray] = None,
    lstm_pred: Optional[np.ndarray] = None,
    weights: Optional[Dict[str, float]] = None,
    extra_preds: Optional[Dict[str, np.ndarray]] = None,
) -> pd.Series:
    """融合得分。extra_preds 用于 Qlib 风格模型 {key: pred_array}。"""
    w = weights or {"b1": 0.25, "b2": 0.20, "lgb": 0.25, "lstm": 0.30}
    score = pd.Series(0.0, index=df.index)

    if w.get("b1", 0) > 0 and "b1_score" in df.columns:
        score = score + w["b1"] * (df["b1_score"] / 100.0)
    if w.get("b2", 0) > 0 and "b2_score" in df.columns:
        score = score + w["b2"] * (df["b2_score"] / 100.0)
    if w.get("lgb", 0) > 0 and lgb_pred is not None:
        score = score + w["lgb"] * pd.Series(lgb_pred, index=df.index).rank(pct=True)
    if w.get("lstm", 0) > 0 and lstm_pred is not None:
        score = score + w["lstm"] * pd.Series(lstm_pred, index=df.index).rank(pct=True)

    used_extra = set()
    if extra_preds:
        for key, arr in extra_preds.items():
            if w.get(key, 0) <= 0:
                continue
            if arr is not None:
                score = score + w[key] * pd.Series(arr, index=df.index).rank(pct=True)
                used_extra.add(key)
            elif f"{key}_pred" in df.columns:
                score = score + w[key] * df[f"{key}_pred"].rank(pct=True)
                used_extra.add(key)

    # 列中已有预测且权重>0，但未通过 extra_preds 传入时也计入
    for key, wk in w.items():
        if wk <= 0 or key in ("b1", "b2", "lgb", "lstm") or key in used_extra:
            continue
        col = f"{key}_pred"
        if col in df.columns:
            score = score + wk * df[col].rank(pct=True)

    return score


def build_latest_snapshot(
    raw: pd.DataFrame,
    codes: List[str],
    b1_params: dict,
    b2_params: dict,
) -> pd.DataFrame:
    """候选池最新截面特征。"""
    rows = []
    from progress_utils import ProgressBar
    bar = ProgressBar(len(codes), desc="  最新截面", unit="股")
    for code in codes:
        sub = raw[raw["code"] == code]
        if len(sub) < 70:
            continue
        ind = compute_b1_indicators(sub.sort_values("date"))
        c = ind["close"]
        ind["ma20"] = c.rolling(20).mean()
        ind["ret5"] = c.pct_change(5)
        ind["ret10"] = c.pct_change(10)
        ind["volatility20"] = c.pct_change().rolling(20).std()
        row = ind.iloc[-1]
        b1, _ = score_b1_row(row, b1_params)
        _, b2, _ = detect_b2_signal(ind, len(ind) - 1, b2_params)
        ma20 = row.get("ma20", row["close"])
        rows.append({
            "code": code,
            "date": row["date"],
            "close": row["close"],
            "b1_score": b1,
            "b2_score": b2,
            "kdj_j": row.get("kdj_j"),
            "rsi3": row.get("rsi3"),
            "vol_ratio5": row.get("vol_ratio5"),
            "macd_dif": row.get("macd_dif"),
            "bias20": row["close"] / ma20 - 1 if ma20 else 0,
            "ret5": row.get("ret5", 0),
            "ret10": row.get("ret10", 0),
            "volatility20": row.get("volatility20", 0),
            "turnover": row.get("turnover", 0),
        })
        bar.update(1, postfix=str(code))
    bar.close()
    return pd.DataFrame(rows)
