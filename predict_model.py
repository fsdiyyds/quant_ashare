"""机器学习预测模块（LightGBM / RandomForest，参考 Qlib 模型层设计）。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False


@dataclass
class TrainResult:
    model: Any
    feature_cols: List[str]
    train_rmse: float
    valid_rmse: float
    valid_r2: float
    feature_importance: pd.DataFrame


def _split_by_date(
    panel: pd.DataFrame,
    train_ratio: float = 0.8,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    dates = sorted(panel["date"].unique())
    cut = int(len(dates) * train_ratio)
    train_dates = set(dates[:cut])
    train = panel[panel["date"].isin(train_dates)]
    valid = panel[~panel["date"].isin(train_dates)]
    return train, valid


def _build_model(algorithm: str):
    algo = (algorithm or "lightgbm").lower()
    if algo == "lightgbm":
        if not HAS_LGB:
            raise ImportError("请安装 lightgbm: pip install lightgbm")
        return lgb.LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1,
        )
    if algo == "random_forest":
        return RandomForestRegressor(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=20,
            random_state=42,
            n_jobs=-1,
        )
    if algo == "gradient_boosting":
        return GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            random_state=42,
        )
    raise ValueError(f"未知算法: {algorithm}")


def train_model(
    panel: pd.DataFrame,
    feature_cols: List[str],
    algorithm: str = "lightgbm",
    train_ratio: float = 0.8,
) -> TrainResult:
    train_df, valid_df = _split_by_date(panel, train_ratio)

    X_train = train_df[feature_cols].values
    y_train = train_df["label"].values
    X_valid = valid_df[feature_cols].values
    y_valid = valid_df["label"].values

    model = _build_model(algorithm)
    model.fit(X_train, y_train)

    pred_train = model.predict(X_train)
    pred_valid = model.predict(X_valid)

    train_rmse = float(np.sqrt(mean_squared_error(y_train, pred_train)))
    valid_rmse = float(np.sqrt(mean_squared_error(y_valid, pred_valid)))
    valid_r2 = float(r2_score(y_valid, pred_valid))

    if hasattr(model, "feature_importances_"):
        imp = pd.DataFrame({
            "feature": feature_cols,
            "importance": model.feature_importances_,
        }).sort_values("importance", ascending=False)
    else:
        imp = pd.DataFrame({"feature": feature_cols, "importance": 0.0})

    return TrainResult(
        model=model,
        feature_cols=feature_cols,
        train_rmse=train_rmse,
        valid_rmse=valid_rmse,
        valid_r2=valid_r2,
        feature_importance=imp,
    )


def predict_scores(
    model: Any,
    latest: pd.DataFrame,
    feature_cols: List[str],
) -> pd.DataFrame:
    out = latest[["code", "date", "close"]].copy()
    out["pred_return"] = model.predict(latest[feature_cols].values)
    out["score"] = out["pred_return"].rank(pct=True)
    return out.sort_values("pred_return", ascending=False)


def direction_label(pred_return: float, threshold: float = 0.0) -> str:
    if pred_return > threshold + 0.01:
        return "看涨"
    if pred_return < threshold - 0.01:
        return "看跌"
    return "震荡"
