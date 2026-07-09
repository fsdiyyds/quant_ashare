"""
回测引擎：训练集 / 验证集 / 测试集 分离，评估方向准确率、Rank IC、Top-K 收益
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ensemble_model import TABULAR_FEATURES, ensemble_score, predict_lgb


@dataclass
class BacktestMetrics:
    direction_accuracy: float = 0.0
    rank_ic: float = 0.0
    rank_ic_ir: float = 0.0
    top_k_win_rate: float = 0.0
    top_k_avg_return: float = 0.0
    top_k_total_return: float = 0.0
    max_drawdown: float = 0.0
    n_test_days: int = 0
    passed: bool = False
    details: Dict = None

    def to_dict(self) -> dict:
        return {
            "direction_accuracy": round(self.direction_accuracy, 4),
            "rank_ic": round(self.rank_ic, 4),
            "rank_ic_ir": round(self.rank_ic_ir, 4),
            "top_k_win_rate": round(self.top_k_win_rate, 4),
            "top_k_avg_return": round(self.top_k_avg_return, 4),
            "top_k_total_return": round(self.top_k_total_return, 4),
            "max_drawdown": round(self.max_drawdown, 4),
            "n_test_days": self.n_test_days,
            "passed": self.passed,
        }


def _daily_ic(day: pd.DataFrame, pred_col: str = "pred") -> float:
    if len(day) < 5:
        return np.nan
    try:
        ic, _ = spearmanr(day[pred_col], day["label"])
        return float(ic) if not np.isnan(ic) else 0.0
    except Exception:
        return 0.0


def run_backtest(
    test_panel: pd.DataFrame,
    lgb_model: Any = None,
    lstm_preds: Optional[pd.DataFrame] = None,
    feature_cols: Optional[List[str]] = None,
    top_k: int = 50,
    weights: Optional[dict] = None,
    thresholds: Optional[dict] = None,
    extra_preds: Optional[Dict[str, Any]] = None,
) -> BacktestMetrics:
    """
    在测试集上逐日截面回测。
    lstm_preds: 需含 date, code, lstm_pred 列（与 test_panel 对齐）
    extra_preds: Qlib 风格模型预测 {key: ndarray}，长度与 test_panel 对齐
    """
    thresholds = thresholds or {}
    min_acc = thresholds.get("min_direction_accuracy", 0.52)
    min_ic = thresholds.get("min_rank_ic", 0.02)
    min_win = thresholds.get("min_top_k_win_rate", 0.45)

    feature_cols = feature_cols or TABULAR_FEATURES
    df = test_panel.copy()

    if lgb_model is not None:
        df["lgb_pred"] = predict_lgb(lgb_model, df, feature_cols)
    else:
        df["lgb_pred"] = 0.0

    if lstm_preds is not None and not lstm_preds.empty:
        df = df.merge(
            lstm_preds[["date", "code", "lstm_pred"]],
            on=["date", "code"], how="left",
        )
        df["lstm_pred"] = df["lstm_pred"].fillna(0)
    else:
        df["lstm_pred"] = 0.0

    extra = dict(extra_preds or {})
    for key, arr in list(extra.items()):
        col = f"{key}_pred"
        if col not in df.columns and arr is not None:
            try:
                df[col] = arr
            except Exception:
                pass

    df["pred"] = ensemble_score(
        df,
        lgb_pred=df["lgb_pred"].values,
        lstm_pred=df["lstm_pred"].values,
        weights=weights,
        extra_preds=extra or None,
    )

    # 方向准确率（全样本）
    df["pred_dir"] = (df["pred"] > df["pred"].median()).astype(int)
    direction_acc = float((df["pred_dir"] == df["label_dir"]).mean())

    # 每日 Rank IC
    ics = []
    top_returns = []
    win_flags = []
    equity = [1.0]

    test_dates = list(df.groupby("date").groups.keys())
    from progress_utils import ProgressBar
    bar = ProgressBar(len(test_dates), desc="  回测逐日", unit="日")

    for dt, day in df.groupby("date"):
        if len(day) < top_k:
            bar.update(1)
            continue
        ic = _daily_ic(day)
        if not np.isnan(ic):
            ics.append(ic)

        picks = day.nlargest(top_k, "pred")
        avg_ret = float(picks["label"].mean())
        top_returns.append(avg_ret)
        win_flags.append(avg_ret > 0)
        equity.append(equity[-1] * (1 + avg_ret))
        bar.update(1)

    bar.close(f"IC均值={np.mean(ics):.4f}" if ics else "无有效日")

    rank_ic = float(np.mean(ics)) if ics else 0.0
    rank_ic_ir = float(np.mean(ics) / (np.std(ics) + 1e-8)) if ics else 0.0
    top_k_win = float(np.mean(win_flags)) if win_flags else 0.0
    top_k_avg = float(np.mean(top_returns)) if top_returns else 0.0
    total_ret = equity[-1] - 1 if len(equity) > 1 else 0.0

    eq = pd.Series(equity)
    dd = float(((eq - eq.cummax()) / eq.cummax()).min()) if len(eq) > 1 else 0.0

    passed = (
        direction_acc >= min_acc
        and rank_ic >= min_ic
        and top_k_win >= min_win
    )

    return BacktestMetrics(
        direction_accuracy=direction_acc,
        rank_ic=rank_ic,
        rank_ic_ir=rank_ic_ir,
        top_k_win_rate=top_k_win,
        top_k_avg_return=top_k_avg,
        top_k_total_return=total_ret,
        max_drawdown=dd,
        n_test_days=len(ics),
        passed=passed,
        details={"daily_ics": ics[:20], "equity_curve": equity},
    )


def format_backtest_report(m: BacktestMetrics, thresholds: dict) -> str:
    lines = [
        "## 测试集回测结果",
        "",
        f"| 指标 | 数值 | 门槛 | 是否达标 |",
        f"|------|------|------|----------|",
        f"| 方向准确率 | {m.direction_accuracy:.2%} | {thresholds.get('min_direction_accuracy', 0.52):.0%} | {'✓' if m.direction_accuracy >= thresholds.get('min_direction_accuracy', 0.52) else '✗'} |",
        f"| Rank IC | {m.rank_ic:.4f} | {thresholds.get('min_rank_ic', 0.02):.2f} | {'✓' if m.rank_ic >= thresholds.get('min_rank_ic', 0.02) else '✗'} |",
        f"| Top-K 胜率 | {m.top_k_win_rate:.2%} | {thresholds.get('min_top_k_win_rate', 0.45):.0%} | {'✓' if m.top_k_win_rate >= thresholds.get('min_top_k_win_rate', 0.45) else '✗'} |",
        f"| Top-K 日均收益 | {m.top_k_avg_return:.2%} | - | - |",
        f"| 累计模拟收益 | {m.top_k_total_return:.2%} | - | - |",
        f"| 最大回撤 | {m.max_drawdown:.2%} | - | - |",
        f"| 测试交易日 | {m.n_test_days} | - | - |",
        "",
        f"**综合评估: {'通过，可输出推荐' if m.passed else '未通过，建议谨慎参考'}**",
    ]
    return "\n".join(lines)
