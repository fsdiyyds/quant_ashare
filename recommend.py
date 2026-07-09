"""生成详细购买理由（B1 战法 + LSTM 综合）。"""

from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd

from b1_selector import compute_b1_indicators, score_b1_row


def _tag(ok: bool, yes: str, no: str) -> str:
    return yes if ok else no


def build_buy_reason(row: pd.Series, checks: Dict[str, bool], forward_days: int = 5) -> str:
    """一段话购买理由。"""
    parts: List[str] = []

    name = row.get("name", row.get("code", ""))
    j = row.get("kdj_j", 0)
    b1 = row.get("b1_score", 0)

    parts.append(f"【{name}】符合 B1 战法调整蓄势特征（得分 {b1:.0f}/100）。")

    if checks.get("kdj_low"):
        parts.append(f"KDJ 的 J 值={j:.1f}，处于超卖低位，卖盘枯竭，符合「逆小势」低吸逻辑。")
    else:
        parts.append(f"KDJ J 值={j:.1f}，情绪偏低但未达理想超卖区。")

    if checks.get("vol_shrink"):
        parts.append("成交量较 5 日均量与 34 日 EMA 量明显萎缩，呈现缩量洗盘，主力未明显出货。")
    if checks.get("above_bbi") and checks.get("ma60_up"):
        parts.append("股价站在 BBI 多空线上方且 MA60 走平向上，中期趋势仍偏多（顺大势）。")
    elif checks.get("above_ma60"):
        parts.append("收盘价不低于 MA60，中期趋势未破。")

    if checks.get("macd_bull"):
        parts.append("MACD 的 DIF>0，多头区域运行。")
    if checks.get("zhixing"):
        parts.append("知行线多头排列（短>长），趋势结构健康。")
    if checks.get("rsi3_low"):
        parts.append(f"RSI(3)={row.get('rsi3', 0):.1f} 超卖，短线反弹概率提升。")

    b2 = row.get("b2_score", 0) or 0
    if b2 >= 60:
        parts.append(f"B2放量启动信号强（得分{b2:.0f}/100）：长阳配合放量，主力进攻意图明确。")
    elif b2 >= 40:
        parts.append(f"B2特征部分满足（得分{b2:.0f}），可关注是否出现放量长阳确认。")
    else:
        parts.append(f"B2尚未确认（得分{b2:.0f}），严格战法应等待B2放量后再介入。")

    pred = row.get("lstm_pred_return", row.get("lstm_pred", 0)) or 0
    pred_pct = row.get("lstm_pred_pct", pred * 100 if abs(pred) < 1 else pred)

    if pred > 0.02:
        parts.append(
            f"LSTM 模型基于近 30 日价量换手序列，预测未来 {forward_days} 日收益约 {pred_pct:+.2f}%，"
            "AI 信号偏多。"
        )
    elif pred > 0:
        parts.append(f"LSTM 预测未来 {forward_days} 日小幅上涨（约 {pred_pct:+.2f}%），可作辅助参考。")
    else:
        parts.append(f"LSTM 预测未来 {forward_days} 日收益 {pred_pct:+.2f}%，建议等待 B2 放量阳线确认后再介入。")

    return "".join(parts)


def build_reason_tags(checks: Dict[str, bool], row: pd.Series) -> str:
    """简短标签，用于表格展示。"""
    tags = []
    if checks.get("kdj_low"):
        tags.append("KDJ超卖")
    if checks.get("vol_shrink"):
        tags.append("缩量洗盘")
    if checks.get("above_bbi"):
        tags.append("BBI上")
    if checks.get("ma60_up"):
        tags.append("MA60向上")
    if checks.get("macd_bull"):
        tags.append("MACD多头")
    if checks.get("zhixing"):
        tags.append("知行多头")
    b2 = row.get("b2_score", 0) or 0
    if b2 >= 60:
        tags.append("B2放量")
    elif b2 >= 40:
        tags.append("B2酝酿")
    pred = row.get("lstm_pred_return", row.get("lstm_pred", 0)) or 0
    if pred > 0.02:
        tags.append("LSTM强看涨")
    elif pred > 0:
        tags.append("LSTM看涨")
    return " | ".join(tags) if tags else "综合信号"


def enrich_with_reasons(
    merged: pd.DataFrame,
    raw: pd.DataFrame,
    b1_params: Optional[dict] = None,
    forward_days: int = 5,
) -> pd.DataFrame:
    """为 Top 结果附加名称、详细理由、标签。"""
    from stock_info import attach_names

    if merged.empty:
        return merged

    params = b1_params or {}
    out = attach_names(merged)
    reasons, tags_list, checks_json = [], [], []

    from progress_utils import ProgressBar
    bar = ProgressBar(len(out), desc="  生成推荐理由", unit="股")
    for _, row in out.iterrows():
        code = str(row["code"]).zfill(6)
        sub = raw[raw["code"] == code]
        if sub.empty:
            checks = {}
            snap = row
        else:
            ind = compute_b1_indicators(sub)
            snap = ind.iloc[-1]
            _, checks = score_b1_row(snap, params)

        merged_row = {**row.to_dict(), **snap.to_dict()}
        sr = pd.Series(merged_row)
        reasons.append(build_buy_reason(sr, checks, forward_days))
        tags_list.append(build_reason_tags(checks, sr))
        checks_json.append(checks)
        bar.update(1)

    bar.close()

    out["buy_reason"] = reasons
    out["reason_tags"] = tags_list
    out["display_name"] = out.apply(
        lambda r: f"{r['name']}({r['code']})" if r.get("name") != r.get("code") else str(r["code"]),
        axis=1,
    )
    return out
