"""Plotly 可视化：股价、成交量、换手率、KDJ + LSTM 预测路径。"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

from b1_selector import compute_b1_indicators


def _require_plotly():
    if not HAS_PLOTLY:
        raise ImportError("请安装 plotly: pip install plotly")


def build_lstm_backtest_figure(
    hist: pd.DataFrame,
    code: str,
    name: str = "",
    forward_days: int = 5,
    max_points: int = 60,
) -> "go.Figure":
    """
    LSTM walk-forward backtest: predicted vs actual target price + error panel.
    Requires saved model at output/latest/lstm_model.keras (or .h5).
    """
    _require_plotly()
    from pathlib import Path
    from lstm_model import HAS_TF, lstm_backtest_metrics, predict_lstm_historical

    if not HAS_TF:
        raise ImportError("需要 tensorflow 才能展示 LSTM 回测")

    root = Path(__file__).resolve().parent
    model_path = root / "output" / "latest" / "lstm_model.keras"
    if not model_path.exists():
        model_path = root / "output" / "latest" / "lstm_model.h5"
    if not model_path.exists():
        raise FileNotFoundError("未找到已训练的 LSTM 模型，请先运行含 LSTM 的训练任务")

    import tensorflow as tf
    model = tf.keras.models.load_model(str(model_path))
    bt = predict_lstm_historical(model, hist, forward_days=forward_days, max_points=max_points)
    if bt.empty:
        raise ValueError("历史数据不足，无法生成 LSTM 回测")

    metrics = lstm_backtest_metrics(bt)
    title = f"{name}({code})" if name and name != code else code

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
        row_heights=[0.65, 0.35],
        subplot_titles=(
            f"{title} — LSTM {forward_days}日后目标价：预测 vs 真实",
            "预测偏差 (%) = (预测价-真实价)/真实价",
        ),
    )

    fig.add_trace(
        go.Scatter(
            x=bt["target_date"], y=bt["actual_close"],
            name="真实收盘价(目标日)", line=dict(color="#2563eb", width=2),
            mode="lines+markers", marker=dict(size=4),
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=bt["target_date"], y=bt["pred_close"],
            name=f"LSTM预测价({forward_days}日后)", line=dict(color="#ef4444", width=2, dash="dash"),
            mode="lines+markers", marker=dict(size=5, symbol="diamond"),
        ),
        row=1, col=1,
    )

    colors = ["#22c55e" if h else "#ef4444" for h in bt["direction_hit"]]
    fig.add_trace(
        go.Bar(x=bt["target_date"], y=bt["error_pct"], name="偏差%", marker_color=colors, opacity=0.75),
        row=2, col=1,
    )
    fig.add_hline(y=0, line_dash="dot", line_color="gray", row=2, col=1)

    ann = (
        f"样本={metrics.get('n_points', 0)} | "
        f"方向准确率={metrics.get('direction_accuracy', 0):.1%} | "
        f"均价误差={metrics.get('mean_error_pct', 0):+.2f}% | "
        f"MAE={metrics.get('mae_price', 0):.3f} | "
        f"收益相关={metrics.get('return_correlation', 0):.3f}"
    )
    fig.update_layout(
        height=650, template="plotly_white", hovermode="x unified",
        title=ann, margin=dict(l=50, r=30, t=80, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    fig.update_yaxes(title_text="价格", row=1, col=1)
    fig.update_yaxes(title_text="偏差%", row=2, col=1)
    return fig


def build_stock_figure(
    df: pd.DataFrame,
    code: str,
    name: str = "",
    lstm_pred_return: float = 0.0,
    forward_days: int = 5,
    tail_days: int = 120,
) -> "go.Figure":
    """多面板：价格+均线 / 成交量 / 换手率 / KDJ + LSTM 预测延伸。"""
    _require_plotly()

    g = compute_b1_indicators(df.sort_values("date")).tail(tail_days).copy()
    if g.empty:
        raise ValueError("无有效行情数据")

    title = f"{name}({code})" if name and name != code else code
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.04,
        row_heights=[0.45, 0.2, 0.15, 0.2],
        subplot_titles=(
            f"{title} — 价格 & BBI/MA60 & LSTM前瞻",
            "成交量", "换手率 / 量比", "KDJ-J（B1核心指标）",
        ),
    )

    dates = g["date"]

    # --- 价格 ---
    fig.add_trace(
        go.Scatter(x=dates, y=g["close"], name="收盘价", line=dict(color="#2563eb", width=2)),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=dates, y=g["ma60"], name="MA60", line=dict(color="#f59e0b", width=1, dash="dot")),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=dates, y=g["bbi"], name="BBI", line=dict(color="#8b5cf6", width=1, dash="dash")),
        row=1, col=1,
    )

    # LSTM forward projection (latest signal only, dashed red line)
    last_date = g["date"].iloc[-1]
    last_close = float(g["close"].iloc[-1])
    pred_close = last_close * (1 + lstm_pred_return)
    pred_dates = pd.date_range(last_date, periods=forward_days + 1, freq="B")[1:]
    if len(pred_dates) == 0:
        pred_dates = [last_date + pd.Timedelta(days=forward_days)]
    pred_y = np.linspace(last_close, pred_close, len(pred_dates) + 1)[1:]

    fig.add_trace(
        go.Scatter(
            x=[last_date] + list(pred_dates),
            y=[last_close] + list(pred_y),
            name=f"LSTM前瞻预测({forward_days}日 {lstm_pred_return*100:+.2f}%)",
            line=dict(color="#ef4444", width=2, dash="dash"),
            mode="lines+markers",
            marker=dict(size=6, symbol="diamond"),
        ),
        row=1, col=1,
    )

    # --- 成交量 ---
    colors = ["#ef4444" if c >= o else "#22c55e" for c, o in zip(g["close"], g["open"])]
    fig.add_trace(
        go.Bar(x=dates, y=g["volume"], name="成交量", marker_color=colors, opacity=0.7),
        row=2, col=1,
    )
    fig.add_trace(
        go.Scatter(x=dates, y=g["vol_ma5"], name="5日均量", line=dict(color="#64748b", width=1)),
        row=2, col=1,
    )

    # --- 换手率 / 量比 ---
    turnover = g["turnover"].replace(0, np.nan)
    if turnover.notna().any() and turnover.max() > 0:
        fig.add_trace(
            go.Scatter(x=dates, y=turnover, name="换手率%", line=dict(color="#0ea5e9", width=1.5)),
            row=3, col=1,
        )
    else:
        fig.add_trace(
            go.Scatter(x=dates, y=g["vol_ratio5"], name="量比(5日)", line=dict(color="#0ea5e9", width=1.5)),
            row=3, col=1,
        )

    # --- KDJ J ---
    fig.add_trace(
        go.Scatter(x=dates, y=g["kdj_j"], name="KDJ-J", line=dict(color="#a855f7", width=1.5)),
        row=4, col=1,
    )
    fig.add_hline(y=13, line_dash="dash", line_color="green", annotation_text="B1理想J≤13", row=4, col=1)
    fig.add_hline(y=15, line_dash="dot", line_color="orange", annotation_text="J≤15", row=4, col=1)
    fig.add_hline(y=0, line_dash="dot", line_color="gray", row=4, col=1)

    fig.update_layout(
        height=900,
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=50, r=30, t=80, b=40),
    )
    fig.update_yaxes(title_text="价格", row=1, col=1)
    fig.update_yaxes(title_text="量", row=2, col=1)
    fig.update_yaxes(title_text="%", row=3, col=1)
    fig.update_yaxes(title_text="J", row=4, col=1)

    return fig


def build_lstm_feature_heatmap(
    df: pd.DataFrame,
    code: str,
    seq_len: int = 30,
) -> "go.Figure":
    """LSTM 输入特征热力图（最近 seq_len 日）。"""
    _require_plotly()
    from lstm_model import SEQ_FEATURES, build_lstm_sequences

    g = df[df["code"] == code] if "code" in df.columns else df
    X, _, feats = build_lstm_sequences(g, seq_len=seq_len)
    if len(X) == 0:
        raise ValueError("序列不足")

    mat = X[-1].T  # (features, seq_len)
    fig = go.Figure(data=go.Heatmap(
        z=mat,
        x=list(range(1, seq_len + 1)),
        y=feats,
        colorscale="RdBu",
        zmid=0,
    ))
    fig.update_layout(
        title=f"{code} LSTM 输入特征（最近 {seq_len} 日）",
        xaxis_title="交易日（距今日）",
        yaxis_title="特征",
        height=350,
    )
    return fig


def build_model_backtest_figure(
    hist: "pd.DataFrame",
    model_label: str,
    code: str = "",
    name: str = "",
    forward_days: int = 5,
    metrics: Optional[dict] = None,
) -> "go.Figure":
    """单模型回测：预测目标价 vs 真实目标价 + 偏差柱。"""
    _require_plotly()
    if hist is None or hist.empty:
        raise ValueError("无回测数据")

    title = f"{name}({code})" if name and name != code else (code or model_label)
    metrics = metrics or {}
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
        row_heights=[0.65, 0.35],
        subplot_titles=(
            f"{title} — {model_label}：{forward_days}日后目标价 预测 vs 真实",
            "预测偏差 (%)",
        ),
    )
    fig.add_trace(
        go.Scatter(
            x=hist["target_date"], y=hist["actual_close"],
            name="真实目标价", line=dict(color="#2563eb", width=2),
            mode="lines+markers", marker=dict(size=4),
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=hist["target_date"], y=hist["pred_close"],
            name=f"{model_label} 预测价", line=dict(color="#ef4444", width=2, dash="dash"),
            mode="lines+markers", marker=dict(size=5, symbol="diamond"),
        ),
        row=1, col=1,
    )
    colors = ["#22c55e" if h else "#ef4444" for h in hist["direction_hit"]]
    fig.add_trace(
        go.Bar(x=hist["target_date"], y=hist["error_pct"], name="偏差%", marker_color=colors, opacity=0.75),
        row=2, col=1,
    )
    fig.add_hline(y=0, line_dash="dot", line_color="gray", row=2, col=1)
    ann = (
        f"样本={metrics.get('n_points', len(hist))} | "
        f"方向准确率={metrics.get('direction_accuracy', 0):.1%} | "
        f"均价误差={metrics.get('mean_error_pct', 0):+.2f}% | "
        f"收益相关={metrics.get('return_correlation', 0):.3f}"
    )
    fig.update_layout(
        height=560, template="plotly_white", hovermode="x unified",
        title=ann, margin=dict(l=50, r=30, t=80, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    fig.update_yaxes(title_text="价格", row=1, col=1)
    fig.update_yaxes(title_text="偏差%", row=2, col=1)
    return fig


def build_multi_model_accuracy_figure(summary: "pd.DataFrame") -> "go.Figure":
    """多模型回测准确度对比柱状图。"""
    _require_plotly()
    df = summary.copy()
    df = df[df["状态"] == "成功"].dropna(subset=["方向准确率"])
    if df.empty:
        raise ValueError("无成功模型可对比")

    fig = make_subplots(
        rows=1, cols=2, subplot_titles=("方向准确率", "预测收益%（未来）"),
        horizontal_spacing=0.12,
    )
    fig.add_trace(
        go.Bar(
            x=df["模型"], y=df["方向准确率"],
            name="方向准确率", marker_color="#2563eb",
            text=[f"{v:.1%}" for v in df["方向准确率"]],
            textposition="outside",
        ),
        row=1, col=1,
    )
    pred_col = [c for c in df.columns if c.startswith("预测") and c.endswith("收益%")]
    if pred_col:
        vals = df[pred_col[0]].fillna(0)
        colors = ["#22c55e" if v >= 0 else "#ef4444" for v in vals]
        fig.add_trace(
            go.Bar(
                x=df["模型"], y=vals, name="预测收益%",
                marker_color=colors,
                text=[f"{v:+.2f}%" for v in vals],
                textposition="outside",
            ),
            row=1, col=2,
        )
    fig.update_layout(
        height=420, template="plotly_white", showlegend=False,
        title="多模型回测准确度 & 未来收益预测对比",
        margin=dict(l=40, r=20, t=70, b=80),
    )
    fig.update_yaxes(tickformat=".0%", row=1, col=1)
    return fig


def build_future_forecast_figure(
    hist: "pd.DataFrame",
    model_results: list,
    code: str = "",
    name: str = "",
    forward_days: int = 5,
    tail_days: int = 80,
) -> "go.Figure":
    """历史收盘价 + 各模型未来目标价预测射线。"""
    _require_plotly()
    g = hist.sort_values("date").tail(tail_days).copy()
    title = f"{name}({code})" if name and name != code else code
    last_date = pd.to_datetime(g["date"].iloc[-1])
    last_close = float(g["close"].iloc[-1])
    pred_dates = pd.date_range(last_date, periods=forward_days + 1, freq="B")[1:]
    if len(pred_dates) == 0:
        pred_dates = pd.DatetimeIndex([last_date + pd.Timedelta(days=forward_days)])

    palette = [
        "#ef4444", "#f59e0b", "#8b5cf6", "#0ea5e9",
        "#22c55e", "#ec4899", "#14b8a6", "#a855f7",
    ]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=g["date"], y=g["close"], name="历史收盘价",
        line=dict(color="#2563eb", width=2),
    ))

    ok = [r for r in model_results if not getattr(r, "error", None) and getattr(r, "hist", None) is not None]
    for i, r in enumerate(ok):
        if getattr(r, "hist", pd.DataFrame()).empty and not getattr(r, "future_pred_close", None):
            continue
        pred_close = float(getattr(r, "future_pred_close", last_close))
        pred_ret = float(getattr(r, "future_pred_return", 0.0))
        ys = np.linspace(last_close, pred_close, len(pred_dates) + 1)[1:]
        color = palette[i % len(palette)]
        label = getattr(r, "label", getattr(r, "key", f"model{i}"))
        fig.add_trace(go.Scatter(
            x=[last_date] + list(pred_dates),
            y=[last_close] + list(ys),
            name=f"{label} {pred_ret*100:+.2f}% → {pred_close:.2f}",
            line=dict(color=color, width=2, dash="dash"),
            mode="lines+markers",
            marker=dict(size=6, symbol="diamond"),
        ))

    fig.update_layout(
        height=480, template="plotly_white", hovermode="x unified",
        title=f"{title} — 多模型未来 {forward_days} 日股价预测",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=50, r=30, t=80, b=40),
        yaxis_title="价格",
    )
    return fig


def build_multi_model_overlay_backtest(
    model_results: list,
    code: str = "",
    name: str = "",
    max_points: int = 40,
) -> "go.Figure":
    """多模型预测目标价叠加（与真实价对比）。"""
    _require_plotly()
    title = f"{name}({code})" if name and name != code else code
    fig = go.Figure()
    actual_drawn = False
    palette = [
        "#ef4444", "#f59e0b", "#8b5cf6", "#0ea5e9",
        "#22c55e", "#ec4899", "#14b8a6", "#a855f7",
    ]
    for i, r in enumerate(model_results):
        h = getattr(r, "hist", None)
        if h is None or getattr(h, "empty", True) or getattr(r, "error", None):
            continue
        hh = h.tail(max_points)
        if not actual_drawn:
            fig.add_trace(go.Scatter(
                x=hh["target_date"], y=hh["actual_close"],
                name="真实目标价", line=dict(color="#2563eb", width=2.5),
            ))
            actual_drawn = True
        fig.add_trace(go.Scatter(
            x=hh["target_date"], y=hh["pred_close"],
            name=getattr(r, "label", r.key),
            line=dict(color=palette[i % len(palette)], width=1.5, dash="dot"),
        ))
    if not actual_drawn:
        raise ValueError("无可用回测曲线")
    fig.update_layout(
        height=480, template="plotly_white", hovermode="x unified",
        title=f"{title} — 多模型回测：预测目标价叠加",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        yaxis_title="价格",
    )
    return fig


def save_stock_chart(
    df: pd.DataFrame,
    code: str,
    name: str,
    out_path: Path,
    lstm_pred_return: float = 0.0,
    forward_days: int = 5,
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig = build_stock_figure(
        df, code, name=name,
        lstm_pred_return=lstm_pred_return,
        forward_days=forward_days,
    )
    fig.write_html(str(out_path), include_plotlyjs="cdn")
    return out_path


def save_html_report(
    top_df: pd.DataFrame,
    raw: pd.DataFrame,
    out_path: Path,
    forward_days: int = 5,
    max_charts: int = 20,
) -> Path:
    """生成含 Top N 图表的 HTML 报告。"""
    _require_plotly()

    sections = [
        "<html><head><meta charset='utf-8'>",
        "<title>B1+LSTM 可视化报告</title>",
        "<style>body{font-family:sans-serif;max-width:1200px;margin:auto;padding:20px}"
        "h2{border-bottom:2px solid #2563eb;padding-bottom:8px}"
        ".reason{background:#f0f9ff;padding:12px;border-radius:8px;margin:8px 0}</style>",
        "</head><body>",
        "<h1>B1战法 + LSTM 可视化报告</h1>",
    ]

    for _, row in top_df.head(max_charts).iterrows():
        code = str(row["code"]).zfill(6)
        name = row.get("name", code)
        sub = raw[raw["code"] == code]
        if sub.empty:
            continue
        fig = build_stock_figure(
            sub, code, name=name,
            lstm_pred_return=float(row.get("lstm_pred_return", 0)),
            forward_days=forward_days,
        )
        chart_div = fig.to_html(full_html=False, include_plotlyjs=False)
        reason = row.get("buy_reason", row.get("advice", ""))
        sections.append(f"<h2>#{row.get('rank','')} {name} ({code})</h2>")
        sections.append(f"<div class='reason'>{reason}</div>")
        sections.append(chart_div)

    sections.append("<p><em>免责声明：仅供学习研究，不构成投资建议。</em></p></body></html>")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    html = "\n".join(sections)
    html = html.replace("</body>", '<script src="https://cdn.plot.ly/plotly-latest.min.js"></script></body>', 1)
    out_path.write_text(html, encoding="utf-8")
    return out_path
