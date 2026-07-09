#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""量化选股 — 多模型组合 + 最新行情 + 可视化 Web 展示。"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

ROOT = Path(__file__).resolve().parent
LATEST_DIR = ROOT / "output" / "latest"
OUTPUT_DIR = ROOT / "output" / "b1_lstm"
RUNTIME_CFG = ROOT / "output" / "runtime_config.yaml"
CACHE_DIR = ROOT / "data" / "cache"

sys.path.insert(0, str(ROOT))
from ensemble_config import (
    DEFAULT_WEIGHTS,
    MODEL_DESC,
    MODEL_KEYS,
    MODEL_LABELS,
    NATIVE_KEYS,
    QLIB_MODEL_KEYS,
    QLIB_SEQ_KEYS,
    QLIB_TABULAR_KEYS,
    ModelConfig,
    available_qlib_models,
)

st.set_page_config(page_title="量化选股", page_icon="📈", layout="wide")

st.title("A股量化选股 · 多模型组合")
st.caption(
    "B1 / B2 / LightGBM / LSTM + Qlib 风格模型 Zoo · 自定义权重 · "
    "每次运行自动拉取最新成交数据"
)


def _load_csv(name: str) -> pd.DataFrame:
    p = LATEST_DIR / name
    if p.exists():
        return pd.read_csv(p)
    files = sorted(OUTPUT_DIR.glob(name.replace("_latest", "_*")), reverse=True)
    return pd.read_csv(files[0]) if files else pd.DataFrame()


def _load_model_config() -> dict:
    p = LATEST_DIR / "model_config.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    bt = LATEST_DIR / "backtest_metrics.json"
    if bt.exists():
        data = json.loads(bt.read_text(encoding="utf-8"))
        return data.get("model_config", {})
    return {}


def _data_asof() -> str:
    p = LATEST_DIR / "data_asof.txt"
    if p.exists():
        return p.read_text(encoding="utf-8").strip()
    for f in sorted(CACHE_DIR.glob("hist_*.pkl"), reverse=True):
        try:
            raw = pd.read_pickle(f)
            return str(pd.to_datetime(raw["date"]).max().date())
        except Exception:
            continue
    return "未知"


def _load_hist(code: str) -> pd.DataFrame:
    # 优先最新别名缓存
    for f in [CACHE_DIR / "hist_latest.pkl", *sorted(CACHE_DIR.glob("hist_*.pkl"), reverse=True)]:
        if not f.exists():
            continue
        try:
            raw = pd.read_pickle(f)
            sub = raw[raw["code"].astype(str).str.zfill(6) == code.zfill(6)]
            if not sub.empty:
                return sub
        except Exception:
            continue
    from data_fetcher import fetch_daily_hist
    os.environ.setdefault("QUANT_DATA_SOURCE", "sina")
    return fetch_daily_hist(code, start_date="20230101")


@st.cache_data(ttl=3600)
def _get_hist_cached(code: str) -> pd.DataFrame:
    return _load_hist(code)


def _render_weight_pie(norm_weights: dict) -> None:
    labels, values = [], []
    for k in MODEL_KEYS:
        if norm_weights.get(k, 0) > 0:
            labels.append(MODEL_LABELS.get(k, k))
            values.append(norm_weights[k])
    if not labels:
        st.warning("请至少启用一个模型")
        return
    try:
        import plotly.graph_objects as go
        fig = go.Figure(data=[go.Pie(
            labels=labels, values=values, hole=0.45,
            textinfo="label+percent", textposition="outside",
        )])
        fig.update_layout(
            height=360, margin=dict(t=20, b=20, l=20, r=20),
            showlegend=False, title="融合权重分布（归一化后）",
        )
        st.plotly_chart(fig, use_container_width=True)
    except ImportError:
        st.bar_chart(pd.Series(values, index=labels))


def _build_runtime_yaml(
    base_path: Path,
    enabled: dict,
    weights: dict,
    skip_gate: bool,
    force_refresh: bool,
) -> Path:
    cfg = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    cfg.setdefault("ensemble", {})
    cfg["ensemble"]["enabled"] = enabled
    cfg["ensemble"]["weights"] = weights
    cfg.setdefault("data", {})["force_refresh"] = force_refresh
    if skip_gate:
        cfg.setdefault("backtest", {})["require_pass_to_recommend"] = False
    RUNTIME_CFG.parent.mkdir(parents=True, exist_ok=True)
    RUNTIME_CFG.write_text(
        yaml.dump(cfg, allow_unicode=True, default_flow_style=False), encoding="utf-8",
    )
    return RUNTIME_CFG


def _apply_preset(preset: str) -> tuple[dict, dict]:
    presets = {
        "仅 B1 战法": (
            {"b1": True},
            {"b1": 1.0},
        ),
        "仅 B2 放量": (
            {"b2": True},
            {"b2": 1.0},
        ),
        "仅 LSTM": (
            {"lstm": True},
            {"lstm": 1.0},
        ),
        "仅 LightGBM": (
            {"lgb": True},
            {"lgb": 1.0},
        ),
        "B1+B2 规则组合": (
            {"b1": True, "b2": True},
            {"b1": 0.55, "b2": 0.45},
        ),
        "原生四模型默认": (
            {k: True for k in NATIVE_KEYS},
            {k: DEFAULT_WEIGHTS[k] for k in NATIVE_KEYS},
        ),
        "Qlib 表格全家桶": (
            {k: True for k in QLIB_TABULAR_KEYS},
            {k: 1.0 for k in QLIB_TABULAR_KEYS},
        ),
        "Qlib 时序全家桶": (
            {k: True for k in QLIB_SEQ_KEYS},
            {k: 1.0 for k in QLIB_SEQ_KEYS},
        ),
        "原生+Qlib 均衡": (
            {
                "b1": True, "b2": True, "lgb": True, "lstm": True,
                "ridge": True, "xgb": True, "gru": True,
            },
            {
                "b1": 0.15, "b2": 0.10, "lgb": 0.15, "lstm": 0.20,
                "ridge": 0.10, "xgb": 0.15, "gru": 0.15,
            },
        ),
    }
    pe, pw = presets.get(preset, ({k: True for k in NATIVE_KEYS}, {k: DEFAULT_WEIGHTS[k] for k in NATIVE_KEYS}))
    enabled = {k: pe.get(k, False) for k in MODEL_KEYS}
    weights = {k: float(pw.get(k, 0.0)) for k in MODEL_KEYS}
    return enabled, weights


# ── 侧边栏：最新数据 ──
with st.sidebar:
    st.markdown("### 最新行情数据")
    st.caption(f"当前缓存截至：**{_data_asof()}**")
    refresh_n = st.slider("刷新股票数", 50, 800, 200, 50, key="refresh_n")
    if st.button("立即拉取最新成交数据", use_container_width=True):
        os.environ.setdefault("QUANT_DATA_SOURCE", "sina")
        with st.spinner("正在增量拉取最新历史成交..."):
            try:
                from data_fetcher import get_all_a_codes, refresh_market_data
                codes = get_all_a_codes()[:refresh_n]
                df = refresh_market_data(
                    codes=codes,
                    start_date="20230101",
                    max_stocks=refresh_n,
                    workers=6,
                    cache_dir=CACHE_DIR,
                )
                if df.empty:
                    st.error("拉取失败，请检查网络或 QUANT_DATA_SOURCE")
                else:
                    asof = str(pd.to_datetime(df["date"]).max().date())
                    (LATEST_DIR).mkdir(parents=True, exist_ok=True)
                    (LATEST_DIR / "data_asof.txt").write_text(asof, encoding="utf-8")
                    st.success(f"已更新 {df['code'].nunique()} 只，截至 {asof}")
                    st.cache_data.clear()
            except Exception as e:
                st.error(f"刷新失败: {e}")

    st.divider()
    st.markdown("""
**模型说明**
- **B1 / B2**：规则战法
- **LGB / LSTM**：原生 ML
- **Qlib Zoo**：Linear/Ridge/XGB/CatBoost/GRU/ALSTM/Transformer 等

**组合示例**
- 保守：仅 B1
- 激进：B2 + LSTM + GRU
- 研究：Qlib 表格全家桶

> 仅供学习，不构成投资建议。
""")


tab_list, tab_chart, tab_run, tab_data = st.tabs(
    ["推荐列表", "个股可视化", "模型组合 & 运行", "数据管理"]
)

top50 = _load_csv("top50_latest.csv")
b1pool = _load_csv("b1_pool_latest.csv")
model_info = _load_model_config()
avail = available_qlib_models()

# ── 推荐列表 ──
with tab_list:
    if top50.empty:
        st.warning("暂无结果。请在「模型组合 & 运行」页配置并开始训练。")
    else:
        mix = model_info.get("summary") or str(top50.get("model_mix", pd.Series([""])).iloc[0])
        if mix:
            st.info(f"当前模型组合：**{mix}**")

        asof = top50.get("data_asof", pd.Series([_data_asof()])).iloc[0] if "data_asof" in top50.columns else _data_asof()
        bt_path = LATEST_DIR / "backtest_metrics.json"
        if bt_path.exists():
            bt = json.loads(bt_path.read_text(encoding="utf-8"))
            st.subheader("测试集回测")
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("方向准确率", f"{bt.get('direction_accuracy', 0):.1%}")
            c2.metric("Rank IC", f"{bt.get('rank_ic', 0):.4f}")
            c3.metric("Top-K 胜率", f"{bt.get('top_k_win_rate', 0):.1%}")
            c4.metric("累计收益", f"{bt.get('top_k_total_return', 0):.1%}")
            c5.metric("评估", "通过" if bt.get("passed") else "未通过")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("B1 初选", len(b1pool) if not b1pool.empty else "-")
        c2.metric("最终推荐", len(top50))
        c3.metric("更新", str(top50.get("date", pd.Series(["-"])).iloc[0])[:10])
        c4.metric("行情截至", str(asof)[:10])

        st.subheader("最值得购买的股票")
        for _, row in top50.iterrows():
            rank = row.get("rank", "")
            disp = row.get("display_name") or row.get("name") or row["code"]
            conf = row.get("confidence", "")
            score = row.get("ensemble_score", 0)
            with st.expander(
                f"#{rank} [{conf}] {disp} | 融合分={score:.2f} | "
                f"B1={row.get('b1_score', 0):.0f} B2={row.get('b2_score', 0):.0f} | "
                f"LSTM {row.get('lstm_pred_pct', 0):+.2f}%",
                expanded=(rank == 1),
            ):
                st.markdown(f"**购买理由：** {row.get('buy_reason', row.get('advice', ''))}")
                tags = row.get("reason_tags", "")
                if tags:
                    st.caption(f"标签：{tags}")

        st.download_button(
            "下载推荐 CSV",
            top50.to_csv(index=False).encode("utf-8-sig"),
            f"top_recommend_{datetime.now():%Y%m%d}.csv",
        )

# ── 个股可视化 ──
with tab_chart:
    st.subheader("历史走势 + LSTM 回测验证")
    if top50.empty:
        st.info("请先运行选股任务生成结果。")
    else:
        options = {}
        for _, r in top50.iterrows():
            label = r.get("display_name") or f"{r.get('name', r['code'])}({r['code']})"
            options[label] = str(r["code"]).zfill(6)

        choice = st.selectbox("选择股票", list(options.keys()))
        code = options[choice]
        row = top50[top50["code"].astype(str).str.zfill(6) == code].iloc[0]

        lstm_summary_path = LATEST_DIR / "lstm_backtest_history.csv"
        if lstm_summary_path.exists():
            try:
                from lstm_model import lstm_backtest_metrics
                hdf = pd.read_csv(lstm_summary_path)
                hdf["code"] = hdf["code"].astype(str).str.zfill(6)
                h_code = hdf[hdf["code"] == code]
                if not h_code.empty:
                    sm = lstm_backtest_metrics(h_code)
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("LSTM方向准确率", f"{sm.get('direction_accuracy', 0):.1%}")
                    c2.metric("均价偏差", f"{sm.get('mean_error_pct', 0):+.2f}%")
                    c3.metric("价格MAE", f"{sm.get('mae_price', 0):.3f}")
                    c4.metric("收益相关", f"{sm.get('return_correlation', 0):.3f}")
            except Exception:
                pass

        with st.spinner("加载行情与绘制图表..."):
            try:
                from visualize import (
                    build_lstm_backtest_figure,
                    build_lstm_feature_heatmap,
                    build_stock_figure,
                )
                hist = _get_hist_cached(code)
                if hist.empty:
                    st.error("无法获取行情数据")
                else:
                    pred = float(row.get("lstm_pred_return", row.get("lstm_pred", 0)))
                    name = row.get("name", code)

                    st.markdown("**① 价量/KDJ 总览**（红色虚线为 *最新一日* 的前瞻预测，非历史回测）")
                    fig = build_stock_figure(
                        hist, code, name=name,
                        lstm_pred_return=pred, forward_days=5,
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    st.markdown("**② LSTM 真实回测：预测目标价 vs 真实目标价**")
                    model_files = [
                        LATEST_DIR / "lstm_model.keras",
                        LATEST_DIR / "lstm_model.h5",
                    ]
                    if any(p.exists() for p in model_files):
                        try:
                            bt_fig = build_lstm_backtest_figure(hist, code, name=name, forward_days=5)
                            st.plotly_chart(bt_fig, use_container_width=True)
                        except Exception as e:
                            st.warning(f"LSTM 回测图暂不可用: {e}")
                    else:
                        st.info("未找到 LSTM 模型。请在「模型组合 & 运行」中启用 LSTM 并完成训练。")

                    st.markdown(f"**购买理由：** {row.get('buy_reason', row.get('advice', ''))}")
                    with st.expander("LSTM 输入特征热力图"):
                        try:
                            hm = build_lstm_feature_heatmap(hist, code)
                            st.plotly_chart(hm, use_container_width=True)
                        except Exception as e:
                            st.caption(f"特征图不可用: {e}")
                    c1, c2, c3, c4, c5 = st.columns(5)
                    c1.metric("B1 得分", f"{row.get('b1_score', 0):.0f}")
                    c2.metric("B2 得分", f"{row.get('b2_score', 0):.0f}")
                    c3.metric("KDJ-J", f"{row.get('kdj_j', 0):.1f}")
                    c4.metric("LSTM 5日", f"{row.get('lstm_pred_pct', pred * 100):+.2f}%")
                    c5.metric("融合分", f"{row.get('ensemble_score', 0):.2f}")
            except ImportError:
                st.error("请安装 plotly: pip install plotly")

# ── 模型组合 & 运行 ──
with tab_run:
    st.subheader("选择模型并设置融合比例")
    st.caption("勾选原生模型与 Qlib 风格模型，拖动滑块设置权重（自动归一化）。每次训练默认拉取最新成交数据。")

    preset = st.selectbox(
        "快捷预设",
        [
            "自定义", "仅 B1 战法", "仅 B2 放量", "仅 LSTM", "仅 LightGBM",
            "B1+B2 规则组合", "原生四模型默认",
            "Qlib 表格全家桶", "Qlib 时序全家桶", "原生+Qlib 均衡",
        ],
        index=6,
    )

    if preset != "自定义":
        preset_enabled, preset_weights = _apply_preset(preset)
    else:
        preset_enabled = {k: (k in NATIVE_KEYS) for k in MODEL_KEYS}
        preset_weights = dict(DEFAULT_WEIGHTS)

    col_cfg, col_viz = st.columns([1.35, 1])

    with col_cfg:
        st.markdown("**① 原生模型**")
        enabled = {}
        for k in NATIVE_KEYS:
            default_on = preset_enabled.get(k, True) if preset != "自定义" else True
            enabled[k] = st.checkbox(
                f"{MODEL_LABELS[k]} — {MODEL_DESC.get(k, '')}",
                value=default_on, key=f"en_{k}",
            )

        st.markdown("**② Qlib 表格模型**（参考 microsoft/qlib Model Zoo）")
        with st.expander("展开选择表格模型", expanded=(preset.startswith("Qlib") or preset == "原生+Qlib 均衡")):
            for k in QLIB_TABULAR_KEYS:
                ok = avail.get(k, False)
                label = f"{MODEL_LABELS[k]} — {MODEL_DESC.get(k, '')}"
                if not ok:
                    label += " ⚠️ 依赖未安装"
                default_on = preset_enabled.get(k, False) if preset != "自定义" else False
                enabled[k] = st.checkbox(label, value=default_on and ok, key=f"en_{k}", disabled=not ok)

        st.markdown("**③ Qlib 时序模型**（需 TensorFlow）")
        with st.expander("展开选择时序模型", expanded=preset in ("Qlib 时序全家桶", "原生+Qlib 均衡")):
            for k in QLIB_SEQ_KEYS:
                ok = avail.get(k, False)
                label = f"{MODEL_LABELS[k]} — {MODEL_DESC.get(k, '')}"
                if not ok:
                    label += " ⚠️ 需 tensorflow"
                default_on = preset_enabled.get(k, False) if preset != "自定义" else False
                enabled[k] = st.checkbox(label, value=default_on and ok, key=f"en_{k}", disabled=not ok)

        st.markdown("**④ 设置权重**（仅对已启用的模型生效）")
        weights = {}
        for k in MODEL_KEYS:
            if enabled.get(k):
                default_w = float(preset_weights.get(k, DEFAULT_WEIGHTS.get(k, 0.1) or 0.1))
                if default_w <= 0:
                    default_w = 0.1
                weights[k] = st.slider(
                    f"{MODEL_LABELS[k]} 权重",
                    0.0, 1.0, min(default_w, 1.0), 0.05,
                    key=f"w_{k}",
                )
            else:
                weights[k] = 0.0

        mc = ModelConfig(enabled=enabled, weights=weights)
        norm = mc.normalized_weights()

        st.markdown("**⑤ 运行参数**")
        max_stocks = st.slider("扫描股票数", 50, 500, 150, 50)
        top_k = st.slider("输出 Top N 推荐", 10, 100, 50, 10)
        force_refresh = st.checkbox("运行时拉取最新成交数据", value=True)
        skip_gate = st.checkbox("忽略回测门槛（仍输出推荐）", value=True)

    with col_viz:
        _render_weight_pie(norm)
        if mc.active_keys():
            st.success(f"将使用：**{mc.summary()}**")
            st.caption(f"启用 {len(mc.active_keys())} 个模型")
        else:
            st.error("请至少启用一个模型")

        st.markdown("**环境可用性**")
        rows = []
        for k in QLIB_MODEL_KEYS:
            rows.append({
                "模型": MODEL_LABELS[k],
                "可用": "✓" if avail.get(k) else "✗",
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    st.divider()
    log_box = st.empty()
    prog = st.progress(0, text="等待开始...")

    if st.button("开始训练并生成推荐", type="primary", disabled=not mc.active_keys()):
        import re

        base_cfg = ROOT / "config" / "cloud_settings.yaml"
        cfg_path = _build_runtime_yaml(base_cfg, enabled, weights, skip_gate, force_refresh)
        cfg_data = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        cfg_data.setdefault("lstm", {})["top_k"] = top_k
        cfg_path.write_text(
            yaml.dump(cfg_data, allow_unicode=True, default_flow_style=False), encoding="utf-8",
        )

        active = mc.active_keys()
        w_str = ",".join(str(weights.get(k, 1.0)) for k in active)
        cmd = [
            sys.executable, "-u", str(ROOT / "b1_lstm_daily.py"),
            "--config", str(cfg_path),
            "--max-stocks", str(max_stocks),
            "--models", ",".join(active),
            "--weights", w_str,
        ]
        if skip_gate:
            cmd.append("--skip-backtest-gate")
        if force_refresh:
            cmd.append("--force-refresh")
        else:
            cmd.append("--no-refresh")

        env = {**os.environ, "QUANT_DATA_SOURCE": "sina", "PYTHONUNBUFFERED": "1"}
        lines: list = []
        step_pat = re.compile(r"\[(\d+)/(\d+)\]")

        with st.spinner("训练中，请查看下方实时日志..."):
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, env=env, cwd=str(ROOT),
                encoding="utf-8", errors="replace",
            )
            for line in proc.stdout:
                line = line.rstrip()
                if line:
                    lines.append(line)
                    log_box.code("\n".join(lines[-45:]), language=None)
                    m = step_pat.search(line)
                    if m:
                        cur, tot = int(m.group(1)), int(m.group(2))
                        prog.progress(min(cur / tot, 1.0), text=f"步骤 {cur}/{tot} — {mc.summary()}")
            proc.wait()

        if proc.returncode == 0:
            prog.progress(1.0, text="完成!")
            st.success(f"训练完成！模型组合：{mc.summary()}")
            st.info("请切换到「推荐列表」查看最值得购买的股票。")
            st.cache_data.clear()
            st.rerun()
        else:
            st.error(f"运行失败 (exit {proc.returncode})，请查看日志。")

# ── 数据管理 ──
with tab_data:
    st.subheader("历史成交数据管理")
    st.markdown(
        "每次在「模型组合 & 运行」中训练时，默认会**增量拉取最新成交日**并合并进缓存。"
        "也可在此页或侧边栏单独刷新。"
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("行情截至", _data_asof())
    pkls = list(CACHE_DIR.glob("hist_*.pkl"))
    c2.metric("缓存文件数", len(pkls))
    if (CACHE_DIR / "hist_latest.pkl").exists():
        try:
            raw = pd.read_pickle(CACHE_DIR / "hist_latest.pkl")
            c3.metric("缓存股票数", raw["code"].nunique())
            st.dataframe(
                raw.groupby("code")["date"].max().reset_index()
                .rename(columns={"date": "最新日期"})
                .sort_values("最新日期", ascending=False)
                .head(20),
                use_container_width=True,
            )
        except Exception as e:
            st.caption(f"读取缓存失败: {e}")
    else:
        c3.metric("缓存股票数", "-")
        st.info("尚无 hist_latest.pkl，请先刷新数据或运行一次训练。")
