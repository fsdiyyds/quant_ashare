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
    "单股多模型回测/预测 · 每次运行自动拉取最新成交数据"
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

**单股分析**
- 打开「单股多模型分析」
- 输入代码 → 多选模型 → 回测准确度 + 未来预测图

> 仅供学习，不构成投资建议。
""")


tab_list, tab_chart, tab_single, tab_run, tab_data = st.tabs(
    ["推荐列表", "个股可视化", "单股多模型分析", "模型组合 & 运行", "数据管理"]
)

top50 = _load_csv("top50_latest.csv")
b1pool = _load_csv("b1_pool_latest.csv")
model_info = _load_model_config()
avail = available_qlib_models()

# ── 推荐列表 ──
with tab_list:
    banner = st.session_state.get("train_result_banner")
    if banner and banner.get("kind") == "success":
        st.success(banner.get("msg", "训练已完成，以下为最新推荐。"))

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

def _tf_status() -> tuple[bool, str]:
    try:
        import tensorflow as tf
        return True, f"TensorFlow {tf.__version__}"
    except Exception as e:
        return False, f"未安装/不可用: {e}"


# ── 单股多模型分析 ──
with tab_single:
    st.subheader("输入股票代码 · 多模型回测与未来预测")
    st.caption(
        "对单只股票分别训练 LightGBM / Qlib 表格模型 / LSTM / GRU 等，"
        "输出回测准确度对比图、预测目标价叠加图，以及未来 N 日股价预测。"
    )

    tf_ok, tf_msg = _tf_status()
    if tf_ok:
        st.success(f"时序模型环境就绪：{tf_msg}")
    else:
        st.error(
            f"LSTM/GRU 等时序模型不可用 — {tf_msg}\n\n"
            "解决：Streamlit Advanced settings 选 **Python 3.11 或 3.12** 后 Redeploy；"
            "本地执行 `pip install \"tensorflow>=2.15,<2.20\"`。"
        )

    c1, c2, c3 = st.columns([1.2, 1.5, 1])
    with c1:
        code_in = st.text_input("股票代码", value="600519", placeholder="如 600519 / 000858")
    with c2:
        single_model_opts = {
            "lgb": "LightGBM",
            "ridge": "Qlib Ridge",
            "lasso": "Qlib Lasso",
            "elasticnet": "Qlib ElasticNet",
            "rf": "Qlib RandomForest",
            "xgb": "Qlib XGBoost",
            "catboost": "Qlib CatBoost",
            "double_ensemble": "Qlib DoubleEnsemble",
            "lstm": "LSTM",
            "gru": "Qlib GRU",
            "alstm": "Qlib ALSTM",
            "transformer": "Qlib Transformer",
            "mlp_seq": "Qlib MLP(时序)",
        }
        seq_keys_ui = {"lstm", "gru", "alstm", "transformer", "mlp_seq"}
        default_sel = ["lgb", "ridge", "xgb"] + (["lstm", "gru"] if tf_ok else [])
        selected_labels = st.multiselect(
            "选择模型（可多选）",
            options=list(single_model_opts.values()),
            default=[single_model_opts[k] for k in default_sel if k in single_model_opts],
        )
        label_to_key = {v: k for k, v in single_model_opts.items()}
        selected_keys = [label_to_key[x] for x in selected_labels]
        if any(k in seq_keys_ui for k in selected_keys) and not tf_ok:
            st.warning("已选时序模型，但当前环境无 TensorFlow，运行后这些模型会标记为失败并显示原因。")
    with c3:
        fwd = st.selectbox("预测天数", [3, 5, 10], index=1)
        epochs_single = st.slider("时序训练轮数", 5, 25, 10, 1)

    run_single = st.button("开始分析该股票", type="primary", disabled=not selected_keys)
    single_log_box = st.empty()

    if run_single:
        code_clean = "".join(ch for ch in code_in if ch.isdigit()).zfill(6)
        if len(code_clean) != 6:
            st.error("请输入 6 位股票代码")
        else:
            logs: list = []
            with st.spinner(f"正在拉取 {code_clean} 行情并训练 {len(selected_keys)} 个模型..."):
                try:
                    from stock_analyzer import analyze_stock

                    result = analyze_stock(
                        code_clean,
                        model_keys=selected_keys,
                        forward_days=int(fwd),
                        epochs=int(epochs_single),
                        force_refresh=True,
                        log_lines=logs,
                    )
                    st.session_state["single_analysis"] = result
                    st.session_state["single_analysis_logs"] = logs
                    failed = [r for r in result.models if r.error]
                    if failed:
                        st.warning(
                            "部分模型失败：\n" + "\n".join(f"- {r.label}: {r.error}" for r in failed)
                        )
                    st.success(
                        f"{result.name}({result.code}) 分析完成 | "
                        f"行情截至 {result.asof} | 最新收盘 {result.last_close:.2f}"
                    )
                except Exception as e:
                    import traceback
                    st.session_state.pop("single_analysis", None)
                    st.session_state["single_analysis_logs"] = logs + [
                        f"FATAL: {e}", traceback.format_exc(),
                    ]
                    st.error(f"分析失败: {e}")
                    st.code(traceback.format_exc(), language=None)

            if logs:
                with st.expander("运行日志（点击展开）", expanded=True):
                    single_log_box.code("\n".join(logs[-80:]), language=None)

    # 保留上次日志
    if "single_analysis_logs" in st.session_state and not run_single:
        with st.expander("上次运行日志", expanded=False):
            st.code("\n".join(st.session_state["single_analysis_logs"][-80:]), language=None)

    result = st.session_state.get("single_analysis")
    if result is not None:
        st.markdown(f"### {result.name}（{result.code}）")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("最新收盘", f"{result.last_close:.2f}")
        m2.metric("行情截至", result.asof)
        ok_n = sum(1 for r in result.models if not r.error)
        m3.metric("成功模型", f"{ok_n}/{len(result.models)}")
        m4.metric("预测窗口", f"{result.forward_days} 日")

        failed_models = [r for r in result.models if r.error]
        if failed_models:
            st.error(
                "失败模型详情：\n"
                + "\n".join(f"- **{r.label}**: `{r.error}`" for r in failed_models)
            )

        st.markdown("#### ① 多模型回测准确度 & 未来收益对比")
        st.dataframe(result.summary, use_container_width=True, hide_index=True)
        try:
            from visualize import (
                build_future_forecast_figure,
                build_model_backtest_figure,
                build_multi_model_accuracy_figure,
                build_multi_model_overlay_backtest,
            )
            fig_acc = build_multi_model_accuracy_figure(result.summary)
            st.plotly_chart(fig_acc, use_container_width=True)
        except Exception as e:
            st.warning(f"准确度对比图暂不可用: {e}")

        st.markdown("#### ② 多模型回测：预测目标价叠加")
        try:
            fig_ov = build_multi_model_overlay_backtest(
                result.models, code=result.code, name=result.name,
            )
            st.plotly_chart(fig_ov, use_container_width=True)
        except Exception as e:
            st.warning(f"叠加回测图暂不可用: {e}")

        st.markdown("#### ③ 未来股价预测（各模型射线）")
        try:
            fig_fut = build_future_forecast_figure(
                result.hist, result.models,
                code=result.code, name=result.name,
                forward_days=result.forward_days,
            )
            st.plotly_chart(fig_fut, use_container_width=True)
        except Exception as e:
            st.warning(f"未来预测图暂不可用: {e}")

        st.markdown("#### ④ 分模型回测详情")
        for r in result.models:
            with st.expander(f"{r.label} — {'失败' if r.error else '成功'}", expanded=bool(r.error)):
                if r.error:
                    st.error(r.error)
                    continue
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("方向准确率", f"{r.metrics.get('direction_accuracy', 0):.1%}")
                c2.metric("均价误差", f"{r.metrics.get('mean_error_pct', 0):+.2f}%")
                c3.metric("收益相关", f"{r.metrics.get('return_correlation', 0):.3f}")
                c4.metric(
                    f"预测{result.forward_days}日",
                    f"{r.future_pred_return*100:+.2f}% → {r.future_pred_close:.2f}",
                )
                try:
                    from visualize import build_model_backtest_figure
                    fig_one = build_model_backtest_figure(
                        r.hist, r.label, code=result.code, name=result.name,
                        forward_days=result.forward_days, metrics=r.metrics,
                    )
                    st.plotly_chart(fig_one, use_container_width=True)
                except Exception as e:
                    st.caption(f"单模型图不可用: {e}")

        st.download_button(
            "下载分析汇总 CSV",
            result.summary.to_csv(index=False).encode("utf-8-sig"),
            f"stock_analysis_{result.code}_{datetime.now():%Y%m%d}.csv",
        )

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
        tf_ok_native, tf_msg_native = _tf_status()
        for k in NATIVE_KEYS:
            default_on = preset_enabled.get(k, True) if preset != "自定义" else True
            label = f"{MODEL_LABELS[k]} — {MODEL_DESC.get(k, '')}"
            if k == "lstm" and not tf_ok_native:
                label += " ⚠️ 当前环境无 TensorFlow"
            enabled[k] = st.checkbox(label, value=default_on, key=f"en_{k}")
        if enabled.get("lstm") and not tf_ok_native:
            st.warning(
                f"已勾选 LSTM，但 {tf_msg_native}。训练会跳过 LSTM，推荐结果可能与预期不符。"
                "请将 Streamlit Python 改为 3.11/3.12。"
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
    # 展示上次训练结果横幅（避免 rerun 后提示消失）
    last_banner = st.session_state.get("train_result_banner")
    if last_banner:
        kind = last_banner.get("kind", "info")
        msg = last_banner.get("msg", "")
        if kind == "success":
            st.success(msg)
        elif kind == "error":
            st.error(msg)
        else:
            st.warning(msg)
        if last_banner.get("logs"):
            with st.expander("上次训练日志", expanded=(kind != "success")):
                st.code("\n".join(last_banner["logs"][-80:]), language=None)

    tf_ok_run, tf_msg_run = _tf_status()
    if any(enabled.get(k) for k in ("lstm", "gru", "alstm", "transformer", "mlp_seq")):
        if tf_ok_run:
            st.info(f"已启用时序模型，环境：{tf_msg_run}。Cloud 上建议扫描股票数 ≤ 100，避免内存不足。")
        else:
            st.error(
                f"已启用 LSTM/时序模型，但 {tf_msg_run}。"
                "训练会跳过或失败。请将 Streamlit Python 改为 3.11/3.12 后 Redeploy。"
            )

    log_box = st.empty()
    prog = st.progress(0, text="等待开始...")

    if st.button("开始训练并生成推荐", type="primary", disabled=not mc.active_keys()):
        import re
        import traceback
        from datetime import datetime as _dt

        base_cfg = ROOT / "config" / "cloud_settings.yaml"
        cfg_path = _build_runtime_yaml(base_cfg, enabled, weights, skip_gate, force_refresh)
        cfg_data = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        cfg_data.setdefault("lstm", {})["top_k"] = top_k
        # Cloud 上 LSTM 降负载
        if any(enabled.get(k) for k in ("lstm", "gru", "alstm", "transformer", "mlp_seq")):
            cfg_data.setdefault("lstm", {})["epochs"] = min(
                int(cfg_data.get("lstm", {}).get("epochs", 15)), 12,
            )
        cfg_path.write_text(
            yaml.dump(cfg_data, allow_unicode=True, default_flow_style=False), encoding="utf-8",
        )

        top50_path = LATEST_DIR / "top50_latest.csv"
        before_mtime = top50_path.stat().st_mtime if top50_path.exists() else 0.0

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

        env = {
            **os.environ,
            "QUANT_DATA_SOURCE": "sina",
            "PYTHONUNBUFFERED": "1",
            "TF_CPP_MIN_LOG_LEVEL": "2",
        }
        lines: list = []
        step_pat = re.compile(r"\[(\d+)/(\d+)\]")
        done_pat = re.compile(r"\[DONE\].*top50_latest")
        wrote_recommend = False
        lines.append(f"$ {' '.join(cmd)}")
        lines.append(f"[env] {tf_msg_run}")
        lines.append(f"[time] start {_dt.now():%Y-%m-%d %H:%M:%S}")
        log_box.code("\n".join(lines[-45:]), language=None)

        with st.spinner("训练中，请查看下方实时日志（含 LSTM 时可能较久）..."):
            try:
                proc = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, env=env, cwd=str(ROOT),
                    encoding="utf-8", errors="replace",
                )
                assert proc.stdout is not None
                for line in proc.stdout:
                    line = line.rstrip()
                    if line:
                        lines.append(line)
                        if done_pat.search(line):
                            wrote_recommend = True
                        log_box.code("\n".join(lines[-60:]), language=None)
                        m = step_pat.search(line)
                        if m:
                            cur, tot = int(m.group(1)), int(m.group(2))
                            prog.progress(
                                min(cur / tot, 1.0),
                                text=f"步骤 {cur}/{tot} — {mc.summary()}",
                            )
                proc.wait()
            except Exception as e:
                lines.append(f"FATAL subprocess: {e}")
                lines.append(traceback.format_exc())
                log_box.code("\n".join(lines[-60:]), language=None)
                st.session_state["train_result_banner"] = {
                    "kind": "error",
                    "msg": f"启动训练进程失败: {e}",
                    "logs": lines,
                }
                st.error(f"启动训练进程失败: {e}")
                st.stop()

        after_exists = top50_path.exists()
        after_mtime = top50_path.stat().st_mtime if after_exists else 0.0
        file_updated = after_exists and after_mtime > before_mtime
        n_rows = 0
        if after_exists:
            try:
                n_rows = len(pd.read_csv(top50_path))
            except Exception:
                n_rows = -1

        lines.append(f"[time] end {_dt.now():%Y-%m-%d %H:%M:%S}")
        lines.append(
            f"[check] exit={proc.returncode} wrote_flag={wrote_recommend} "
            f"file_updated={file_updated} top50_rows={n_rows}"
        )
        log_box.code("\n".join(lines[-60:]), language=None)
        st.session_state["last_train_logs"] = lines

        # 137/-9 常见于 Cloud OOM
        oom = proc.returncode in (137, -9, 247)
        success = (proc.returncode == 0 and (wrote_recommend or file_updated)) or (
            file_updated and wrote_recommend
        )

        if success or file_updated:
            prog.progress(1.0, text="完成!")
            banner = (
                f"训练完成！模型：{mc.summary()} | "
                f"已更新 top50_latest.csv（{n_rows} 只）| "
                f"请打开「推荐列表」查看。"
            )
            if proc.returncode != 0 and file_updated:
                banner = (
                    f"进程异常退出 (exit {proc.returncode})，但推荐文件已更新（{n_rows} 只）。"
                    f"可打开「推荐列表」。日志末尾可能有 LSTM 后处理被中断。"
                )
            st.session_state["train_result_banner"] = {
                "kind": "success",
                "msg": banner,
                "logs": lines,
            }
            st.cache_data.clear()
            st.rerun()
        else:
            prog.progress(1.0, text="失败")
            reason = f"运行失败 (exit {proc.returncode})"
            if oom:
                reason += " — 疑似内存不足(OOM)。请减小扫描股票数，或暂时关闭 LSTM/GRU。"
            if not file_updated:
                reason += " — 推荐文件未更新，故「推荐列表」不会变化。"
            st.session_state["train_result_banner"] = {
                "kind": "error",
                "msg": reason,
                "logs": lines,
            }
            st.error(reason)
            with st.expander("完整失败日志", expanded=True):
                st.code(
                    "\n".join(lines[-120:]) if lines else "(无输出，进程可能被系统直接杀掉)",
                    language=None,
                )
            st.warning(
                "排查建议：\n"
                "1) Streamlit Advanced settings 选 Python **3.11/3.12**（装 TensorFlow）\n"
                "2) 扫描股票数先设 **50~100**，确认能跑通\n"
                "3) 先只用 B1+LightGBM，确认推荐页会更新，再加 LSTM\n"
                "4) 看日志是否出现 `[DONE] 已写入推荐`"
            )

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
