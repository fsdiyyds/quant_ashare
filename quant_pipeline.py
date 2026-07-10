"""
量化选股完整流水线：训练 → 回测验证 → 通过后输出推荐
支持原生 B1/B2/LGB/LSTM + Qlib 风格模型 Zoo 选取与加权组合。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from backtest_engine import BacktestMetrics, format_backtest_report, run_backtest
from b1_selector import screen_b1_universe
from data_fetcher import fetch_universe_hist_parallel, get_all_a_codes, get_st_list
from ensemble_config import ModelConfig
from ensemble_model import (
    TABULAR_FEATURES,
    build_historical_panel,
    build_latest_snapshot,
    ensemble_score,
    predict_lgb,
    train_lightgbm,
)
from lstm_model import (
    HAS_TF,
    build_panel_sequences,
    lstm_backtest_metrics,
    predict_latest_lstm,
    predict_lstm_historical,
    predict_lstm_panel,
    train_lstm,
)
from progress_utils import StepProgress
from qlib_models import (
    predict_qlib_seq_latest,
    predict_qlib_tabular,
    train_qlib_seq,
    train_qlib_tabular,
)
from recommend import enrich_with_reasons


@dataclass
class PipelineResult:
    b1_pool: pd.DataFrame
    final_top: pd.DataFrame
    backtest: BacktestMetrics
    lstm_valid_ic: float
    backtest_passed: bool
    report_paths: dict


def _split_panel(panel, train_ratio=0.6, valid_ratio=0.15):
    dates = sorted(panel["date"].unique())
    n = len(dates)
    t1 = int(n * train_ratio)
    t2 = int(n * (train_ratio + valid_ratio))
    return (
        panel[panel["date"].isin(set(dates[:t1]))],
        panel[panel["date"].isin(set(dates[t1:t2]))],
        panel[panel["date"].isin(set(dates[t2:]))],
    )


def _build_candidate_codes(
    raw: pd.DataFrame,
    model_cfg: ModelConfig,
    b1_pool: pd.DataFrame,
    b1_top: int,
    min_score: float,
) -> list:
    """按启用的规则模型决定候选股票池。"""
    if model_cfg.enabled.get("b1", False) and not b1_pool.empty:
        return b1_pool["code"].tolist()

    codes = []
    for code, sub in raw.groupby("code"):
        if len(sub) < 70:
            continue
        codes.append(code)
    return codes[:b1_top] if b1_top else codes


def run_pipeline(
    cfg: dict,
    max_stocks: Optional[int] = None,
    workers: int = 8,
    force_refresh: bool = True,
) -> PipelineResult:
    b1_cfg = cfg.get("b1", {})
    b2_cfg = cfg.get("b2", {})
    lstm_cfg = cfg.get("lstm", {})
    bt_cfg = cfg.get("backtest", {})
    ens_cfg = cfg.get("ensemble", {})
    data_cfg = cfg.get("data", {})
    out_cfg = cfg.get("output", {})

    b1_params = {
        "j_threshold": b1_cfg.get("j_threshold", 15),
        "j_best": b1_cfg.get("j_best", 13),
        "price_range_pct": b1_cfg.get("price_range_pct", 1.0),
    }
    b2_params = {
        "up_pct_threshold": b2_cfg.get("up_pct_threshold", 0.05),
        "vol_multiple": b2_cfg.get("vol_multiple", 1.5),
        "lookback_n": b2_cfg.get("lookback_n", 10),
        "j_threshold": b2_cfg.get("j_threshold", 15),
    }
    model_cfg = ModelConfig.from_cfg(ens_cfg)
    norm_weights = model_cfg.normalized_weights()
    if not model_cfg.active_keys():
        raise ValueError("请至少启用一个模型（B1 / B2 / LightGBM / LSTM / Qlib 模型）")

    # 配置可覆盖 force_refresh；默认每次运行刷新最新行情
    if "force_refresh" in data_cfg:
        force_refresh = bool(data_cfg["force_refresh"])

    print(f"  模型组合: {model_cfg.summary()}", flush=True)
    thresholds = {
        "min_direction_accuracy": bt_cfg.get("min_direction_accuracy", 0.52),
        "min_rank_ic": bt_cfg.get("min_rank_ic", 0.02),
        "min_top_k_win_rate": bt_cfg.get("min_top_k_win_rate", 0.45),
    }
    forward_days = lstm_cfg.get("forward_days", 5)
    seq_len = lstm_cfg.get("seq_len", 30)
    top_k = lstm_cfg.get("top_k", 50)
    b1_top = b1_cfg.get("top_n", 520)
    start = data_cfg.get("start_date", "2023-01-01").replace("-", "")
    cache_dir = Path(__file__).resolve().parent / data_cfg.get("cache_dir", "data/cache")
    out_dir = Path(__file__).resolve().parent / out_cfg.get("dir", "output/b1_lstm")

    # 云端配置里的 max_stocks 接入
    if max_stocks is None and data_cfg.get("max_stocks"):
        max_stocks = int(data_cfg["max_stocks"])

    step_names = ["获取股票池与最新行情"]
    step_names.append("构建历史训练面板")
    if model_cfg.needs_lgb():
        step_names.append("训练 LightGBM")
    if model_cfg.needs_qlib_tabular():
        step_names.append("训练 Qlib 表格模型")
    if model_cfg.needs_lstm():
        step_names.append("训练 LSTM")
    if model_cfg.needs_qlib_seq():
        step_names.append("训练 Qlib 时序模型")
    step_names.extend(["测试集回测", "生成今日推荐"])

    steps = StepProgress(step_names, title="量化选股（训练→回测→推荐）")

    # 1. 数据（默认强制增量刷新最新成交）
    steps.start(1)
    codes = get_all_a_codes(
        exclude_gem=b1_cfg.get("exclude_gem", False),
        exclude_star=b1_cfg.get("exclude_star", False),
    )
    if max_stocks:
        codes = codes[:max_stocks]
    print(f"  股票池: {len(codes)} 只 | 刷新最新数据={force_refresh}", flush=True)

    raw = fetch_universe_hist_parallel(
        codes, start_date=start, cache_dir=cache_dir, workers=workers,
        force_refresh=force_refresh,
    )
    if raw.empty:
        raise RuntimeError("未获取到行情数据")
    latest_dt = pd.to_datetime(raw["date"]).max()
    steps.done(f"{len(raw)} 条记录, {raw['code'].nunique()} 只股票, 最新日={latest_dt:%Y-%m-%d}")

    # 2. 历史面板
    min_b1 = b1_cfg.get("min_score", 30) if model_cfg.enabled.get("b1", False) else 0
    steps.start(2, "B1+B2+标签，较慢请耐心等待")
    panel = build_historical_panel(
        raw, b1_params, b2_params, forward_days=forward_days,
        min_b1_score=min_b1,
    )
    needs_ml = (
        model_cfg.needs_lgb() or model_cfg.needs_lstm()
        or model_cfg.needs_qlib_tabular() or model_cfg.needs_qlib_seq()
    )
    min_samples = 100 if not needs_ml else 500
    if len(panel) < min_samples:
        raise RuntimeError(f"训练样本不足: {len(panel)}（至少需要 {min_samples}）")

    train, valid, test = _split_panel(
        panel,
        train_ratio=bt_cfg.get("train_ratio", 0.6),
        valid_ratio=bt_cfg.get("valid_ratio", 0.15),
    )
    steps.done(f"训练 {len(train)} | 验证 {len(valid)} | 测试 {len(test)}")

    step_idx = 3
    lgb_model = None
    if model_cfg.needs_lgb():
        steps.start(step_idx)
        step_idx += 1
        lgb_model = train_lightgbm(panel, TABULAR_FEATURES)
        steps.done("LightGBM 完成")

    qlib_bundle = None
    if model_cfg.needs_qlib_tabular():
        steps.start(step_idx)
        step_idx += 1
        qlib_bundle = train_qlib_tabular(
            panel, TABULAR_FEATURES, model_cfg.active_qlib_tabular(),
            train_ratio=bt_cfg.get("train_ratio", 0.6),
            valid_ratio=bt_cfg.get("valid_ratio", 0.15),
        )
        steps.done(f"已训练: {list(qlib_bundle.tabular.keys())}")

    lstm_valid_ic = 0.0
    lstm_model = None
    if model_cfg.needs_lstm():
        steps.start(step_idx)
        step_idx += 1
        if HAS_TF:
            try:
                b1_codes = panel["code"].unique().tolist()
                X, y = build_panel_sequences(raw, codes=b1_codes, seq_len=seq_len, forward_days=forward_days)
                print(f"  LSTM 序列样本: {len(X)}", flush=True)
                if len(X) >= 100:
                    lstm_result = train_lstm(
                        X, y, seq_len=seq_len,
                        epochs=lstm_cfg.get("epochs", 20),
                        batch_size=lstm_cfg.get("batch_size", 64),
                    )
                    lstm_model = lstm_result.model
                    lstm_valid_ic = lstm_result.valid_ic
                    steps.done(f"验证 IC={lstm_valid_ic:.4f}")
                else:
                    steps.done(f"样本不足 ({len(X)})，跳过 LSTM")
            except Exception as e:
                print(f"  LSTM 训练失败（将继续其他模型）: {e}", flush=True)
                steps.done(f"失败已跳过: {e}")
        else:
            print("  警告: 未安装 tensorflow，LSTM 权重将不生效", flush=True)
            steps.done("未安装 tensorflow，跳过")

    qlib_seq_models = {}
    if model_cfg.needs_qlib_seq():
        steps.start(step_idx)
        step_idx += 1
        if HAS_TF:
            try:
                seq_codes = panel["code"].unique().tolist()
                X, y = build_panel_sequences(raw, codes=seq_codes, seq_len=seq_len, forward_days=forward_days)
                print(f"  Qlib时序 序列样本: {len(X)}", flush=True)
                if len(X) >= 100:
                    qlib_seq_models = train_qlib_seq(
                        X, y, model_cfg.active_qlib_seq(),
                        seq_len=seq_len,
                        epochs=lstm_cfg.get("epochs", 15),
                        batch_size=lstm_cfg.get("batch_size", 128),
                    )
                    steps.done(f"已训练: {list(qlib_seq_models.keys())}")
                else:
                    steps.done(f"样本不足 ({len(X)})，跳过")
            except Exception as e:
                print(f"  Qlib时序训练失败（将继续）: {e}", flush=True)
                steps.done(f"失败已跳过: {e}")
        else:
            print("  警告: 未安装 tensorflow，Qlib 时序模型跳过", flush=True)
            steps.done("未安装 tensorflow，跳过")

    # 回测：组装测试集预测
    steps.start(step_idx)
    step_idx += 1
    lstm_preds = None
    if model_cfg.needs_lstm() and lstm_model is not None:
        lstm_preds = predict_lstm_panel(
            lstm_model, raw, test, seq_len=seq_len, forward_days=forward_days,
        )
        if not lstm_preds.empty:
            merged = test.merge(lstm_preds, on=["date", "code"], how="inner")
            if len(merged) > 10:
                lstm_ic = float(np.corrcoef(merged["lstm_pred"], merged["label"])[0, 1])
                lstm_dir = float(((merged["lstm_pred"] > 0) == (merged["label"] > 0)).mean())
                print(f"  LSTM test IC={lstm_ic:.4f} dir_acc={lstm_dir:.1%}", flush=True)

    extra_test_preds = {}
    if qlib_bundle is not None and qlib_bundle.tabular:
        tab_preds = predict_qlib_tabular(qlib_bundle, test)
        for k, arr in tab_preds.items():
            extra_test_preds[k] = arr
            test[f"{k}_pred"] = arr

    if qlib_seq_models:
        # 时序模型在测试集上用截面近似：按 code 最新序列预测填入（简化）
        seq_latest = predict_qlib_seq_latest(
            qlib_seq_models, raw, test["code"].unique().tolist(), seq_len=seq_len,
        )
        for k, sdf in seq_latest.items():
            if sdf.empty:
                continue
            col = f"{k}_pred"
            merged = test.merge(sdf, on="code", how="left")
            test[col] = merged[col].fillna(0).values
            extra_test_preds[k] = test[col].values

    bt = run_backtest(
        test,
        lgb_model=lgb_model if model_cfg.needs_lgb() else None,
        lstm_preds=lstm_preds,
        feature_cols=TABULAR_FEATURES,
        top_k=min(top_k, 20),
        weights=norm_weights,
        thresholds=thresholds,
        extra_preds=extra_test_preds or None,
    )
    steps.done(
        f"准确率={bt.direction_accuracy:.1%} IC={bt.rank_ic:.4f} "
        f"{'通过' if bt.passed else '未通过'}"
    )

    if bt_cfg.get("require_pass_to_recommend", True) and not bt.passed:
        print("  提示: 回测未达门槛，推荐将标记为「低置信度」", flush=True)

    steps.start(step_idx)
    st_codes = get_st_list() if b1_cfg.get("exclude_st", True) else set()
    b1_pool = screen_b1_universe(
        raw, top_n=b1_top, min_score=b1_cfg.get("min_score", 35),
        params=b1_params, st_codes=st_codes,
    )
    candidate_codes = _build_candidate_codes(
        raw, model_cfg, b1_pool, b1_top, b1_cfg.get("min_score", 35),
    )
    if not candidate_codes:
        raise RuntimeError("候选股票池为空，请调整模型组合或扫描范围")

    snapshot = build_latest_snapshot(raw, candidate_codes, b1_params, b2_params)
    if model_cfg.needs_lgb() and lgb_model is not None:
        snapshot["lgb_pred"] = predict_lgb(lgb_model, snapshot, TABULAR_FEATURES)
    else:
        snapshot["lgb_pred"] = 0.0

    if model_cfg.needs_lstm() and lstm_model is not None:
        lstm_pred = predict_latest_lstm(lstm_model, raw, snapshot["code"].tolist(), seq_len=seq_len)
        snapshot = snapshot.merge(
            lstm_pred[["code", "lstm_pred_return"]].rename(columns={"lstm_pred_return": "lstm_pred"}),
            on="code", how="left",
        )
        snapshot["lstm_pred"] = snapshot["lstm_pred"].fillna(0)
    else:
        snapshot["lstm_pred"] = 0.0

    extra_snap = {}
    if qlib_bundle is not None and qlib_bundle.tabular:
        tab_preds = predict_qlib_tabular(qlib_bundle, snapshot)
        for k, arr in tab_preds.items():
            snapshot[f"{k}_pred"] = arr
            extra_snap[k] = arr

    if qlib_seq_models:
        seq_latest = predict_qlib_seq_latest(
            qlib_seq_models, raw, snapshot["code"].tolist(), seq_len=seq_len,
        )
        for k, sdf in seq_latest.items():
            col = f"{k}_pred"
            if sdf.empty:
                snapshot[col] = 0.0
            else:
                snapshot = snapshot.merge(sdf, on="code", how="left")
                snapshot[col] = snapshot[col].fillna(0)
            extra_snap[k] = snapshot[col].values

    lgb_arr = snapshot["lgb_pred"].values if model_cfg.needs_lgb() else None
    lstm_arr = snapshot["lstm_pred"].values if model_cfg.needs_lstm() else None
    snapshot["ensemble_score"] = ensemble_score(
        snapshot, lgb_pred=lgb_arr, lstm_pred=lstm_arr,
        weights=norm_weights, extra_preds=extra_snap or None,
    )
    snapshot = snapshot.sort_values("ensemble_score", ascending=False).reset_index(drop=True)
    snapshot["rank"] = range(1, len(snapshot) + 1)
    snapshot["lstm_pred_pct"] = (snapshot["lstm_pred"] * 100).round(2)
    snapshot["backtest_passed"] = bt.passed
    snapshot["model_mix"] = model_cfg.summary()
    snapshot["data_asof"] = str(latest_dt.date())
    snapshot["confidence"] = snapshot.apply(
        lambda r: "高" if bt.passed and r["ensemble_score"] > 0.6 else ("中" if bt.passed else "低"),
        axis=1,
    )

    final = enrich_with_reasons(
        snapshot.head(top_k), raw, b1_params=b1_params, forward_days=forward_days,
    )
    steps.done(f"Top {len(final)} 推荐 | 数据截至 {latest_dt:%Y-%m-%d}")

    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    final.to_csv(out_dir / f"top50_buy_{ts}.csv", index=False, encoding="utf-8-sig")
    b1_pool.to_csv(out_dir / f"b1_pool_{ts}.csv", index=False, encoding="utf-8-sig")

    bt_report = format_backtest_report(bt, thresholds)
    md_lines = [
        "# 量化选股报告",
        f"\n生成时间: {datetime.now():%Y-%m-%d %H:%M:%S}",
        f"\n行情截至: {latest_dt:%Y-%m-%d}",
        f"\n模型组合: {model_cfg.summary()}\n",
        bt_report,
        f"\nLSTM 验证 IC: {lstm_valid_ic:.4f}",
        f"\n回测门槛: {'通过' if bt.passed else '未通过'} | 推荐置信度: 见 confidence 列\n",
        "## Top 推荐\n",
        final[["rank", "display_name", "b1_score", "b2_score", "lstm_pred_pct", "confidence", "buy_reason"]].to_string(index=False),
        "\n\n免责声明: 仅供学习研究，不构成投资建议。",
    ]
    report_md = out_dir / f"report_{ts}.md"
    report_md.write_text("\n".join(md_lines), encoding="utf-8")

    bt_json = out_dir / "backtest_metrics.json"
    bt_json.write_text(json.dumps({
        **bt.to_dict(),
        "lstm_valid_ic": lstm_valid_ic,
        "thresholds": thresholds,
        "model_config": model_cfg.to_dict(),
        "data_asof": str(latest_dt.date()),
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    latest = Path(__file__).resolve().parent / "output" / "latest"
    latest.mkdir(parents=True, exist_ok=True)

    # 先写推荐结果，避免后续 LSTM 存模型/逐股回测 OOM 导致推荐页不更新
    final.to_csv(latest / "top50_latest.csv", index=False, encoding="utf-8-sig")
    b1_pool.to_csv(latest / "b1_pool_latest.csv", index=False, encoding="utf-8-sig")
    import shutil
    shutil.copy(report_md, latest / "report_latest.md")
    shutil.copy(bt_json, latest / "backtest_metrics.json")
    (latest / "model_config.json").write_text(
        json.dumps(model_cfg.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8",
    )
    (latest / "data_asof.txt").write_text(str(latest_dt.date()), encoding="utf-8")
    print(
        f"[DONE] 已写入推荐: {latest / 'top50_latest.csv'} "
        f"({len(final)} 只) model={model_cfg.summary()}",
        flush=True,
    )

    if lstm_model is not None:
        try:
            print("  保存 LSTM 模型...", flush=True)
            lstm_model.save(str(latest / "lstm_model.keras"))
        except Exception as e:
            print(f"  保存 keras 失败: {e}", flush=True)
            try:
                lstm_model.save(str(latest / "lstm_model.h5"))
            except Exception as e2:
                print(f"  保存 h5 失败: {e2}", flush=True)

        try:
            # Cloud 内存有限：只对 Top5 做轻量历史回测
            n_bt = min(5, len(final))
            print(f"  生成 LSTM 历史回测图数据 (Top{n_bt})...", flush=True)
            bt_hist_parts = []
            for code in final["code"].head(n_bt).astype(str).str.zfill(6):
                sub = raw[raw["code"] == code]
                h = predict_lstm_historical(
                    lstm_model, sub, seq_len=seq_len, forward_days=forward_days, max_points=40,
                )
                if not h.empty:
                    h["code"] = code
                    bt_hist_parts.append(h)
            if bt_hist_parts:
                bt_hist = pd.concat(bt_hist_parts, ignore_index=True)
                bt_hist.to_csv(latest / "lstm_backtest_history.csv", index=False, encoding="utf-8-sig")
                summary = lstm_backtest_metrics(bt_hist)
                (latest / "lstm_backtest_summary.json").write_text(
                    json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8",
                )
                print(f"  LSTM 回测历史已写入 ({len(bt_hist)} 点)", flush=True)
        except Exception as e:
            print(f"  LSTM 历史回测跳过（不影响推荐）: {e}", flush=True)

    print("[DONE] pipeline finished OK", flush=True)
    return PipelineResult(
        b1_pool=b1_pool,
        final_top=final,
        backtest=bt,
        lstm_valid_ic=lstm_valid_ic,
        backtest_passed=bt.passed,
        report_paths={"md": str(report_md), "metrics": str(bt_json)},
    )
