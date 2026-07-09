#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A股 AI 量化分析主程序

架构参考（GitHub 开源）:
  - 数据: akfamily/akshare          https://github.com/akfamily/akshare
  - 研究平台: microsoft/qlib         https://github.com/microsoft/qlib
  - A股工作台: qlib-research-workbench https://github.com/ioiochen11/qlib-research-workbench
  - 机构思路: hualin6/quant-ashare   https://github.com/hualin6/quant-ashare

用法:
  python ashare_quant.py                    # 默认沪深300，训练+预测+回测
  python ashare_quant.py --universe zz500   # 中证500
  python ashare_quant.py --codes 600519 000858  # 自定义股票
  python ashare_quant.py --max-stocks 50    # 限制下载数量（调试）
  python ashare_quant.py --no-backtest      # 跳过回测

免责声明: 本脚本仅供学习研究，不构成投资建议。股市有风险，投资需谨慎。
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from data_fetcher import fetch_universe_hist, get_st_list, get_universe
from features import latest_feature_row, prepare_panel
from predict_model import predict_scores, train_model
from strategy import build_recommendations, simple_backtest

try:
    from tabulate import tabulate
except ImportError:
    tabulate = None


def load_config(path: Path) -> dict:
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def save_report(rec: pd.DataFrame, out_dir: Path, train_info: dict, bt: dict) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    csv_path = out_dir / f"recommendations_{ts}.csv"
    rec.to_csv(csv_path, index=False, encoding="utf-8-sig")

    md_path = out_dir / f"report_{ts}.md"
    lines = [
        "# A股 AI 量化分析报告",
        "",
        f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 模型表现（验证集）",
        f"- 训练 RMSE: {train_info['train_rmse']:.6f}",
        f"- 验证 RMSE: {train_info['valid_rmse']:.6f}",
        f"- 验证 R²: {train_info['valid_r2']:.4f}",
        "",
        "## 回测摘要（验证集后段，Top-K 等权）",
        f"- 总收益率: {bt.get('total_return', 0)*100:.2f}%",
        f"- 最大回撤: {bt.get('max_drawdown', 0)*100:.2f}%",
        f"- 期末资产: {bt.get('final_equity', 0):,.0f} 元",
        "",
        "> 回测未完全模拟 A 股 T+1、涨跌停等规则，结果仅供参考。",
        "",
        "## 今日 AI 选股推荐",
        "",
    ]

    if tabulate and not rec.empty:
        show = rec[["rank", "code", "close", "pred_return_pct", "direction", "advice"]]
        lines.append(tabulate(show, headers="keys", tablefmt="github", showindex=False))
    else:
        lines.append(rec.to_string(index=False))

    lines += [
        "",
        "## 重要因子（Top 10）",
        "",
        train_info["top_features"].to_string(index=False),
        "",
        "---",
        "**免责声明**: 本报告由机器学习模型自动生成，不构成任何投资建议。",
    ]

    md_path.write_text("\n".join(lines), encoding="utf-8")
    return md_path


def main():
    parser = argparse.ArgumentParser(description="A股 AI 量化分析")
    parser.add_argument("--config", default=str(ROOT / "config" / "settings.yaml"))
    parser.add_argument("--universe", default=None, help="hs300|zz500|custom")
    parser.add_argument("--codes", nargs="*", help="自定义股票代码")
    parser.add_argument("--start", default=None, help="开始日期 YYYY-MM-DD")
    parser.add_argument("--max-stocks", type=int, default=None, help="限制股票数量")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--forward-days", type=int, default=None)
    parser.add_argument("--algorithm", default=None)
    parser.add_argument("--no-backtest", action="store_true")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    strat_cfg = cfg.get("strategy", {})
    bt_cfg = cfg.get("backtest", {})
    out_cfg = cfg.get("output", {})

    universe = args.universe or data_cfg.get("universe", "hs300")
    custom = args.codes or data_cfg.get("custom_codes") or []
    start = (args.start or data_cfg.get("start_date", "2022-01-01")).replace("-", "")
    cache_dir = ROOT / data_cfg.get("cache_dir", "data/cache")
    out_dir = ROOT / out_cfg.get("dir", "output")

    forward_days = args.forward_days or model_cfg.get("forward_days", 5)
    algorithm = args.algorithm or model_cfg.get("algorithm", "lightgbm")
    train_ratio = model_cfg.get("train_ratio", 0.8)
    top_k = args.top_k or strat_cfg.get("top_k", 10)

    print("=" * 55)
    print("A股 AI 量化分析")
    print("=" * 55)
    print(f"股票池: {universe}")
    print(f"算法: {algorithm} | 预测周期: {forward_days} 日")
    print()

    # 1. 股票池
    print("[1/5] 获取股票池...")
    if args.codes:
        codes = [c.zfill(6) for c in args.codes]
        universe = "custom"
    else:
        codes = get_universe(universe, custom)
    if args.max_stocks:
        codes = codes[: args.max_stocks]
    print(f"  共 {len(codes)} 只股票")

    # 2. 下载数据
    print("[2/5] 下载历史行情 (AkShare)...")
    raw = fetch_universe_hist(
        codes, start_date=start, cache_dir=cache_dir, max_stocks=None,
    )
    if raw.empty:
        print("错误: 未获取到行情数据，请检查网络或 akshare 版本。")
        sys.exit(1)
    print(f"  共 {len(raw)} 条日线记录")

    # 3. 因子 + 标签
    print("[3/5] 计算技术因子与训练标签...")
    panel, feature_cols = prepare_panel(
        raw,
        forward_days=forward_days,
        exclude_limit=strat_cfg.get("exclude_limit", True),
    )
    if panel.empty:
        print("错误: 因子计算后无有效样本。")
        sys.exit(1)
    print(f"  有效样本: {len(panel)} | 因子数: {len(feature_cols)}")

    # 4. 训练模型
    print("[4/5] 训练 AI 预测模型...")
    result = train_model(
        panel, feature_cols,
        algorithm=algorithm,
        train_ratio=train_ratio,
    )
    print(f"  训练 RMSE: {result.train_rmse:.6f}")
    print(f"  验证 RMSE: {result.valid_rmse:.6f}")
    print(f"  验证 R2:   {result.valid_r2:.4f}")

    # 5. 预测 + 建议
    print("[5/5] 生成 AI 选股建议...")
    latest = latest_feature_row(panel, feature_cols)
    scores = predict_scores(result.model, latest, feature_cols)

    st_codes = get_st_list() if strat_cfg.get("exclude_st", True) else set()
    rec = build_recommendations(
        scores,
        top_k=top_k,
        min_score=strat_cfg.get("min_score", 0.0),
        st_codes=st_codes,
        exclude_st=strat_cfg.get("exclude_st", True),
    )

    bt = {}
    if not args.no_backtest:
        print("  运行简易回测...")
        bt = simple_backtest(
            panel, feature_cols, result.model,
            top_k=top_k,
            initial_cash=bt_cfg.get("initial_cash", 1_000_000),
            commission=bt_cfg.get("commission", 0.0003),
            stamp_tax=bt_cfg.get("stamp_tax", 0.001),
            slippage=bt_cfg.get("slippage", 0.001),
        )
        print(f"  回测总收益: {bt['total_return']*100:.2f}% | 最大回撤: {bt['max_drawdown']*100:.2f}%")

    train_info = {
        "train_rmse": result.train_rmse,
        "valid_rmse": result.valid_rmse,
        "valid_r2": result.valid_r2,
        "top_features": result.feature_importance.head(10),
    }
    report_path = save_report(rec, out_dir, train_info, bt)

    print()
    print("=" * 55)
    print("AI 选股 Top 推荐")
    print("=" * 55)
    if not rec.empty:
        for _, row in rec.iterrows():
            print(
                f"  #{row['rank']} {row['code']} "
                f"现价 {row['close']:.2f} | "
                f"预测{forward_days}日收益 {row['pred_return_pct']:+.2f}% | "
                f"{row['direction']} | {row['advice']}"
            )
    else:
        print("  暂无满足条件的推荐。")

    print()
    print(f"报告已保存: {report_path}")
    print()
    print("参考开源项目（可进一步扩展）:")
    print("  - Qlib:     https://github.com/microsoft/qlib")
    print("  - AkShare:  https://github.com/akfamily/akshare")
    print("  - Workbench: https://github.com/ioiochen11/qlib-research-workbench")
    print()
    print("免责声明: 仅供学习研究，不构成投资建议。")


if __name__ == "__main__":
    main()
