#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
B1+B2+多模型 量化选股主程序

用法:
  python -u b1_lstm_daily.py              # -u 无缓冲，进度条实时刷新
  python -u b1_lstm_daily.py --max-stocks 200
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Windows terminal: avoid garbled progress bar (GBK vs UTF-8)
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from quant_pipeline import run_pipeline

try:
    from visualize import save_html_report, save_stock_chart
    HAS_VIZ = True
except ImportError:
    HAS_VIZ = False


def main():
    parser = argparse.ArgumentParser(description="B1+B2+多模型 量化选股")
    parser.add_argument("--config", default=str(ROOT / "config" / "b1_settings.yaml"))
    parser.add_argument("--max-stocks", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--skip-backtest-gate", action="store_true")
    parser.add_argument(
        "--models", type=str, default=None,
        help="启用的模型，逗号分隔，如 b1,b2,lstm",
    )
    parser.add_argument(
        "--weights", type=str, default=None,
        help="模型权重（与 --models 顺序对应），如 0.4,0.3,0.3",
    )
    parser.add_argument(
        "--force-refresh", action="store_true", default=None,
        help="强制增量拉取最新行情（默认开启）",
    )
    parser.add_argument(
        "--no-refresh", action="store_true",
        help="跳过行情刷新，直接使用本地缓存",
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    if args.skip_backtest_gate:
        cfg.setdefault("backtest", {})["require_pass_to_recommend"] = False

    if args.models:
        from ensemble_config import MODEL_KEYS
        keys = [k.strip().lower() for k in args.models.split(",") if k.strip()]
        ens = cfg.setdefault("ensemble", {})
        ens["enabled"] = {k: (k in keys) for k in MODEL_KEYS}
        if args.weights:
            ws = [float(x.strip()) for x in args.weights.split(",")]
            if len(ws) != len(keys):
                parser.error("--weights 数量须与 --models 一致")
            ens["weights"] = dict(zip(keys, ws))

    workers = args.workers or cfg.get("data", {}).get("workers", 8)
    force_refresh = True
    if args.no_refresh:
        force_refresh = False
    elif args.force_refresh is True:
        force_refresh = True

    result = run_pipeline(
        cfg, max_stocks=args.max_stocks, workers=workers, force_refresh=force_refresh,
    )

    bt = result.backtest
    print()
    print("=" * 60)
    print("回测摘要（测试集）")
    print("=" * 60)
    print(f"  方向准确率: {bt.direction_accuracy:.2%}")
    print(f"  Rank IC:     {bt.rank_ic:.4f}")
    print(f"  Top-K 胜率:  {bt.top_k_win_rate:.2%}")
    print(f"  累计收益:    {bt.top_k_total_return:.2%}")
    print(f"  最大回撤:    {bt.max_drawdown:.2%}")
    print(f"  LSTM验证IC:  {result.lstm_valid_ic:.4f}")
    print(f"  评估:        {'通过' if result.backtest_passed else '未通过'}")

    print()
    print("=" * 60)
    print(f"Top {len(result.final_top)} 推荐")
    print("=" * 60)
    for _, row in result.final_top.iterrows():
        disp = row.get("display_name") or row.get("name") or row["code"]
        print(
            f"  #{row['rank']} [{row.get('confidence','?')}] {disp} "
            f"B1={row.get('b1_score',0):.0f} B2={row.get('b2_score',0):.0f} "
            f"LSTM={row.get('lstm_pred_pct',0):+.2f}%"
        )
        reason = row.get("buy_reason", "")
        if reason:
            print(f"       {reason[:100]}...")

    if HAS_VIZ and not result.final_top.empty:
        out_dir = ROOT / cfg.get("output", {}).get("dir", "output/b1_lstm")
        try:
            import pandas as pd
            raw_cache = None
            cache_dir = ROOT / "data" / "cache"
            for f in sorted(cache_dir.glob("hist_*.pkl"), reverse=True):
                raw_cache = pd.read_pickle(f)
                break
            if raw_cache is not None:
                save_html_report(
                    result.final_top, raw_cache,
                    out_path=out_dir / "report_visual_latest.html",
                    max_charts=10,
                )
        except Exception:
            pass

    print(f"\n报告: {result.report_paths.get('md')}")
    print(f"回测指标: {result.report_paths.get('metrics')}")
    print("\n免责声明: 仅供学习研究，不构成投资建议。")


if __name__ == "__main__":
    main()
