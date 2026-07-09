"""Terminal progress bar (ASCII-only, Windows GBK safe)."""

from __future__ import annotations

import sys
import time

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# Windows console often uses GBK; keep labels ASCII-only.
DESC_EN = {
    "下载行情": "download",
    "  下载行情": "download",
    "构建训练面板": "build-panel",
    "  构建训练面板": "build-panel",
    "最新截面": "snapshot",
    "  最新截面": "snapshot",
    "生成推荐理由": "reasons",
    "  生成推荐理由": "reasons",
    "回测逐日": "backtest-day",
    "  回测逐日": "backtest-day",
    "LSTM序列": "lstm-seq",
    "  LSTM序列": "lstm-seq",
    "LSTM预测": "lstm-predict",
    "  LSTM预测": "lstm-predict",
    "LSTM回测": "lstm-backtest",
    "  LSTM回测": "lstm-backtest",
    "B1评分扫描": "b1-scan",
    "  B1评分扫描": "b1-scan",
}

UNIT_EN = {"股": "stk", "日": "day", "it": "it"}


def _ascii_label(text: str, default: str = "") -> str:
    raw = text or ""
    if raw in DESC_EN:
        return DESC_EN[raw]
    stripped = raw.strip()
    if stripped in DESC_EN:
        return DESC_EN[stripped]
    safe = "".join(ch if ord(ch) < 128 else " " for ch in raw)
    return " ".join(safe.split()) or default


def _ascii_unit(unit: str) -> str:
    return UNIT_EN.get(unit, _ascii_label(unit, "it"))


class ProgressBar:
    """Single-task progress bar."""

    def __init__(self, total: int, desc: str = "", unit: str = "it"):
        self.total = max(int(total), 1)
        self.desc = _ascii_label(desc, "progress")
        self.unit = _ascii_unit(unit)
        self.n = 0
        self._tqdm = None
        self._last_print = 0.0

        if HAS_TQDM:
            self._tqdm = tqdm(
                total=self.total,
                desc=self.desc,
                unit=self.unit,
                ncols=88,
                file=sys.stdout,
                leave=True,
                ascii=True,
                bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
            )
        else:
            print(f"  [{self.desc}] 0/{self.total} 0%", flush=True)

    def update(self, n: int = 1, postfix: str = "") -> None:
        self.n = min(self.n + n, self.total)
        pf = _ascii_label(postfix, "") if postfix else ""
        if self._tqdm is not None:
            self._tqdm.update(n)
            if pf:
                self._tqdm.set_postfix_str(pf, refresh=True)
        else:
            now = time.time()
            if now - self._last_print >= 0.3 or self.n >= self.total:
                self._last_print = now
                pct = int(self.n / self.total * 100)
                bar_len = 30
                filled = int(bar_len * self.n / self.total)
                bar = "#" * filled + "-" * (bar_len - filled)
                msg = f"  [{self.desc}] [{bar}] {self.n}/{self.total} ({pct}%)"
                if pf:
                    msg += f" {pf}"
                print(msg, flush=True)

    def close(self, final_msg: str = "") -> None:
        if self._tqdm is not None:
            if final_msg:
                self._tqdm.set_postfix_str(_ascii_label(final_msg, ""), refresh=True)
            self._tqdm.close()
        elif final_msg:
            print(f"  [{self.desc}] done -- {_ascii_label(final_msg, final_msg)}", flush=True)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


STEP_EN = {
    "获取股票池与行情": "fetch-data",
    "构建历史训练面板": "build-panel",
    "训练 LightGBM": "train-lgb",
    "训练 LSTM": "train-lstm",
    "测试集回测": "backtest",
    "生成今日推荐": "recommend",
}


class StepProgress:
    """Multi-step progress (e.g. 1/6, 2/6)."""

    def __init__(self, steps: list, title: str = ""):
        self.steps = steps
        self.total = len(steps)
        self.title = title
        self.current = 0
        if title:
            print("=" * 60, flush=True)
            print("quant-pipeline", flush=True)
            print("=" * 60, flush=True)

    def start(self, idx: int, detail: str = "") -> None:
        self.current = idx
        name = self.steps[idx - 1] if 0 < idx <= self.total else ""
        en = STEP_EN.get(name, _ascii_label(name, name))
        line = f"\n[{idx}/{self.total}] {en or 'step'}"
        if detail:
            d = _ascii_label(detail, "")
            if d:
                line += f" | {d}"
        print(line, flush=True)

    def done(self, msg: str = "") -> None:
        if msg:
            print(f"  [OK] {_ascii_label(msg, msg)}", flush=True)


def iter_progress(items, desc: str = "", unit: str = "it"):
    """Iterable progress wrapper."""
    if HAS_TQDM:
        yield from tqdm(
            items, desc=_ascii_label(desc, "progress"), unit=_ascii_unit(unit),
            ncols=88, file=sys.stdout, ascii=True,
        )
    else:
        total = len(items) if hasattr(items, "__len__") else None
        bar = ProgressBar(total or 1, desc=desc, unit=unit)
        for i, item in enumerate(items, 1):
            if total:
                bar.n = i - 1
                bar.update(1)
            yield item
        bar.close()
