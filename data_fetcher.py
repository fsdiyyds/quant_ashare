"""A股行情数据获取（多数据源：AkShare / 新浪 / 东方财富，自动降级 + 本地缓存）。"""

from __future__ import annotations

import os
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd

try:
    import requests
except ImportError:
    requests = None  # type: ignore

try:
    import akshare as ak
    HAS_AK = True
except ImportError:
    HAS_AK = False

# 数据源优先级: akshare > sina > eastmoney（可通过环境变量覆盖）
# 本地网络不通东方财富时，自动使用新浪
DATA_SOURCE = os.environ.get("QUANT_DATA_SOURCE", "auto").lower()

_HTTP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Referer": "https://finance.sina.com.cn/",
}


def _today_str() -> str:
    return datetime.now().strftime("%Y%m%d")


def _norm_code(code: str) -> str:
    return str(code).strip().zfill(6)


def _sina_symbol(code: str) -> str:
    code = _norm_code(code)
    return f"sh{code}" if code.startswith(("5", "6", "9")) else f"sz{code}"


def _cache_root() -> Path:
    root = Path(__file__).resolve().parent / "data" / "cache"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _load_code_cache(name: str = "all_a_codes.txt") -> List[str]:
    path = _cache_root() / name
    if path.exists():
        codes = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if codes:
            return codes
    return []


def _save_code_cache(codes: List[str], name: str = "all_a_codes.txt") -> None:
    path = _cache_root() / name
    path.write_text("\n".join(codes), encoding="utf-8")


def _fetch_all_a_sina() -> List[str]:
    """新浪 A 股列表（沪深，不含北交所）。"""
    if requests is None:
        return []

    url = (
        "http://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/"
        "Market_Center.getHQNodeData"
    )
    codes: List[str] = []
    page = 1
    while page <= 80:
        try:
            resp = requests.get(
                url,
                params={
                    "page": page, "num": 80, "sort": "symbol", "asc": 1,
                    "node": "hs_a", "symbol": "", "_s_r_a": "init",
                },
                headers=_HTTP_HEADERS,
                timeout=20,
            )
            resp.raise_for_status()
            data = resp.json()
            if not data:
                break
            for item in data:
                sym = item.get("symbol", "")
                if sym.startswith("sh") or sym.startswith("sz"):
                    codes.append(_norm_code(sym[2:]))
            if len(data) < 80:
                break
            page += 1
            time.sleep(0.12)
        except Exception:
            break

    # 去重保序
    seen = set()
    out = []
    for c in codes:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def _fetch_hist_sina(
    code: str,
    start_date: str = "20220101",
    end_date: Optional[str] = None,
    datalen: int = 1023,
) -> pd.DataFrame:
    """新浪日线（前复权由接口内部处理，datalen 最大约 1023）。"""
    if requests is None:
        return pd.DataFrame()

    start_date = start_date.replace("-", "")
    end_date = (end_date or _today_str()).replace("-", "")
    url = (
        "https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/"
        "CN_MarketData.getKLineData"
    )

    for attempt in range(3):
        try:
            resp = requests.get(
                url,
                params={
                    "symbol": _sina_symbol(code),
                    "scale": 240,
                    "ma": "no",
                    "datalen": datalen,
                },
                headers=_HTTP_HEADERS,
                timeout=20,
            )
            resp.raise_for_status()
            data = resp.json()
            if not data:
                return pd.DataFrame()

            rows = []
            for bar in data:
                day = bar.get("day", "").replace("-", "")
                if not day:
                    continue
                if day < start_date or day > end_date:
                    continue
                close = float(bar["close"])
                open_ = float(bar["open"])
                rows.append({
                    "date": bar["day"],
                    "open": open_,
                    "close": close,
                    "high": float(bar["high"]),
                    "low": float(bar["low"]),
                    "volume": float(bar["volume"]),
                    "amount": 0.0,
                    "pct_chg": (close / open_ - 1) * 100 if open_ else 0.0,
                    "turnover": 0.0,
                    "code": _norm_code(code),
                })

            if not rows:
                return pd.DataFrame()
            df = pd.DataFrame(rows)
            df["date"] = pd.to_datetime(df["date"])
            return df.sort_values("date").reset_index(drop=True)
        except Exception:
            time.sleep(0.3 * (attempt + 1))
    return pd.DataFrame()


def _fetch_hist_em(
    code: str,
    start_date: str = "20220101",
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    if requests is None:
        raise ImportError("请安装 requests")

    end_date = (end_date or _today_str()).replace("-", "")
    start_date = start_date.replace("-", "")
    code = _norm_code(code)
    secid = f"1.{code}" if code.startswith(("5", "6", "9")) else f"0.{code}"

    url = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
    params = {
        "secid": secid,
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
        "klt": "101", "fqt": "1", "beg": start_date, "end": end_date,
    }
    resp = requests.get(url, params=params, headers=_HTTP_HEADERS, timeout=15)
    resp.raise_for_status()
    klines = (resp.json().get("data") or {}).get("klines") or []
    if not klines:
        return pd.DataFrame()

    rows = []
    for line in klines:
        parts = line.split(",")
        if len(parts) < 11:
            continue
        rows.append({
            "date": parts[0],
            "open": float(parts[1]), "close": float(parts[2]),
            "high": float(parts[3]), "low": float(parts[4]),
            "volume": float(parts[5]), "amount": float(parts[6]),
            "pct_chg": float(parts[8]) if parts[8] not in ("", "-") else 0.0,
            "turnover": float(parts[10]) if parts[10] not in ("", "-") else 0.0,
            "code": code,
        })
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def _default_demo_codes() -> List[str]:
    return [
        "600519", "000858", "601318", "600036", "000333",
        "601166", "600900", "000001", "601888", "300750",
    ]


def get_all_a_codes(exclude_gem: bool = False, exclude_star: bool = False) -> List[str]:
    """获取全 A 股代码，多源降级 + 本地缓存。"""
    cached = _load_code_cache()
    codes: List[str] = []

    if DATA_SOURCE in ("auto", "akshare") and HAS_AK:
        try:
            df = ak.stock_info_a_code_name()
            codes = [_norm_code(c) for c in df["code"].tolist()]
        except Exception:
            pass

    if not codes and DATA_SOURCE in ("auto", "sina"):
        codes = _fetch_all_a_sina()

    if not codes and DATA_SOURCE in ("auto", "eastmoney") and requests:
        url = "https://push2.eastmoney.com/api/qt/clist/get"
        for page in range(1, 60):
            try:
                resp = requests.get(
                    url,
                    params={
                        "pn": page, "pz": 100, "po": 1, "np": 1,
                        "fltt": 2, "invt": 2, "fid": "f12",
                        "fs": "m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23",
                        "fields": "f12",
                    },
                    headers=_HTTP_HEADERS,
                    timeout=15,
                )
                diff = (resp.json().get("data") or {}).get("diff") or []
                if not diff:
                    break
                codes.extend(_norm_code(item["f12"]) for item in diff if item.get("f12"))
                if len(diff) < 100:
                    break
                time.sleep(0.15)
            except Exception:
                break

    if not codes:
        codes = cached or _default_demo_codes()
        if not cached:
            print("  警告: 在线获取股票列表失败，使用内置示例池（请部署到云端或检查网络）")
    else:
        _save_code_cache(codes)

    if exclude_gem:
        codes = [c for c in codes if not c.startswith(("300", "301"))]
    if exclude_star:
        codes = [c for c in codes if not c.startswith("688")]
    return codes


def fetch_daily_hist(
    code: str,
    start_date: str = "20220101",
    end_date: Optional[str] = None,
    adjust: str = "qfq",
    sleep: float = 0.15,
) -> pd.DataFrame:
    end_date = end_date or _today_str()
    start_date = start_date.replace("-", "")
    end_date = end_date.replace("-", "")
    code = _norm_code(code)

    sources = []
    if DATA_SOURCE == "sina":
        sources = ["sina"]
    elif DATA_SOURCE == "eastmoney":
        sources = ["eastmoney"]
    elif DATA_SOURCE == "akshare":
        sources = ["akshare", "sina", "eastmoney"]
    else:
        sources = ["akshare", "sina", "eastmoney"]

    for src in sources:
        if src == "akshare" and HAS_AK:
            try:
                df = ak.stock_zh_a_hist(
                    symbol=code, period="daily",
                    start_date=start_date, end_date=end_date, adjust=adjust,
                )
                if df is not None and not df.empty:
                    df = df.rename(columns={
                        "日期": "date", "开盘": "open", "收盘": "close",
                        "最高": "high", "最低": "low", "成交量": "volume",
                        "成交额": "amount", "涨跌幅": "pct_chg", "换手率": "turnover",
                    })
                    df["code"] = code
                    df["date"] = pd.to_datetime(df["date"])
                    return df.sort_values("date").reset_index(drop=True)
            except Exception:
                time.sleep(sleep)

        if src == "sina":
            df = _fetch_hist_sina(code, start_date, end_date)
            if not df.empty:
                return df
            time.sleep(sleep * 0.5)

        if src == "eastmoney":
            try:
                df = _fetch_hist_em(code, start_date, end_date)
                if not df.empty:
                    return df
            except Exception:
                time.sleep(sleep)

    return pd.DataFrame()


def _hist_cache_path(
    cache_dir: Path,
    codes: List[str],
    start_date: str,
    end_date: Optional[str],
) -> Path:
    tag = f"all_{start_date}_{end_date or 'latest'}_{len(codes)}"
    return cache_dir / f"hist_{tag}.pkl"


def _merge_hist(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    if old is None or old.empty:
        return new
    if new is None or new.empty:
        return old
    out = pd.concat([old, new], ignore_index=True)
    out["code"] = out["code"].astype(str).str.zfill(6)
    out["date"] = pd.to_datetime(out["date"])
    out = out.drop_duplicates(subset=["code", "date"], keep="last")
    return out.sort_values(["code", "date"]).reset_index(drop=True)


def fetch_universe_hist_parallel(
    codes: List[str],
    start_date: str,
    end_date: Optional[str] = None,
    cache_dir: Optional[Path] = None,
    workers: int = 8,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """并行拉取行情。

    force_refresh=True 时：若已有缓存则做增量更新（从缓存最大日期起补齐到今天），
    保证每次运行都能并入最新成交日数据。
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    cache_dir = Path(cache_dir) if cache_dir else None
    cache_file = None
    cached: Optional[pd.DataFrame] = None

    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = _hist_cache_path(cache_dir, codes, start_date, end_date)
        if cache_file.exists():
            print(f"  使用缓存: {cache_file.name}", flush=True)
            cached = pd.read_pickle(cache_file)
            if not force_refresh:
                return cached

    fetch_start = start_date
    if force_refresh and cached is not None and not cached.empty:
        max_dt = pd.to_datetime(cached["date"]).max()
        # 从倒数第 3 个交易日起重拉，覆盖可能未收盘/修正的数据
        fetch_start = (max_dt - pd.Timedelta(days=5)).strftime("%Y%m%d")
        print(f"  增量刷新行情: {fetch_start} → {end_date or _today_str()}", flush=True)
    elif force_refresh:
        print("  强制全量拉取最新行情...", flush=True)

    frames: List[pd.DataFrame] = []
    total = len(codes)
    from progress_utils import ProgressBar
    bar = ProgressBar(total, desc="  下载行情", unit="股")

    def _one(c: str) -> pd.DataFrame:
        return fetch_daily_hist(c, start_date=fetch_start, end_date=end_date, sleep=0.05)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_one, c): c for c in codes}
        for fut in as_completed(futures):
            code = futures[fut]
            df = fut.result()
            if not df.empty:
                frames.append(df)
            bar.update(1, postfix=code)

    bar.close(f"成功 {len(frames)}/{total} 只")

    if not frames:
        if cached is not None and not cached.empty:
            print("  增量拉取失败，回退使用旧缓存", flush=True)
            return cached
        return pd.DataFrame()

    fresh = pd.concat(frames, ignore_index=True)
    out = _merge_hist(cached, fresh) if force_refresh else fresh

    if cache_dir and cache_file is not None:
        out.to_pickle(cache_file)
        # 同步写一份固定名，便于 Streamlit / 可视化读取
        latest_alias = cache_dir / "hist_latest.pkl"
        out.to_pickle(latest_alias)
        print(
            f"  已写入缓存: {cache_file.name} | 最新日期={pd.to_datetime(out['date']).max():%Y-%m-%d}",
            flush=True,
        )
    return out


def refresh_market_data(
    codes: Optional[List[str]] = None,
    start_date: str = "20230101",
    max_stocks: Optional[int] = None,
    workers: int = 6,
    cache_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """供 Streamlit / CLI 调用：强制拉取并合并最新历史成交数据。"""
    cache_dir = Path(cache_dir) if cache_dir else _cache_root()
    if codes is None:
        codes = get_all_a_codes()
    if max_stocks:
        codes = codes[:max_stocks]
    return fetch_universe_hist_parallel(
        codes,
        start_date=start_date.replace("-", ""),
        cache_dir=cache_dir,
        workers=workers,
        force_refresh=True,
    )


def fetch_universe_hist(
    codes: List[str],
    start_date: str,
    end_date: Optional[str] = None,
    cache_dir: Optional[Path] = None,
    max_stocks: Optional[int] = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    if max_stocks:
        codes = codes[:max_stocks]
    return fetch_universe_hist_parallel(
        codes, start_date=start_date, end_date=end_date,
        cache_dir=cache_dir, workers=4, force_refresh=force_refresh,
    )


def get_universe(universe: str, custom_codes: Optional[List[str]] = None) -> List[str]:
    universe = (universe or "hs300").lower()
    if universe == "custom" and custom_codes:
        return [_norm_code(c) for c in custom_codes]
    if universe in ("all_a", "all"):
        return get_all_a_codes()
    return get_all_a_codes()[:300]


def get_st_list() -> set:
    if HAS_AK:
        try:
            df = ak.stock_zh_a_st_em()
            if df is not None and not df.empty:
                col = "代码" if "代码" in df.columns else df.columns[0]
                return {_norm_code(c) for c in df[col]}
        except Exception:
            pass
    return set()
