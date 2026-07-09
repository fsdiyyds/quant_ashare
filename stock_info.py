"""股票代码 ↔ 名称映射（新浪数据源 + 本地缓存）。"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Optional

try:
    import requests
except ImportError:
    requests = None

_CACHE_PATH = Path(__file__).resolve().parent / "data" / "cache" / "stock_names.json"
_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0",
    "Referer": "https://finance.sina.com.cn/",
}


def _load_cache() -> Dict[str, str]:
    if _CACHE_PATH.exists():
        try:
            return json.loads(_CACHE_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _save_cache(mapping: Dict[str, str]) -> None:
    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _CACHE_PATH.write_text(
        json.dumps(mapping, ensure_ascii=False, indent=0),
        encoding="utf-8",
    )


def fetch_all_names() -> Dict[str, str]:
    """从新浪拉取全 A 股 code→name。"""
    if requests is None:
        return {}

    url = (
        "http://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/"
        "Market_Center.getHQNodeData"
    )
    mapping: Dict[str, str] = {}
    for page in range(1, 80):
        try:
            resp = requests.get(
                url,
                params={
                    "page": page, "num": 80, "sort": "symbol", "asc": 1,
                    "node": "hs_a", "symbol": "", "_s_r_a": "init",
                },
                headers=_HEADERS,
                timeout=20,
            )
            data = resp.json()
            if not data:
                break
            for item in data:
                sym = item.get("symbol", "")
                name = item.get("name", "")
                if sym.startswith(("sh", "sz")) and name:
                    mapping[sym[2:].zfill(6)] = name
            if len(data) < 80:
                break
            time.sleep(0.1)
        except Exception:
            break

    if mapping:
        _save_cache(mapping)
    return mapping


def get_name_map(refresh: bool = False) -> Dict[str, str]:
    if not refresh:
        cached = _load_cache()
        if cached:
            return cached
    return fetch_all_names() or _load_cache()


def get_stock_name(code: str, name_map: Optional[Dict[str, str]] = None) -> str:
    code = str(code).strip().zfill(6)
    mp = name_map or get_name_map()
    return mp.get(code, code)


def attach_names(df, code_col: str = "code", name_col: str = "name"):
    import pandas as pd
    if df is None or (hasattr(df, "empty") and df.empty):
        return df
    mp = get_name_map()
    out = df.copy()
    out[name_col] = out[code_col].astype(str).str.zfill(6).map(lambda c: mp.get(c, c))
    return out
