"""多模型组合配置：原生模型 + Qlib 风格模型，启用开关 + 权重归一化。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from qlib_models import (
    DEFAULT_QLIB_WEIGHTS,
    QLIB_MODEL_DESC,
    QLIB_MODEL_KEYS,
    QLIB_MODEL_LABELS,
    QLIB_SEQ_KEYS,
    QLIB_TABULAR_KEYS,
    available_qlib_models,
)

# 原生规则 / ML 模型
NATIVE_KEYS: List[str] = ["b1", "b2", "lgb", "lstm"]
NATIVE_LABELS: Dict[str, str] = {
    "b1": "B1 缩量战法",
    "b2": "B2 放量启动",
    "lgb": "LightGBM 表格模型",
    "lstm": "LSTM 时序模型",
}

# 全部可选模型 = 原生 + Qlib Zoo
MODEL_KEYS: List[str] = NATIVE_KEYS + list(QLIB_MODEL_KEYS)
MODEL_LABELS: Dict[str, str] = {**NATIVE_LABELS, **QLIB_MODEL_LABELS}
MODEL_DESC: Dict[str, str] = {
    "b1": "缩量超卖，KDJ-J 低位",
    "b2": "放量启动，突破 B1 平台",
    "lgb": "表格因子 LightGBM 回归",
    "lstm": "价量时序深度学习",
    **QLIB_MODEL_DESC,
}

DEFAULT_ENABLED = {k: (k in NATIVE_KEYS) for k in MODEL_KEYS}
DEFAULT_WEIGHTS = {
    "b1": 0.25,
    "b2": 0.20,
    "lgb": 0.25,
    "lstm": 0.30,
    **DEFAULT_QLIB_WEIGHTS,
}

# 兼容旧代码
__all__ = [
    "MODEL_KEYS",
    "MODEL_LABELS",
    "MODEL_DESC",
    "DEFAULT_ENABLED",
    "DEFAULT_WEIGHTS",
    "NATIVE_KEYS",
    "QLIB_MODEL_KEYS",
    "QLIB_TABULAR_KEYS",
    "QLIB_SEQ_KEYS",
    "ModelConfig",
    "available_qlib_models",
]


@dataclass
class ModelConfig:
    enabled: Dict[str, bool] = field(default_factory=lambda: dict(DEFAULT_ENABLED))
    weights: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))

    @classmethod
    def from_cfg(cls, ens_cfg: dict) -> "ModelConfig":
        raw_en = ens_cfg.get("enabled") or {}
        raw_w = ens_cfg.get("weights") or {}
        if raw_en:
            # 显式配置：未写出的模型视为关闭（便于选取/组合）
            enabled = {k: bool(raw_en.get(k, False)) for k in MODEL_KEYS}
        else:
            enabled = dict(DEFAULT_ENABLED)
        weights = {**DEFAULT_WEIGHTS, **raw_w}
        for k in MODEL_KEYS:
            weights.setdefault(k, 0.0)
        return cls(enabled=enabled, weights=weights)

    def active_keys(self) -> List[str]:
        return [k for k in MODEL_KEYS if self.enabled.get(k, False)]

    def active_native(self) -> List[str]:
        return [k for k in NATIVE_KEYS if self.enabled.get(k, False)]

    def active_qlib_tabular(self) -> List[str]:
        return [k for k in QLIB_TABULAR_KEYS if self.enabled.get(k, False)]

    def active_qlib_seq(self) -> List[str]:
        return [k for k in QLIB_SEQ_KEYS if self.enabled.get(k, False)]

    def normalized_weights(self) -> Dict[str, float]:
        """仅对启用的模型按输入比例归一化，未启用的权重为 0。"""
        active = self.active_keys()
        if not active:
            return {k: 0.0 for k in MODEL_KEYS}

        raw = {k: max(float(self.weights.get(k, 0)), 0.0) for k in active}
        total = sum(raw.values())
        if total <= 0:
            share = 1.0 / len(active)
            return {k: (share if k in active else 0.0) for k in MODEL_KEYS}

        return {k: (raw[k] / total if k in active else 0.0) for k in MODEL_KEYS}

    def needs_lgb(self) -> bool:
        return self.enabled.get("lgb", False)

    def needs_lstm(self) -> bool:
        return self.enabled.get("lstm", False)

    def needs_qlib_tabular(self) -> bool:
        return bool(self.active_qlib_tabular())

    def needs_qlib_seq(self) -> bool:
        return bool(self.active_qlib_seq())

    def needs_panel(self) -> bool:
        """是否需要构建历史训练面板（ML 模型或回测）。"""
        return len(self.active_keys()) > 0

    def summary(self) -> str:
        w = self.normalized_weights()
        parts = [f"{MODEL_LABELS.get(k, k)} {w[k]:.0%}" for k in self.active_keys()]
        return " + ".join(parts) if parts else "（未选择模型）"

    def to_dict(self) -> dict:
        return {
            "enabled": dict(self.enabled),
            "weights": dict(self.weights),
            "normalized": self.normalized_weights(),
            "summary": self.summary(),
            "active": self.active_keys(),
        }
