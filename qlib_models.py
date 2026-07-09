"""
Qlib 风格模型 Zoo（轻量复刻，不强制安装 pyqlib）。

参考 microsoft/qlib Model Zoo，在本项目面板特征上提供可训练/可预测接口，
便于在 Streamlit / CLI 中选取与加权组合。

表格类: linear / ridge / lasso / elasticnet / rf / xgb / catboost / double_ensemble
时序类: gru / alstm / transformer / mlp_seq  （依赖 TensorFlow，可选）
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
    from sklearn.preprocessing import StandardScaler
    HAS_SK = True
except ImportError:
    HAS_SK = False

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import catboost as cb
    HAS_CAT = True
except ImportError:
    HAS_CAT = False

try:
    import tensorflow as tf
    from tensorflow.keras import layers, models, callbacks
    HAS_TF = True
except ImportError:
    HAS_TF = False


# ── 模型注册表（与 ensemble_config.MODEL_KEYS 对齐）──────────────────────────

QLIB_TABULAR_KEYS = [
    "linear", "ridge", "lasso", "elasticnet",
    "rf", "xgb", "catboost", "double_ensemble",
]
QLIB_SEQ_KEYS = ["gru", "alstm", "transformer", "mlp_seq"]
QLIB_MODEL_KEYS = QLIB_TABULAR_KEYS + QLIB_SEQ_KEYS

QLIB_MODEL_LABELS: Dict[str, str] = {
    "linear": "Qlib Linear",
    "ridge": "Qlib Ridge",
    "lasso": "Qlib Lasso",
    "elasticnet": "Qlib ElasticNet",
    "rf": "Qlib RandomForest",
    "xgb": "Qlib XGBoost",
    "catboost": "Qlib CatBoost",
    "double_ensemble": "Qlib DoubleEnsemble",
    "gru": "Qlib GRU",
    "alstm": "Qlib ALSTM",
    "transformer": "Qlib Transformer",
    "mlp_seq": "Qlib MLP(时序)",
}

QLIB_MODEL_DESC: Dict[str, str] = {
    "linear": "普通最小二乘线性回归",
    "ridge": "L2 正则线性模型",
    "lasso": "L1 稀疏线性模型",
    "elasticnet": "L1+L2 弹性网络",
    "rf": "随机森林回归",
    "xgb": "XGBoost GBDT（需 xgboost）",
    "catboost": "CatBoost GBDT（需 catboost）",
    "double_ensemble": "样本+特征重加权双集成（LightGBM）",
    "gru": "门控循环网络时序模型",
    "alstm": "Attention-LSTM 时序模型",
    "transformer": "轻量 Transformer 编码器",
    "mlp_seq": "时序展平后的 MLP",
}

DEFAULT_QLIB_WEIGHTS = {k: 0.0 for k in QLIB_MODEL_KEYS}


@dataclass
class QlibModelBundle:
    """已训练的 Qlib 风格模型集合。"""
    tabular: Dict[str, Any]
    seq: Dict[str, Any]
    scaler: Optional[Any] = None
    feature_cols: Optional[List[str]] = None


def available_qlib_models() -> Dict[str, bool]:
    """返回各模型当前环境是否可用。"""
    avail = {k: HAS_SK for k in ("linear", "ridge", "lasso", "elasticnet", "rf")}
    avail["xgb"] = HAS_XGB
    avail["catboost"] = HAS_CAT
    try:
        import lightgbm  # noqa: F401
        avail["double_ensemble"] = True
    except ImportError:
        avail["double_ensemble"] = False
    for k in QLIB_SEQ_KEYS:
        avail[k] = HAS_TF
    return avail


def _split_xy(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    feature_cols: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Any]:
    scaler = StandardScaler() if HAS_SK else None
    X_tr = train[feature_cols].fillna(0).values.astype(np.float32)
    y_tr = train["label"].values.astype(np.float32)
    X_va = valid[feature_cols].fillna(0).values.astype(np.float32)
    y_va = valid["label"].values.astype(np.float32)
    if scaler is not None:
        X_tr = scaler.fit_transform(X_tr)
        X_va = scaler.transform(X_va)
    return X_tr, y_tr, X_va, y_va, scaler


def _fit_tabular_one(key: str, X_tr, y_tr, X_va, y_va) -> Optional[Any]:
    if not HAS_SK and key in ("linear", "ridge", "lasso", "elasticnet", "rf"):
        return None

    if key == "linear":
        m = LinearRegression()
        m.fit(X_tr, y_tr)
        return m
    if key == "ridge":
        m = Ridge(alpha=1.0, random_state=42)
        m.fit(X_tr, y_tr)
        return m
    if key == "lasso":
        m = Lasso(alpha=0.001, max_iter=5000, random_state=42)
        m.fit(X_tr, y_tr)
        return m
    if key == "elasticnet":
        m = ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=5000, random_state=42)
        m.fit(X_tr, y_tr)
        return m
    if key == "rf":
        m = RandomForestRegressor(
            n_estimators=120, max_depth=8, min_samples_leaf=20,
            n_jobs=-1, random_state=42,
        )
        m.fit(X_tr, y_tr)
        return m
    if key == "xgb":
        if not HAS_XGB:
            print("  跳过 XGBoost：未安装 xgboost", flush=True)
            return None
        m = xgb.XGBRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8,
            reg_lambda=1.0, n_jobs=-1, random_state=42,
            verbosity=0,
        )
        m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        return m
    if key == "catboost":
        if not HAS_CAT:
            print("  跳过 CatBoost：未安装 catboost", flush=True)
            return None
        m = cb.CatBoostRegressor(
            iterations=200, learning_rate=0.05, depth=6,
            loss_function="RMSE", verbose=False, random_seed=42,
        )
        m.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=False)
        return m
    if key == "double_ensemble":
        return _fit_double_ensemble(X_tr, y_tr, X_va, y_va)
    return None


def _fit_double_ensemble(X_tr, y_tr, X_va, y_va, n_rounds: int = 3):
    """简化版 DoubleEnsemble：多轮样本重加权 + 特征子集集成。"""
    try:
        import lightgbm as lgb
    except ImportError:
        print("  跳过 DoubleEnsemble：未安装 lightgbm", flush=True)
        return None

    n_feat = X_tr.shape[1]
    models_list = []
    sample_w = np.ones(len(y_tr), dtype=np.float64)
    rng = np.random.RandomState(42)

    for r in range(n_rounds):
        # 每轮随机保留约 70% 特征
        mask = rng.rand(n_feat) > 0.3
        if mask.sum() < max(3, n_feat // 3):
            mask = np.ones(n_feat, dtype=bool)
        cols = np.where(mask)[0]
        m = lgb.LGBMRegressor(
            n_estimators=80, learning_rate=0.05, max_depth=5,
            num_leaves=31, random_state=42 + r, verbose=-1,
        )
        m.fit(
            X_tr[:, cols], y_tr,
            sample_weight=sample_w,
            eval_set=[(X_va[:, cols], y_va)],
        )
        pred = m.predict(X_tr[:, cols])
        err = np.abs(pred - y_tr)
        # 加大难样本权重
        sample_w = 0.5 + err / (err.mean() + 1e-8)
        sample_w = sample_w / sample_w.mean()
        models_list.append((m, cols))

    return {"type": "double_ensemble", "models": models_list}


def _predict_tabular_one(key: str, model: Any, X: np.ndarray) -> np.ndarray:
    if model is None:
        return np.zeros(len(X))
    if key == "double_ensemble" and isinstance(model, dict):
        preds = []
        for m, cols in model["models"]:
            preds.append(m.predict(X[:, cols]))
        return np.mean(preds, axis=0)
    return np.asarray(model.predict(X), dtype=np.float64)


def train_qlib_tabular(
    panel: pd.DataFrame,
    feature_cols: List[str],
    enabled_keys: List[str],
    train_ratio: float = 0.6,
    valid_ratio: float = 0.15,
) -> QlibModelBundle:
    """训练启用的表格类 Qlib 风格模型。"""
    dates = sorted(panel["date"].unique())
    n = len(dates)
    t1 = int(n * train_ratio)
    t2 = int(n * (train_ratio + valid_ratio))
    train = panel[panel["date"].isin(set(dates[:t1]))]
    valid = panel[panel["date"].isin(set(dates[t1:t2]))]
    if train.empty or valid.empty:
        return QlibModelBundle(tabular={}, seq={})

    X_tr, y_tr, X_va, y_va, scaler = _split_xy(train, valid, feature_cols)
    tabular: Dict[str, Any] = {}
    for key in enabled_keys:
        if key not in QLIB_TABULAR_KEYS:
            continue
        print(f"  训练 Qlib/{QLIB_MODEL_LABELS.get(key, key)} ...", flush=True)
        try:
            m = _fit_tabular_one(key, X_tr, y_tr, X_va, y_va)
            if m is not None:
                tabular[key] = m
                print(f"    ✓ {key}", flush=True)
            else:
                print(f"    ✗ {key} 不可用", flush=True)
        except Exception as e:
            print(f"    ✗ {key} 失败: {e}", flush=True)

    return QlibModelBundle(
        tabular=tabular, seq={}, scaler=scaler, feature_cols=list(feature_cols),
    )


def predict_qlib_tabular(
    bundle: QlibModelBundle,
    df: pd.DataFrame,
    keys: Optional[List[str]] = None,
) -> Dict[str, np.ndarray]:
    """对截面/面板预测，返回 {model_key: pred_array}。"""
    if not bundle.tabular or not bundle.feature_cols:
        return {}
    keys = keys or list(bundle.tabular.keys())
    X = df[bundle.feature_cols].fillna(0).values.astype(np.float32)
    if bundle.scaler is not None:
        X = bundle.scaler.transform(X)
    out = {}
    for key in keys:
        if key in bundle.tabular:
            out[key] = _predict_tabular_one(key, bundle.tabular[key], X)
    return out


# ── 时序模型（复用 lstm_model 的序列特征）────────────────────────────────────

def _build_seq_model(key: str, seq_len: int, n_feat: int):
    if not HAS_TF:
        return None
    inp = layers.Input(shape=(seq_len, n_feat))
    if key == "gru":
        x = layers.GRU(64, return_sequences=True)(inp)
        x = layers.GRU(32)(x)
    elif key == "alstm":
        x = layers.LSTM(64, return_sequences=True)(inp)
        # 简化 attention：对时间维做加权平均
        score = layers.Dense(1, activation="tanh")(x)
        score = layers.Softmax(axis=1)(score)
        x = layers.Multiply()([x, score])
        x = layers.Lambda(lambda t: tf.reduce_sum(t, axis=1))(x)
        x = layers.Dense(32, activation="relu")(x)
    elif key == "transformer":
        # 轻量单层 self-attention
        attn = layers.MultiHeadAttention(num_heads=2, key_dim=16)(inp, inp)
        x = layers.Add()([inp, attn])
        x = layers.LayerNormalization()(x)
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(32, activation="relu")(x)
    elif key == "mlp_seq":
        x = layers.Flatten()(inp)
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(64, activation="relu")(x)
    else:
        return None
    out = layers.Dense(1)(x)
    model = models.Model(inp, out)
    model.compile(optimizer="adam", loss="mse")
    return model


def train_qlib_seq(
    X: np.ndarray,
    y: np.ndarray,
    enabled_keys: List[str],
    seq_len: int,
    epochs: int = 12,
    batch_size: int = 128,
) -> Dict[str, Any]:
    """训练启用的时序类 Qlib 风格模型。X: (N, seq_len, F)。"""
    if not HAS_TF or len(X) < 100:
        return {}
    n = len(X)
    split = int(n * 0.85)
    X_tr, y_tr = X[:split], y[:split]
    X_va, y_va = X[split:], y[split:]
    n_feat = X.shape[-1]
    out: Dict[str, Any] = {}

    for key in enabled_keys:
        if key not in QLIB_SEQ_KEYS:
            continue
        print(f"  训练 Qlib/{QLIB_MODEL_LABELS.get(key, key)} ...", flush=True)
        try:
            model = _build_seq_model(key, seq_len, n_feat)
            if model is None:
                print(f"    ✗ {key} 不可用", flush=True)
                continue
            es = callbacks.EarlyStopping(
                monitor="val_loss", patience=3, restore_best_weights=True,
            )
            model.fit(
                X_tr, y_tr,
                validation_data=(X_va, y_va),
                epochs=epochs, batch_size=batch_size,
                callbacks=[es], verbose=0,
            )
            out[key] = model
            print(f"    ✓ {key}", flush=True)
        except Exception as e:
            print(f"    ✗ {key} 失败: {e}", flush=True)
    return out


def predict_qlib_seq_latest(
    seq_models: Dict[str, Any],
    raw: pd.DataFrame,
    codes: List[str],
    seq_len: int = 30,
) -> Dict[str, pd.DataFrame]:
    """对最新截面做时序模型预测，返回 {key: DataFrame[code, pred]}。"""
    from lstm_model import build_lstm_sequences

    rows_cache: Dict[str, list] = {k: [] for k in seq_models}

    for code in codes:
        sub = raw[raw["code"] == code]
        if len(sub) < seq_len + 5:
            continue
        X, _, _ = build_lstm_sequences(sub, seq_len=seq_len, forward_days=5)
        if len(X) == 0:
            continue
        x_last = X[-1:]
        for key, model in seq_models.items():
            try:
                pred = float(model.predict(x_last, verbose=0).ravel()[0])
                rows_cache[key].append({"code": code, f"{key}_pred": pred})
            except Exception:
                continue

    out = {}
    for key, rows in rows_cache.items():
        out[key] = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["code", f"{key}_pred"])
    return out


def predict_qlib_seq_panel(
    seq_models: Dict[str, Any],
    raw: pd.DataFrame,
    panel: pd.DataFrame,
    seq_len: int = 30,
    forward_days: int = 5,
) -> Dict[str, pd.DataFrame]:
    """测试集面板预测（按 code 对齐最近序列）。"""
    from lstm_model import build_lstm_sequences

    out: Dict[str, list] = {k: [] for k in seq_models}
    codes = panel["code"].unique().tolist()
    for code in codes:
        sub = raw[raw["code"] == code].sort_values("date")
        X, y, _ = build_lstm_sequences(sub, seq_len=seq_len, forward_days=forward_days)
        if len(X) == 0:
            continue
        # 用序列末尾日期近似对齐 panel
        dates = sub["date"].iloc[seq_len:seq_len + len(X)].values
        for key, model in seq_models.items():
            try:
                preds = model.predict(X, verbose=0).ravel()
                for dt, p in zip(dates, preds):
                    out[key].append({"date": pd.Timestamp(dt), "code": code, f"{key}_pred": float(p)})
            except Exception:
                continue

    return {
        k: (pd.DataFrame(v) if v else pd.DataFrame())
        for k, v in out.items()
    }
