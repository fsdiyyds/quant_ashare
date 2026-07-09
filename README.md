# A股 AI 量化分析

基于 GitHub 主流开源方案整合的 A 股量化研究脚本：**AkShare 取数 + 技术因子 + LightGBM 预测 + Top-K 选股建议**。

## 参考开源项目

| 项目 | Stars | 用途 |
|------|-------|------|
| [microsoft/qlib](https://github.com/microsoft/qlib) | 4.4万+ | AI 量化研究平台，Alpha158 因子、LightGBM/LSTM |
| [akfamily/akshare](https://github.com/akfamily/akshare) | 1万+ | A 股免费行情/财务数据接口 |
| [ioiochen11/qlib-research-workbench](https://github.com/ioiochen11/qlib-research-workbench) | - | A 股 Qlib 工作台，滚动训练+日报 |
| [hualin6/quant-ashare](https://github.com/hualin6/quant-ashare) | - | 机构级 A 股因子中性化、多 Agent 决策 |
| [vnpy/vnpy](https://github.com/vnpy/vnpy) | 2.8万+ | 实盘交易框架（进阶对接） |

本仓库为**轻量入门版**，便于快速跑通；深度研究建议安装 Qlib 完整流水线。

## 本地网络不通？→ 云端部署

东方财富/AkShare 无法访问时，使用 **新浪数据源** 或 **免费云端**：

```powershell
# 本地改用新浪（已验证可用）
$env:QUANT_DATA_SOURCE="sina"
python b1_lstm_daily.py --max-stocks 200
```

**免费云端 + Streamlit 可视化**（详见 [GITHUB_SETUP.md](GITHUB_SETUP.md)）：

| 方案 | 说明 |
|------|------|
| **GitHub Actions** | 每天自动跑，结果含可视化 HTML |
| **Streamlit Cloud** | 浏览器看 Top50 + K线图 + 购买理由（推荐） |
| **Google Colab** | 零部署手动跑 |
| **Render Docker** | 容器托管 |

**Streamlit 快速部署：** https://share.streamlit.io → Main file: `streamlit_app.py` → Python **3.11** → Secret: `QUANT_DATA_SOURCE=sina`  
仓库：https://github.com/fsdiyyds/quant_ashare  

> Cloud 展示用 `requirements.txt`（无 TensorFlow）；完整训练用 `pip install -r requirements-train.txt`（需 Python ≤3.12）。

---

## B1战法 + LSTM 每日选股（新）

基于 **B1战法（少妇战法）** + **LSTM 深度学习** 的两阶段选股：

1. **B1初选**：全 A 股扫描，按 BBI+KDJ+缩量+MACD+知行线 评分，取 Top **520**
2. **LSTM精筛**：用 30 日价格/成交量/换手率时序训练 LSTM，输出 Top **50**

参考开源：[StockTradebyZ](https://github.com/Noctis-lzy/StockTradebyZ) BBIKDJSelector

```powershell
pip install tensorflow
python b1_lstm_daily.py --max-stocks 200    # 调试：限制扫描数量
python b1_lstm_daily.py                     # 全市场（耗时较长，建议缓存）
python b1_lstm_daily.py --b1-top 520 --lstm-top 50
```

输出目录：`output/b1_lstm/`（B1池 CSV + Top50 CSV + Markdown 报告）

---

## 快速开始

```powershell
cd quant_ashare
pip install -r requirements.txt
python ashare_quant.py --max-stocks 30
```

## 常用命令

```powershell
# 沪深300 全量分析
python ashare_quant.py

# 中证500
python ashare_quant.py --universe zz500

# 指定股票
python ashare_quant.py --codes 600519 000858 601318

# 调试：只拉 20 只
python ashare_quant.py --max-stocks 20 --no-backtest

# 换模型
python ashare_quant.py --algorithm random_forest
```

## 目录结构

```
quant_ashare/
├── ashare_quant.py      # 主入口
├── data_fetcher.py      # AkShare 数据下载
├── features.py          # 技术因子（简化 Alpha158）
├── predict_model.py     # LightGBM / RF 训练预测
├── strategy.py          # 选股建议 + 简易回测
├── config/settings.yaml # 配置文件
├── data/cache/          # 行情缓存
└── output/              # 报告输出
```

## 功能说明

1. **数据层**：AkShare 拉取沪深300/中证500/自定义股票前复权日线
2. **因子层**：MA、RSI、MACD、KDJ、动量、波动率等 20+ 技术因子
3. **AI 预测**：LightGBM 回归预测未来 N 日收益率
4. **投资建议**：Top-K 选股 + 看涨/看跌/震荡标签 + 文字建议
5. **简易回测**：验证集后段 Top-K 等权调仓（未含 T+1/涨跌停完整模拟）

## 多模型组合（原生 + Qlib 风格 Zoo）

本项目已内置 [microsoft/qlib](https://github.com/microsoft/qlib) Model Zoo 的轻量复刻，可在 Streamlit / CLI 中选取、加权组合：

| 类别 | 模型 key |
|------|----------|
| 原生 | `b1` `b2` `lgb` `lstm` |
| Qlib 表格 | `linear` `ridge` `lasso` `elasticnet` `rf` `xgb` `catboost` `double_ensemble` |
| Qlib 时序 | `gru` `alstm` `transformer` `mlp_seq` |

```powershell
# 每次运行默认增量拉取最新成交
python -u b1_lstm_daily.py --max-stocks 150 --models b1,lgb,ridge,xgb --force-refresh

# Streamlit：侧边栏可单独刷新行情；「模型组合」页勾选 Zoo
streamlit run streamlit_app.py
```

完整 Qlib 平台（可选，非必须）：

```powershell
pip install pyqlib
python -m qlib.run.get_data qlib_data --target_dir ~/.qlib/qlib_data/cn_data --region cn
```

## 每日自动化部署

对着操作即可：**GitHub Actions 每日训练 + 结果推送 + Streamlit 查看**。

→ 详见 **[SERVER_DEPLOY.md](SERVER_DEPLOY.md)**（推荐）  
→ 补充：[GITHUB_SETUP.md](GITHUB_SETUP.md) / [DEPLOY.md](DEPLOY.md)

## 风险提示

- 机器学习预测**不能保证**未来收益
- 回测存在过拟合、前视偏差、幸存者偏差等风险
- **不构成任何投资建议**，实盘需自行承担风险
