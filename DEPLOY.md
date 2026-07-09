# 云端部署指南

本地网络无法访问东方财富/AkShare 时，可将 **B1战法 + LSTM 选股** 部署到免费云端，由云端拉取新浪行情并生成报告。

> **推荐对照操作完整教程**（每日自动训练 + 结果推送 + Streamlit）：  
> → **[SERVER_DEPLOY.md](SERVER_DEPLOY.md)**

---

## 方案一：GitHub Actions（推荐，完全免费）

**优点**：每天自动跑、结果上传 Artifacts、可写回仓库 `output/latest/`  
**缺点**：需 GitHub 账号；公开仓库免费

### 步骤

1. 在 [GitHub](https://github.com) 新建**独立仓库** `quant_ashare`，仓库根目录即本项目

2. 工作流已配置：`.github/workflows/daily_b1_lstm.yml`
   - 每工作日 **北京时间 18:00** 自动运行
   - 使用 **新浪数据源**（`QUANT_DATA_SOURCE=sina`）
   - 扫描约 2500 只股票（可在 `config/cloud_settings.yaml` 调整）

3. **手动触发**：仓库 → Actions → `B1 LSTM Daily Pick` → Run workflow

4. **查看结果**：
   - Actions 运行页 → Artifacts → 下载 `b1-lstm-results`
   - 或仓库内 `output/latest/top50_latest.csv`

### 本地同步结果

```powershell
git pull
# 打开 output/latest/top50_latest.csv
```

---

## 方案二：Streamlit Community Cloud（免费 Web 界面）

**优点**：浏览器查看 Top50、可在线试跑  
**地址**：https://streamlit.io/cloud

### 步骤

1. 代码推送到 GitHub（同上）

2. 登录 Streamlit Cloud → New app

3. 配置：
   - Repository: 你的仓库
   - Branch: main
   - **Main file path**: `streamlit_app.py`

4. 环境变量（Advanced settings）：
   ```
   QUANT_DATA_SOURCE=sina
   ```

5. Deploy 后访问生成的 URL，在「查看结果 / 云端运行」页使用

---

## 方案三：Google Colab（免费，零部署）

适合不想建仓库、偶尔手动跑一遍。

1. 打开 [Google Colab](https://colab.research.google.com)

2. 新建笔记本，运行：

```python
!git clone https://github.com/fsdiyyds/quant_ashare.git
%cd quant_ashare
!pip install -r requirements.txt -q
import os
os.environ["QUANT_DATA_SOURCE"] = "sina"
!python b1_lstm_daily.py --config config/cloud_settings.yaml --max-stocks 500
```

3. 下载结果：

```python
from google.colab import files
files.download("output/b1_lstm/top50_buy_*.csv")  # 按实际文件名
```

---

## 方案四：Render / Railway（Docker，免费额度）

1. 使用仓库根目录 `Dockerfile`

2. Render 新建 **Web Service**：
   - Build: Docker
   - Root Directory: 仓库根（留空）
   - 免费档约 750 小时/月

3. 环境变量：`QUANT_DATA_SOURCE=sina`

> Streamlit 服务适合展示；长时间全市场扫描仍建议用 GitHub Actions。

---

## 数据源说明

| 环境变量 | 说明 |
|---------|------|
| `auto`（默认） | AkShare → 新浪 → 东方财富 自动降级 |
| `sina` | 强制新浪（**云端/本地网络差时推荐**） |
| `eastmoney` | 东方财富 |
| `akshare` | AkShare |

本地试跑新浪源：

```powershell
cd quant_ashare
$env:QUANT_DATA_SOURCE="sina"
python b1_lstm_daily.py --max-stocks 100
```

---

## 配置文件

| 文件 | 用途 |
|------|------|
| `config/b1_settings.yaml` | 本地全量 |
| `config/cloud_settings.yaml` | 云端（限制 2500 只、epochs=15） |

---

## 常见问题

**Q: 为什么本地显示 0 只股票？**  
东方财富被墙/拦截。设置 `QUANT_DATA_SOURCE=sina` 或改用 GitHub Actions。

**Q: 云端跑不完 5000 只？**  
免费 runner 有 120 分钟超时，默认 `max_stocks: 2500`。可分批或升级自建 runner。

**Q: 如何改定时？**  
编辑 `.github/workflows/daily_b1_lstm.yml` 中 `cron`（UTC 时间）。

---

**免责声明**：仅供学习研究，不构成投资建议。
