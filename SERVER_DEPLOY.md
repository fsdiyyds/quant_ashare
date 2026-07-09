# 开源服务器部署教程：每日自动训练 + 推送结果

本教程针对**独立仓库 `quant_ashare`**（仓库根目录即本项目），实现：

1. **每个交易日自动**拉取最新 A 股成交数据  
2. **自动训练**（原生模型 + Qlib 风格模型组合）  
3. **自动推送结果**到仓库 `output/latest/`，并在 Streamlit 网页查看  

推荐组合：**GitHub Actions（训练） + Streamlit Cloud（展示）**，全程免费。

当前仓库示例：`https://github.com/fsdiyyds/quant_ashare`

---

## 目录

1. [架构说明](#一架构说明)
2. [准备：推送到 GitHub](#二准备推送到-github)
3. [GitHub Actions 每日自动训练](#三github-actions-每日自动训练)
4. [结果推送与查看](#四结果推送与查看)
5. [Streamlit Cloud 可视化](#五streamlit-cloud-可视化)
6. [邮件 / 微信推送（可选）](#六邮件--微信推送可选)
7. [自建 VPS / Docker（可选）](#七自建-vps--docker可选)
8. [日常操作清单](#八日常操作清单)
9. [常见问题](#九常见问题)

---

## 一、架构说明

```
┌─────────────────────┐     每天 18:00(北京)      ┌──────────────────────┐
│  新浪行情 API        │ ◄────────────────────── │  GitHub Actions      │
│  (QUANT_DATA_SOURCE │     增量拉取最新成交      │  ubuntu-latest       │
│   =sina)            │                          │  b1_lstm_daily.py    │
└─────────────────────┘                          └──────────┬───────────┘
                                                            │
                                                            ▼
                                                 commit → output/latest/
                                                 top50_latest.csv
                                                 report_latest.md
                                                 backtest_metrics.json
                                                            │
                                                            ▼
                                                 ┌──────────────────────┐
                                                 │  Streamlit Cloud     │
                                                 │  浏览器看推荐/K线     │
                                                 └──────────────────────┘
```

| 组件 | 作用 | 费用 |
|------|------|------|
| GitHub Actions | 定时训练、写回结果 | 公开仓库免费 |
| Streamlit Cloud | Web 展示与手动试跑 | 免费档可用 |
| 新浪数据源 | 云端可访问的行情 | 免费 |

---

## 二、准备：推送到 GitHub

### 2.1 创建独立仓库（方案 B）

1. 打开 https://github.com/new  
2. Repository name：`quant_ashare`  
3. 选 **Public**  
4. **不要**勾选 README / .gitignore  
5. Create repository  

### 2.2 本地推送（在 quant_ashare 目录内）

```powershell
cd c:\Users\fenglei\PycharmProjects\pythonProject\waterRPA\quant_ashare

# 若尚未 init
git init
git branch -M main

git add .
git commit -m "feat: A股量化选股独立仓库"

git remote add origin https://github.com/fsdiyyds/quant_ashare.git
# 若 remote 已存在：git remote set-url origin https://github.com/fsdiyyds/quant_ashare.git

git push -u origin main
```

> 密码处填 **Personal Access Token**（勾选 `repo` + `workflow`）。  
> 若推送因网络中断失败，可加大缓冲后重试：  
> `git config http.postBuffer 524288000`  
> 然后再次 `git push -u origin main`

### 2.3 确认仓库根目录文件

推送成功后，GitHub 根目录应直接看到：

```
.github/workflows/daily_b1_lstm.yml
b1_lstm_daily.py
streamlit_app.py
config/cloud_settings.yaml
SERVER_DEPLOY.md
requirements.txt
```

**不要**再出现外层 `waterRPA/`、`flood_design/` 等无关目录。

---

## 三、GitHub Actions 每日自动训练

### 3.1 启用 Actions

1. 打开 https://github.com/fsdiyyds/quant_ashare → **Actions**  
2. 启用 workflows  
3. 左侧 **B1 LSTM Daily Pick** → **Run workflow**（建议先手动跑通）

### 3.2 定时规则

```yaml
cron: "0 10 * * 1-5"   # UTC 10:00 = 北京 18:00，周一~周五
```

### 3.3 调整模型 / 扫描数量

编辑根目录 `config/cloud_settings.yaml` 后 push，或手动触发时填写 `max_stocks` / `models`。

---

## 四、结果推送与查看

| 路径 | 说明 |
|------|------|
| `output/latest/top50_latest.csv` | 今日 Top 推荐 |
| `output/latest/b1_pool_latest.csv` | B1 初选池 |
| `output/latest/report_latest.md` | Markdown 报告 |
| `output/latest/backtest_metrics.json` | 回测指标 |
| `output/latest/data_asof.txt` | 行情截至日期 |
| `output/latest/model_config.json` | 本次模型组合 |

```powershell
cd c:\Users\fenglei\PycharmProjects\pythonProject\waterRPA\quant_ashare
git pull
```

---

## 五、Streamlit Cloud 可视化

| 配置项 | 填写内容 |
|--------|----------|
| Repository | `fsdiyyds/quant_ashare` |
| Branch | `main` |
| **Main file path** | `streamlit_app.py` |
| Python version | **Advanced settings 选 `3.11` 或 `3.12`**（勿用 3.14） |

Secrets：

```toml
QUANT_DATA_SOURCE = "sina"
```

> 若日志出现 `No matching distribution found for tensorflow`：  
> 1）确认已推送最新 `requirements.txt`（已不含 TF）；  
> 2）Streamlit 控制台 → Manage app → Reboot / Redeploy；  
> 3）Advanced settings 把 Python 改为 3.11。

本地预览：

```powershell
cd c:\Users\fenglei\PycharmProjects\pythonProject\waterRPA\quant_ashare
$env:QUANT_DATA_SOURCE="sina"
streamlit run streamlit_app.py
```

---

## 六、邮件 / 微信推送（可选）

在仓库 **Settings → Secrets** 配置 SMTP 后，可在 `.github/workflows/daily_b1_lstm.yml` 末尾增加发信步骤（读取 `output/latest/report_latest.md`）。详见历史版本示例或自行用 webhook。

---

## 七、自建 VPS / Docker（可选）

```bash
cd /opt/quant_ashare
docker build -t quant-ashare .
docker run -d -p 8501:8501 -e QUANT_DATA_SOURCE=sina quant-ashare
```

crontab 示例：

```bash
5 18 * * 1-5 cd /opt/quant_ashare && \
  QUANT_DATA_SOURCE=sina python3 -u b1_lstm_daily.py \
  --config config/cloud_settings.yaml --force-refresh --skip-backtest-gate \
  >> /var/log/quant_daily.log 2>&1
```

---

## 八、日常操作清单

1. 等 Actions 跑完（或手动 Run workflow）  
2. 打开 Streamlit → **推荐列表**  
3. 需要时改 `config/cloud_settings.yaml` 后 push  

本地调试：

```powershell
cd c:\Users\fenglei\PycharmProjects\pythonProject\waterRPA\quant_ashare
$env:QUANT_DATA_SOURCE="sina"
python -u b1_lstm_daily.py --max-stocks 100 --models b1,lgb,ridge --force-refresh --skip-backtest-gate
streamlit run streamlit_app.py
```

---

## 九、常见问题

**Q: 推送 Connection was reset？**  
多半是大文件或网络不稳。确认 `.gitignore` 已排除 `data/cache/*.pkl`，并执行：

```powershell
git config http.postBuffer 524288000
git push -u origin main
```

**Q: Streamlit Main file 填什么？**  
独立仓库填 `streamlit_app.py`（不是 `quant_ashare/streamlit_app.py`）。

**Q: 旧的 waterRPA / xuangu 仓库怎么办？**  
可在 GitHub 网页 Settings → Delete repository 删除，或留着不管；日常只用 `quant_ashare`。

**免责声明**：仅供学习研究，不构成投资建议。
