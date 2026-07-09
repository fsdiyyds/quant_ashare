# GitHub 建仓 + Streamlit 可视化部署完整指南

本指南针对**独立仓库 `quant_ashare`**（仓库根目录即项目代码）。

当前仓库：`https://github.com/fsdiyyds/quant_ashare`

---

## 目录

1. [GitHub 建仓与推送](#一github-建仓与推送)
2. [GitHub Actions 每日自动选股](#二github-actions-每日自动选股)
3. [Streamlit Cloud 可视化部署（重点）](#三streamlit-cloud-可视化部署重点)
4. [本地 Streamlit 预览](#四本地-streamlit-预览)
5. [每日使用流程](#五每日使用流程)
6. [常见问题](#六常见问题)

---

## 一、GitHub 建仓与推送

### 1.1 创建空仓库

1. 打开 https://github.com/new
2. **Repository name**: `quant_ashare`
3. **Public**
4. **不要**勾选 README / .gitignore
5. **Create repository**

### 1.2 本地推送

```powershell
cd c:\Users\fenglei\PycharmProjects\pythonProject\waterRPA\quant_ashare

git init
git branch -M main
git add .
git commit -m "feat: A股量化选股独立仓库"
git remote add origin https://github.com/fsdiyyds/quant_ashare.git
git push -u origin main
```

> 密码处填 Personal Access Token（勾选 `repo` + `workflow`）。

---

## 二、GitHub Actions 每日自动选股

1. 仓库 → **Actions** → 启用 workflows  
2. **B1 LSTM Daily Pick** → **Run workflow**  
3. 等待完成后查看 Artifacts 或 `output/latest/`

工作流文件：`.github/workflows/daily_b1_lstm.yml`（已在仓库根目录）。

---

## 三、Streamlit Cloud 可视化部署（重点）

| 配置项 | 填写内容 |
|--------|----------|
| Repository | `fsdiyyds/quant_ashare` |
| Branch | `main` |
| Main file path | `streamlit_app.py` |
| Python version | **务必在 Advanced settings 选 `3.11` 或 `3.12`**（不要用 3.14） |

Secrets：

```toml
QUANT_DATA_SOURCE = "sina"
```

> `requirements.txt` 已去掉 TensorFlow，即使误选 3.14 也能装依赖并展示结果。  
> LSTM 训练请用 GitHub Actions（`requirements-train.txt`，Python 3.11）。

---

## 四、本地 Streamlit 预览

```powershell
cd c:\Users\fenglei\PycharmProjects\pythonProject\waterRPA\quant_ashare
pip install -r requirements.txt
$env:QUANT_DATA_SOURCE="sina"
streamlit run streamlit_app.py
```

---

## 五、每日使用流程

```
1. 等待 GitHub Actions 跑完（或手动 Run workflow）
2. git pull 拉取最新 output/latest/
3. 打开 Streamlit → 「推荐列表」
4. 「个股可视化」看 K 线 + LSTM
```

---

## 六、常见问题

**Q: Main file 还要写 quant_ashare/streamlit_app.py 吗？**  
不需要。独立仓库根目录就是项目，填 `streamlit_app.py`。

**Q: 推送失败 / 连接重置？**  
不要提交 `data/cache/*.pkl` 大文件；见 `.gitignore`。可执行：

```powershell
git config http.postBuffer 524288000
git push -u origin main
```

## 相关文档

- [SERVER_DEPLOY.md](SERVER_DEPLOY.md) — 每日自动训练完整教程  
- [DEPLOY.md](DEPLOY.md) — Colab / Render 补充  
- [README.md](README.md) — 功能说明  

**免责声明**：仅供学习研究，不构成投资建议。
