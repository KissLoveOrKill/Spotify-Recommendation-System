# Spotify Recommender 

一个本地的 Spotify 风格推荐演示项目，旨在展示如何基于音频特征实现内容推荐。

## 项目特点

- **前端框架**：基于 Flask 的轻量级 Web 应用。
- **推荐算法**：使用 PyTorch 实现的 MLP Autoencoder，将音频特征映射为 32 维向量。
- **推荐逻辑**：通过计算向量相似度（余弦相似度或点积）实现内容推荐。
- **数据管理**：支持本地离线歌曲特征数据库，方便快速加载和处理。

## 目录结构（关键文件）

- `spotify_rec_system/app.py` - Flask 应用主入口，负责路由和服务启动。
- `spotify_rec_system/recommender.py` - 推荐引擎核心模块，包含 MLP Autoencoder 和相似度搜索逻辑。
- `spotify_rec_system/dataset_service.py` - 数据加载服务，支持从 CSV 文件加载离线数据。
- `spotify_rec_system/templates/` - 前端模板文件夹，包含 HTML 页面。
- `spotify_rec_system/data/` - 本地数据存储文件夹（如歌曲特征数据集）。

## 快速开始

以下是本地运行项目的完整流程，无需泄露任何个人信息：

1. **克隆项目**

   将本项目克隆或下载到本地：

   ```powershell
   git clone https://github.com/KissLoveOrKill/Spotify-Recommendation-System.git
   cd Spotify-Recommendation-System
   ```

## 获取 Spotify API 凭证

- 如果你还没有 Spotify Developer 账户或应用，请先前往开发者控制台创建应用：

   https://developer.spotify.com/dashboard/applications

- 创建应用后，在应用设置中添加重定向 URI：`http://127.0.0.1:5000/callback`（与本项目 `SPOTIPY_REDIRECT_URI` 保持一致）。
- 在应用详情页复制 `Client ID` 和 `Client Secret`，并填入本地 `.env`（见下一节）。


2. **配置环境变量**

   复制 `.env.example` 文件为 `.env`，并填写你的 Spotify API 密钥：

   ```powershell
   copy .env.example .env
   ```

   > ⚠️ **注意**：`.env` 文件仅用于本地开发，务必不要提交到 GitHub。

3. **安装依赖**

   使用 pip 安装项目依赖：

   ```powershell
   pip install -r spotify_rec_system\requirements.txt
   ```

4. **启动服务**

   运行以下命令启动 Flask 应用：

   ```powershell
   python spotify_rec_system\app.py
   ```

5. **访问应用**

   打开浏览器，访问 [http://127.0.0.1:5000](http://127.0.0.1:5000) 即可体验推荐功能。

## 注意事项

- **数据管理**：
  - 推荐将大数据文件（如 `dataset.csv`）和模型权重保留在本地，避免推送到远程仓库。
  - 数据文件夹 `data/` 已在 `.gitignore` 中配置，无需额外操作。

- **安全性**：
  - 请勿在公共仓库中泄露 API 密钥或其他敏感信息。
  - 使用 `.env` 文件管理本地开发环境的敏感配置。

## 数据集（下载与解压）

- 本项目的离线歌曲特征数据应放在 `spotify_rec_system/data/dataset.csv`。
- 为避免直接将超大 CSV 推送到仓库，建议在仓库中保留压缩包 `spotify_rec_system/data/dataset.zip`，并在本地解压使用。
- 若仓库中已有 `dataset.zip`（或你已从外部下载），请在项目根目录通过 PowerShell 解压：

```powershell
# 在项目根目录运行（Windows PowerShell）
Expand-Archive -Path "spotify_rec_system\\data\\dataset.zip" -DestinationPath "spotify_rec_system\\data" -Force

# 解压后验证 CSV 是否存在：
Test-Path "spotify_rec_system\\data\\dataset.csv"
```

- 如果你从外部下载数据集，请将 `dataset.csv` 放到 `spotify_rec_system/data/` 下并确保文件名为 `dataset.csv`。
- 注意：数据集可能较大，推荐使用云存储（Google Drive、OneDrive 等）分享下载链接，或使用 Git LFS 管理大文件。

## 技术栈

- **前端**：Flask
- **机器学习**：PyTorch
- **数据库**：本地 CSV 文件
- **推荐算法**：MLP Autoencoder + 向量相似度搜索

## 贡献指南

欢迎对本项目提出建议或贡献代码！请提交 Pull Request 或创建 Issue 与我们交流。


