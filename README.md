# Spotify Recommender (Local)

一个本地的 Spotify 风格推荐演示项目。

- Flask 前端 + 本地离线歌曲特征数据库
- 使用 PyTorch MLP Autoencoder 将音频特征映射为 32 维向量
- 基于向量相似度（余弦/点积）做内容推荐

目录结构（关键文件）：

- `spotify_rec_system/app.py` - Flask 应用主入口
- `spotify_rec_system/recommender.py` - 推荐引擎（MLP Autoencoder + 相似度搜索）
- `spotify_rec_system/dataset_service.py` - 离线数据加载器（CSV）
- `spotify_rec_system/templates/` - 前端模板
- `data/` - 建议本地保留的大数据文件夹（已在 `.gitignore` 中忽略）

快速运行（示例）：
```powershell
cd "C:\Users\LYK\Desktop\Study\Practice\Spotify"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r spotify_rec_system\requirements.txt
python spotify_rec_system\app.py
```

**注意：**
- 请不要把大型数据文件（`data/`）和模型权重直接推送到公共仓库；如果需要版本化模型，请使用 Git LFS。

**环境变量（`.env`）**

- 本项目使用 `python-dotenv` 的 `load_dotenv()`（见 `spotify_rec_system/app.py`），它会从当前工作目录或父目录中查找并加载 `.env` 文件。
- 推荐做法：把 `.env` 放在项目根目录（与本 `README.md` 同级）。示例文件为 `.env.example`，已包含在仓库中；复制并填写后请勿提交真实的 `.env`。示例：

```dotenv
# .env (不要提交到仓库)
SPOTIPY_CLIENT_ID=your_spotify_client_id_here
SPOTIPY_CLIENT_SECRET=your_spotify_client_secret_here
SPOTIPY_REDIRECT_URI=http://127.0.0.1:5000/callback
FLASK_SECRET=your_flask_secret_optional
```

- 如果你确实需要把 `.env` 放到 `spotify_rec_system/` 子目录也是可以的（`load_dotenv()` 会向上查找），但为了可预测性与 IDE/部署一致性，推荐放在项目根目录。

**本地启动（示例）**

```powershell
cd "C:\Users\LYK\Desktop\Study\Practice\Spotify"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r spotify_rec_system\requirements.txt
python spotify_rec_system\app.py
```

其他注意事项：
- `spotify_rec_system/data/dataset.zip` 较大（约 76MB），GitHub 会提示使用 Git LFS。建议把大型原始数据放在外部存储或使用 Git LFS 管理。
