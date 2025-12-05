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

注意：
- 请不要把大型数据文件（`data/`）和模型权重直接推送到公共仓库；如果需要版本化模型，请使用 Git LFS。
- 将您的 `SPOTIPY_CLIENT_ID`、`SPOTIPY_CLIENT_SECRET` 放在本地 `.env` 文件中（该文件被 `.gitignore` 忽略）。
