import os
# Fix for OpenMP runtime error on Windows
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from flask import Flask, session, request, redirect, render_template, url_for, jsonify
import spotipy
from spotipy.oauth2 import SpotifyOAuth, SpotifyClientCredentials
import time
import pandas as pd
import threading
from dotenv import load_dotenv
# from recommender import ContentBasedRecommender  <-- Moved to inside init_model_background

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['SESSION_COOKIE_NAME'] = 'Spotify Cookie'

# Initialize Recommender Engine (Global Instance)
# 使用后台线程初始化，避免阻塞 Flask 启动
global_recommender = None
is_model_ready = False
is_training_started = False # 新增标志位
init_progress = {'percent': 0, 'message': '等待初始化...'}

def update_progress(percent, message):
    global init_progress
    init_progress['percent'] = percent
    init_progress['message'] = message

def init_model_background():
    global global_recommender, is_model_ready
    print("="*50)
    print("[SYSTEM] 正在初始化全局推荐引擎 (后台运行)...")
    try:
        # Lazy import to prevent startup crashes due to Torch/OpenMP conflicts
        from recommender import ContentBasedRecommender
        
        global_recommender = ContentBasedRecommender(progress_callback=update_progress)
        is_model_ready = True
        print("[SYSTEM] 全局推荐引擎初始化完成！")
    except Exception as e:
        print(f"[ERROR] 初始化失败: {e}")
        import traceback
        traceback.print_exc()
        update_progress(0, f"初始化失败: {str(e)}")
    print("="*50)

# 移除自动启动，改为在 /status 请求时触发
# init_thread = threading.Thread(target=init_model_background)
# init_thread.start()

@app.route('/status')
def get_status():
    global is_training_started
    
    # 当前端 loading 页面第一次轮询状态时，才启动训练线程
    # 这样可以确保用户已经看到了 loading 页面
    if not is_training_started and not is_model_ready:
        is_training_started = True
        threading.Thread(target=init_model_background).start()
        
    return jsonify({
        'ready': is_model_ready,
        'progress': init_progress
    })

@app.before_request
def check_model_ready():
    # 允许静态资源和状态检查请求通过
    if request.endpoint in ['static', 'get_status']:
        return
    
    # 如果模型未就绪，拦截所有页面请求并显示加载页
    if not is_model_ready:
        return render_template('loading.html')

# Spotify Configuration
# Ensure no whitespace issues
SPOTIPY_CLIENT_ID = os.getenv('SPOTIPY_CLIENT_ID', '').strip()
SPOTIPY_CLIENT_SECRET = os.getenv('SPOTIPY_CLIENT_SECRET', '').strip()
SPOTIPY_REDIRECT_URI = 'http://127.0.0.1:5000/callback'
SCOPE = 'user-library-read playlist-read-private playlist-read-collaborative user-read-private user-read-email'

def create_spotify_oauth():
    return SpotifyOAuth(
        client_id=SPOTIPY_CLIENT_ID,
        client_secret=SPOTIPY_CLIENT_SECRET,
        redirect_uri=SPOTIPY_REDIRECT_URI,
        scope=SCOPE
    )

def get_spotify_client():
    """
    Get a Spotify client using Client Credentials Flow.
    This is better for fetching public data (like audio features) 
    as it is less likely to run into user-specific permission issues.
    """
    # Disable cache to prevent using stale tokens
    client_credentials_manager = SpotifyClientCredentials(
        client_id=SPOTIPY_CLIENT_ID,
        client_secret=SPOTIPY_CLIENT_SECRET,
        cache_handler=None
    )
    return spotipy.Spotify(client_credentials_manager=client_credentials_manager)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login')
def login():
    # Clear cache automatically before login to ensure fresh permissions
    try:
        for filename in os.listdir('.'):
            if filename.startswith('.cache'):
                os.remove(filename)
    except Exception as e:
        print(f"Error removing cache: {e}")

    # Force show_dialog=True to ensure user can switch accounts if needed
    # and to force a fresh token grant
    sp_oauth = create_spotify_oauth()
    # 添加 show_dialog=True，强制弹出授权页面
    auth_url = sp_oauth.get_authorize_url() + "&show_dialog=true"
    return redirect(auth_url)

@app.route('/logout')
def logout():
    session.clear()
    # Remove all cache files (handle .cache and .cache-username)
    try:
        for filename in os.listdir('.'):
            if filename.startswith('.cache'):
                os.remove(filename)
    except Exception as e:
        print(f"Error removing cache: {e}")
    return redirect(url_for('index'))

@app.route('/callback')
def callback():
    sp_oauth = create_spotify_oauth()
    session.clear()
    code = request.args.get('code')
    
    token_info = None
    max_retries = 3
    last_error = None

    for i in range(max_retries):
        try:
            # 尝试获取 Token，增加重试机制以应对网络波动 (503 Error)
            # check_cache=False 避免读取旧缓存
            token_info = sp_oauth.get_access_token(code, check_cache=False)
            break
        except Exception as e:
            last_error = e
            print(f"[WARN] Token exchange failed (Attempt {i+1}/{max_retries}): {e}")
            if i < max_retries - 1:
                time.sleep(2) # 等待2秒后重试
    
    if not token_info:
        return render_template('error.html', message=f"登录失败，无法连接到 Spotify 服务器。<br>请检查您的网络连接（可能需要开启全局代理）并重试。<br>错误详情: {last_error}")

    session['token_info'] = token_info
    return redirect(url_for('select_playlist'))

def get_token():
    token_info = session.get('token_info', None)
    if not token_info:
        return None
    now = int(time.time())
    is_expired = token_info['expires_at'] - now < 60
    if is_expired:
        sp_oauth = create_spotify_oauth()
        token_info = sp_oauth.refresh_access_token(token_info['refresh_token'])
    return token_info

@app.route('/select_playlist')
def select_playlist():
    token_info = get_token()
    if not token_info:
        return redirect(url_for('login'))
    
    sp = spotipy.Spotify(auth=token_info['access_token'])
    
    # Debug: Print current user info to console to verify email
    try:
        current_user = sp.current_user()
        print(f"\n[DEBUG] 当前登录用户: {current_user.get('display_name')}")
        print(f"[DEBUG] 注册邮箱 (请确保此邮箱在白名单中): {current_user.get('email')}\n")
    except Exception as e:
        print(f"[DEBUG] 获取用户信息失败: {e}")

    # Fetch user's playlists
    playlists = []
    results = sp.current_user_playlists(limit=50)
    while results:
        for item in results['items']:
            # Show all playlists, regardless of track count
            playlists.append({
                'id': item['id'],
                'name': item['name'],
                'total_tracks': item['tracks']['total'],
                'image': item['images'][0]['url'] if item['images'] else None
            })
        if results['next']:
            results = sp.next(results)
        else:
            results = None
            
    # Get current user profile for display
    user_profile = None
    try:
        user_profile = sp.current_user()
    except:
        pass

    return render_template('select_playlist.html', playlists=playlists, user_profile=user_profile)

@app.route('/recommend', methods=['POST'])
def recommend():
    token_info = get_token()
    if not token_info:
        return redirect(url_for('login'))
    
    # Use User Token for accessing playlist (User Data)
    sp_user = spotipy.Spotify(auth=token_info['access_token'])
    # Use Client Token for analysis (Public Data) - More stable
    sp_public = get_spotify_client()
    
    playlist_id = request.form.get('playlist_id')
    
    # 1. Get tracks from the selected playlist
    results = sp_user.playlist_tracks(playlist_id)
    tracks = results['items']
    while results['next']:
        results = sp_user.next(results)
        tracks.extend(results['items'])
        
    # 2. Extract Track IDs
    track_ids = []
    for item in tracks:
        # Ensure it's a track (not episode), has an ID, and is not a local file
        if (item['track'] and 
            item['track']['id'] and 
            item['track']['type'] == 'track' and 
            not item['track'].get('is_local', False)):
            track_ids.append(item['track']['id'])
            
    if not track_ids:
        return render_template('error.html', message="该歌单似乎是空的，或者包含的歌曲无法被 Spotify 识别（例如本地文件）。<br>请尝试选择另一个歌单。")

    # Limit to analyzing first 100 tracks to avoid rate limits and slowness for this demo
    track_ids = track_ids[:100]
    
    # 3. 使用全局推荐引擎 (Global Recommender)
    # from recommender import ContentBasedRecommender
    # recommender = ContentBasedRecommender()
    
    print("[INFO] 正在调用全局推荐算法...")
    try:
        # 传入所有 track_ids 作为种子，让算法自己计算平均值
        # 注意：算法内部会过滤掉不在数据库中的 ID
        rec_results = global_recommender.recommend(track_ids, limit=50)
        
        rec_tracks = []
        if rec_results:
            print(f"[SUCCESS] 算法成功推荐了 {len(rec_results)} 首歌曲")
            for item in rec_results:
                # 构造前端需要的数据格式
                # 注意：CSV 中的列名可能需要适配
                rec_tracks.append({
                    'id': item['id'], # 添加 ID 以便前端链接
                    'name': item.get('track_name', item.get('name', 'Unknown')),
                    'artist': item.get('artists', item.get('artist_name', 'Unknown')),
                    # CSV 通常没有封面图，这里可以用默认图，或者再调用一次 Spotify API 获取封面 (如果需要)
                    'album_art': None, 
                    'preview_url': None, # CSV 通常没有试听链接
                    'external_url': f"https://open.spotify.com/track/{item['id']}"
                })
                
            # 可选：如果想要封面图，可以批量调用一次 Spotify API (tracks endpoint)
            # 但为了速度和避免 API 限制，这里先留空或使用占位符
            # 如果您非常需要封面，可以取消下面这段注释
            try:
                rec_ids = [item['id'] for item in rec_results][:50]
                sp_tracks = sp_public.tracks(rec_ids)
                for i, t_info in enumerate(sp_tracks['tracks']):
                    if t_info and i < len(rec_tracks):
                        rec_tracks[i]['album_art'] = t_info['album']['images'][0]['url'] if t_info['album']['images'] else None
                        rec_tracks[i]['preview_url'] = t_info['preview_url']
                        rec_tracks[i]['external_url'] = t_info['external_urls']['spotify'] # 更新为官方链接
            except Exception as e_img:
                print(f"[WARN] 获取推荐歌曲封面失败: {e_img}")
            
            # 获取当前用户信息用于显示
            user_profile = None
            try:
                user_profile = sp_user.current_user()
            except:
                pass

            return render_template('results.html', tracks=rec_tracks, user_profile=user_profile, playlist_id=playlist_id)
            
        else:
            print("[WARN] 推荐算法未返回任何结果 (可能是种子歌曲都不在数据库中)")
            return render_template('error.html', message="无法生成推荐：您的歌单中的歌曲似乎都不在我们的离线数据库中。<br>请尝试选择包含更多热门歌曲的歌单。")

    except Exception as e:
        print(f"[ERROR] 推荐算法运行出错: {e}")
        import traceback
        traceback.print_exc()
        return render_template('error.html', message=f"推荐算法内部错误: {e}")

    # (以下旧代码已废弃)
    """
    # 4. Calculate Average Features (The "Algorithm")
    # We create a profile of the user's taste based on this playlist
    df = pd.DataFrame(all_audio_features)
    ...
    """

@app.route('/playlist', methods=['GET'])
def playlist_detail():
    token_info = get_token()
    if not token_info:
        return redirect(url_for('login'))
    
    sp_user = spotipy.Spotify(auth=token_info['access_token'])
    playlist_id = request.args.get('playlist_id')
    
    if not playlist_id:
        return redirect(url_for('select_playlist'))

    try:
        # Get Playlist Metadata
        playlist_meta = sp_user.playlist(playlist_id, fields="name,images,description")
        playlist_info = {
            'name': playlist_meta['name'],
            'image': playlist_meta['images'][0]['url'] if playlist_meta['images'] else None,
            'description': playlist_meta['description']
        }

        # Get tracks (Basic Info Only)
        results = sp_user.playlist_tracks(playlist_id)
        tracks = results['items']
        while results['next']:
            results = sp_user.next(results)
            tracks.extend(results['items'])
            
        track_data = []
        for item in tracks:
            if (item['track'] and item['track']['id'] and item['track']['type'] == 'track'):
                track_data.append({
                    'id': item['track']['id'],
                    'name': item['track']['name'],
                    'artist': item['track']['artists'][0]['name'],
                    'album_art': item['track']['album']['images'][0]['url'] if item['track']['album']['images'] else None
                })
        
        return render_template('playlist.html', tracks=track_data, playlist_id=playlist_id, playlist_info=playlist_info, user_profile=sp_user.current_user())
    except Exception as e:
        return render_template('error.html', message=f"获取歌单失败: {e}")

@app.route('/track/<track_id>')
def track_detail(track_id):
    token_info = get_token()
    if not token_info:
        return redirect(url_for('login'))
    
    sp_user = spotipy.Spotify(auth=token_info['access_token'])
    sp_public = get_spotify_client()
    
    from dataset_service import SpotifyDataset
    dataset = SpotifyDataset.get_instance()
    
    try:
        # 1. Get Online Metadata (Name, Artist, Popularity, Release Date)
        track_info = sp_user.track(track_id)
        track_name = track_info['name']
        artist_name = track_info['artists'][0]['name']
        
        # 2. Get Offline Features (CSV)
        # First try by ID
        offline_feats = dataset.get_track_features(track_id)
        
        # If not found by ID, try by Name + Artist (Fuzzy Match)
        if not offline_feats:
            print(f"[INFO] ID {track_id} 未命中，尝试使用名称匹配: {track_name} - {artist_name}")
            offline_feats = dataset.get_track_features_by_name(track_name, artist_name)
            if offline_feats:
                print(f"[SUCCESS] 名称匹配成功!")
        
        # 3. Get Genre Logic
        # Priority: CSV Genre > Spotify Artist Genre
        genres = "暂无流派信息 (Unknown)"
        
        # Try to use CSV genre first
        if offline_feats and offline_feats.get('genre') and str(offline_feats['genre']).lower() != 'unknown':
            genres = str(offline_feats['genre']).title()
        else:
            # Fallback to Spotify Artist API
            try:
                artist_id = track_info['artists'][0]['id']
                artist_info = sp_public.artist(artist_id)
                if artist_info and artist_info.get('genres'):
                    genres_list = [g.title() for g in artist_info.get('genres', [])[:3]]
                    genres = ", ".join(genres_list)
            except Exception as e:
                print(f"[WARN] 获取在线流派失败: {e}")

        # Merge Data
        def fmt_feat(val):
            try:
                v = float(val)
                # 对于 BPM (Tempo)，不需要显示"极低"
                if v > 30 and v < 250: # 简单的 BPM 范围判断
                    return f"{v:.0f}" # BPM 通常取整数
                
                if v < 0.01:
                    return "Low"
                return f"{v:.3f}"
            except:
                return '-'

        features = {
            'danceability': fmt_feat(offline_feats.get('danceability')) if offline_feats else '-',
            'energy': fmt_feat(offline_feats.get('energy')) if offline_feats else '-',
            'valence': fmt_feat(offline_feats.get('valence')) if offline_feats else '-',
            'acousticness': fmt_feat(offline_feats.get('acousticness')) if offline_feats else '-',
            'instrumentalness': fmt_feat(offline_feats.get('instrumentalness')) if offline_feats else '-',
            'tempo': fmt_feat(offline_feats.get('tempo')) if offline_feats else '-',
            'genre': genres
        }
        
        track_display = {
            'name': track_info['name'],
            'artist': track_info['artists'][0]['name'],
            'album_art': track_info['album']['images'][0]['url'] if track_info['album']['images'] else None,
            'release_date': track_info['album']['release_date'],
            'popularity': track_info['popularity'],
            'external_url': track_info['external_urls']['spotify']
        }
        
        source = "本地数据集 (CSV)" if offline_feats else "无 (未在数据库中找到)"
        
        return render_template('track.html', track=track_display, features=features, source=source)
        
    except Exception as e:
        return render_template('error.html', message=f"获取歌曲详情失败: {e}")

@app.route('/recommend_from_playlist', methods=['POST'])
def recommend_from_playlist():
    # Wrapper to reuse existing logic but handle input from playlist page
    playlist_id = request.form.get('playlist_id')
    
    # We need to fetch track IDs first to pass to recommender
    token_info = get_token()
    if not token_info:
        return redirect(url_for('login'))
    sp_user = spotipy.Spotify(auth=token_info['access_token'])
    
    try:
        results = sp_user.playlist_tracks(playlist_id)
        tracks = results['items']
        while results['next']:
            results = sp_user.next(results)
            tracks.extend(results['items'])
            
        track_ids = [item['track']['id'] for item in tracks if item['track'] and item['track']['id']]
        
        # Call the recommender logic (reusing code from previous /recommend or implementing here)
        # Let's implement it cleanly here
        from recommender import ContentBasedRecommender
        recommender = ContentBasedRecommender()
        
        rec_results = recommender.recommend(track_ids, limit=50)
        
        rec_tracks = []
        if rec_results:
            # Optional: Fetch covers
            sp_public = get_spotify_client()
            try:
                rec_ids = [item['id'] for item in rec_results][:50]
                # Split into chunks of 50 for API
                sp_tracks_info = []
                for i in range(0, len(rec_ids), 50):
                    chunk = rec_ids[i:i+50]
                    resp = sp_public.tracks(chunk)
                    sp_tracks_info.extend(resp['tracks'])
                
                for i, item in enumerate(rec_results):
                    if i < len(sp_tracks_info) and sp_tracks_info[i]:
                        t_info = sp_tracks_info[i]
                        rec_tracks.append({
                            'id': t_info['id'],
                            'name': t_info['name'],
                            'artist': t_info['artists'][0]['name'],
                            'album_art': t_info['album']['images'][0]['url'] if t_info['album']['images'] else None,
                            'preview_url': t_info['preview_url'],
                            'external_url': t_info['external_urls']['spotify']
                        })
                    else:
                        # Fallback if API fails for some reason
                        rec_tracks.append({
                            'id': item['id'],
                            'name': item.get('track_name', 'Unknown'),
                            'artist': item.get('artist_name', 'Unknown'),
                            'album_art': None,
                            'external_url': f"https://open.spotify.com/track/{item['id']}"
                        })
            except Exception as e:
                print(f"[WARN] Fetching covers failed: {e}")
                # Fallback without covers
                for item in rec_results:
                    rec_tracks.append({
                        'id': item['id'],
                        'name': item.get('track_name', 'Unknown'),
                        'artist': item.get('artist_name', 'Unknown'),
                        'album_art': None,
                        'external_url': f"https://open.spotify.com/track/{item['id']}"
                    })
            
            # Get current user profile for display
            user_profile = None
            try:
                user_profile = sp_user.current_user()
            except:
                pass

            return render_template('results.html', tracks=rec_tracks, user_profile=user_profile, playlist_id=playlist_id)
        else:
             return render_template('error.html', message="无法生成推荐：您的歌单中的歌曲似乎都不在我们的离线数据库中。")

    except Exception as e:
        return render_template('error.html', message=f"推荐失败: {e}")

# Deprecated routes (kept for reference or removed)
# @app.route('/get_features', methods=['POST']) ... 


if __name__ == '__main__':
    print("Starting Flask server on port 5000...")
    # use_reloader=False prevents the "forrtl: error (200)" crash with PyTorch on Windows
    try:
        app.run(debug=True, use_reloader=False, port=5000)
    except Exception as e:
        print(f"Flask failed to start: {e}")
