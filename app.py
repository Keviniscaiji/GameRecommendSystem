from flask import Flask, request, jsonify, render_template
import pickle
import faiss
import numpy as np
from datetime import datetime
import ast
import plotly.express as px
import pandas as pd
from sklearn.decomposition import PCA
from utils.steam_profile_generator import SteamProfileGenerator
from dotenv import load_dotenv
import os
import random
app = Flask(__name__)
# 加载环境变量
load_dotenv()
STEAM_API_KEY = os.environ.get("STEAM_API_KEY")
if not STEAM_API_KEY:
    raise ValueError("STEAM_API_KEY not set in .env")


# 初始化 SteamProfileGenerator
generator = SteamProfileGenerator(STEAM_API_KEY)
# 加载数据和 Faiss 索引
loaded_df = None
index = None

def load_embeddings(filename='game_embeddings.pkl'):
    with open(filename, 'rb') as f:
        loaded_df = pickle.load(f)
    return loaded_df

def preprocess_data(df):
    def parse_date(date_str):
        try:
            return datetime.strptime(date_str, '%Y/%m/%d').timestamp()
        except:
            return 0
    df['release_timestamp'] = df['release_date'].apply(parse_date)

    df['rating'] = df['positive'] / (df['positive'] + df['negative']).replace(0, 1)
    df['rating'] = df['rating'].fillna(0)

    def parse_genres(genre_str):
        try:
            return ast.literal_eval(genre_str) if isinstance(genre_str, str) else []
        except:
            return []
    df['genres_list'] = df['genres'].apply(parse_genres)

    return df

def create_faiss_index(df):
    embedding_matrix = np.stack(df['Description_Embedding'].values).astype('float32')
    faiss.normalize_L2(embedding_matrix)
    index = faiss.IndexFlatIP(embedding_matrix.shape[1])
    index.add(embedding_matrix)
    return index

# 初始化数据和索引
loaded_df = load_embeddings('game_embeddings.pkl')
loaded_df = preprocess_data(loaded_df)
index = create_faiss_index(loaded_df)

def jaccard_similarity(list1, list2):
    set1, set2 = set(list1), set(list2)
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0

import numpy as np

def get_weighted_recommendations(game_id, df, index, top_n=10,w_description = 0.5, w_date=0.2, w_rating=0.3, w_genre=0.3, w_recommend_num=0.2):
    game_idx_list = df.index[df['appid'] == game_id].tolist()
    if len(game_idx_list) == 0:
        return []
    game_idx = game_idx_list[0]

    query_vector = df.iloc[game_idx]['Description_Embedding']
    query_vector = np.array(query_vector).astype('float32').reshape(1, -1)
    faiss.normalize_L2(query_vector)

    # 从 faiss 索引中检索更多的候选元素（top_n * 100）
    top_n = min(top_n, len(df) - 1)  # 确保不超过数据集大小
    distances, indices = index.search(query_vector, min(top_n * 10, len(df) - 1))

    query_genres = df.iloc[game_idx]['genres_list']
    recommendations = []
    max_timestamp = df['release_timestamp'].max()
    max_review_num = df['num_reviews_total'].max()  # 获取 review_num 的最大值

    # 对 review_num 应用对数变换
    log_max_review_num = np.log1p(max_review_num)  # log1p(x) = log(x + 1)

    # 计算所有候选元素的加权分数
    for i in range(1, len(indices[0])):  # 从 1 开始，跳过查询游戏本身
        idx = indices[0][i]
        sim = distances[0][i]

        release_score = (df.iloc[idx]['release_timestamp'] / max_timestamp) if max_timestamp > 0 else 0
        rating_score = df.iloc[idx]['rating']
        review_num = df.iloc[idx]['num_reviews_total']
        if review_num < 0:
            review_num = 0
        # 对数变换并归一化 review_num
        log_review_num = np.log1p(review_num)  # log1p(x) = log(x + 1)
        review_num_score = (log_review_num / log_max_review_num) if log_max_review_num > 0 else 0
        
        genre_score = jaccard_similarity(query_genres, df.iloc[idx]['genres_list'])
        weighted_score = sim * w_description + w_date * release_score + w_rating * rating_score + w_genre * genre_score + w_recommend_num * review_num_score

        recommendations.append({
            'Index': int(idx),
            'ID': int(df.iloc[idx]['appid']),
            'Name': df.iloc[idx]['name'],
            'Similarity': float(sim),
            'Weighted_Score': float(weighted_score),
            'Num_of_reviews': int(review_num),
            'Description': df.iloc[idx]['short_description'][:200],
            'Release_Date': df.iloc[idx]['release_date'],
            'Rating': float(rating_score),
            'Genres': df.iloc[idx]['genres_list']
        })

    # 根据加权分数排序并返回前 top_n 个结果
    recommendations = sorted(recommendations, key=lambda x: x['Weighted_Score'], reverse=True)[:top_n]
    return recommendations

def generate_3d_plot(df, query_idx, recs):
    # 选取查询游戏和推荐游戏的向量
    indices = [query_idx] + [r['Index'] for r in recs]
    vectors = np.array([df.iloc[i]['Description_Embedding'] for i in indices]).astype('float32')
    
    # 使用 PCA 将向量降到 3 维
    pca = PCA(n_components=3)
    vectors_3d = pca.fit_transform(vectors)

    # 准备绘图数据
    plot_data = []
    for i, idx in enumerate(indices):
        if i == 0:
            plot_data.append({
                'PC1': vectors_3d[i, 0],
                'PC2': vectors_3d[i, 1],
                'PC3': vectors_3d[i, 2],
                'Name': f"[Query] {df.iloc[idx]['name']}",
                'Type': 'Query',
                'Similarity': 1.0,
                'Weighted_Score': 1.0,
                'Rating': df.iloc[idx]['rating'],
                'Genres': df.iloc[idx]['genres_list']
            })
        else:
            rec = recs[i-1]
            plot_data.append({
                'PC1': vectors_3d[i, 0],
                'PC2': vectors_3d[i, 1],
                'PC3': vectors_3d[i, 2],
                'Name': rec['Name'],
                'Type': 'Recommendation',
                'Similarity': rec['Similarity'],
                'Weighted_Score': rec['Weighted_Score'],
                'Rating': rec['Rating'],
                'Genres': rec['Genres']
            })
    
    plot_df = pd.DataFrame(plot_data)

    # 利用 Plotly Express 创建 3D 散点图
    fig = px.scatter_3d(
        plot_df,
        x='PC1',
        y='PC2',
        z='PC3',
        color='Type',
        size=[30 if t == 'Query' else 10 for t in plot_df['Type']],
        symbol='Type',
        hover_data=['Name', 'Similarity', 'Weighted_Score', 'Rating', 'Genres'],
        text='Name',
        color_discrete_map={'Query': 'red', 'Recommendation': 'blue'},
        title='Query & Recommendations (PCA 3D)'
    )

    # 设置图表样式以及图例
    fig.update_traces(
        marker=dict(line=dict(width=1, color='DarkSlateGrey')),
        textfont=dict(size=12, color='black')
    )

    fig.update_layout(
        width=800,
        height=600,
        title=dict(x=0.5, font=dict(size=20)),
        scene=dict(
            xaxis_title="PC1",
            yaxis_title="PC2",
            zaxis_title="PC3"
        ),
        showlegend=True,
        legend=dict(
            x=0.8,
            y=0.9,
            bgcolor='rgba(255, 255, 255, 0.5)',
            bordercolor='black',
            borderwidth=1,
            xanchor='right',
            yanchor='top'
        ),
        hovermode='closest'
    )
    
    return fig.to_html(full_html=False)

def get_3d_coordinates(df, query_idx, recs):
    # 收集 Query 游戏和推荐游戏的索引
    indices = [query_idx] + [r['Index'] for r in recs]
    # 获取对应向量
    vectors = [df.iloc[i]['Description_Embedding'] for i in indices]
    vectors = np.array(vectors).astype('float32')

    # PCA 降至 3 维
    pca = PCA(n_components=3)
    vectors_3d = pca.fit_transform(vectors)

    plot_data = []
    for i, idx in enumerate(indices):
        if i == 0:
            plot_data.append({
                'PC1': float(vectors_3d[i, 0]),
                'PC2': float(vectors_3d[i, 1]),
                'PC3': float(vectors_3d[i, 2]),
                'Name': f"[Query] {df.iloc[idx]['name']}",
                'Type': 'Query',
                'Similarity': 1.0,
                'Weighted_Score': 1.0,
                'Rating': df.iloc[idx]['rating'],
                'Genres': df.iloc[idx]['genres_list']
            })
        else:
            rec = recs[i-1]
            plot_data.append({
                'PC1': float(vectors_3d[i, 0]),
                'PC2': float(vectors_3d[i, 1]),
                'PC3': float(vectors_3d[i, 2]),
                'Name': rec['Name'],
                'Type': 'Recommendation',
                'Similarity': rec['Similarity'],
                'Weighted_Score': rec['Weighted_Score'],
                'Rating': rec['Rating'],
                'Genres': rec['Genres']
            })
    return plot_data


# 主页路由（重命名为 home 避免与全局变量 index 冲突）
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/user')
def contact():
    return render_template('user.html')

# 搜索游戏路由
@app.route('/search', methods=['POST'])
def search():
    query = request.form['query'].strip().lower()
    if not query:
        return jsonify({'games': []})

    # 模糊匹配游戏名称
    matched_games = loaded_df[loaded_df['name'].str.lower().str.contains(query, na=False)]
    games_list = matched_games[['appid', 'name']].to_dict(orient='records')
    return jsonify({'games': games_list})

# 推荐路由
@app.route('/recommend', methods=['POST'])
def recommend():
    game_id = int(request.form['game_id'])
    top_n = int(request.form.get('top_n', 10))

    recommendations = get_weighted_recommendations(game_id, loaded_df, index, top_n=top_n)
    if not recommendations:
        return jsonify({'error': f"Game ID {game_id} not found"}), 404

    query_idx = loaded_df.index[loaded_df['appid'] == game_id][0]
    # 获取 PCA 降维后的数据（坐标数据）
    plot_data = get_3d_coordinates(loaded_df, query_idx, recommendations)

    return jsonify({
        'query_game': loaded_df[loaded_df['appid'] == game_id]['name'].iloc[0],
        'recommendations': recommendations,
        'plot_data': plot_data
    })

@app.route('/api/profile', methods=['GET'])
def get_steam_profile():
    steam_id = request.args.get('steam_id')
    if not steam_id:
        return jsonify({"error": "Steam ID is required"}), 400
    
    profile = generator.generate_user_profile(steam_id, max_games_for_achievements=50)
    if not profile:
        return jsonify({"error": "Failed to generate profile"}), 500
    return jsonify(profile)

@app.route('/user_recommend', methods=['POST'])
def user_recommend():
    steam_id = request.form.get('query')
    if not steam_id:
        return jsonify({"error": "Steam ID is required"}), 400

    # 获取 Steam 个人资料
    profile = generator.generate_user_profile(steam_id)
    if not profile or "games" not in profile:
        return jsonify({"error": "Failed to fetch user profile or no games found"}), 400

    games = profile["games"]
    total_games = len(games)

    if total_games == 0:
        return jsonify({"error": "No games found in user's library"}), 400

    # 确定选择游戏数量和每游戏推荐数量
    base_num = 5  # 目标选择 5 个游戏
    rec_per_game = 10  # 每游戏默认推荐 10 个

    if total_games < base_num:
        selected_games = games
        num_selected = len(selected_games)
        rec_per_game = max(1, 50 // num_selected)  # 调整推荐数量，尽量接近 50 个总推荐
    else:
        selected_games = random.sample(games, base_num)

    # 为每个选中的游戏生成推荐
    all_recommendations = []
    for game in selected_games:
        game_id = game["app_id"]
        recs = get_weighted_recommendations(game_id, loaded_df, index, top_n=rec_per_game)
        all_recommendations.extend(recs)

    # 从所有推荐中随机选择 10 个
    final_recommendations = random.sample(all_recommendations, min(10, len(all_recommendations)))

    # 构造返回数据
    result = {
        "user_info": profile["user_info"],
        "selected_games": [{"name": g["name"], "app_id": g["app_id"]} for g in selected_games],
        "recommendations": final_recommendations,
        "message": f"Selected {len(selected_games)} games, each with {rec_per_game} recommendations, final 10 picked from {len(all_recommendations)} total."
    }
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
