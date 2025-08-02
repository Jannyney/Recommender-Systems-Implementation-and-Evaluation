import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
import os
import math

K_VALUES = [1, 3, 5]
TOP_N = 50  # Initial filter size before ranking

GENRES = ['Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime',
          'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical',
          'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

def load_data():
    ratings = pd.read_csv("ml-latest-small/ml-latest-small/ratings.csv")
    movies = pd.read_csv("ml-latest-small/ml-latest-small/movies.csv")
    return ratings, movies

def split_data(ratings):
    train = ratings.iloc[::2].copy()
    test = ratings.iloc[1::2].copy()
    return train, test

def build_genre_matrix(movies):
    genre_matrix = []
    for genres in movies['genres']:
        genre_vec = [0] * len(GENRES)
        for g in genres.split('|'):
            if g in GENRES:
                genre_vec[GENRES.index(g)] = 1
        genre_matrix.append(genre_vec)
    return np.array(genre_matrix)

def build_user_profiles(train, movies, movie_genre_matrix):
    user_profiles = defaultdict(lambda: np.zeros(len(GENRES)))
    movie_id_to_idx = {mid: i for i, mid in enumerate(movies['movieId'])}

    for _, row in train.iterrows():
        if row['rating'] >= 2.5 and row['movieId'] in movie_id_to_idx:
            idx = movie_id_to_idx[row['movieId']]
            user_profiles[row['userId']] += movie_genre_matrix[idx]
    return dict(user_profiles)

def build_user_item_matrix(train):
    return train.pivot(index='userId', columns='movieId', values='rating').fillna(0)

def collaborative_topn(user_item_matrix, top_n):
    sim_matrix = cosine_similarity(user_item_matrix)
    sim_df = pd.DataFrame(sim_matrix, index=user_item_matrix.index, columns=user_item_matrix.index)

    recommendations = defaultdict(list)
    for user_id in user_item_matrix.index:
        similar_users = sim_df[user_id].drop(user_id).nlargest(10).index
        user_seen = set(user_item_matrix.columns[user_item_matrix.loc[user_id] > 0])
        scores = defaultdict(float)

        for sim_user in similar_users:
            sim_score = sim_df.loc[user_id, sim_user]
            for movie_id, rating in user_item_matrix.loc[sim_user].items():
                if movie_id not in user_seen and rating > 0:
                    scores[movie_id] += sim_score * rating

        top_movies = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        recommendations[user_id] = dict(top_movies)
    return recommendations

def content_topn(user_profiles, movie_genre_matrix, movie_ids, train, top_n):
    recommendations = {}
    train_user_movies = defaultdict(set)
    for _, row in train.iterrows():
        train_user_movies[row['userId']].add(row['movieId'])

    for user_id, profile in user_profiles.items():
        sims = cosine_similarity([profile], movie_genre_matrix).flatten()
        seen = train_user_movies[user_id]
        candidates = [(mid, score) for mid, score in zip(movie_ids, sims) if mid not in seen]
        top_movies = sorted(candidates, key=lambda x: x[1], reverse=True)[:top_n]
        recommendations[user_id] = dict(top_movies)
    return recommendations

def hybrid_rank_by_content(cf_recs, user_profiles, movie_genre_matrix, movie_ids):
    movie_id_to_idx = {mid: i for i, mid in enumerate(movie_ids)}
    final_recs = {}
    for user_id, candidates in cf_recs.items():
        if user_id not in user_profiles:
            continue
        profile = user_profiles[user_id]
        ranked = []
        for mid, _ in candidates.items():
            if mid in movie_id_to_idx:
                idx = movie_id_to_idx[mid]
                sim = cosine_similarity([profile], [movie_genre_matrix[idx]])[0][0]
                ranked.append((mid, sim))
        final_recs[user_id] = [mid for mid, _ in sorted(ranked, key=lambda x: x[1], reverse=True)]
    return final_recs

def hybrid_rank_by_cf(content_recs, user_item_matrix):
    sim_matrix = cosine_similarity(user_item_matrix)
    sim_df = pd.DataFrame(sim_matrix, index=user_item_matrix.index, columns=user_item_matrix.index)
    final_recs = {}
    
    for user_id, candidates in content_recs.items():
        if user_id not in user_item_matrix.index:
            continue
        
        similar_users = sim_df[user_id].drop(user_id).nlargest(10).index
        scores = defaultdict(float)
        
        for sim_user in similar_users:
            sim_score = sim_df.loc[user_id, sim_user]
            for mid in candidates:
                rating = user_item_matrix.loc[sim_user].get(mid, 0)
                if rating > 0:
                    scores[mid] += sim_score * rating
        
        # Add fallback CB items if fewer than 10 CF-ranked items
        if len(scores) < 10:
            missing = [mid for mid in candidates if mid not in scores]
            scores.update({mid: 0.0001 for mid in missing[:10 - len(scores)]})

        # Ensure top-N items are returned
        final_recs[user_id] = [mid for mid, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]
    
    return final_recs


def evaluate(recommendations, test, k_values, model_name):
    test_by_user = defaultdict(set)
    for _, row in test.iterrows():
        test_by_user[row['userId']].add(row['movieId'])

    rows = []
    for k in k_values:
        total_precision, total_recall, total_f1, total_mrr, total_ndcg, total_diversity = 0, 0, 0, 0, 0, 0
        valid_users = 0
        for user_id, recs in recommendations.items():
            if user_id not in test_by_user:
                continue
            recs_k = recs[:k]
            if not recs_k:  # Prevent division on empty list
                continue
            recs_k = recs[:k]
            rel = test_by_user[user_id]
            hits = [int(mid in rel) for mid in recs_k]
            precision = np.mean(hits)
            recall = np.sum(hits) / len(rel)
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
            mrr = 1 / (hits.index(1) + 1) if 1 in hits else 0
            dcg = sum((2**hit - 1) / math.log2(idx + 2) for idx, hit in enumerate(hits))
            idcg = sum(1 / math.log2(i + 2) for i in range(min(len(rel), k)))
            ndcg = dcg / idcg if idcg else 0
            diversity = len(set(recs_k)) / k

            total_precision += precision
            total_recall += recall
            total_f1 += f1
            total_mrr += mrr
            total_ndcg += ndcg
            total_diversity += diversity
            valid_users += 1

        if valid_users:
            rows.append({
                "Model": model_name,
                "k": k,
                "Precision": total_precision / valid_users,
                "Recall": total_recall / valid_users,
                "F1": total_f1 / valid_users,
                "MRR": total_mrr / valid_users,
                "nDCG": total_ndcg / valid_users,
                "Diversity": total_diversity / valid_users
            })
    return pd.DataFrame(rows)

def main():
    ratings, movies = load_data()
    train, test = split_data(ratings)
    movie_ids = movies['movieId'].tolist()
    genre_matrix = build_genre_matrix(movies)
    user_profiles = build_user_profiles(train, movies, genre_matrix)
    user_item_matrix = build_user_item_matrix(train)

    cf_filtered = collaborative_topn(user_item_matrix, TOP_N)
    cb_filtered = content_topn(user_profiles, genre_matrix, movie_ids, train, TOP_N)

    hybrid1 = hybrid_rank_by_content(cf_filtered, user_profiles, genre_matrix, movie_ids)
    hybrid2 = hybrid_rank_by_cf(cb_filtered, user_item_matrix)

    eval1 = evaluate(hybrid1, test, K_VALUES, "hybrid-cf-then-cb")
    eval2 = evaluate(hybrid2, test, K_VALUES, "hybrid-cb-then-cf")
    os.makedirs("../results", exist_ok=True)
    pd.concat([eval1, eval2]).to_csv("../results/hybrid_results.csv", index=False)

    # Save top 10 recs
    pd.DataFrame([
        {"userId": uid, "recommended_movies": recs[:10]}
        for uid, recs in hybrid1.items()
    ]).to_csv("../results/hybrid_cf_then_cb_top10.csv", index=False)

    pd.DataFrame([
        {"userId": uid, "recommended_movies": recs[:10]}
        for uid, recs in hybrid2.items()
    ]).to_csv("../results/hybrid_cb_then_cf_top10.csv", index=False)

main()
