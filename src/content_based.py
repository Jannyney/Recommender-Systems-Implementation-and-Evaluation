# content_based.py
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import os
import math

K_VALUES = [1, 3, 5]

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

def generate_recommendations(user_profiles, movie_genre_matrix, movie_ids, train, top_k):
    recommendations = {}
    train_user_movies = defaultdict(set)
    for _, row in train.iterrows():
        train_user_movies[row['userId']].add(row['movieId'])

    for user_id, profile in user_profiles.items():
        sims = cosine_similarity([profile], movie_genre_matrix).flatten()
        seen = train_user_movies[user_id]
        candidates = [(mid, score) for mid, score in zip(movie_ids, sims) if mid not in seen]
        top_movies = sorted(candidates, key=lambda x: x[1], reverse=True)[:10]
        recommendations[user_id] = [mid for mid, _ in top_movies]
    return recommendations

def evaluate(recommendations, test, k_values):
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
                "Model": "content-based",
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
    recommendations = generate_recommendations(user_profiles, genre_matrix, movie_ids, train, K_VALUES)
    results = evaluate(recommendations, test, K_VALUES)

    os.makedirs("../results", exist_ok=True)
    results.to_csv("../results/content_based_results.csv", index=False)
    print(results)

    # Save top-10 recommendations for each user
    top10_df = pd.DataFrame([
        {"userId": user_id, "recommended_movies": recs}
        for user_id, recs in recommendations.items()
    ])
    top10_df.to_csv("../results/content_based_top10.csv", index=False)

if __name__ == '__main__':
    main()