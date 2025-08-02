# collaborative_filtering.py
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
import os
import math

K_VALUES = [1, 3, 5]

def load_data():
    ratings = pd.read_csv("ml-latest-small/ml-latest-small/ratings.csv")
    return ratings

def split_data(ratings):
    train = ratings.iloc[::2].copy()
    test = ratings.iloc[1::2].copy()
    return train, test

def build_user_item_matrix(train):
    user_item = train.pivot(index='userId', columns='movieId', values='rating').fillna(0)
    return user_item

def generate_recommendations(user_item_matrix, top_k):
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

        top_movies = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k[-1]]
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
                "Model": "collaborative-filtering",
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
    ratings = load_data()
    train, test = split_data(ratings)
    user_item_matrix = build_user_item_matrix(train)
    recommendations = generate_recommendations(user_item_matrix, K_VALUES)
    results = evaluate(recommendations, test, K_VALUES)

    os.makedirs("../results", exist_ok=True)
    results.to_csv("../results/collaborative_filtering_results.csv", index=False)
    print(results)

    # Save top-10 recommendations for each user
    top10_df = pd.DataFrame([
        {"userId": user_id, "recommended_movies": recs[:10]}
        for user_id, recs in recommendations.items()
    ])
    top10_df.to_csv("../results/collaborative_filtering_top10.csv", index=False)


if __name__ == '__main__':
    main()
