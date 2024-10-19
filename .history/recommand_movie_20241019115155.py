import pandas as pd
import numpy as np
from functools import wraps
import time

def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args,**kwargs):
        start_time = time.perf_counter()
        result = func(*args,**kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'function {func.__name__} run_time: {total_time:.4f} s')
        return result
    return timeit_wrapper

class RecommandMovie:
    def __init__(self,data_path,recommand_method):
        self.rating_matrix = pd.read_csv(self.data_path + 'train_ratings.csv')
        self.movie2user_data = []
        self.user2movie_data = []
        self.user_similarity = self.cosine_similarity_user()

    def preprocess_data(self):
        self.user_movie_matrix 
        self.movie_user_matrix
    def train_model(self):
        raise NotImplementedError
    def cosine_similarity_item(self):
        raise NotImplementedError
    def cosine_similarity_user(self):
        # Normalize the rating matrix
        norm_matrix = self.rating_matrix.sub(self.rating_matrix.mean(axis=1), axis=0).fillna(0)
        dot_product = np.dot(norm_matrix, norm_matrix.T)
        norm = np.linalg.norm(norm_matrix, axis=1)
        similarity = dot_product / np.outer(norm, norm)
        np.fill_diagonal(similarity, 0)  # Fill diagonal with zeros to exclude self-similarity
        return pd.DataFrame(similarity, index=self.rating_matrix.index, columns=self.rating_matrix.index)

    
    def predict_ratings_for_user(self,user_id, rating_matrix, user_similarity, k=10):
        # 获取与目标用户相似的其他用户
        similar_users = user_similarity[user_id].drop(user_id).dropna()
        # 选择最相似的前 k 个用户
        similar_users = similar_users.nlargest(k)
        
        # 获取用户未评分的电影
        user_ratings = rating_matrix.loc[user_id]
        unrated_movies = user_ratings[user_ratings.isna()].index
        
        # 只选择最相似的用户评分矩阵
        similar_user_ratings = rating_matrix.loc[similar_users.index, unrated_movies]
        
        # 对于没有评分的电影，先用平均值填充
        global_mean = rating_matrix.mean(axis=0)
        similar_user_ratings = similar_user_ratings.fillna(global_mean[unrated_movies])

        # 进行加权平均的向量化计算
        weights = np.array(similar_users.values).reshape(-1, 1)
        weighted_sum = np.dot(weights.T, similar_user_ratings)
        normalization = np.sum(weights)
        
        # 计算预测评分
        predicted_ratings = weighted_sum.flatten() / normalization
        
        return pd.Series(predicted_ratings, index=unrated_movies)
    
    def evaluate_model(self):
        raise NotImplementedError
    
    @timeit
    def recommend_movies_user(self,user_id, rating_matrix, user_similarity, top_n=10, k=10):
        # 预测用户对未评分电影的评分
        predicted_ratings = self.predict_ratings_for_user(user_id, rating_matrix, user_similarity, k)
        # 按照预测评分排序，推荐前 top_n 个电影
        recommendations = predicted_ratings.sort_values(ascending=False).head(top_n)
        return recommendations
    
    

if __name__ == '__main__':
    