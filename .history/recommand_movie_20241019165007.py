import pandas as pd
import numpy as np
from functools import wraps
import time
import os

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

def printLog(func):
    @wraps(func)
    def print_wrapper(*args,**kwargs):
        func_name = func.__name__
        print(f"func {func_name} is starting")
        result = func(*args,**kwargs)
        print(f"func {func_name} is finished")
        return result
    return print_wrapper

class RecommandMovie:
    @printLog
    def __init__(self,data_path):
        self.rating_data = pd.read_csv(data_path)
        self.preprocess_data()
        self.user_similarity = self.cosine_similarity_user()
        print(self.user_similarity)
        
    @printLog
    def preprocess_data(self): 
        self.rating_matrix = self.rating_data.pivot_table(values='rating',index='user_id',columns=
                                        'movie_id')

    def train_model(self):
        raise NotImplementedError
    def cosine_similarity_item(self):
        raise NotImplementedError
    
    @printLog
    def cosine_similarity_user(self):
        # Normalize the rating matrix
        norm_matrix = self.rating_matrix.sub(self.rating_matrix.mean(axis=1), axis=0).fillna(0)
        dot_product = np.dot(norm_matrix, norm_matrix.T)
        norm = np.linalg.norm(norm_matrix, axis=1)
        epsilon = 1e-8  # avoid norm all is 0
        similarity = dot_product / (np.outer(norm, norm)+epsilon)
        np.fill_diagonal(similarity, 0)  # Fill diagonal with zeros to exclude self-similarity
        return pd.DataFrame(similarity, index=self.rating_matrix.index, columns=self.rating_matrix.index)

    @printLog
    def predict_ratings_for_user(self,user_id, k=10):
        # 获取与目标用户相似的其他用户
        similar_users = self.user_similarity[user_id].drop(user_id).dropna()
        # 选择最相似的前 k 个用户
        similar_users = similar_users.nlargest(k)
        
        # 获取用户未评分的电影
        user_ratings = self.rating_matrix.loc[user_id]
        unrated_movies = user_ratings[user_ratings.isna()].index
        
        # 只选择最相似的用户评分矩阵
        similar_user_ratings = self.rating_matrix.loc[similar_users.index, unrated_movies]
        
        # 对于没有评分的电影，先用平均值填充
        global_mean = self.rating_matrix.mean(axis=0)
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
    @printLog
    def recommend_movies_user_userrc(self,top_n=1, k=10):
        result = []
        user_ids = list(self.rating_data.index)[:2]
        # 预测用户对未评分电影的评分
        for user_id in user_ids:
            predicted_ratings = self.predict_ratings_for_user(user_id+1, k)
        # 按照预测评分排序，推荐前 top_n 个电影
            recommendations = predicted_ratings.sort_values(ascending=False).head(top_n)
            
            print([recommendations.index[0],recommendations.)
        return result
    
    @timeit
    def recommend_movies_user_itemrc(self,top_n=10,k=10):
        raise NotImplementedError


if __name__ == '__main__':
    #data_path = os.path.join()"./data/train_ratings.csv"
    current_file_path = os.path.abspath(__file__)
    dir_name = os.path.dirname(current_file_path)
    data_path = os.path.join(dir_name,"data/train_ratings.csv")
    result_list = RecommandMovie(data_path).recommend_movies_user_userrc()
    for i in result_list:
        print(" ".join(i))