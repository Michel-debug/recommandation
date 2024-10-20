import pandas as pd
import numpy as np
from functools import wraps
import time
import os
import psutil
from sklearn.model_selection import train_test_split


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

def measure_memory(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # 获取当前进程
        process = psutil.Process(os.getpid())
        # 测量函数执行前的内存使用
        mem_before = process.memory_info().rss / 1024 / 1024  # 转换为MB
        # 执行函数
        result = func(*args, **kwargs)
        # 测量函数执行后的内存使用
        mem_after = process.memory_info().rss / 1024 / 1024  # 转换为MB
        # 计算内存使用差异
        mem_diff = mem_after - mem_before
        print(f"function {func.__name__} stockage use:")
        print(f"before run: {mem_before:.2f} MB")
        print(f"after run: {mem_after:.2f} MB")
        print(f"add: {mem_diff:.2f} MB")
        return result
    return wrapper

def printLog(func):
    @wraps(func)
    def print_wrapper(*args,**kwargs):
        func_name = func.__name__
        print(f"func {func_name}  is starting")
        result = func(*args,**kwargs)
        print(f"func {func_name}  is finished")
        return result
    return print_wrapper

class RecommandMovie:
    @printLog
    def __init__(self,data_path):
        self.rating_data = pd.read_csv(data_path)
        self.preprocess_data()
        self.user_similarity = self.cosine_similarity_user()
        
    @printLog
    def preprocess_data(self):
        
        self.rating_matrix = self.rating_data.pivot_table(values='rating',index='user_id',columns=
                                        'movie_id')


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
        print(predicted_ratings)
        print(list(zip(unrated_movies,predicted_ratings)))
        return pd.Series(predicted_ratings, index=unrated_movies)
    


    @printLog
    def evaluate_model(self, test_size=0.2, k=10):
        # split
        train_data, test_data = train_test_split(self.rating_data, test_size=test_size, random_state=42)
        
        # 2. 使用训练集构建评分矩阵和用户相似度矩阵
        rating_matrix = train_data.pivot_table(values='rating', index='user_id', columns='movie_id')
        user_similarity = self.compute_user_similarity(rating_matrix)
        
        # 3. 预测测试集中的评分
        test_data = test_data.copy()
        test_data['predicted_rating'] = test_data.apply(
            lambda row: self.predict_rating(row['user_id'], row['movie_id'],rating_matrix,user_similarity,k), axis=1
        )
        
        # 4. 计算评估指标（RMSE）
        mse = np.mean((test_data['rating'] - test_data['predicted_rating']) ** 2)
        rmse = np.sqrt(mse)
        print(f'模型的RMSE为: {rmse:.4f}')
        # 全局平均评分
global_mean = self.rating_data['rating'].mean()
# 测试集上的RMSE
mse_baseline = np.mean((test_data['rating'] - global_mean) ** 2)
rmse_baseline = np.sqrt(mse_baseline)
print(f'基线模型（全局平均评分）的RMSE为: {rmse_baseline:.4f}')
        return rmse
    
    def predict_rating(self, user_id, movie_id, rating_matrix, user_similarity, k=10):
        # 检查用户和电影是否在矩阵中
        if user_id not in rating_matrix.index or movie_id not in rating_matrix.columns:
            # 返回全局平均评分
            global_mean = rating_matrix.values.mean()
            return global_mean

        # 获取与目标用户相似的其他用户
        similar_users = user_similarity[user_id].drop(user_id).dropna()
        if similar_users.empty:
            # 返回用户平均评分或全局平均评分
            user_mean = rating_matrix.loc[user_id].mean()
            if np.isnan(user_mean):
                global_mean = rating_matrix.values.mean()
                return global_mean
            else:
                return user_mean

        # 选择最相似的前 k 个用户
        similar_users = similar_users.nlargest(k)
        
        # 获取这些用户对目标电影的评分
        similar_users_ratings = rating_matrix.loc[similar_users.index, movie_id]
        
        # 如果所有相似用户都未对该电影评分，返回全局平均评分
        if similar_users_ratings.isnull().all():
            global_mean = rating_matrix.values.mean()
            return global_mean
        
        # 去除未评分的用户
        similar_users_ratings = similar_users_ratings.dropna()
        if similar_users_ratings.empty:
            global_mean = rating_matrix.values.mean()
            return global_mean
        
        weights = similar_users[similar_users_ratings.index]
        
        # 计算加权平均评分
        numerator = np.dot(weights, similar_users_ratings)
        denominator = np.sum(weights)
        if denominator == 0:
            global_mean = rating_matrix.values.mean()
            return global_mean
        predicted_rating = numerator / denominator
        
        # 将预测评分限制在评分范围内（例如1到5）
        predicted_rating = np.clip(predicted_rating, 1, 5)
        return predicted_rating

    def compute_user_similarity(self, rating_matrix):
        # Normalize the rating matrix
        norm_matrix = rating_matrix.sub(rating_matrix.mean(axis=1), axis=0).fillna(0)
        dot_product = np.dot(norm_matrix, norm_matrix.T)
        norm = np.linalg.norm(norm_matrix, axis=1)
        epsilon = 1e-8  # avoid division by zero
        similarity = dot_product / (np.outer(norm, norm) + epsilon)
        np.fill_diagonal(similarity, 0)  # Exclude self-similarity
        return pd.DataFrame(similarity, index=rating_matrix.index, columns=rating_matrix.index)

    @timeit
    @measure_memory
    @printLog
    def recommend_movies_user_userrc(self,top_n=1, k=10):
        result = []
        user_ids = list(self.rating_data.index)[:50]
        # 预测用户对未评分电影的评分
        for user_id in user_ids:
            predicted_ratings = self.predict_ratings_for_user(user_id+1, k)
            # 按照预测评分排序，推荐前 top_n 个电影
            recommendations = predicted_ratings.sort_values(ascending=False).head(top_n)
            result.append([user_id,recommendations.index[0],round(recommendations.values[0],1)])
            print(recommendations)
        return result
    
    @timeit
    @measure_memory
    @timeit
    def recommend_movies_user_itemrc(self,top_n=10,k=10):
        raise NotImplementedError


if __name__ == '__main__':
    #data_path = os.path.join()"./data/train_ratings.csv"
    current_file_path = os.path.abspath(__file__)
    dir_name = os.path.dirname(current_file_path)
    data_path = os.path.join(dir_name,"data/train_ratings.csv")
    # result_list = RecommandMovie(data_path).recommend_movies_user_userrc()
    # for i in result_list:
    #     print(" ".join(map(str,i)))
    RecommandMovie(data_path).evaluate_model(k=3000)