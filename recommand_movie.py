import multiprocess.managers
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from typing import List
import multiprocessing
import logging
import psutil
from decorator import timeit, measure_memory, printLog


#log_config
logging.basicConfig(filename='logs/app.log',level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

#config test record
evaluate_logger = logging.getLogger('evaluate_logger')
evaluate_handler = logging.FileHandler('logs/evaluate.log')
evaluate_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
evaluate_handler.setFormatter(formatter)
evaluate_logger.addHandler(evaluate_handler)


class RecommandMovie:
    @printLog
    def __init__(self,data_path,output_file,chooce_model="user-user"):
        self.chooce_model = chooce_model
        # # if exist the output file, read directly
        # if chooce_model == "user-user" and os.path.exists(output_file):
        #     with open(output_file,"r") as f:
        #         for line in f:
        #             print(line.strip())
        #     return
        # elif chooce_model == "item-item" and os.path.exists(output_file):
        #     with open(output_file,"r") as f:
        #         for line in f:
        #             print(line.strip())
        #     return
        # else:
        self.rating_data = pd.read_csv(data_path)
        self.cpu_count =  multiprocess.cpu_count() # start multiprocess
        self.output_file = output_file
        self.preprocess_data()
        if self.chooce_model == "user-user":
            self.user_similarity = self.cosine_similarity_user()
            # print("user_similary_matrix",self.user_similarity)
        elif self.chooce_model == "item-item":
            self.item_similarity = self.cosine_similarity_item()
            # print("item_similary_matrix",self.item_similarity)
        else:
            raise ValueError("The chooce_model is not correct, please check it")
            
        print("\n\n\033[92m*** The entire reasoning model time is expected to be 5min-10min. please attend ***")
        print("*** The eval model time is expected to be 1min-2min. please attend ***")
        print("** if you want see some procesus info, please check the logs folder's app.log **")
        print("****** The current algo is ",chooce_model," ******")
        print("****** When the programme is running, the terminal will output the predict's result, but is not the sort result, the final result will output at the last moment ****** \033[0m\n\n")
        
        
    @printLog
    def preprocess_data(self):
        self.rating_matrix = self.rating_data.pivot_table(values='rating',index='user_id',columns=
                                        'movie_id')


    @printLog
    def cosine_similarity_item(self):
        rating_matrix = self.rating_matrix.T
        norm_matrix = rating_matrix.sub(rating_matrix.mean(axis=1),axis=0).fillna(0)
        # item_user matrix
        dot_product = np.dot(norm_matrix,norm_matrix.T)
        norm =  np.linalg.norm(norm_matrix,axis=1)
        epsilon = 1e-8
        similarity = dot_product / (np.outer(norm, norm) + epsilon)
        np.fill_diagonal(similarity, 0)
        return pd.DataFrame(similarity, index=rating_matrix.index, columns=rating_matrix.index)
    
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

    def predict_ratings_for_user(self,user_id, similarity_threshold=0.05):
        # get similar users
        similar_users = self.user_similarity[user_id].drop(user_id).dropna()
        # choose the similary value > threshold
        similar_users = similar_users[similar_users > similarity_threshold]
        
        # get the unrating movie
        user_ratings = self.rating_matrix.loc[user_id]
        unrated_movies = user_ratings[user_ratings.isna()].index
        
        # only choose the similar user rating
        similar_user_ratings = self.rating_matrix.loc[similar_users.index, unrated_movies]
        
        # for the similar user rating is null, fill the global mean
        global_mean = self.rating_matrix.mean(axis=0)
        similar_user_ratings = similar_user_ratings.fillna(global_mean[unrated_movies])

        # average the weight
        weights = np.array(similar_users.values).reshape(-1, 1)
        weighted_sum = np.dot(weights.T, similar_user_ratings)
        normalization = np.sum(weights)
        
        # calculate the predicted rating
        predicted_ratings = weighted_sum.flatten() / normalization
        return pd.Series(predicted_ratings, index=unrated_movies)


    @timeit
    def evaluate_model_item_based(self, test_size=0.2, threshold=0.05):
   
        train_data, test_data = train_test_split(self.rating_data, test_size=test_size, random_state=42)

    
        rating_matrix = train_data.pivot_table(values='rating', index='user_id', columns='movie_id')
        item_similarity = self.compute_item_similarity(rating_matrix)
        #  predict the rating
        test_data = test_data.copy()
        test_data['predicted_rating'] = test_data.apply(
            lambda row: self.predict_rating_item_based(row['user_id'], row['movie_id'], rating_matrix, item_similarity, threshold), axis=1
        )

        # calcule the MSE
        mse = np.mean((test_data['rating'] - test_data['predicted_rating']) ** 2)
        rmse = np.sqrt(mse)
        evaluate_logger.info(f'An item-based collaborative, parameter: threshold {threshold} filtering model RMSE : {rmse:.4f}')
        # the test'rmse
        global_mean = self.rating_data['rating'].mean()
        mse_baseline = np.mean((test_data['rating'] - global_mean) ** 2)
        rmse_baseline = np.sqrt(mse_baseline)
        evaluate_logger.info(f'base line (global mean) RMSE : {rmse_baseline:.4f}')
        return rmse


    @timeit
    def evaluate_model_user_based(self, test_size=0.2, threshold=0.05):
        # split
        train_data, test_data = train_test_split(self.rating_data, test_size=test_size, random_state=42)
        
        rating_matrix = train_data.pivot_table(values='rating', index='user_id', columns='movie_id')
        user_similarity = self.compute_user_similarity(rating_matrix)
        
        test_data = test_data.copy()
        test_data['predicted_rating'] = test_data.apply(
            lambda row: self.predict_rating_user_based(row['user_id'], 
                                                       row['movie_id'],
                                                       rating_matrix,
                                                       user_similarity,
                                                        threshold), axis=1
        )
        
        mse = np.mean((test_data['rating'] - test_data['predicted_rating']) ** 2)
        rmse = np.sqrt(mse)
        evaluate_logger.info(f'An user-based collaborative parameter: threshold {threshold} filtering model RMSE: {rmse:.4f}')
     
        global_mean = self.rating_data['rating'].mean()
        
        mse_baseline = np.mean((test_data['rating'] - global_mean) ** 2)
        rmse_baseline = np.sqrt(mse_baseline)
        evaluate_logger.info(f'base line（global mean） RMSE : {rmse_baseline:.4f}')
        return rmse
    
    # This is for eval
    def predict_rating_item_based(self, user_id, movie_id, rating_matrix, item_similarity, threshold=0.00003):
        # check if user and movie in the matrix
        if user_id not in rating_matrix.index:
            # return globle mean
            global_mean = rating_matrix.values.mean()
            return global_mean
        if movie_id not in item_similarity.index:
            # return user mean or global mean
            user_mean = rating_matrix.loc[user_id].mean()
            if np.isnan(user_mean):
                global_mean = rating_matrix.values.mean()
                return global_mean
            else:
                return user_mean
            
        # get the similar item
        similar_items = item_similarity[movie_id].drop(movie_id).dropna()
        if similar_items.empty:
            # return user mean or global mean
            user_mean = rating_matrix.loc[user_id].mean()
            if np.isnan(user_mean):
                global_mean = rating_matrix.values.mean()
                return global_mean
            else:
                return user_mean
        # choose the similar item > threshold
        similar_items = similar_items[similar_items > threshold]
        # get the related user rating
        user_ratings = rating_matrix.loc[user_id, similar_items.index]

        # 如果用户没有对这些物品评分，返回用户平均评分
        if user_ratings.isnull().all():
            user_mean = rating_matrix.loc[user_id].mean()
            if np.isnan(user_mean):
                global_mean = rating_matrix.values.mean()
                return global_mean
            else:
                return user_mean

        # remove the unrated user
        user_ratings = user_ratings.dropna()
        if user_ratings.empty:
            user_mean = rating_matrix.loc[user_id].mean()
            if np.isnan(user_mean):
                global_mean = rating_matrix.values.mean()
                return global_mean
            else:
                return user_mean

        # update the similar item
        similar_items = similar_items[user_ratings.index]

        # calculate the weighted average rating
        numerator = np.dot(similar_items, user_ratings)
        denominator = np.sum(np.abs(similar_items))
        if denominator == 0:
            user_mean = rating_matrix.loc[user_id].mean()
            if np.isnan(user_mean):
                global_mean = rating_matrix.values.mean()
                return global_mean
            else:
                return user_mean

        predicted_rating = numerator / denominator

        # 将预测评分限制在1到5之间
        predicted_rating = np.clip(predicted_rating, 1, 5)
        return predicted_rating
    
    # This is for eval
    def predict_rating_user_based(self, user_id, movie_id, rating_matrix, user_similarity, threshold=0.05):
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
        similar_users = similar_users[similar_users > threshold]
        
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
        predicted_rating = np.clip(predicted_rating, 0, 5)
        return predicted_rating
    # This is for eval
    def compute_user_similarity(self, rating_matrix):
        # Normalize the rating matrix
        norm_matrix = rating_matrix.sub(rating_matrix.mean(axis=1), axis=0).fillna(0)
        dot_product = np.dot(norm_matrix, norm_matrix.T)
        norm = np.linalg.norm(norm_matrix, axis=1)
        epsilon = 1e-8  # avoid division by zero
        similarity = dot_product / (np.outer(norm, norm) + epsilon)
        np.fill_diagonal(similarity, 0)  # Exclude self-similarity
        return pd.DataFrame(similarity, index=rating_matrix.index, columns=rating_matrix.index)
    
    # This is for eval
    def compute_item_similarity(self, rating_matrix):
        
        rating_matrix_T = rating_matrix.T
        # Nomralize the rating matrix
        norm_matrix = rating_matrix_T.sub(rating_matrix_T.mean(axis=1), axis=0).fillna(0)
        dot_product = np.dot(norm_matrix, norm_matrix.T)
        norm = np.linalg.norm(norm_matrix, axis=1)
        epsilon = 1e-8  # division by zero
        similarity = dot_product / (np.outer(norm, norm) + epsilon)
        np.fill_diagonal(similarity, 0)  # 排除自身相似度
        return pd.DataFrame(similarity, index=norm_matrix.index, columns=norm_matrix.index)
    

    def process_users(self,start_i,end_i,user_ids,result,top_n,threshold,mem_usage):
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / 1024 / 1024
        for user_id in user_ids[start_i:end_i]:
            predicted_ratings = self.predict_ratings_for_user(user_id+1, threshold)
            # 按照预测评分排序，推荐前 top_n 个电影
            recommendations = predicted_ratings.sort_values(ascending=False).head(top_n)
            
            logging.info(f"(user_userFC) User {user_id} recommended movie {recommendations.index[0]} with predicted rating {round(recommendations.values[0], 1)}")
            result.append([user_id,recommendations.index[0],round(recommendations.values[0],1)])
            print(f"{user_id} {recommendations.index[0]} {round(recommendations.values[0],1)}")
        mem_after = process.memory_info().rss / 1024 / 1024
        mem_used = mem_after - mem_before
        mem_usage.append(mem_used)
    
    @printLog
    def recommend_movies_user_userrc(self,top_n=10, threshold=0.05):
        # result = []
        manager = multiprocessing.Manager()
        result = manager.list()
        mem_usage = manager.list()
        user_ids = list(self.rating_matrix.index)[:]
        processes = []      
        for i in range(self.cpu_count):
            start_i,end_i = i*len(user_ids) // self.cpu_count, (i+1)*len(user_ids)//self.cpu_count
            p = multiprocessing.Process(target=self.process_users,args=(start_i,end_i,user_ids,result,top_n,threshold,mem_usage))
            processes.append(p)
            p.start()
        # 预测用户对未评分电影的评分
        for p in processes:
            p.join()
        total_mem_usage = sum(mem_usage)
        logging.info(f"Total memory usage: {total_mem_usage:.2f} MB")
        result.append(f"Total memory usage: {total_mem_usage:.2f} MB")
        RecommandMovie.printResult(list(result),output_file=self.output_file)
        
        
    def process_user_item(self, start_i, end_i, user_ids, top_n, threshold, result,memory_usage):
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / 1024 / 1024
        for user_id in user_ids[start_i:end_i]:
            user_ratings = self.rating_matrix.loc[user_id + 1]
            unrated_movies = user_ratings[user_ratings.isna()].index
            predictions = {}
            for movie_id in unrated_movies:
                predicted_rating = self.predict_rating_item_based(user_id + 1, movie_id, self.rating_matrix, self.item_similarity, threshold)
                predictions[movie_id] = predicted_rating
            # 按照预测评分排序，推荐前 top_n 个电影
            recommended_movies = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:top_n]
            result.append([user_id, recommended_movies[0][0], round(recommended_movies[0][1], 1)])
            logging.info(f"(item-itemFc) User {user_id} recommended movie {recommended_movies[0][0]} with predicted rating {round(recommended_movies[0][1], 1)}")
            print(f"{user_id} {recommended_movies[0][0]} {round(recommended_movies[0][1], 1)}")
        mem_after = process.memory_info().rss / 1024 / 1024
        mem_used = mem_after - mem_before
        memory_usage.append(mem_used)
    
    @printLog
    def recommend_movies_user_itemrc(self, top_n=10, threshold=0.05):
        user_ids = list(self.rating_matrix.index)[:]
        manager = multiprocessing.Manager()
        result = manager.list()  # use share list
        mem_usage = manager.list()
        processes = []
        cpu_count = multiprocessing.cpu_count()
        for i in range(cpu_count):
            start_i = i * len(user_ids) // cpu_count
            end_i = (i + 1) * len(user_ids) // cpu_count
            p = multiprocessing.Process(target=self.process_user_item, args=(start_i, end_i, user_ids, top_n, threshold, result,mem_usage))
            processes.append(p)
            p.start()

        for p in processes:
            p.join()
        total_mem_usage = sum(mem_usage)
        logging.info(f"Total memory usage: {total_mem_usage:.2f} MB")
        result.append(f"Total memory usage: {total_mem_usage:.2f} MB")
        RecommandMovie.printResult(list(result),output_file=self.output_file)
        

    
    @timeit
    def recommend_movies(self,threshold=0.05):
        if self.chooce_model == "user-user":
            self.recommend_movies_user_userrc(threshold=threshold)
        elif self.chooce_model == "item-item":
            self.recommend_movies_user_itemrc(threshold=threshold)

            
    
    @staticmethod
    def printResult(result_sets:List,output_file:str):
        print("****** The final result ******")
        with open(output_file,"w") as f:
            for i in sorted(result_sets[:-1]):
                line = " ".join(map(str,i)) + "\n"
                f.write(line) # write file
                print(line.strip())
            line = result_sets[-1] + "\n"
            f.write(line)
            print(line.strip())



if __name__ == '__main__':
    #data_path = os.path.join()"./data/train_ratings.csv"
    current_file_path = os.path.abspath(__file__)
    dir_name = os.path.dirname(current_file_path)
    data_path = os.path.join(dir_name,"data/train_ratings.csv")
    choose_model = ['item-item','user-user']
    

    # RecommandMovie(data_path,chooce_model=choose_model[0]).recommend_movies_user_itemrc(threshold=0.05)
    # RecommandMovie(data_path).recommend_movies_user_userrc(threshold=0.04)
    RecommandMovie(data_path,chooce_model=choose_model[1]).recommend_movies(threshold=0.04)