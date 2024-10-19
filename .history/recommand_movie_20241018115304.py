import pandas as pd
from functools import wraps

def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args,**kwargs):
        start_time = time.perf_count()


class RecommandMovie:
    def __init__(self,data_path):
        self.rating_data = pd.read_csv(self.data_path + 'train_ratings.csv')
        self.movie2user_data = []
        self.user2movie_data = []

    def preprocess_data(self):
        self.user_movie_matrix 
        self.movie_user_matrix
    def train_model(self):
        raise NotImplementedError
    def cosine_similarity_item(self):
        raise NotImplementedError
    def cosine_similarity_user(self):
        raise NotImplementedError
    def predict_rating(self):
        raise NotImplementedError
    def evaluate_model(self):
        raise NotImplementedError
    def user_user_filter_collaborative(self):
        raise NotImplementedError
    def item_item_filter_collaborative(self):
        raise NotImplementedError
    

if __name__ == '__main__':
    