import pandas as pd

class RecommandMovie:
    def __init__(self, rating_data, movie_data, user_data):
        self.rating_data = rating_data
        self.movie_data = movie_data
        self.user_data = user_data

    def load_data(self):
        self.rating_data = pd.read_csv('rating.csv')
        raise NotImplementedError
    def preprocess_data(self):
        raise NotImplementedError
    def train_model(self):
        raise NotImplementedError
    def recommend_movie_filter(self):
        raise NotImplementedError

