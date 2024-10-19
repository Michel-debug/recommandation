import pandas as pd

class RecommandMovie:
    def __init__(self, rating_data, movie_data, user_data,data_path):
        self.rating_data = None
        self.movie_data = movie_data
        self.user_data = user_data

    def load_data(self):
        self.rating_data = pd.read_csv(self.data_path + 'train_ratings.csv')

    def preprocess_data(self):
        raise NotImplementedError
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
    