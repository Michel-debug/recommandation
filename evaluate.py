import os
import numpy as np
import multiprocessing
from recommand_movie import RecommandMovie

def evaluate_threshold(threshold, data_path):
    model = RecommandMovie(data_path)
    # choose model and evaluate
    model.evaluate_model_user_based(threshold=threshold)
    model.evaluate_model_item_based(threshold=threshold)

if __name__ == '__main__':
    current_file_path = os.path.abspath(__file__)
    dir_name = os.path.dirname(current_file_path)
    data_path = os.path.join(dir_name, "data/train_ratings.csv")
    thresholds = np.linspace(0.01, 0.1, 10).tolist()
    
    with multiprocessing.Pool(processes=4) as pool:
        pool.starmap(evaluate_threshold, [(threshold, data_path) for threshold in thresholds])