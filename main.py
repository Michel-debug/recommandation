import os
import argparse
import pandas as pd
from recommand_movie import RecommandMovie
# define the parameters of command line
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Recommand cinema")
    parser.add_argument('rating_file', type=str, help='Path to the rating file(relate path to main.py)')
    parser.add_argument('similarity_threshold',type=float,default=0.5,help='Similarity threshold')

    # parse the command line
    args = parser.parse_args()

    choose_model = ['item-item','user-user']
    # define the path to the rating file
    data_path = os.path.join(os.path.dirname(__file__), args.rating_file)


    # define the similarity threshold
    similarity_threshold = args.similarity_threshold

    RecommandMovie(data_path).recommend_movies_user_itemrc(threshold=similarity_threshold)