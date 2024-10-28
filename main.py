import os
import argparse
import pandas as pd
from recommand_movie import RecommandMovie
import time


if not os.path.exists('logs'):
        os.makedirs('logs')

# define the parameters of command line
if __name__ == '__main__':

    # remove the app.log if exists
    if os.path.exists('./logs/app.log'):
        os.remove('./logs/app.log')
    
    parser = argparse.ArgumentParser(description="Recommand cinema")
    parser.add_argument('rating_file', type=str, help='Path to the rating file(relate path to main.py)')
    parser.add_argument('similarity_threshold',type=float,default=0.05,help='Similarity threshold')

    # parse the command line
    args = parser.parse_args()

    choose_model = ['item-item','user-user']
    output_file = ['item-item-fc.out','user-user-fc.out']
   
    # define the path to the rating file
    data_path = os.path.join(os.path.dirname(__file__), args.rating_file)


    # define the similarity threshold
    similarity_threshold = args.similarity_threshold
    # print(f"Similarity threshold: {similarity_threshold}")
    start_time = time.perf_counter()
    RecommandMovie(data_path=data_path,chooce_model=choose_model[0],output_file=output_file[0]).recommend_movies(similarity_threshold)
    end_time = time.perf_counter()
    print(f"Total time: {end_time - start_time:.4f} s")
    with open(output_file[0],'a') as f:
        f.write(f"Total time: {end_time - start_time:.4f} s\n")