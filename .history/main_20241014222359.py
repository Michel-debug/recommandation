import os
import argparse
# define the parameters of command line
parser = argparse.ArgumentParser(description="Recommand cinema")
parser.add_argument('rating_file', type=str, help='Path to the rating file(relate path to main.py)')
parser.add_argument('similarity_threshold',type=float,default=0.5,help='Similarity threshold')

# parse the command line
args = parser.parse_args()

# define the path to the rating file
rating_file = os.path.join(os.path.dirname(__file__), args.rating_file)

# define the similarity threshold
similarity_threshold = args.similarity_threshold
print(rating_file)

print(similarity_threshold)