import os
import argparse
# define the parameters of command line
parser = argparse.ArgumentParser(description="Recommand cinema")
parser.add_argument('rating_file', type=str, help='Path to the rating file(relate path to main.py)')
parser.add_argument('similarity