#! /bin/bash
rating_file = $1
similarity_threshold = $(2:-0.5)
python3 main.py $1 $2