#! /bin/bash
# set python path
export PYTHONPATH=$(pwd)
rating_file = $1
similarity_threshold = ${2:-0.5}
python3 main.py "$rating_file" "$similarity_threshold"
