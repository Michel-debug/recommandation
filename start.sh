#! /bin/bash
# set python path
export PATH="$PATH"
rating_file=$1
if [ -z "$rating_file" ]; then
    echo "Error: You must provide a rating file as the first argument."
    exit 1
fi
similarity_threshold=${2:-0.05}
python3 main.py "$rating_file" "$similarity_threshold"

