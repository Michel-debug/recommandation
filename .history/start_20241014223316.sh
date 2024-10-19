#! /bin/bash
# set python path

rating_file=$1
if [ -z "$RATING_FILE" ]; then
    echo "Error: You must provide a rating file as the first argument."
    exit 1
fi
similarity_threshold = ${2:-0.5}
python3 main.py "$rating_file" "$similarity_threshold"
