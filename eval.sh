#!/bin/bash

if [ $# -ne 1 ]; then
    echo "Error: The command should be of the format: $0 <path to npy file>"
    exit 1
fi

input_file="$1"

if [ ! -f "$input_file" ]; then
    echo "Error: File '$input_file' not found."
    exit 1
fi

python eval.py "$input_file"
