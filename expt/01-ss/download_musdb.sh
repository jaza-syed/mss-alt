#!/bin/bash

# Check if DATA_ROOT is set
if [ -z "$DATA_ROOT" ]; then
    echo "Error: DATA_ROOT environment variable is not set." >&2
    exit 1
fi

# Check if DATA_ROOT directory exists
if [ ! -d "$DATA_ROOT" ]; then
    echo "Error: DATA_ROOT directory '$DATA_ROOT' does not exist." >&2
    exit 1
fi

wget https://zenodo.org/api/records/3338373/files/musdb18hq.zip/content $DATA_ROOT/musdb18hq.zip
unzip $DATA_ROOT/musdb18hq.zip -d $DATA_ROOT/musdb18hq
