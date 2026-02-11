#!/usr/bin/env bash
set -euo pipefail

if [ ! -d "data" ]; then
    echo "data dir does not exist: please create and extract Bench2Drive data into it."
    exit 1
fi

if [ ! -d "map" ]; then
    echo "map dir does not exist: please create and extract nuScenes data into it."
    exit 1
fi

docker build -t bench_converter .
mkdir -p output
docker run -t --rm \
    --user $(id -u):$(id -g) \
    -v $(pwd)/data:/data -v $(pwd)/map:/map -v $(pwd)/output:/output \
    bench_converter python3 bench_to_rosbag.py -d /data -m /map -o /output "$@"
