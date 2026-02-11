# bench to rosbag2

> _Convert [Bench2Drive](https://github.com/Thinklab-SJTU/Bench2Drive) data with [nuScenes](https://www.nuscenes.org/) map into [rosbag2](https://docs.ros.org/en/jazzy/index.html) format_

## Introduction

nuScenes is a large-scale dataset of autonomous driving in urban environments, provided free for non-commercial use. This project provides helper scripts to download the nuScenes dataset and convert scenes into [rosbag2](https://docs.ros.org/en/jazzy/index.html) files for easy viewing in tools such as [Foxglove](https://foxglove.dev/).

## Usage

### Prepare Bench2Drive dataset

1. Download the [Bench2Drive mini dataset](https://github.com/Thinklab-SJTU/Bench2Drive/blob/main/tools/download_mini.sh)
1. Extract the following files into the `data/` directory:
```bash
hf download --repo-type dataset rethinklab/Bench2Drive --include "DynamicObjectCrossing_Town02_Route13_Weather6.tar.gz" --local-dir data
cd data
tar -xvf HardBreakRoute_Town01_Route30_Weather3.tar.gz
```

### Prepare nuScenes dataset

1. Download the [nuScenes mini dataset](https://nuscenes.org/nuscenes). You will need to make an account and agree to the terms of use.
1. Extract the following files into the `map/` directory:
    1. `can_bus.zip` to `map/`
    1. `nuScenes-map-expansion-v1.3.zip` to `map/maps`
    1. `v1.0-mini.tgz` to `map/`

### Build docker image and run

1. Build and run the converter container with `./convert_bench_scenes.sh`

## License

bench_to_rosbag is licensed under the [MIT License](https://opensource.org/licenses/MIT).
