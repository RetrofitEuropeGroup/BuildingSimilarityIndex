![3d project0002](https://github.com/user-attachments/assets/429e8d3f-425a-4f78-82b8-e90f867c2fdb)

<h1 align="center">Welcome to FF_BuildingSimilarityIndex 👋</h1>
<p>
  <img alt="Version" src="https://img.shields.io/badge/version-0.0.1-blue.svg?cacheSeconds=2592000" />
  <a href="https://github.com/RetrofitEuropeGroup/FF_BuildingSimilarityIndex#readme" target="_blank">
    <img alt="Documentation" src="https://img.shields.io/badge/documentation-yes-brightgreen.svg" />
  </a>
  <a href="https://github.com/RetrofitEuropeGroup/FF_BuildingSimilarityIndex/graphs/commit-activity" target="_blank">
    <img alt="Maintenance" src="https://img.shields.io/badge/Maintained%3F-yes-green.svg" />
  </a>
</p>


A Python library based on [3d-building-metrics repo](https://github.com/tudelft3d/3d-building-metrics) to create a pipeline for the comparison of buildings. The library automatically collects the data from the [3D-BAG](https://docs.3dbag.nl/en/) and compares building based on the results of a turning function and metrics from the 3d-building-metrics repo. It consists of three modules: collection, processing and similarity_calculation. The modules are combined in one large module: BuildingsSimilarity. 

The <b>collection</b> module downloads cityjson from the 3D-BAG based on a list of [BAG IDs](https://www.geobasisregistraties.nl/basisregistraties/adressen-en-gebouwen). The <b>processing</b> module processes the buildings by merging the downloaded cityjson, calculating 2D/3D metrics, filtering out unsuitable buildings (based on abnormal metric values) and a turning function. Its output is a geopackage with the BAG IDs and the metrics. Finally the calculates the <b>similarity_calculation</b> module calculates the distance between buildings. This can be the distance between two individual buildings or multiple buildings at once.

## Prerequisites

- python >=3.10

## Usage
All modules can be used individually but are combined in BuildingSimilarity. Go to the demo directory for an idea of how to use the module.

## Installation

### Windows
You can follow the instructions here or go to the [val3dity repo](https://github.com/tudelft3d/val3dity). However, you have to make sure that the executable is working and in the processing/metrics/val3dity directory so it is recommended to follow this instruction. <br></br>

1. Clone the repo & install dependencies

```bash
git clone https://github.com/RetrofitEuropeGroup/FF_BuildingSimilarityIndex.git
cd FF_BuildingSimilarityIndex
pip install -r requirements.txt
```

2. [Download](https://github.com/tudelft3d/val3dity/releases/download/2.4.0/val3dity-win64-v240.zip) the val3dity exectuable from their github page
3. Unzip val3dity-win64-v240
4. Unzip val3dity-win64
5. Copy all the files from the unzipped directory <i>val3dity-win64</i> to this specific place in the repo: <i>processing/metrics/val3dity</i>


### Linux (Ubuntu 22.04)
You can follow the instructions here or go to the val3dity repo page. However, you have to make sure that the package is working and in the processing/metrics/val3dity directory so it is recommended to follow this instruction. <br></br>

1. Start to install the required packages. They are needed in order for the pyvista library & val3dity
```bash
sudo apt-get update
sudo apt-get install libxrender1 cmake g++ git libboost-all-dev libeigen3-dev libgeos++-dev libcgal-dev libgl1
````


2. Create the CMAKE package
```bash
git clone https://github.com/tudelft3d/val3dity.git
mkdir val3dity/build
cd val3dity/build
cmake ..
make
cd ../..
```

3. Clone the repo, copy the cmake package into the repo & install dependencies
```bash
git clone https://github.com/RetrofitEuropeGroup/FF_BuildingSimilarityIndex.git
mkdir FF_BuildingSimilarityIndex/processing/metrics/val3dity
cp val3dity/build/val3dity FF_BuildingSimilarityIndex/processing/metrics/val3dity/val3dity
cd FF_BuildingSimilarityIndex
pip install -r requirements.txt
```


## Show your support

Give a ⭐️ if this project helped you!
