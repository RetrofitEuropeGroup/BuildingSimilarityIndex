![3d project0002](https://github.com/user-attachments/assets/429e8d3f-425a-4f78-82b8-e90f867c2fdb)

<h1 align="center">Welcome to BuildingSimilarityIndex 👋</h1>
<p>
  <img alt="Version" src="https://img.shields.io/badge/version-0.0.1-blue.svg?cacheSeconds=2592000" />
  <a href="https://github.com/RetrofitEuropeGroup/FF_BuildingSimilarityIndex#readme" target="_blank">
    <img alt="Documentation" src="https://img.shields.io/badge/documentation-yes-brightgreen.svg" />
  </a>
  <a href="https://github.com/RetrofitEuropeGroup/FF_BuildingSimilarityIndex/graphs/commit-activity" target="_blank">
    <img alt="Maintenance" src="https://img.shields.io/badge/Maintained%3F-yes-green.svg" />
  </a>
</p>


A Python library inspired by the [3d-building-metrics repo](https://github.com/tudelft3d/3d-building-metrics) designed to create a pipeline for assessing building similarity. This library automatically collects data from the [3D-BAG](https://docs.3dbag.nl/en/) and compares buildings using a turning function, metrics from the 3d-building-metrics repo, and public BAG data. It consists of three main modules: collection, processing, and similarity_calculation. These modules are integrated into a comprehensive module called BuildingsSimilarity.

- The **collection** module downloads CityJSON files from the 3D-BAG based on a list of [BAG IDs](https://www.geobasisregistraties.nl/basisregistraties/adressen-en-gebouwen) or a [neighborhood ID](https://catalogus.kadaster.nl/brt/en/page/Buurt?clang=nl).
- The **processing** module merges the downloaded CityJSON files, calculates 2D/3D metrics, filters out unsuitable buildings (based on abnormal metric values), and applies a turning function. The output is a Pandas Dataframe containing the BAG IDs and the calculated metrics.
- The **similarity_calculation** module preprocesses the data for similarity calcuation by scaling, normalizing, handeling N/A values and selecting relevant columns. The most basic form of similarity_calculation is to calculate the distance between two individual buildings. More advanced options are to calculate a (reference) distance matrix or to run ML-algorithms such as DBSCAN or K-Means

## Prerequisites

- python==3.11
- A API key for the BAG. If you don't have one, you can [apply](https://www.kadaster.nl/zakelijk/producten/adressen-en-gebouwen/bag-api-individuele-bevragingen) for it free of charge

## Usage
All modules can be used individually but are combined in BuildingSimilarity. Go to the demo directory for an example of how to use the module.

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
6. Provide the `BAG_API_KEY` as an environmental variable. You can do this by creating a `.env` file in the root directory of the project and adding the following line:

```
BAG_API_KEY=your_api_key_here
```

Make sure to replace `your_api_key_here` with your actual BAG API key.


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


4. Provide the `BAG_API_KEY` as an environmental variable. You can do this by creating a `.env` file in the root directory of the project and adding the following line:

```
BAG_API_KEY=your_api_key_here
```

Make sure to replace `your_api_key_here` with your actual BAG API key.



## Show your support

Give a ⭐️ if this project helped you!
