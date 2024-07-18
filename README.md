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

A Python library that combines 3D-BAG data and the repo github.com/tudelft3d/val3dity to combine a pipeline for the comparison of buildings. It consists of three modules: collection, processing and similarity_calculation. 

The <b>collection</b> module downloads cityjson from the 3D-BAG based on a list of [BAG IDs](https://www.geobasisregistraties.nl/basisregistraties/adressen-en-gebouwen). The <b>processing</b> module processes the buildings by merging the downloaded cityjson, calculating 2D/3D metrics, filtering out unsuitable buildings (based on abnormal metric values) and a turning function. Its output is a geopackage with the BAG IDs and the metrics. Finally the calculates the <b>similarity_calculation</b> module calculates the distance between buildings. This can be the distance between two individual buildings or multiple buildings at once.

## Prerequisites

- python >=3.10

## Usage
All modules can be used individually. The main.py file in the root folder serves as an example of how to use the modules together.

```bash
python main.py
```


## Installation

### Install the val3dity libary (Windows). 
You can follow the instructions here or go to the [val3dity repo](https://github.com/tudelft3d/val3dity). However, you have to make sure that the executable is working and in the processing/metrics/val3dity directory so it is recommended to follow this instruction. 

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


### Install the val3dity library (Linux - Ubuntu 22.04)
You can follow the instructions here or go to the val3dity repo page. However, you have to make sure that the package is working and in the processing/metrics/val3dity directory so it is recommended to follow this instruction. 

Start to install the required packages. They are needed in order for the pyvista library & val3dity
```bash
sudo apt-get update
sudo apt-get install libxrender1 cmake g++ git libboost-all-dev
````

Create the CMAKE package.
```bash
git clone https://github.com/tudelft3d/val3dity.git
mkdir val3dity/build
cd val3dity/build
cmake ..
make
cd ../..
```

Clone the repo & install dependencies, assuming the working directory of your terminal is still 
```bash
git clone https://github.com/RetrofitEuropeGroup/FF_BuildingSimilarityIndex.git
cd FF_BuildingSimilarityIndex
pip install -r requirements.txt
```


## Author

* Github: [@RetrofitEuropeGroup](https://github.com/RetrofitEuropeGroup)

## Show your support

Give a ⭐️ if this project helped you!

***
_This README was generated with ❤️ by [readme-md-generator](https://github.com/kefranabg/readme-md-generator)_