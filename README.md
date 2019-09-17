Description
===========
Deep learning for multi-modal classification of cloud, shadow and land cover scenes in 
high-resolution satellite imagery implemented using [Keras](https://keras.io/) as described in:
[Shendryk, Y., Rist, Y., Ticehurst, C. and Thorburn, P. (2019). "Deep learning for multi-modal classification 
of cloud, shadow and land cover scenes in PlanetScope and Sentinel-2 imagery." 
ISPRS Journal of Photogrammetry and Remote Sensing 157: 124-136.](https://www.sciencedirect.com/science/article/pii/S0924271619302023)

![Graphical_Abstract](Graphical_Abstract.png)

Acknowledgements
================
This work exists thanks to:
1) [Yannik Rist](https://smp.uq.edu.au/profile/7543/yannik-rist) 
2) [Catherine Ticehurst](https://www.researchgate.net/profile/Catherine_Ticehurst)
3) [Peter Thorburn](https://scholar.google.nl/citations?user=URFZ6xEAAAAJ&hl=en)

and [CSIRO's Digiscape Future Science Platform](https://research.csiro.au/digiscape/)


## Requirements
- `conda`
- `git`

## Installation
1. Clone the repository.
    ```bash
    cd path/to/where/project/lives
    git clone --recursive https://github.com/yurithefury/ChipClassification.git
    ```

2. Conda environment install on local machine
   `conda-env create -f env.yaml` will create an appropriate environment called 'keras'.  In one already exists, you will 
   have to edit the first line of `env.yaml` to something else, eg `name: veryniceenvironment`.  If your computer
   does NOT have a GPU, you will have to edit `env.yaml` to change the following line 
        ```
              - keras-gpu=2.2.4
        ```
        to read
        ```
              - keras=2.2.4
        ```

## Project Contents

#### `create_final_combined_dataset.py`
This script creates the datasets for training the final models used for inference.

#### `inference_example.py`
This is an example of how to perform inference using a single Keras module and the `utils.inference` module.  It works on `TIF` 
scenes.  If you run this you should be able to look at the output in the `scrap/` subdirectory.

#### `nanscan.py` 
A small utility script to check for NaNs in a HDF5 file.

#### `plot_training_history.py`
Exactly what you'd think.

#### `test_keras.sh`
A Bash script for testing whether your environment works and whether Keras can find your GPUs.

####  `train_sugarbyte.py`
A training script, for use with DistributedDuctTape, but can also be run as a standalone script.

#### `qgis/`
Contains some QGIS style-files for visualisation of outputs.

#### `models/`
Inference-ready models live in here.

#### `hdf5cvtools/`
This is a collection of functions for turning geospatial raster data into HDF5 datasets suitable for machine
learning tasks.  It is a one-file module.  See the source code for docs.

#### `data/`
Here you can find T-PS and T-S2 datasets, which is also available at [Mendeley Data](https://data.mendeley.com/datasets/6gdybpjnwh/1).  A-PS data could be found at
[Planet: Understanding the Amazon from Space](https://www.kaggle.com/c/planet-understanding-the-amazon-from-space/data)