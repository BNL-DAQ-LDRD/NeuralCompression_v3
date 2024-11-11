# Variable Rate Neural Compression for Sparse Detector Data

## Install `MinkowskiEngine`
Installing `MinkowskiEngine` may not be super straightforward. 
Please see the documentation [here](documents/README_install_MinkowskiEngine.md). 

## Get prepared for Point-of-Interest Detection with Sparse Convolution

### Clone the repository
Get a clone of the `sparse_poi` by running
```
git clone https://github.com/pphuangyi/sparse_poi/tree/main
```
then
```
cd sparse_poi
```
Install the package by running
```
python setup.py develop
```
Again, please consider forking the repo and clone the fork.
Let us make it better together.

### Download the Time-Projection Chamber data
(to be completed)


### Set up environment variables


Set the root to the data by running
```
export DATAROOT=/path/to/your/data
```
Then you can check whether the data root is set correct by running the
unit test of the TPC dataset API
```
python -m unittest ./tests/test_dataset_tpc.py
```

This is optional, but if you want to develop your own code and
use `pylint` for disciplined coding as I do, please add the following
line so that `pylint` can locate your packages.
```
export PYTHONPATH=${CONDA_PREFIX}/lib/python3.11/site-packages:$PYTHONPATH
```
