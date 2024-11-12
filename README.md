# Variable Rate Neural Compression for Sparse Detector Data

In this repository, we explain how to train a Bicephalous Convolutional 
AutoEncoder model enabling variable compression ratio for Sparse data 
(`BCAE-VS`). The motivation for designing the model is to compress highly
sparse data collected from the time project chamber (TPC), the main
tracking device in colliders.

The key feature of `BCAE-VS` is that it compresses data not by 
downsizing the input tensor to a uniform (smaller) size but by 
downsampling the nonzero values (signal) in the sparse input. 
More specifically, the encoder of `BCAE-VS`'s tags each signal 
with an importance score, and only those with high importance will
be saved and used later for decoding. The compression scheme of 
`BCAE-VS` implies the sparser the input, the smaller the compressed 
data.

The encoder part of `BCAE-VS` is implemented with sparse convolution
which utilizes the sparsity of the input by avoiding
matrix multiplications with all-zero operands. In this study,
we use the [`MinkowskiEngine`'s](https://github.com/NVIDIA/MinkowskiEngine)
implementation for sparse convolution kernels. 

In the remainder of this read-me file, we will show how to install 
the `MinkowskiEngine` sparse convolution library and how to train
a `BCAE-VS` model on TPC data.

## Install `MinkowskiEngine`
Installing `MinkowskiEngine` may not be super straightforward. 
Please see the documentation [here](documents/README_install_MinkowskiEngine.md). 

After installing the `MinkowskiEngine` following the instruction, 
we should have a `conda` environment called `py311-me`. 

Please activate the environment by running
```
conda activate py311-me
```

## Clone the repository
Get a clone of the `NeuralCompression_v3` repository by running
```
git clone https://github.com/pphuangyi/sparse_poi/tree/main
```
then
```
cd NeuralCompression_v3
```
Install the package by running
```
python setup.py develop
```
Again, please consider forking the repo and clone the fork.
Let us make it better together.

## Download the Time-Projection Chamber data
The TPC data can be downloaded from [Zenodo](https://zenodo.org/records/14064045).
Please download both the `occupancy_by_wedge.csv` and `outer.tgz`.
The `outer.tgz` contains training and test data for the neural compression model.
And the `occupancy_by_wedge.csv` is needed for evaluation.

Decompress `outer.tgz` by running
```
tar -xvzf outer.tgz
```
It will produce a folder called `outer`. 
Please move the `occupancy_by_wedge.csv` into the `outer` folder.

Later on, the root of the TPC data will be `path_to_outer/outer`.

## Set up environment variables
Set the root to the data by running
```
export DATAROOT=/path/to/your/data
```
For example, for the TPC data, we can run 
```
export DATAROOT=path_to_outer/outer
```
 
## Train 
To train a `BCAE-VS` model on TPC data with the default config, cd into the `train` folder and run
```
python train.py config.yaml
```
> Note: during training, if the `keep_ratio_soft` remains around .5, we may consider restart the training. 

## Evaluate or compress
Assuming a pretrained model is saved at `checkpoints/bi_lambda-30_lb-10/model_last.pth`, to evaluate its performance on TPC data, we can run the following command:
```
python evaluate/evaluate.py checkpoints/bi_lambda-30_lb-10 --split test --device cuda --gpu-id 0 --precision full --compressed-path ./compresse --result-csv-path ./result.csv
```
If `--compressed-path` is not used, compressed data will not be saved.

If we just want to compress the data, we can run the following command:
```
python compress.py ../checkpoints/bi_lambda-30_lb-10/ --split test --device cuda --gpu-id 1 --compressed-path ./compresse
```
