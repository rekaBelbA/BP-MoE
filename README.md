
## Overview

This repo is the open-sourced code for our work *BP-MoE: Behavior Pattern-aware Mixture-of-Experts for Temporal Graph Representation Learning*.

## Requirements
- python >= 3.6.13
- pytorch >= 1.8.1
- pandas >= 1.1.5
- numpy >= 1.19.5
- dgl >= 0.6.1
- pyyaml >= 5.4.1
- tqdm >= 4.61.0
- pybind11 >= 2.6
- g++ >= 7.5.0
- openmp >= 201511

bulid up temporal sampler 
> python setup.py build_ext --inplace

## Dataset
Download the datasets from AWS S3 bucket using the `down.sh` script. 

### Dynamic Link Prediction
>python train.py --data \<DATA> 


### Dynamic Node Classification
>python train_node.py --data \<DATA> --config \<PathToConfigFile> --model \<PathToSavedModel>


Thanks to the publicly released codes of [TGN],[TGL]and [Graph-Mixer], we implement BP-MoE based on them. 