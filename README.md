# phyddle

A pipeline-based toolkit for fiddling around with phylogenetic models and deep learning 

*NOTE:*

*<b>This private beta version of phyddle is still under development.</b> It is generally tested and stable with respect to a few use standard cases. Much of the documentation and some key features are still missing. Most phyddle development occurs on a 16-core Intel Macbook Pro laptop and a 64-core Ubuntu server, so there are also unknown portability/scalability issues to correct. Any feedback is appreciated! michael.landis@wustl.edu*


## Overview

<img align="right" src="https://github.com/landislab/landislab.github.io/blob/5bb4685a12ebf4c99dd773de6d87b44cc3c47090/assets/research/img/phyddle_pipeline.png?raw=true" width="35%">

A standard phyddle analysis performs the following tasks for you:

- **Pipeline configuration** applies analysis settings provided through a config file and/or command line arguments.
- **Model configuration** constructs a base simulating model to be *Simulated* (states, events, rates).
- **Simulating** simulates a large training dataset under the model to be *Formatted* (parallelized, partly compressed). Saves output in the [`raw_data`](workspace/raw_data) directory.
- **Formatting** encodes the raw simulated data into tensor format for *Learning* (parallelized, compressed). Saves output in the [`tensor_data`](workspace/tensor_data) directory.
- **Learning** shuffles and splits training data, builds a network, then trains and saves the network with the data for *Prediction*. Saves output in the [`network`](workspace/network) directory.
- **Predicting** estimates model parameters for a new dataset with the trained network. Saves output in the [`predict`](workspace/predict) directory.
- **Plotting** generates figures that summarize the training data (*Formatting*), the network and its training (*Learning*), and any new predictions (*Predicting*). Saves output in the [`plot`](workspace/plot) directory.



## Quick start

To run a phyddle analysis enter the `scripts` directory:
```shell
cd ~/projects/phyddle/scripts
```

Then create and run a pipeline under the settings you've specified in `my_config.py`:
```shell
./run_phyddle.sh --cfg my_config
```

This will run a phyddle analysis for a simple 3-region GeoSSE model with just 500 training examples. In practice, you'll want to generate a larger training dataset with anywhere from 10k to 1M examples, depending on the model.

To add new examples to your training set
```shell
# simulate new training examples and store in raw_data/my_project
./run_phyddle.sh -s sim -c my_config --start_idx 500 --end_idx 15000

# encode all raw_data examples as tensors in tensor_data/my_project
./run_phyddle.sh -s fmt -c my_config --start_idx 0 --end_idx 15000

# train network with tensor_data, but override batch size, then store in network/my_project
./run_phyddle.sh -s lrn -c my_config --batch_size 256

# make prediction with example dataset
./run_phyddle.sh -s prd -c my_config

# generate figures and store in plot
./run_phyddle.sh -s plt -c my_config
```

Pipeline options are applied to all pipeline stages. See the full list of currently supported options with
```shell
./run_phyddle.sh --help
```

## Features

Current features:
- trained network generates parameter estimates and coverage-calibrated prediction intervals (CPIs) for input datasets
- provides several state-dependent birth-death model types and variants (more to come)
- parallelized simulating, formatting, and learning
- encoding of phylogenetic-state tensor from serial and extant-only input with multiple states (CBLV+S and CDV+S extensions)
- encoding of auxiliary data tensor from automatically computed summary statistics and "known" parameter (e.g. sampling rate)
- HDF5 with gzip compression for tensor data
- shuffles and splits input tensors into training, test, validation, and calibration datasets for supervised learning
- builds network with convolution, pooling, and dense layers that match input tensors
- trains network and saves history
- automatic figure generation with Matplotlib

## About
Thanks for your interest in phyddle. The phyddle project emerged from a phylogenetic deep learning study led by Ammon Thompson ([paper](https://www.biorxiv.org/content/10.1101/2023.02.08.527714v2)). The goal of phyddle is to provide its users with a generalizable pipeline workflow for phylogenetic modeling and deep learning. This hopefully will make it easier for phylogenetic model enthusiasts and developers to explore and apply models that do not have tractable likelihood functions. It's also intended for use by methods developers who want to characterize how deep learning methods perform under different conditions for standard phylogenetic estimation tasks.

The phyddle project is developed by [Michael Landis](https://landislab.org) and [Ammon Thompson](https://scholar.google.com/citations?user=_EpmmTwAAAAJ&hl=en&oi=ao).

