# phyddle 0.2.0

#### Software to fiddle around with deep learning for phylogenetic models

> *NOTE: This beta version of phyddle is still under development.*

## User guide
Visit https://mlandis.github.io/phyddle to learn more about phyddle.

## Overview

<img align="right" src="https://github.com/landislab/landislab.github.io/blob/5bb4685a12ebf4c99dd773de6d87b44cc3c47090/assets/research/img/phyddle_pipeline.png?raw=true" width="35%">

A standard phyddle analysis performs the following tasks for you:

- **Pipeline configuration** applies analysis settings provided through a config file and/or command line arguments.
- **Simulate** simulates a large training dataset using a user-designed simulator.
- **Format** encodes the raw simulated data (from *Simulate*) into tensor format for *Train*.
- **Train** loads and splits training data (from *Format*), builds a network, then trains and saves the network.
- **Estimate** estimates model parameters for a new dataset with the trained network (from *Train*).
- **Plot** generates figures that summarize the training data (*Format*), the network and its training (*Train*), and any new predictions (*Estimate*).

## Quick start

To run a phyddle analysis enter the `scripts` directory:
```shell
cd ~/projects/phyddle
```

Then create and run a pipeline under the settings you've specified in `workspace/example/config.py`:
```shell
cd workspace/example
phyddle --cfg config.py
```

This will run a phyddle analysis with 1000 simulations using R and the castor package for a simple birth-death model with one 3-state character. In practice, you'll want to generate a larger training dataset with anywhere from 10k to 1M examples, depending on the model.

To add new examples to your training set
```shell
# simulate new training examples and store in simulate
phyddle -s S -c config.py --sim_more 14000

# encode all raw_data examples as tensors in format
phyddle -s F -c config.py

# train network with tensor data, but override batch size, then store in train
phyddle -s T -c config.py --trn_batch_size 256

# make prediction for empirical example in dataset
phyddle -s E -c config.py

# generate figures and store in plot
phyddle -s P -c config.py
```

To see a full list of all options currently supported by phyddle
```shell
phyddle --help
```

## Installation

A stable version of phyddle can be installed using the Python package manager, pip:

```shell
python3 -m pip install --upgrade phyddle
# ... install ...
phyddle
```

...or using conda:

```shell
conda create -n phyddle_env -c bioconda -c landismj phyddle
# ... install ...
conda activate phyddle_env
phyddle
```

The newest development version of phyddle must be manually installed through a local build:
```shell
git clone git@github.com:mlandis/phyddle.git
cd phyddle
git checkout development
git pull
./build.sh
```

phyddle uses third-party simulators to generate training datasets. Example workflows assume that [R](https://cran.r-project.org), [RevBayes](https://revbayes.github.io), [Phylojunction](https://phylojunction.org/build/html/index.html), or [BEAST](https://www.beast2.org/) with [MASTER](https://github.com/tgvaughan/MASTER) (plugin) is installed on your machine and can be executed as a command from terminal. The documentation explains how to configure R for use with phyddle.

## Note on code stability

Code on the [main](https://github.com/mlandis/phyddle/tree/main) branch is tested and stable with respect to the standard use cases. Code on the [development](https://github.com/mlandis/phyddle/tree/development) branch contains new features, but is not as rigorously tested. Most phyddle development occurs on a 16-core Intel Macbook Pro laptop and a 64-core Intel Ubuntu server. Any feedback is appreciated! [michael.landis@wustl.edu](mailto:michael.landis@wustl.edu)

## About
Thanks for your interest in phyddle. The phyddle project emerged from a phylogenetic deep learning study led by Ammon Thompson ([paper](https://doi.org/10.1093/sysbio/syad074)). The goal of phyddle is to provide its users with a generalizable pipeline workflow for phylogenetic modeling and deep learning. This hopefully will make it easier for phylogenetic model enthusiasts and developers to explore and apply models that do not have tractable likelihood functions. It's also intended for use by methods developers who want to characterize how deep learning methods perform under different conditions for standard phylogenetic estimation tasks.

The phyddle project is developed by [Michael Landis](https://landislab.org) and [Ammon Thompson](https://scholar.google.com/citations?user=_EpmmTwAAAAJ&hl=en&oi=ao).
