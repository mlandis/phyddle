# phyddle 0.0.8

#### Software to fiddle around with deep learning for phylogenetic models

> *NOTE: This private beta version of phyddle is still under development.*

## User guide
Visit https://mlandis.github.io/phyddle to learn more about phyddle.

## Overview

<img align="right" src="https://github.com/landislab/landislab.github.io/blob/5bb4685a12ebf4c99dd773de6d87b44cc3c47090/assets/research/img/phyddle_pipeline.png?raw=true" width="35%">

A standard phyddle analysis performs the following tasks for you:

- **Pipeline configuration** applies analysis settings provided through a config file and/or command line arguments.
- **Simulate** simulates a large training dataset under the model to be *Formatted* (parallelized, partly compressed). Saves output in the [`simulate`](workspace/simulate) directory.
- **Format** encodes the raw simulated data into tensor format for *Train*. Saves output in the [`format`](workspace/format) directory.
- **Train** loads and splits training data, builds a network, then trains and saves the network with the data for *Estimate*. Saves output in the [`train`](workspace/train) directory.
- **Estimate** estimates model parameters for a new dataset with the trained network. Saves output in the [`train`](workspace/train) directory.
- **Plot** generates figures that summarize the training data (*Formatting*), the network and its training (*Train*), and any new predictions (*Estimate*). Saves output in the [`plot`](workspace/plot) directory.


## Quick installation

A stable version of phyddle can be installed using pip:

```shell
# install pip
python -m ensurepip --upgrade
# install phyddle
python3 -m pip install --upgrade phyddle
```

The newest development version of phyddle must be manually installed through a local build:
```shell
# clone phyddle repo
git clone git@github.com:mlandis/phyddle.git
# enter phyddle
cd phyddle
# switch to development branch
git checkout development
# pull to acquire newest changes (not needed after first clone)
git pull
# build local phyddle package
./build.sh
```

phyddle uses third-party simulators to generate training datasets. The standard workflow assumes that [R](https://cran.r-project.org) v4.2.2, [RevBayes](https://revbayes.github.io) v1.2.1, or [BEAST](https://www.beast2.org/) v2.7.3 with [MASTER](https://github.com/tgvaughan/MASTER) v7.0.0 (plugin) is installed on your machine and be executed as a command from terminal. The documentation explains how to configure R for use with phyddle.

## Quick start

To run a phyddle analysis enter the `scripts` directory:
```shell
cd ~/projects/phyddle/scripts
```

Then create and run a pipeline under the settings you've specified in `my_config.py`:
```shell
./run_phyddle.py --cfg config
```

This will run a phyddle analysis with 1000 simulations from R and the ape package for a simple birth-death model with two 3-state characters. In practice, you'll want to generate a larger training dataset with anywhere from 10k to 1M examples, depending on the model.

To add new examples to your training set
```shell
# simulate new training examples and store in simulate/my_project
./run_phyddle.py -s S -c config --sim_more 14000

# encode all raw_data examples as tensors in format/my_project
./run_phyddle.py -s F -c config

# train network with tensor_data, but override batch size, then store in train/my_project
./run_phyddle.py -s T -c config --batch_size 256

# make prediction with example dataset
./run_phyddle.py -s E -c config

# generate figures and store in plot
./run_phyddle.spy -s P -c config
```

Pipeline options are applied to all pipeline stages. See the full list of currently supported options with
```shell
./run_phyddle.sh --help
```

## Note on code stability

Code on the [main](https://github.com/mlandis/phyddle/tree/main) branch is tested and stable with respect to the standard use cases. Code on the [development](https://github.com/mlandis/phyddle/tree/development) branch contains new features, but is not as rigorously tested. Most phyddle development occurs on a 16-core Intel Macbook Pro laptop and a 64-core Intel Ubuntu server, so there are also unknown portability/scalability issues to correct. Any feedback is appreciated! michael.landis@wustl.edu*

## About
Thanks for your interest in phyddle. The phyddle project emerged from a phylogenetic deep learning study led by Ammon Thompson ([paper](https://www.biorxiv.org/content/10.1101/2023.02.08.527714v2)). The goal of phyddle is to provide its users with a generalizable pipeline workflow for phylogenetic modeling and deep learning. This hopefully will make it easier for phylogenetic model enthusiasts and developers to explore and apply models that do not have tractable likelihood functions. It's also intended for use by methods developers who want to characterize how deep learning methods perform under different conditions for standard phylogenetic estimation tasks.

The phyddle project is developed by [Michael Landis](https://landislab.org) and [Ammon Thompson](https://scholar.google.com/citations?user=_EpmmTwAAAAJ&hl=en&oi=ao).

