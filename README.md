# phyddle

A pipeline-based toolkit for fiddling around with phylogenetic models and deep learning 

*NOTE:*

*<b>This private beta version of phyddle is still under development.</b> It is generally tested and stable with respect to a few use standard cases. Much of the documentation and some key features are still missing. Most phyddle development occurs on a 16-core Intel Macbook Pro laptop and a 64-core Ubuntu server, so there are also unknown portability/scalability issues to correct. Any feedback is appreciated! michael.landis@wustl.edu*


## Overview

<img align="right" src="https://github.com/landislab/landislab.github.io/blob/5bb4685a12ebf4c99dd773de6d87b44cc3c47090/assets/research/img/phyddle_pipeline.png?raw=true" width="35%">

A standard phyddle analysis performs the following tasks for you:

- **Pipeline configuration** applies analysis settings provided through a config file and/or command line arguments.
- **Model configuration** constructs a base simulating model to be *Simulated* (states, events, rates).
- **Simulating** simulates a large training dataset under the model to be *Formatted* (parallelized, partly compressed). Saves output in the [`raw_data`](raw_data) directory.
- **Formatting** encodes the raw simulated data into tensor format for *Learning* (parallelized, compressed). Saves output in the [`tensor_data`](tensor_data) directory.
- **Learning** shuffles and splits training data, builds a network, then trains and saves the network with the data for *Prediction*. Saves output in the [`network`](network) directory.
- **Predicting** estimates model parameters for a new dataset with the trained network. Saves output in the [`predict`](predict) directory.
- **Plotting** generates figures that summarize the training data (*Formatting*), the network and its training (*Learning*), and any new predictions (*Predicting*). Saves output in the [`plot`](plot) directory.



## Quick start

To run a phyddle analysis enter the `code` directory:
```shell
cd ~/projects/phyddle/code
```

Then create and run a pipeline under the settings you've specified in `my_config.py`:
```shell
./run_pipeline.sh --cfg my_config
```

This will run a phyddle analysis for a simple 3-region GeoSSE model with just 500 training examples. In practice, you'll want to generate a larger training dataset with anywhere from 10k to 1M examples, depending on the model.

To add new examples to your training set
```shell
# simulate new training examples and store in raw_data/my_project
./run_simulate.sh --cfg my_config --start_idx 500 --end_idx 15000

# encode all raw_data examples as tensors in tensor_data/my_project
./run_format.sh --cfg my_config --start_idx 0 --end_idx 15000

# train network with tensor_data, but override batch size, then store in network/my_project
./run_learn.sh --cfg my_config --batch_size 256

# make prediction with example dataset
./run_predict.sh --cfg my_config

# generate figures and store in plot
./run_plot.sh --cfg my_config
```

Pipeline options are applied to all pipeline stages. See the full list of currently supported options with
```shell
./run_pipeline.sh --help
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

Planned features:
- better back-end documentation for developers/hackers/etc.
- expanded library of model types/variants for discrete and continuous state types
- expanded support for standard simulators and a generic script-based simulator interface
- better parallelization for hdf5-chunking of very large datasets
- better subsampling support
- expansion of standard prediction tasks
- expansion of unit/integration testing



## Installation & Requirements

To install phyddle on your computer, you can either clone the repository or you can [download](https://github.com/mlandis/phyddle/archive/refs/heads/main.zip) and unzip the current version of the main branch.

phyddle was last tested using with Mac OS X 11.6.4 (Intel CPU) using Python 3.11.3 (installed with homebrew) with the following versions of the required third-party package dependencies (installed with pip):
```
PIL 9.5.0
PyPDF2 3.0.1
argparse 1.1
h5py 3.8.0
joblib 1.2.0
keras 2.12.0
matplotlib 3.7.1
numpy 1.23.5
pandas 2.0.0
pydot_ng 2.0.0
scipy 1.10.1
sklearn 1.2.2
tensorflow 2.12.0
tqdm 4.65.0
```

phyddle is also used with a 64-core Ubuntu LTS 22.04 server using Python 3.xx.xx (aptitude) and similar package versions. phyddle has yet not been tested using conda, Windows, M1 Macs, various GPUs, etc.

phyddle currently relies on the BEAST plugin MASTER for simulation. The operating system must be able to call BEAST from anywhere in the filesystem through the `beast` command. This can be done by adding the BEAST executable to be covered by the `$PATH` shell variable. Creating a symbolic link (shortcut) to the BEAST binary `beast` with `ln -s` in `~/.local/bin` is one an easy way to make `beast` globally accessible on Mac OS X.
```
$ ls -lart /Users/mlandis/.local/bin/beast
lrwxr-xr-x  1 mlandis  staff  35 Feb 14 10:32 /Users/mlandis/.local/bin/beast -> /Applications/BEAST 2.7.3/bin/beast
$  which beast
/Users/mlandis/.local/bin/beast
$ beast -version                                                                                                       1 ↵
BEAST v2.7.3
---
BEAST.base v2.7.3
MASTER v7.0.0
BEAST.app v2.7.3
---
Java version 17.0.5
```

## Developer guide
(to be written)

We will host documentation built with `sphinx` to help developers read, use, and modify the back-end Python source code for their purposes. For now, anyone who clones the repository can make local changes to the code, though with some risk of introducing errors to how the tested version of phyddle behaves.


## User guide
(under development)

This guide provides phyddle users with an overview for how the toolkit works, how to configure it, where it stores files, and how to interpret files and figures.

In general, the pipeline assumes that the user supplies runs scripts in `code` using a consistent *project name* (e.g. `my_project`) to coordinate the analysis across the `raw_data`, `tensor_data`, `network`, and `plot` directories.

### Example dataset

The `example` project bundled with phyddle was generated using the command `./code/run_pipeline.sh --cfg config --proj example --end_idx 25000`. This corresponds to a 3-region equal-rates GeoSSE model. All directories are populated, except `raw_data/example` contains only 20 original examples.

### Directory structure

The repository has five main directories:
- [`code`](code) contains scripts to generate and process data
- [`raw_data`](raw_data) contains raw data generated by simulation
- [`tensor_data`](tensor_data) contains data formatted into tensors for training networks
- [`network`](network) contains trained networks and diagnostics
- [`predict`](predict) contains new test datasets their predictions
- [`plot`](plot) contains figures of training and validation procedures

If a user runs an analysis with the project name `my_project`, then pipeline files created by the analysis would be stored in the following directories
```
raw_data/my_project       # output of Simulating
tensor_data/my_project    # output of Formatting
network/my_project        # output of Learning
predict/my_project        # output of Predicting
plot/my_project           # output of Plotting
```

### Analysis configuration

There are two ways to configure a phyddle analysis: through the config file or (when run via command line) command line options. Settings a config file are overwritten by any provided command line options. Both input systems use the same names for the same settings. All required analysis settings must be provided by config file and command line options, together, for a phyddle analysis to run successfully.

## Config file

The config file is a Python dictionary that specifies various program settings (arguments or `args`) that configure how the underlying pipeline steps behave. Because it's a Python script, you can write code within the config file to specify your analysis, if you find that helpful. The below example defines settings into different blocks based on which pipeline step first needs a given setting. However, any setting might be used by different pipeline steps, so we concatenate all settings into a single dictionary called `args`, which is then used by all pipeline steps.

**NOTE: phyddle assumes you want to use the config file calle `my_config.py`. Use a different config file by calling, e.g. `./run_pipline --cfg my_other_config.py`**

```python
#==============================================================================#
# Default phyddle config file                                                  #
#==============================================================================#

# helper libraries
import scipy as sp

# helper variables
num_char = 3

args = {

    #-------------------------------#
    # Project organization          #
    #-------------------------------#
    'proj'           : 'my_project',        # directory name for pipeline project
    'sim_dir'        : '../raw_data',       # directory for simulated data
    'fmt_dir'        : '../tensor_data',    # directory for tensor-formatted data
    'net_dir'        : '../network',        # directory for trained network
    'plt_dir'        : '../plot',           # directory for plotted figures
    'pred_dir'       : '../predict',        # directory for predictions on new data
    'pred_prefix'    : 'new.1',             # prefix for new dataset to predict

    #-------------------------------#
    # Multiprocessing               #
    #-------------------------------#
    'use_parallel'   : True,                # use multiprocessing to speed up jobs?
    'num_proc'       : -2,                  # how many CPUs to use (-2 means all but 2)

    #-------------------------------#
    # Model Configuration           #
    #-------------------------------#
    'model_type'         : 'geosse',        # model type defines general states and events
    'model_variant'      : 'equal_rates',   # model variant defines rate assignments
    'num_char'           : num_char,        # number of evolutionary characters
    'rv_fn'              : {                # distributions for model parameters
        'w': sp.stats.expon.rvs,
        'e': sp.stats.expon.rvs,
        'd': sp.stats.expon.rvs,
        'b': sp.stats.expon.rvs
    },
    'rv_arg'             : {                # loc/scale/shape for model parameter dists
        'w': { 'scale' : 0.2 },
        'e': { 'scale' : 0.1 },
        'd': { 'scale' : 0.1 },
        'b': { 'scale' : 0.5 }
    },

    #-------------------------------#
    # Simulating Step settings      #
    #-------------------------------#
    'sim_logging'       : 'verbose',        # verbose, compressed, or clean
    'start_idx'         : 0,                # first simulation replicate index
    'end_idx'           : 1000,             # last simulation replicate index
    'sample_population' : ['S'],            # name of population to sample
    'stop_time'         : 10,               # time to stop simulation
    'min_num_taxa'      : 10,               # min number of taxa for valid sim
    'max_num_taxa'      : 500,              # max number of taxa for valid sim

    #-------------------------------#
    # Formatting Step settings      #
    #-------------------------------#
    'tree_type'         : 'extant',         # use model with serial or extant tree
    'tree_width_cats'   : [ 200, 500 ],     # tree size categories for binning phylo-state tensors
    'param_pred'        : [                 # model parameters to predict (labels)
        'w_0', 'e_0', 'd_0_1', 'b_0_1'
    ],
    'param_data'        : [],               # model parameters that are known (aux. data)
    'tensor_format'     : 'hdf5',           # save as compressed HDF5 or raw csv
    'save_phyenc_csv'   : False,            # save intermediate phylo-state vectors to file

    #-------------------------------#
    # Learning Step settings        #
    #-------------------------------#
    'tree_width'        : 500,              # tree size class used to train network
    'num_epochs'        : 20,               # number of training intervals (epochs)
    'prop_test'         : 0.05,             # proportion of sims in test dataset
    'prop_validation'   : 0.05,             # proportion of sims in validation dataset
    'prop_calibration'  : 0.20,             # proportion of sims in CPI calibration dataset
    'cpi_coverage'      : 0.95,             # coverage level for CPIs
    'batch_size'        : 128,              # number of samples in each training batch
    'loss'              : 'mse',            # loss function for learning
    'optimizer'         : 'adam',           # optimizer for network weight/bias parameters
    'metrics'           : ['mae', 'acc'],   # recorded training metrics

    #-------------------------------#
    # Plotting Step settings        #
    #-------------------------------#
    'plot_train_color'      : 'blue',       # plot color for training data
    'plot_test_color'       : 'purple',     # plot color for test data
    'plot_validation_color' : 'red',        # plot color for validation data
    'plot_aux_data_color'   : 'green',      # plot color for input auxiliary data
    'plot_label_color'      : 'orange',     # plot color for labels (params)
    'plot_pred_color'       : 'black'       # plot color for predictions

    #-------------------------------#
    # Predicting Step settings      #
    #-------------------------------#
    # prediction already handled by previously defined settings
    # no prediction-specific settings currently implemented
}
```

### Command line arguments

Settings applied through the config file can be overwritten by setting options when running phyddle from the command line. The names of settings are the same for the command line options and in the config file. Using command line options makes it easy to adjust the behavior of pipeline steps without needing to edit the config file. List all settings that can be adjusted with the command line using the `--help` option:

```
$ ./run_pipeline.py --help

usage: run_simulate.py [-h] [-c] [-p] [--use_parallel] [--num_proc] [--sim_dir] [--fmt_dir] [--net_dir] [--plt_dir] [--pred_dir] [--pred_prefix] [--show_models] [--model_type]
                       [--model_variant] [--num_char] [--sim_logging] [--start_idx] [--end_idx] [--stop_time] [--min_num_taxa] [--max_num_taxa] [--tree_type] [--tree_width_cats]
                       [--tensor_format] [--save_phyenc_csv] [--tree_width] [--num_epochs] [--batch_size] [--prop_test] [--prop_validation] [--prop_calibration] [--cpi_coverage]
                       [--loss] [--optimizer] [--plot_train_color] [--plot_label_color] [--plot_test_color] [--plot_val_color] [--plot_aux_color] [--plot_pred_color]

phyddle pipeline config

options:
  -h, --help           show this help message and exit
  -c , --cfg           Config file name
  -p , --proj          Project name used as directory across pipeline stages
  --use_parallel       Use parallelization? (recommended)
  --num_proc           How many cores for multiprocessing? (e.g. 4 uses 4, -2 uses all but 2)
  --sim_dir            Directory for raw simulated data
  --fmt_dir            Directory for tensor-formatted simulated data
  --net_dir            Directory for trained networks and predictions
  --plt_dir            Directory for plotted results
  --pred_dir           Predict results for dataset located in this directory
  --pred_prefix        Predict results for this dataset
  --show_models        Print all available model types and variants?
  --model_type         Model type
  --model_variant      Model variant
  --num_char           Number of characters
  --sim_logging        Simulation logging style
  --start_idx          Start index for simulation
  --end_idx            End index for simulation
  --stop_time          Maximum duration of evolution for each simulation
  --min_num_taxa       Minimum number of taxa for each simulation
  --max_num_taxa       Maximum number of taxa for each simulation
  --tree_type          Type of tree
  --tree_width_cats    The phylo-state tensor widths for formatting training datasets, space-delimited
  --tensor_format      Storage format for simulation tensors
  --save_phyenc_csv    Save encoded phylogenetic tensor encoding to csv?
  --tree_width         The phylo-state tensor width dataset used for a neural network
  --num_epochs         Number of learning epochs
  --batch_size         Training batch sizes during learning
  --prop_test          Proportion of data used as test examples (demonstrate trained network performance)
  --prop_validation    Proportion of data used as validation examples (diagnose network overtraining)
  --prop_calibration   Proportion of data used as calibration examples (calibrate conformal prediction intervals)
  --cpi_coverage       Expected coverage percent for calibrated prediction intervals
  --loss               Loss function used as optimization criterion
  --optimizer          Method used for optimizing neural network
  --plot_train_color   Plotting color for training data elements
  --plot_label_color   Plotting color for training label elements
  --plot_test_color    Plotting color for test data elements
  --plot_val_color     Plotting color for validation data elements
  --plot_aux_color     Plotting color for auxiliary input data elements
  --plot_pred_color    Plotting color for prediction data elements
```

### Model configuration

Models in phyddle are designed by setting five control variables: `model_type`, `model_variant`, `num_char`, `rv_fn`, `rv_arg`.

Defining a model is only needed if you use phyddle to simulate training data through the Simulating step. The Simulating step describes the format phyddle expects for training datasets.

At a high level, `model_type` defines a class of models that share similar statespaces and eventspaces. The `model_variant` defines how rates are assigned to distinct event patterns in the eventspace. The behavior of how characters evolve and how character states influence other evolutionary dynamics are internally determined by `model_type` and `model_variant`. The number of distinct character supported by the model is set by `num_char`. Lastly, the way base parameter values are drawn for each simulated example in the training dataset are controlled with a set of random variable functions (`rv_fn`) and random variable arguments (`rv_arg`). Both `rv_fn` and `rv_arg` are dictionaries with keys that correspond to event class labels. With `rv_fn` the values are data-generating functions that have arguments and behavior equivalent to `scipy.stats.distribution.rvs`. With `rv_arg` the values are the arguments passed in to the corresponding `rv_fn` functions.

Descriptions of supported built-in models that you can specify with `model_type` and `model_variant` are listed using the `--show_models`. (More models to come. Developer guide will describe how to add new back-end model variants \[easier\] and types \[harder\].)

```shell
$ ./run_simulate.py --show_models                             130 ↵
Type                Variant             Description
============================================================
geosse              --                  Geographic State-dependent Speciation Extinction [GeoSSE]
                    free_rates          rates differ among all events within type
                    equal_rates         rates equal among all events within type
                    density_effect      equal_rates + local density-dependent extinction

sirm                --                  Susceptible-Infected-Recovered-Migration [SIRM]
                    free_rates          rates differ among all events within type
                    equal_rates         rates equal among all events within type
```

Let's create a geographic state-dependent speciation-extinction (GeoSSE) model as a concrete example. GeoSSE models describe how species move and evolve among discrete regions through four event classes: within-region speciation, between-region speciation, dispersal, and local extinction. We'll create a GeoSSE model for a biogeographic system with three regions where all events within a class have equal rates, where rates are exponentially distributed with expected values of 1.0. The settings in the configuration file for this would be

```python
`model_type`     : `geosse`,
`model_variant`  : `equal_rates`,
`num_char`       : 3,
`rv_fn`          : {
    	`w` : sp.stats.expon.rvs,
    	`b` : sp.stats.expon.rvs,
    	`d` : sp.stats.expon.rvs,
    	`e` : sp.stats.expon.rvs
	},
`rv_arg`         : {
    	`w` : { `scale` : 1.0 },
    	`b` : { `scale` : 1.0 },
    	`d` : { `scale` : 1.0 },
    	`e` : { `scale` : 1.0 }
	}

```


### Simulating

Once your model is configured, you can instruct phyddle to simulate your training dataset. Currently, phyddle relies on the MASTER plugin from BEAST to simulate. MASTER was designed primarily to simulate under Susceptible-Infected-Recovered compartment models from epidemiology. These models allow for lineage to evolve according to rates that depend on the state of the entire evolutionary system. For example, the rate of change for one species may depend on its state and the number of other species in that state or other states. See the Requirements section to see how phyddle expects MASTER and BEAST are configured for its use.

Results from simulations are stored based on the `sim_dir` and `proj` settings. `sim_dir` is the directory in phyddle that contains the "raw" simulated output across all projects, and is typically set to `raw_data`. `proj` defines the simulations for a single project. Each individual simulation is assigned a replicate index. You can simulate replicates in different "chunks" with the start (`start_idx`) and end (`end_idx`) index variables, which is especially useful for building up a training dataset for a project over multiple jobs, e.g. on a cluster.

Each replicate being simulated will run for some length of evolutionary time (`stop_time`) and may require some minimum (`min_num_taxa`) and/or maximum (`max_num_taxa`) number of lineages per simulation.

Assuming that `sim_dir == raw_data` and `proj == example`, the standard simulation output will follow this format
```
raw_data/example/sim.0.tre
raw_data/example/sim.0.dat.nex
raw_data/example/sim.0.param_col.csv
raw_data/example/sim.0.param_row.csv
```

The `.tre` file contains a Newick string. The `.dat.nex` contains a Nexus character matrix. These are reformatted as tensors to become the input training dataset. The `.param_col.csv` and `.param_row.csv` contain the simulating parameters in column and row format, with the row format files being converted to a tensor of training labels. 

In addition, MASTER will retain only the certain simulated taxa (populations) from the system, set using `sample_population`. phyddle generates an `xml` file that specifies the MASTER simulation, a `beast.log` file that reports the text generated by BEAST during simulation, and a `json` file that reports metadata about the evolutionary history of the system. These files can be valuable for debugging and postprocessing, but they may become quite large, so the `sim_logging` setting will control whether they are retained, compressed, or deleted.


Note, that downstream steps in the pipeline, such as Formatting, only require that the appropriate files with the appropriate content exist to proceed. They can either be generated with the Simuating step within phyddle or completely outside of phyddle.

### Formatting

Raw simulated data must first boverted into a tensor format to interface with the neural network we'll later train and use for future predictions. For most computational purposes, it is safe to think of a tensor as an n-dimensional array. It is essential that all individual datasets share a standard shape (e.g. numbers of rows and columns) to ensure the training dataset that contains predictable data patterns. phyddle formatting encodes two input tensors and one output tensor.

phyddle saves its formatted tensors to `fmt_dir` in a subdirectory called `proj`. For example, if `fmt_dir == tensor_data` and `proj == example` then the tensors are stored in `tensor_data/example`.

One input tensor is the **phylogenetic-state tensor**. Loosely speaking, these tensors contain information about terminal taxa across columns and information about relevant branch lengths and states per taxon across rows. The phylogenetic-state tensors used by phyddle are based on the compact bijective ladderized vector (CBLV) format of Voznica et al. (2022).

CBLV encodes a phylogenetic tree with $n \leq N$ taxa in to a matrix of with 2 rows that contains branch length sorted across $N$ columns that contain topological information for a tree with taxa serially sampled over time (e.g. epidemiological data). The matrix is then flattened into vector format. Ammon et al. (2022) introduced the CBLV+S format, which allows for multiple characters to be associated with each taxon in a CBLV, constructing a matrix with $2+M$ rows and $N$ columns for a dataset of $n \leq N$ taxa with $M$ characters. Another important tensor type developed by Lambert et al. (2022) is the compact diversified vector (CDV). CDV is a matrix with 2 rows and $N$ columns, with the first row corresponding to node ages and the other recording state values for a single binary character.

CBLV and CDV differ primarily in terms what criteria they use to they order (ladderize) the topology. CBLV ladderizes by minimum terminal-node age per clade and CDV ladderized by maximum subclade branch length. Both formats pack the phylogenetic information from a tree with $n$ taxa into a "wider" tree-width class that allows up to $N$ taxa. The tensor is packed from left-to-right based on an in-order tree traversal, then use zeroes to buffer the all remaining cells until the $N$th column. In phyddle, we use expanded CBLV+S and CDV+S formats that additionally encode terminal branch length formation for the terminal node and the parent node, resulting in $4+M$ rows for our CBLV+S and $3+M$ rows for our CDV+S format. (Will add diagram later.)

The second input is the **auxiliary data tensor**. This tensor contains summary statistics for the phylogeny and character data matrix and "known" parameters for the data generating process. The summary statistics, for example, report things such as the number of taxa, the tree height, the mean and variance of branch lengths and node ages, the state-pattern counts, etc. The known parameters might report things such as the population sizes of a susceptible population or the recovery period in an SIR model.

The output tensor reports **labels** that are generally unknown data generating parameters to be estimated using the neural network.  Depending on the estimation task, all or only some model parameters might be treated as labels for training and prediction. For example, when `model_variant == 'free_rates'` one might want to estimate every rate in the model, but estimate only one parameter per event-class when `model_variant == 'equal_rates'`.

Formatting processes the tree, data matrix, and model parameters for each replicate. This is done in parallel, when the setting is enabled. Simulated data are processed using `CBLV+S` format if `tree_type == 'serial'`. If `tree_type = 'extant'` then all non-extant taxa are pruned, saved as `pruned.tre`, then encoded using CDV+S. The size of each tree ($n$) is then used to identify the largest value in `tree_width_cats` it can fit into. The phylogenetics-state tensors and auxiliary data tensors are then created. If `save_phyenc_csv` is set, then individual csv files are saved for each dataset, which is especially useful for formatting new empirical datasets into an accepted phyddle format. The `param_pred` setting identifies which parameters in the labels tensor you want to treat as downstream prediction targets. The `param_data` setting identifies which of those parameters you want to treat as "known" auxiliary data.

Formatted tensors are then saved to disk either in simple comma-separated value format or in a compressed HDF5 format. For example, suppose we set `fmt_dir == 'tensor_data`, `proj == 'example'`, and `tree_type == 'serial'`. If we set, it produces `tensor_format == 'hdf5'`:

```bash
tensor_data/example/sim.nt200.hdf5
tensor_data/example/sim.nt500.hdf5
```

or if `tensor_format == 'csv'`:

```bash
sim.nt200.cdvs.data.csv
sim.nt200.labels.csv
sim.nt200.summ_stat.csv
sim.nt500.cdvs.data.csv
sim.nt500.labels.csv
sim.nt500.summ_stat.csv
```

These files can then be processed by the Learning step.


### Learning

Learning builds a neural network that can be trained to make predictions based on the tensors made by the Formatting step. This step also shuffles the replicate indices and splits the entire dataset into separate training, test, validation, and calibration subsets. The phylogenetic-state tensor is processed by convolutional and pooling layers, while the auxiliary data is processed by dense layers. All input layers are concatenated then pushed into three branches terminating in output layers to produce point estimates and upper and lower prediction intervals. Lastly, the step runs the training procedure and stores its results, including the history and trained network, to file.

When data are read in, they are shuffled, with some set aside for test data (`prop_test`), validation data (`prop_validation`), and calibration data (`prop_calibration`), with the remaining data being used for `training`. A network must be trained against a particular `tree_width` size (see above). The network also must target a particular prediction interval (e.g. `cpi_coverage == 0.95` means 95% of test predictions are expected contain the true simulating value) for two-sided conformalized quantile regression). Training runs for a number of intervals given by `num_epoch` using batch stochastic gradient descent, with batch sizes given by `batch_size`. Parameter point estimates use a loss function (e.g. `loss == 'mse'`; Tensorflow-supported string or function) while lower/upper prediction intervals must use a pinball loss function (hard-coded). Different optimizers can be used to update network weight and bias parameters (e.g. `optimizer == 'adam'; Tensorflow-supported string or function).

Training is automatically parallelized using CPUs and GPUs, dependent on how Tensorflow was installed and system hardware. Output files are stored in the directory assigned to `<lrn_dir>` in the subdirectory `<proj>`.


### Predicting

Predicting loads a new dataset stored in `<pred_dir>/<proj>` with filenames `<pred_prefix.tre>` and `<pred_prefix>.dat.nex`. This step then loads a pretrained network and has it predict new point estimates and calibrated prediction intervals based on other project settings. New predictions are then stored into the original `<pred_dir>/<proj>`.


### Plotting

Plotting collects all results from the Formatting, Learning, and Predicting steps to compile a set of useful figures, listed below. When a prediction is available, the step will integrate it into other figures to contextualize where that input dataset and predicted labels fall with respect to the training dataset. Plots are stored within `<plot_dir>` in the `<proj>` subdirectory. Colors for plot elements can be modified with `plot_train_color`, `plot_label_color`, `plot_test_color`, `plot_val_color`, `plot_aux_color`, and `plot_pred_color` using common color names or hex codes supported by Matplotlib.

- `summary.pdf` contains all figures in a single plot
- `pred_est_CI.pdf` - simple plot of point estimates and calibrated prediction intervals for prediction
- `histogram_aux.pdf` - histograms of all values in the auxiliary dataset; red line for predicted dataset
- `pca_aux.pdf - pairwise PCA of all values in the auxiliary dataset; red dot for predicted dataset
- `history_.pdf` - loss performance across epochs for test/validation datasets for entire network
- `history_<stat_name>.pdf` - loss, accuracy, error performance across epochs for test/validation datasets for particular statistics (point est., lower CPI, upper CPI)
- `train_<label_name>.pdf` - point estimates and calibrated prediction intervals for training dataset
- `test_<label_name>.pdf` - point estimates and calibrated prediction intervals for test dataset
- `network_architecture.pdf` - visualization of Tensorflow architecture



## Issues & Troubleshooting

Please use [Issues](https://github.com/mlandis/phyddle/issues) to report bugs or feature requests that require modifying phyddle source code. Please contact [Michael Landis](mailto:michael.landis@wustl.edu) to request troubleshooting support using phyddle.



## About
Thanks for your interest in phyddle. The phyddle project emerged from a phylogenetic deep learning study led by Ammon Thompson ([paper](https://www.biorxiv.org/content/10.1101/2023.02.08.527714v2)). The goal of phyddle is to provide its users with a generalizable pipeline workflow for phylogenetic modeling and deep learning. This hopefully will make it easier for phylogenetic model enthusiasts and developers to explore and apply models that do not have tractable likelihood functions. It's also intended for use by methods developers who want to characterize how deep learning methods perform under different conditions for standard phylogenetic estimation tasks.

The phyddle project is developed by [Michael Landis](https://landislab.org) and [Ammon Thompson](https://scholar.google.com/citations?user=_EpmmTwAAAAJ&hl=en&oi=ao).

