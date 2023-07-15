Updates
=======

The complete commit history for phyddle is located here: https://github.com/mlandis/phyddle/commits/main


**phyddle v0.0.5** -- (in progress)

* Pipeline script now prints pipeline steps
* Better handling of FileNotFoundError related to Predicting step
* Tested against Apple M1. Not an easy install, because unsupported by Tensorflow. Thanks Albert and Sean!


**phyddle v0.0.4** -- 23.07.09

* Simulating now supports command-line scripts
* Better backend support for alternative phylostate tensor encodings
* Simplified pipeline scripts and interface
* Docs improved to reflect current code design
* Tests now cover Simulating and Formatting


**phyddle v0.0.3** -- 23.07.02

* Sphinx configuration for documentation
* TestPyPI configuration for package deployment
* GitHub Actions configuration for unit testing


**phyddle v0.0.2** -- 23.06.25

* (first internal working version)
* trained network generates parameter estimates and coverage-calibrated prediction intervals (CPIs) for input datasets
* provides several state-dependent birth-death model types and variants (more to come)
* parallelized simulating, formatting, and learning
* encoding of phylogenetic-state tensor from serial and extant-only input with multiple states (CBLV+S and CDV+S extensions)
* encoding of auxiliary data tensor from automatically computed summary statistics and "known" parameter (e.g. sampling rate)
* HDF5 with gzip compression for tensor data
* shuffles and splits input tensors into training, test, validation, and calibration datasets for supervised learning
* builds network with convolution, pooling, and dense layers that match input tensors
* trains network and saves history
* automatic figure generation with Matplotlib


**phyddle v0.0.1** -- 23.03.16

* (initial development version)


**Planned features**

* better back-end documentation for developers/hackers/etc.
* expanded library of model types/variants for discrete and continuous state types
* expanded support for standard simulators and a generic script-based simulator interface
* better parallelization for hdf5-chunking of very large datasets
* better subsampling support
* expansion of standard prediction tasks
* expansion of unit/integration testing
