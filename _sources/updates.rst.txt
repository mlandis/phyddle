Updates
=======

The complete commit history for phyddle is located here: https://github.com/mlandis/phyddle/commits/main

**phyddle v0.2.0** -- 24.03.19

  * New support for categorical labels (e.g. model testing).
  * New plots for categorical results.
  * Add support for GPU + CUDA + PyTorch to accelerate Train and Estimate.
  * Improved printed feedback for all steps.
  * Better plots for PCA and scatter plot results.
  * Simpler default directory structure. Default phyddle configs, scripts, and outputs now share a proj directory in ./workspace. 
  * Better support for multiple project names within a workspace directory.
  * Rewrote how Format/Estimate/Plot interact so simulated-test/empirical datasets are treated similarly.
  * Better error checking and helpful messages for Simulate-Format config mismatches.


**phyddle v0.1.1** -- 24.02.18

  * Simpler directory settings for analyses
  * Simpler settings application hierarchy
  * Removed workspace and proj arguments
  * Removed requirement for __config_default.py
  * Repo directory restructuring, with clearer distinction between scripts and configs
  * Added example simulation script for PhyloJunction


**phyddle v0.1.0** -- 24.02.08

  * Update minor version from v0.0.10 to v0.1.0 |:partying_face:|
  * Convert from Tensorflow to PyTorch on backend
  * New support for scalable network architecture in Train
  * New support for output precision for floating point output to files
  * New summary csv report for Plot
  * Better screen output for Train progress
  * Better training history plot
  * Better network architecture plot
  * Better handling of cases where small tree size causes tree statistics to return 0/undefined/etc.
  * Fix bug with non-sequential indices among simulated training examples
  * Fix bug with Format for-loop to process summary statistics
  * Switch summary statistics to log-scale


**phyddle v0.0.10** -- 23.11.10
  
  * Basic GitHub Action tests working
  * Format step faster (removed extraneous and costly deepcopy calls)
  * Convert config_R.py to BiSSE example using R + castor
  * Add external validation for phyddle vs. castor
  * Improve config support
  * Cleaning and commenting code
  * Clean and finish sphinx documentation


**phyddle v0.0.9** -- 23.08.23

  * Package now hosted through conda: https://anaconda.org/landismj/phyddle
  * Better errors for missing files (configs, command, scripts, etc.)
  * PyPI package installed through pip now autoinstalls dependencies


**phyddle v0.0.8** -- 23.08.16

  * Minor fixes following 0.0.7 update
  * Documentation visuals fixed


**phyddle v0.0.7** -- 23.08.16

  * Format now splits training from test datasets.
  * Estimate now applies trained model against test dataset.
  * Steps now initialize using settings look-up table.
  * Add PCA contour plot.
  * Simplify filenames within projects for many steps.
  * Timestamp step start/end/duration.
  * Docs now hosted online at http://mlandis.github.io/phyddle
  * Python package now hosted through pip at https://pypi.org/project/phyddle/
  * Add bump2version supports


**phyddle v0.0.6** -- 23.08.09

  * Refreshed documentation, plus added setting tables and glossary.
  * Centralize management of settings.
  * Settings now automatically apply default config, then user config, then command line arguments.
  * Format only enodes for a single tree width category now.
  * Format now downsamples and stores raw num. taxa for each data point.
  * Allow for "all-but-X" CPUs for multiprocessing.
  * Simulate wth --sim_more option to add new replicates to training examples.
  * Format encodes all detected datasets by default.
  * Support to allow scripts to simulate batches (>1) of replicates.
  * Major overhaul of help docs, but not done.

**phyddle v0.0.5** -- 23.07.30

  * Pipeline steps renamed to: Simulate, Format, Train, Estimate, Plot
  * Simulators now in scripts/sim/MASTER, scripts/sim/R, scripts/sim/Rev
  * Streamlined handling of phyddle settings, internally
  * Model configuration code moved outside phyddle
  * MASTER simulator now outside phyddle, in scripts/sim/MASTER
  * Complete source code reorg/rename
  * Pipeline script now prints pipeline steps
  * Support for granular management of flow of projects across pipeline steps
  * Support asymmetric CPI calibration
  * Better handling of FileNotFoundError related to Estimate step
  * Much faster Formatting step (>100x speedup)
  * Pipeline steps now generate logs in ``workspace/log/<project_name>`` to track phyddle is used during a project analysis
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

