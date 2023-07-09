.. _Settings:

Settings
========

There are two ways to configure a phyddle analysis: through the config file or (when run via command line) command line options. Settings a config file are overwritten by any provided command line options. Both input systems use the same names for the same settings. All required analysis settings must be provided by config file and command line options, together, for a phyddle analysis to run successfully.


.. _config_file:

Configuration by file
---------------------

The config file is a Python dictionary that specifies various program settings (arguments or ``args``) that configure how the underlying pipeline steps behave. Because it's a Python script, you can write code within the config file to specify your analysis, if you find that helpful. The below example defines settings into different blocks based on which pipeline step first needs a given setting. However, any setting might be used by different pipeline steps, so we concatenate all settings into a single dictionary called ``args``, which is then used by all pipeline steps.

**NOTE: phyddle assumes you want to use the config file calle ``my_config.py``. Use a different config file by calling, e.g. ``./run_pipline --cfg my_other_config.py``**

.. code-block:: python

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


.. _config_CLI:

Configuration by CLI
--------------------

Settings applied through the config file can be overwritten by setting options when running phyddle from the command line. The names of settings are the same for the command line options and in the config file. Using command line options makes it easy to adjust the behavior of pipeline steps without needing to edit the config file. List all settings that can be adjusted with the command line using the ``--help`` option:

.. code-block:: shell

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


