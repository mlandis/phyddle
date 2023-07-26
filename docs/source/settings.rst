.. _Settings:

Settings
========

There are two ways to configure the settings of a phyddle analysis: through a :ref:`config file <config_file>` or the :ref:`command line <config_cli>`. Command line settings outrank config file settings.

.. _config_file:

Settings by file
----------------

The phyddle config file is a Python dictionary of analysis arguments (``args``) that configure how phyddle pipeline steps behave. Because it's a Python script, you can write code within the config file to specify your analysis, if you find that helpful. The below example defines settings into different blocks based on which pipeline step first needs a given setting. However, any setting might be used by different pipeline steps, so we concatenate all settings into a single dictionary called ``args``, which is then used by all pipeline steps. Settings configured by file can be adjusted through the :ref:`command line <config_cli>`, if desired.

.. note::

    By default, phyddle assumes you want to use the config file called ``config.py``. Use a different config file by calling, e.g. ``./run_pipline --cfg my_other_config.py``

.. code-block:: python

    #==============================================================================#
    # Default phyddle config file                                                  #
    #==============================================================================#

    # external import
    import scipy.stats
    import scipy as sp

    # helper variables
    num_char = 3
    num_states = 2

    args = {

        #-------------------------------#
        # Project organization          #
        #-------------------------------#
        'proj'    : 'my_project',               # project name(s)
        'step'    : 'A',                        # step(s) to run
        'verbose' : True,                       # print verbose phyddle output?
        'sim_dir' : '../workspace/simulate',    # directory for simulated data
        'fmt_dir' : '../workspace/format',      # directory for tensor-formatted data
        'trn_dir' : '../workspace/train',       # directory for trained network
        'plt_dir' : '../workspace/plot',        # directory for plotted figures
        'est_dir' : '../workspace/estimate',    # directory for predictions on new data
        'log_dir' : '../workspace/log',         # directory for analysis logs

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
        'num_states'         : num_states,      # number of states per character
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
        # Simulate Step settings        #
        #-------------------------------#
        'sim_method'        : 'master',         # command, master, [phylojunction], ...
        'sim_command'       : 'beast',          # exact command string, argument is output file prefix
        'sim_logging'       : 'verbose',        # verbose, compressed, or clean
        'start_idx'         : 0,                # first simulation replicate index
        'end_idx'           : 1000,             # last simulation replicate index
        'sample_population' : ['S'],            # name of population to sample
        'stop_time'         : 10,               # time to stop simulation
        'min_num_taxa'      : 10,               # min number of taxa for valid sim
        'max_num_taxa'      : 500,              # max number of taxa for valid sim

        #-------------------------------#
        # Format Step settings          #
        #-------------------------------#
        'tree_type'         : 'extant',         # use model with serial or extant tree
        'chardata_format'   : 'nexus',
        'tree_width_cats'   : [ 200, 500 ],     # tree width categories for phylo-state tensors
        'tree_encode_type'  : 'height_brlen',   # how to encode phylo brlen? height_only or height_brlen
        'char_encode_type'  : 'integer',        # how to encode discrete states? one_hot or integer
        'param_pred'        : [                 # model parameters to predict (labels)
            'w_0', 'e_0', 'd_0_1', 'b_0_1'
        ],
        'param_data'        : [],               # model parameters that are known (aux. data)
        'tensor_format'     : 'hdf5',           # save as compressed HDF5 or raw csv
        'save_phyenc_csv'   : False,            # save intermediate phylo-state vectors to file

        #-------------------------------#
        # Train Step settings           #
        #-------------------------------#
        'trn_objective'     : 'param_est',      # what is the learning task? param_est or model_test
        'tree_width'        : 500,              # tree width category used to train network
        'num_epochs'        : 20,               # number of training intervals (epochs)
        'prop_test'         : 0.05,             # proportion of sims in test dataset
        'prop_validation'   : 0.05,             # proportion of sims in validation dataset
        'prop_calibration'  : 0.20,             # proportion of sims in CPI calibration dataset
        'cpi_coverage'      : 0.95,             # coverage level for CPIs
        'cpi_asymmetric'    : True,             # upper/lower (True) or symmetric (False) CPI adjustments
        'batch_size'        : 128,              # number of samples in each training batch
        'loss'              : 'mse',            # loss function for learning
        'optimizer'         : 'adam',           # optimizer for network weight/bias parameters
        'metrics'           : ['mae', 'acc'],   # recorded training metrics

        #-------------------------------#
        # Estimate Step settings        #
        #-------------------------------#
        'est_prefix'     : 'new.1',             # prefix for new dataset to predict

        #-------------------------------#
        # Plot Step settings            #
        #-------------------------------#
        'plot_train_color'      : 'blue',       # plot color for training data
        'plot_test_color'       : 'purple',     # plot color for test data
        'plot_val_color'        : 'red',        # plot color for validation data
        'plot_aux_color'        : 'green',      # plot color for input auxiliary data
        'plot_label_color'      : 'orange',     # plot color for labels (params)
        'plot_est_color'        : 'black'       # plot color for estimated data/values

    }


.. _config_CLI:

Settings via CLI
----------------

Settings applied through a :ref:`config file <config_file>` can be overwritten by setting options when running phyddle from the command line. The names of settings are the same for the command line options and in the config file. Using command line options makes it easy to adjust the behavior of pipeline steps without needing to edit the config file. List all settings that can be adjusted with the command line using the ``--help`` option:

.. code-block:: shell

	$ ./run_phyddle.py --help
    
    usage: run_phyddle.py [-h] [-c] [-p] [-s] [-v] [--use_parallel] [--num_proc] [--sim_dir] [--fmt_dir] [--trn_dir] [--est_dir] [--plt_dir] [--log_dir] [--show_models] [--model_type] [--model_variant] [--num_char] [--sim_method] [--sim_command] [--sim_logging]
                      [--start_idx] [--end_idx] [--stop_time] [--min_num_taxa] [--max_num_taxa] [--tree_type] [--tree_width_cats] [--tree_encode_type] [--char_encode_type] [--chardata_format] [--tensor_format] [--save_phyenc_csv] [--trn_objective] [--tree_width]
                      [--num_epochs] [--batch_size] [--prop_test] [--prop_validation] [--prop_calibration] [--cpi_coverage] [--cpi_asymmetric] [--loss] [--optimizer] [--est_prefix] [--plot_train_color] [--plot_label_color] [--plot_test_color] [--plot_val_color]
                      [--plot_aux_color] [--plot_est_color]

    phyddle pipeline config

    options:
    -h, --help           show this help message and exit
    -c , --cfg           Config file name
    -p , --proj          Project name used as directory across pipeline stages
    -s , --step          Pipeline step(s) defined with (S)imulate, (F)ormat, (T)rain, (E)stimate, (P)lot, or (A)ll
    -v , --verbose       Verbose output to screen? (recommended)
    --use_parallel       Use parallelization? (recommended)
    --num_proc           How many cores for multiprocessing? (e.g. 4 uses 4, -2 uses all but 2)
    --sim_dir            Directory for raw simulated data
    --fmt_dir            Directory for tensor-formatted simulated data
    --trn_dir            Directory for trained networks and training predictions
    --est_dir            Directory for new datasets and predictions
    --plt_dir            Directory for plotted results
    --log_dir            Directory for logs of analysis metadata
    --show_models        Print all available model types and variants?
    --model_type         Model type
    --model_variant      Model variant
    --num_char           Number of characters
    --sim_method         Simulation method
    --sim_command        Simulation command (when sim_method=='command')
    --sim_logging        Simulation logging style
    --start_idx          Start index for simulation
    --end_idx            End index for simulation
    --stop_time          Maximum duration of evolution for each simulation
    --min_num_taxa       Minimum number of taxa for each simulation
    --max_num_taxa       Maximum number of taxa for each simulation
    --tree_type          Type of tree
    --tree_width_cats    The phylo-state tensor widths for formatting training datasets, space-delimited
    --tree_encode_type   Method for encoding branch length info in tensor
    --char_encode_type   Method for encoding character states in tensor
    --chardata_format    Input format for character matrix data
    --tensor_format      Output format for storing tensors of training dataset
    --save_phyenc_csv    Save encoded phylogenetic tensor encoding to csv?
    --trn_objective      Objective of training procedure
    --tree_width         The phylo-state tensor width dataset used for a neural network
    --num_epochs         Number of training epochs
    --batch_size         Training batch sizes
    --prop_test          Proportion of data used as test examples (demonstrate trained network performance)
    --prop_validation    Proportion of data used as validation examples (diagnose network overtraining)
    --prop_calibration   Proportion of data used as calibration examples (calibrate conformal prediction intervals)
    --cpi_coverage       Expected coverage percent for calibrated prediction intervals (CPIs)
    --cpi_asymmetric     Use asymmetric (True) or symmetric (False) adjustments for CPIs?
    --loss               Loss function used as optimization criterion
    --optimizer          Method used for optimizing neural network
    --est_prefix         Predict results for this dataset
    --plot_train_color   Plotting color for training data elements
    --plot_label_color   Plotting color for training label elements
    --plot_test_color    Plotting color for test data elements
    --plot_val_color     Plotting color for validation data elements
    --plot_aux_color     Plotting color for auxiliary input data elements
    --plot_est_color     Plotting color for new estimation elements


.. _Setting_Description:

Descriptions
------------

This section highlights how to configure some of the more-powerful but also more-complicated phyddle settings.

The ``step`` setting
^^^^^^^^^^^^^^^^^^^^

The ``step`` setting controls which steps should be applied.
Each pipeline step is represented by a capital letter:
`S` for :ref:`Simulate`,`F` for :ref:`Format`, `T` for :ref:`Train`, `E` for :ref:`Estimate`, `P` for :ref:`Plot`, and `A` for all steps.

For example, the following two commands are equivalent
.. code-block:: shell

    ./run_phyddle.py --step A
    ./run_phyddle.py --step SFTEP

whereas calling

.. code-block:: shell

    ./run_phyddle.py --step SF

commands phyddle to perform the Simulate and Format steps, but not the Train, Estimate, or Plot steps.


The ``proj`` setting
^^^^^^^^^^^^^^^^^^^^

The ``proj`` setting controls how project names are assigned to different pipeline steps.
Typically, ``proj`` is provided a single project name that is shared across all pipeline steps.
For example, calling

.. code-block:: shell

    ./run_phyddle.py --proj my_project

causes all results from this phyddle analysis to be stored in a subdirectory called ``my_project``.
The ``proj`` setting can also be used to specify different project names for individual pipeline steps. For example, calling

.. code-block:: shell

    ./run_phyddle.py --proj my_project,E:new_estimate,P:new_plot

would use the project name ``new_estimate`` for the Estimate step (``E``), ``new_plot`` for the Plot step (``P``), and ``my_project`` for all other steps.

