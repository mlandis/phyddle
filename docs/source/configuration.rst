.. _Configuration:

Configuration
=============

.. note:: 
    
    This section describes how to configure settings for a phyddle analysis.
    Visit :ref:`Pipeline` to learn more about how settings determine the
    behavior of a phyddle analysis. Visit :ref:`Glossary` to learn more about
    how phyddle defines different terms.

There are two ways to configure the settings of a phyddle analysis: through a
:ref:`config file <config_file>` or the :ref:`command line <config_cli>`.
Command line settings outrank config file settings.

.. _config_file:

By file
-------

The phyddle config file is a Python dictionary of analysis arguments (``args``)
that configure how phyddle pipeline steps behave. Because it's a Python script,
you can write code within the config file to specify your analysis, if you find
that helpful. The below example defines settings into different blocks based on
which pipeline step first needs a given setting. However, any setting might be
used by different pipeline steps, so we concatenate all settings into a single
dictionary called ``args``, which is then used by all pipeline steps. Settings
configured by file can be adjusted through the :ref:`command line <config_cli>`,
if desired.

.. note::

    By default, phyddle assumes you want to use the config file called
    ``config.py``. Use a different config file by calling, e.g.
    ``./run_pipline --cfg my_other_config.py``

.. code-block:: python

    #==============================================================================#
    # Config:       Default phyddle config file                                    #
    # Authors:      Michael Landis and Ammon Thompson                              #
    # Date:         230804                                                         #
    # Description:  Simple BiSSE model                                             #
    #==============================================================================#

    args = {

        #-------------------------------#
        # Project organization          #
        #-------------------------------#
        'proj'    : 'example',                  # project name(s)
        'step'    : 'SFTEP',                    # step(s) to run
        'verbose' : 'T',                        # print verbose phyddle output?
        'sim_dir' : '../workspace/simulate',    # directory for simulated data
        'fmt_dir' : '../workspace/format',      # directory for tensor-formatted data
        'trn_dir' : '../workspace/train',       # directory for trained network
        'plt_dir' : '../workspace/plot',        # directory for plotted figures
        'est_dir' : '../workspace/estimate',    # directory for predictions on new data
        'log_dir' : '../workspace/log',         # directory for analysis logs

        #-------------------------------#
        # Multiprocessing               #
        #-------------------------------#
        'use_parallel'   : 'T',                 # use multiprocessing to speed up jobs?
        'num_proc'       : -2,                  # how many CPUs to use (-2 means all but 2)

        #-------------------------------#
        # Simulate Step settings        #
        #-------------------------------#
        'sim_command'       : 'Rscript sim/R/sim_one.R',  # exact command string
        'sim_logging'       : 'verbose',        # verbose, compressed, or clean
        'start_idx'         : 0,                # first simulation replicate index
        'end_idx'           : 1000,             # last simulation replicate index
        'sim_batch_size'    : 10,               # number of replicates per simulation

        #-------------------------------#
        # Format Step settings          #
        #-------------------------------#
        'encode_all_sim'    : 'T',
        'num_char'          : 1,                # number of evolutionary characters
        'num_states'        : 2,                # number of states per character
        'min_num_taxa'      : 10,               # min number of taxa for valid sim
        'max_num_taxa'      : 500,              # max number of taxa for valid sim
        'tree_width'        : 500,              # tree width category used to train network
        'tree_encode'       : 'extant',         # use model with serial or extant tree
        'brlen_encode'      : 'height_brlen',   # how to encode phylo brlen? height_only or height_brlen
        'char_encode'       : 'integer',        # how to encode discrete states? one_hot or integer
        'param_est'         : [                 # model parameters to predict (labels)
            'birth_1', 'birth_2', 'death', 'state_rate'
        ],
        'param_data'        : [                 # model parameters that are known (aux. data)
            'sample_frac'
        ],
        'tensor_format'     : 'hdf5',           # save as compressed HDF5 or raw csv
        'char_format'       : 'csv',
        'save_phyenc_csv'   : 'F',              # save intermediate phylo-state vectors to file

        #-------------------------------#
        # Train Step settings           #
        #-------------------------------#
        'trn_objective'     : 'param_est',      # what is the learning task? param_est or model_test
        'num_epochs'        : 10,               # number of training intervals (epochs)
        'prop_test'         : 0.05,             # proportion of sims in test dataset
        'prop_val'          : 0.05,             # proportion of sims in validation dataset
        'prop_cal'          : 0.20,             # proportion of sims in CPI calibration dataset
        'combine_test_val'  : 'T',
        'cpi_coverage'      : 0.95,             # coverage level for CPIs
        'cpi_asymmetric'    : 'T',              # upper/lower ('T') or symmetric ('F') CPI adjustments
        'batch_size'        : 1024,             # number of samples in each training batch
        'loss'              : 'mse',            # loss function for learning
        'optimizer'         : 'adam',           # optimizer for network weight/bias parameters
        'metrics'           : ['mae', 'acc'],   # recorded training metrics

        #-------------------------------#
        # Estimate Step settings        #
        #-------------------------------#
        'est_prefix'     : 'new.0',             # prefix for new dataset to predict

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

Via command line
----------------

Settings applied through a :ref:`config file <config_file>` can be overwritten
by setting options when running phyddle from the command line. The names of
settings are the same for the command line options and in the config file.
Using command line options makes it easy to adjust the behavior of pipeline
steps without needing to edit the config file. List all settings that can be
adjusted with the command line using the ``--help`` option:

.. code-block::

	$ phyddle --help
    
    usage: phyddle [-h] [-c] [-p] [-s] [-v] [-f] [--make_cfg] [--use_parallel] [--num_proc] [--sim_dir] [--fmt_dir] [--trn_dir]
               [--est_dir] [--plt_dir] [--log_dir] [--sim_command] [--sim_logging] [--start_idx] [--end_idx] [--sim_more]
               [--sim_batch_size] [--encode_all_sim] [--num_char] [--num_states] [--min_num_taxa] [--max_num_taxa]
               [--downsample_taxa] [--tree_width] [--tree_encode] [--brlen_encode] [--char_encode] [--param_est] [--param_data]
               [--char_format] [--tensor_format] [--save_phyenc_csv] [--trn_objective] [--num_epochs] [--trn_batch_size]
               [--prop_test] [--prop_val] [--prop_cal] [--cpi_coverage] [--cpi_asymmetric] [--loss] [--optimizer] [--metrics]
               [--est_prefix] [--plot_train_color] [--plot_label_color] [--plot_test_color] [--plot_val_color] [--plot_aux_color]
               [--plot_est_color]

    Software to fiddle around with deep learning for phylogenetic models

    options:
    -h, --help           show this help message and exit
    -c , --cfg           Config file name
    -p , --proj          Project name(s) for pipeline step(s)
    -s , --step          Pipeline step(s) defined with (S)imulate, (F)ormat, (T)rain, (E)stimate, (P)lot, or (A)ll
    -v , --verbose       Verbose output to screen?
    -f, --force          Arguments override config file settings
    --make_cfg           Write default config file to '__config_default.py'?'
    --use_parallel       Use parallelization? (recommended)
    --num_proc           Number of cores for multiprocessing (-N for all but N)
    --sim_dir            Directory for raw simulated data
    --fmt_dir            Directory for tensor-formatted simulated data
    --trn_dir            Directory for trained networks and training output
    --est_dir            Directory for new datasets and estimates
    --plt_dir            Directory for plotted results
    --log_dir            Directory for logs of analysis metadata
    --sim_command        Simulation command to run single job (see documentation)
    --sim_logging        Simulation logging style
    --start_idx          Start replicate index for simulated training dataset
    --end_idx            End replicate index for simulated training dataset
    --sim_more           Add more simulations with auto-generated indices
    --sim_batch_size     Number of replicates per simulation command
    --encode_all_sim     Encode all simulated replicates into tensor?
    --num_char           Number of characters
    --num_states         Number of states per character
    --min_num_taxa       Minimum number of taxa allowed when formatting
    --max_num_taxa       Maximum number of taxa allowed when formatting
    --downsample_taxa    Downsampling strategy taxon count
    --tree_width         Width of phylo-state tensor
    --tree_encode        Encoding strategy for tree
    --brlen_encode       Encoding strategy for branch lengths
    --char_encode        Encoding strategy for character data
    --param_est          Model parameters to estimate
    --param_data         Model parameters treated as data
    --char_format        File format for character data
    --tensor_format      File format for training example tensors
    --save_phyenc_csv    Save encoded phylogenetic tensor encoding to csv?
    --trn_objective      Objective of training procedure
    --num_epochs         Number of training epochs
    --trn_batch_size     Training batch sizes
    --prop_test          Proportion of data used as test examples (assess trained network performance)
    --prop_val           Proportion of data used as validation examples (diagnose network overtraining)
    --prop_cal           Proportion of data used as calibration examples (calibrate CPIs)
    --cpi_coverage       Expected coverage percent for calibrated prediction intervals (CPIs)
    --cpi_asymmetric     Use asymmetric (True) or symmetric (False) adjustments for CPIs?
    --loss               Loss function for optimization
    --optimizer          Method used for optimizing neural network
    --metrics            Recorded training metrics
    --est_prefix         Predict results for this dataset
    --plot_train_color   Plotting color for training data elements
    --plot_label_color   Plotting color for training label elements
    --plot_test_color    Plotting color for test data elements
    --plot_val_color     Plotting color for validation data elements
    --plot_aux_color     Plotting color for auxiliary data elements
    --plot_est_color     Plotting color for new estimation elements

.. _Setting_Summary:

Table summary
-------------

This section summarizes available settings
in phyddle. The `Setting` column is the exact name of the string that appears in
the configuration file and command-line argument list. The `Step(s)` identifies
all steps that use the setting: [S]imulate, [F]ormat, [T]rain, [E]stimate, and
[P]lot. The `Type` column is the Python variable type expected for the setting.
The `Description` gives a brief description of what the setting does. Visit 
:ref:`Pipeline` to learn more about phyddle settings impact different pipeline
analysis steps. 

.. _table_phyddle_settings:

.. tabularcolumns:: p{0.1\linewidth}p{0.1\linewidth}p{0.1\linewidth}p{0.7\linewidth}
.. csv-table:: phyddle settings
   :file: ./tables/settings_phyddle.csv
   :header-rows: 1
   :widths: 10, 10, 10, 70
   :delim: |
   :align: center
   :width: 100%
   :class: longtable


.. _Special_Settings:

Details
-------

This section provides detailed descriptions for several settings that
are not intuitive to specify, but very powerful when used correctly.

.. _setting_description_step:

``step``
^^^^^^^^

The ``step`` setting controls which steps should be applied.
Each pipeline step is represented by a capital letter:
``S`` for :ref:`Simulate`, ``F`` for :ref:`Format`, ``T`` for :ref:`Train`,
``E`` for :ref:`Estimate`, ``P`` for :ref:`Plot`, and ``A`` for all steps.

For example, the following two commands are equivalent

.. code-block:: shell

    phyddle --step A
    phyddle -s SFTEP

whereas calling

.. code-block:: shell

    phyddle -s SF

commands phyddle to perform the Simulate and Format steps, but not the Train,
Estimate, or Plot steps.

.. _setting_description_proj:

``proj``
^^^^^^^^

The ``proj`` setting controls how project names are assigned to different
pipeline steps. Typically, ``proj`` is provided a single project name that is
shared across all pipeline steps. For example, calling either command

.. code-block:: shell

    phyddle --proj my_project
    phyddle -p my_project

causes all results from this phyddle analysis to be stored in a subdirectory
called ``my_project``. The ``proj`` setting can also be used to specify
different project names for individual pipeline steps. For example, calling

.. code-block:: shell

    phyddle --proj my_project,E:new_estimate,P:new_plot

would use ``new_estimate`` as the project name for the ``E`` step (Estimate),
``new_plot`` for the ``P`` step (Plot), and ``my_project`` for all other steps.

