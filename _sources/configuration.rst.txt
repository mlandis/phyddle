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
    ``phyddle --cfg my_other_config.py``

.. note::

    phyddle maintains a number of example config files for different models
    and simulation methods. These are organized as project subdirectories
    within the ``./workspace`` directory. For example,
    ``./workspace/bisse_r/config.py`` simulates under a BiSSE model
    with the R simulation script ``./workspace/bisse_r/sim_bisse.R``.

.. code-block:: python

    #==============================================================================#
    # Config:       Example phyddle config file                                    #
    # Authors:      Michael Landis and Ammon Thompson                              #
    # Date:         230804                                                         #
    # Description:  Simple BiSSE model                                             #
    #==============================================================================#

    work_dir = './'                           # Assumes config is run from local dir
    args = {
        #-------------------------------#
        # Basic                         #
        #-------------------------------#
        'step'               : 'SFTEP',        # Pipeline step(s) defined with
                                               #   (S)imulate, (F)ormat, (T)rain,
                                               #   (E)stimate, (P)lot, or (A)ll
        'verbose'            : 'T',            # Verbose output to screen?
        'output_precision'   : 16,             # Number of digits (precision)
                                               #   for numbers in output files

        #-------------------------------#
        # Analysis                      #
        #-------------------------------#
        'use_parallel'       : 'T',            # Use parallelization? (recommended)
        'num_proc'           : -2,             # Number of cores for multiprocessing 
                                               #   (-N for all but N)

        #-------------------------------#
        # Workspace                     #
        #-------------------------------#
        
        'dir'                : f'{work_dir}',           # Base directory for all step directories
        'sim_dir'            : f'{work_dir}/simulate',  # Directory for raw simulated data
        'emp_dir'            : f'{work_dir}/empirical', # Directory for raw simulated data
        'fmt_dir'            : f'{work_dir}/format',    # Directory for tensor-formatted simulated data
        'trn_dir'            : f'{work_dir}/train',     # Directory for trained networks and training output
        'est_dir'            : f'{work_dir}/estimate',  # Directory for new datasets and estimates
        'plt_dir'            : f'{work_dir}/plot',      # Directory for plotted results
        'log_dir'            : f'{work_dir}/log',       # Directory for logs of analysis metadata
        'prefix'             : 'out',          # Prefix for all output unless step prefix given
        'sim_prefix'         : 'out',          # Prefix for raw simulated data
        'emp_prefix'         : 'out',          # Prefix for raw empirical data
        'fmt_prefix'         : 'out',          # Prefix for tensor-formatted data
        'trn_prefix'         : 'out',          # Prefix for trained networks and training output
        'est_prefix'         : 'out',          # Prefix for new datasets and estimates
        'plt_prefix'         : 'out',          # Prefix for plotted results

        #-------------------------------#
        # Simulate                      #
        #-------------------------------#
        'sim_command'        : f'Rscript {work_dir}/sim_bisse.R', # Simulation command to run single
                                                                  #   job (see documentation)
        'sim_logging'        : 'verbose',                 # Simulation logging style
        'start_idx'          : 0,                         # Start index for simulated training replicates
        'end_idx'            : 1000,                      # End index for simulated training replicates
        'sim_batch_size'     : 10,                        # Number of replicates per simulation command

        #-------------------------------#
        # Format                        #
        #-------------------------------#
        'encode_all_sim'     : 'T',                  # Encode all simulated replicates into tensor?
        'num_char'           : 1,                    # Number of characters
        'num_states'         : 2,                    # Number of states per character
        'min_num_taxa'       : 10,                   # Minimum number of taxa allowed when formatting
        'max_num_taxa'       : 500,                  # Maximum number of taxa allowed when formatting
        'downsample_taxa'    : 'uniform',            # Downsampling strategy taxon count
        'tree_width'         : 500,                  # Width of phylo-state tensor
        'tree_encode'        : 'extant',             # Encoding strategy for tree
        'brlen_encode'       : 'height_brlen',       # Encoding strategy for branch lengths
        'char_encode'        : 'integer',            # Encoding strategy for character data
        'param_est'          : {                     # Unknown model parameters to estimate
            'log10_birth_1'      : 'real',
            'log10_birth_2'      : 'real',
            'log10_death'        : 'real',
            'log10_state_rate'   : 'real',
            'model_type'         : 'cat',
            'root_state'         : 'cat'
        ],
        'param_data'        : {                      # Known model parameters to treat as aux. data
            'sample_frac'        : 'real'
        },
        'char_format'        : 'csv',                # File format for character data
        'tensor_format'      : 'hdf5',               # File format for training example tensors
        'save_phyenc_csv'    : 'F',                  # Save encoded phylogenetic tensor encoding to csv?

        #-------------------------------#
        # Train                         #
        #-------------------------------#
        'num_epochs'         : 20,                   # Number of training epochs
        'trn_batch_size'     : 2048,                 # Training batch sizes
        'prop_test'          : 0.05,                 # Proportion of data used as test examples
                                                     #     (to assess trained network performance)
        'prop_val'           : 0.05,                 # Proportion of data used as validation examples
                                                     #     (to diagnose network overtraining)
        'prop_cal'           : 0.2,                  # Proportion of data used as calibration examples
                                                     #     (to calibrate CPIs)
        'cpi_coverage'       : 0.95,                 # Expected coverage percent for calibrated
                                                     #     prediction intervals (CPIs)
        'cpi_asymmetric'     : 'T',                  # Use asymmetric (True) or symmetric (False)
                                                     #     adjustments for CPIs?
        'loss'               : 'mae',                # Loss function for optimization
        'optimizer'          : 'adam',               # Method used for optimizing neural network
        'phy_channel_plain'  : [64, 96, 128],        # Output channel sizes for plain convolutional
                                                     #     layers for phylogenetic state input
        'phy_channel_stride' : [64, 96],             # Output channel sizes for stride convolutional
                                                     #     layers for phylogenetic state input
        'phy_channel_dilate' : [32, 64],             # Output channel sizes for dilate convolutional
                                                     #     layers for phylogenetic state input
        'aux_channel'        : [128, 64, 32],        # Output channel sizes for dense layers for
                                                     #     auxiliary data input
        'lbl_channel'        : [128, 64, 32],        # Output channel sizes for dense layers for
                                                     #     label outputs
        'phy_kernel_plain'   : [3, 5, 7],            # Kernel sizes for plain convolutional layers
                                                     #     for phylogenetic state input
        'phy_kernel_stride'  : [7, 9],               # Kernel sizes for stride convolutional layers
                                                     #     for phylogenetic state input
        'phy_kernel_dilate'  : [3, 5],               # Kernel sizes for dilate convolutional layers
                                                     #     for phylogenetic state input
        'phy_stride_stride'  : [3, 6],               # Stride sizes for stride convolutional layers
                                                     #     for phylogenetic state input
        'phy_dilate_dilate'  : [3, 5],               # Dilation sizes for dilate convolutional layers
                                                     #     for phylogenetic state input

        #-------------------------------#
        # Estimate                      #
        #-------------------------------#
        # not currently used

        #-------------------------------#
        # Plot                          #
        #-------------------------------#
        'plot_train_color'   : 'blue',               # Plotting color for training data elements
        'plot_label_color'   : 'orange',             # Plotting color for training label elements
        'plot_test_color'    : 'purple',             # Plotting color for test data elements
        'plot_val_color'     : 'red',                # Plotting color for validation data elements
        'plot_aux_color'     : 'green',              # Plotting color for auxiliary data elements
        'plot_emp_color'     : 'black',              # Plotting color for empirical elements
        'plot_num_scatter'   : 50,                   # Number of examples in scatter plot
        'plot_min_emp'       : 5,                    # Minimum number of empirical datasets to plot densities
        'plot_num_emp'       : 10                    # Number of empirical results to plot
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

	usage: phyddle [-h] [-c] [-s] [-v] [--make_cfg]
                   [--output_precision] [--use_parallel] [--num_proc]
                   [--no_emp] [--no_sim] [--dir] [--sim_dir]
                   [--emp_dir] [--fmt_dir] [--trn_dir] [--est_dir]
                   [--plt_dir] [--log_dir] [--prefix] [--sim_prefix]
                   [--emp_prefix] [--fmt_prefix] [--trn_prefix]
                   [--est_prefix] [--plt_prefix] [--sim_command]
                   [--sim_logging] [--start_idx] [--end_idx]
                   [--sim_more] [--sim_batch_size] [--encode_all_sim]
                   [--num_char] [--num_states] [--min_num_taxa]
                   [--max_num_taxa] [--downsample_taxa] [--tree_width]
                   [--tree_encode] [--brlen_encode] [--char_encode]
                   [--param_est] [--param_data] [--char_format]
                   [--tensor_format] [--save_phyenc_csv]
                   [--num_epochs] [--trn_batch_size] [--prop_test]
                   [--prop_val] [--prop_cal] [--cpi_coverage]
                   [--cpi_asymmetric] [--loss_real] [--optimizer]
                   [--log_offset] [--phy_channel_plain]
                   [--phy_channel_stride] [--phy_channel_dilate]
                   [--aux_channel] [--lbl_channel]
                   [--phy_kernel_plain] [--phy_kernel_stride]
                   [--phy_kernel_dilate] [--phy_stride_stride]
                   [--phy_dilate_dilate] [--plot_train_color]
                   [--plot_test_color] [--plot_val_color]
                   [--plot_label_color] [--plot_aux_color]
                   [--plot_emp_color] [--plot_num_scatter]
                   [--plot_min_emp] [--plot_num_emp]
    
    Software to fiddle around with deep learning for phylogenetic
    models
    
    options:
      -h, --help            show this help message and exit
      -c , --cfg            Config file name
      -s , --step           Pipeline step(s) defined with (S)imulate,
                            (F)ormat, (T)rain, (E)stimate, (P)lot, or
                            (A)ll
      -v , --verbose        Verbose output to screen?
      --make_cfg            Write default config file to
                            '__config_default.py'?
      --output_precision    Number of digits (precision) for numbers
                            in output files
      --use_parallel        Use parallelization? (recommended)
      --num_proc            Number of cores for multiprocessing (-N
                            for all but N)
      --no_emp              Disable Format/Estimate steps for
                            empirical data?
      --no_sim              Disable Format/Estimate steps for
                            simulated data?
      --dir                 Parent directory for all step directories
                            unless step directory given
      --sim_dir             Directory for raw simulated data
      --emp_dir             Directory for raw empirical data
      --fmt_dir             Directory for tensor-formatted data
      --trn_dir             Directory for trained networks and
                            training output
      --est_dir             Directory for new datasets and estimates
      --plt_dir             Directory for plotted results
      --log_dir             Directory for logs of analysis metadata
      --prefix              Prefix for all output unless step prefix
                            given
      --sim_prefix          Prefix for raw simulated data
      --emp_prefix          Prefix for raw empirical data
      --fmt_prefix          Prefix for tensor-formatted data
      --trn_prefix          Prefix for trained networks and training
                            output
      --est_prefix          Prefix for estimate results
      --plt_prefix          Prefix for plotted results
      --sim_command         Simulation command to run single job (see
                            documentation)
      --sim_logging         Simulation logging style
      --start_idx           Start replicate index for simulated
                            training dataset
      --end_idx             End replicate index for simulated training
                            dataset
      --sim_more            Add more simulations with auto-generated
                            indices
      --sim_batch_size      Number of replicates per simulation
                            command
      --encode_all_sim      Encode all simulated replicates into
                            tensor?
      --num_char            Number of characters
      --num_states          Number of states per character
      --min_num_taxa        Minimum number of taxa allowed when
                            formatting
      --max_num_taxa        Maximum number of taxa allowed when
                            formatting
      --downsample_taxa     Downsampling strategy taxon count
      --tree_width          Width of phylo-state tensor
      --tree_encode         Encoding strategy for tree
      --brlen_encode        Encoding strategy for branch lengths
      --char_encode         Encoding strategy for character data
      --param_est           Model parameters and variables to estimate
      --param_data          Model parameters and variables treated as
                            data
      --char_format         File format for character data
      --tensor_format       File format for training example tensors
      --save_phyenc_csv     Save encoded phylogenetic tensor encoding
                            to csv?
      --num_epochs          Number of training epochs
      --trn_batch_size      Training batch sizes
      --prop_test           Proportion of data used as test examples
                            (assess trained network performance)
      --prop_val            Proportion of data used as validation
                            examples (diagnose network overtraining)
      --prop_cal            Proportion of data used as calibration
                            examples (calibrate CPIs)
      --cpi_coverage        Expected coverage percent for calibrated
                            prediction intervals (CPIs)
      --cpi_asymmetric      Use asymmetric (True) or symmetric (False)
                            adjustments for CPIs?
      --loss_real           Loss function for real value estimates
      --optimizer           Method used for optimizing neural network
      --log_offset          Offset size c when taking ln(x+c) for
                            zero-valued variables
      --phy_channel_plain   Output channel sizes for plain
                            convolutional layers for phylogenetic
                            state input
      --phy_channel_stride
                            Output channel sizes for stride
                            convolutional layers for phylogenetic
                            state input
      --phy_channel_dilate
                            Output channel sizes for dilate
                            convolutional layers for phylogenetic
                            state input
      --aux_channel         Output channel sizes for dense layers for
                            auxiliary data input
      --lbl_channel         Output channel sizes for dense layers for
                            label outputs
      --phy_kernel_plain    Kernel sizes for plain convolutional
                            layers for phylogenetic state input
      --phy_kernel_stride   Kernel sizes for stride convolutional
                            layers for phylogenetic state input
      --phy_kernel_dilate   Kernel sizes for dilate convolutional
                            layers for phylogenetic state input
      --phy_stride_stride   Stride sizes for stride convolutional
                            layers for phylogenetic state input
      --phy_dilate_dilate   Dilation sizes for dilate convolutional
                            layers for phylogenetic state input
      --plot_train_color    Plotting color for training data elements
      --plot_test_color     Plotting color for test data elements
      --plot_val_color      Plotting color for validation data
                            elements
      --plot_label_color    Plotting color for label elements
      --plot_aux_color      Plotting color for auxiliary data elements
      --plot_emp_color      Plotting color for empirical elements
      --plot_num_scatter    Number of examples in scatter plot
      --plot_min_emp        Minimum number of empirical datasets to
                            plot densities
      --plot_num_emp        Number of empirical results to plot

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
   :file: ./tables/phyddle_settings.csv
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

Step
^^^^

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


.. _setting_description_dir:

Step directories
^^^^^^^^^^^^^^^^

A standard phyddle analysis assumes all work is stored within a single
project directory. Work from each step, however, is stored into different
subdirectories.

Customizing the input and output directories among steps allows users to
quickly explore alternative pipeline designs while leaving previous
pipeline results in place.

The project directory can be set using ``dir``. During analysis, phyddle will
create subdirectories for each step using default names, as needed. For example,
if ``dir`` is set to the local directory ``./``, then a full phyddle analysis 
would use the following directories for the analysis:

.. code-block:: shell

  ./simulate        # default sim_dir
  ./empirical       # default emp_dir
  ./format          # default fmt_dir
  ./train           # default trn_dir
  ./estimate        # default est_dir
  ./plot            # default plt_dir
  ./log             # default log_dir

Individual step directories can be overriden with custom directory locations.
For example, setting ``dir`` to ``./`` but setting ``emp_dir`` to
``/Users/mlandis/datasets/viburnum`` and ``plt_dir`` to 
``/Users/mlandis/projects/viburnum/results`` would cause
phyddle to use the following directories:
 
.. code-block:: shell
    
  ./simulate                                # default sim_dir
  /Users/mlandis/datasets/viburnum          # custom emp_dir
  ./format                                  # default fmt_dir
  ./train                                   # default trn_dir
  ./estimate                                # default est_dir
  /Users/mlandis/projects/viburnum/results  # custom plt_dir
  ./log                                     # default log_dir 
 

.. _setting_description_prefix:

Step prefixes
^^^^^^^^^^^^^

Standard phyddle analyses assume that the files generated by each pipeline
step begin with the filename prefix ``'out'``.

The filename prefix for all pipeline steps can be changed using the ``prefix``
settings. Changing the filename prefix allows you to generate alternative
pipeline filesets without overwriting previous results.

As with the pipeline directory settings (above), prefixes for individual
pipeline steps can be overridden with custom prefixes. This allows you to compare
pipeline performance using different settings, while saving previous work. For
example,

.. code-block:: shell

  phyddle -c config.py \                # load config
          -s TE \                       # run Train and Estimate steps
          --prefix new \                # T & E output has prefix 'new'
          --fmt_prefix out \            # Format input has prefix 'out' 
          --num_epochs 50 \             # Train for 50 epochs
          --trn_batch_size 4096         # Use batch sizes of 4096 samples


.. _setting_description_nosim_noemp:

``no_sim`` and ``no_emp``
^^^^^^^^^^^^^^^^^^^^^^^^^

By default the :ref:`Format` and :ref:`Estimate` steps run in a greedy manner,
against the simulated datasets identified by ``dir`` (or ``sim_dir``) and
``prefix`` (or ``sim_prefix``), and against the empirical datasets identified
by ``dir`` (or ``emp_dir``) and ``prefix`` (or ``emp_prefix``), should those
datasets exist.

Setting ``--no_sim`` during a command-line run will instruct phyddle to skip
the Format and Estimate steps for the simulated datasets (i.e. the train and
test datasets).

Setting ``--no_emp`` during a command-line run will instruct phyddle to skip
the Format and Estimate steps for the empirical datasets.
 
In particular, the ``--no_sim`` flag in particular is useful when you only
need to format new empirical datasets, but do not need to reformat existing
simulated (i.e. training/test) datasets. The flag helps eliminate redundant
formatting tasks during pipeline development. 
