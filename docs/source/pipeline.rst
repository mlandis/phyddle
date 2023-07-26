.. _Pipeline:

Pipeline
========
..
    This guide provides phyddle users with an overview for how the pipeline toolkit works, where it stores files, and how to interpret files and figures. Learn how to configure phyddle analyses by reading the :ref:`Settings` documentation. 

A phyddle pipeline analysis has five steps: :ref:`Simulate`, :ref:`Format`, :ref:`Train`, :ref:`Estimate`, and :ref:`Plot`. Standard analyses run all steps, in order for a single batch of settings. That said, steps can be run multiple times under different settings and orders, which is useful for exploratory and advanced analyses.

This page describes how different analysis settings control phyddle pipeline behavior, whereas the :ref:`Settings` page describes how to apply settings to the software. In addition, the examples in this section follow the :ref:`Example Project Workspace <Example_Project>` layout.

.. _Model_Configuration:

Model Configuration
-------------------

:ref:`Model_Configuration` is only needed if you use phyddle to simulate training data through the :ref:`Simulate` step. The :ref:`Simulate` step describes what formats phyddle expects for training datasets.

Models in phyddle are designed by setting five control variables: ``model_type``, ``model_variant``, ``num_char``, ``rv_fn``, ``rv_arg``.

At a high level, ``model_type`` defines a class of models that share similar statespaces and eventspaces. The ``model_variant`` defines how rates are assigned to distinct event patterns in the eventspace. The behavior of how characters evolve and how character states influence other evolutionary dynamics are internally determined by ``model_type`` and ``model_variant``. The number of distinct character supported by the model is set by ``num_char``. Lastly, the way base parameter values are drawn for each simulated example in the training dataset are controlled with a set of random variable functions (``rv_fn``) and random variable arguments (``rv_arg``). Both ``rv_fn`` and ``rv_arg`` are dictionaries with keys that correspond to event class labels. With ``rv_fn`` the values are data-generating functions that have arguments and behavior equivalent to ``scipy.stats.distribution.rvs``. With ``rv_arg`` the values are the arguments passed in to the corresponding ``rv_fn`` functions.

Descriptions of supported built-in models that you can specify with ``model_type`` and ``model_variant`` are listed using the ``--show_models``. (More models to come. Developer guide will describe how to add new back-end model variants \[easier\] and types \[harder\].)

.. code-block:: shell

	$ ./run_phyddle.py --show_models
	Type                Variant             Description
	============================================================
	geosse              --                  Geographic State-dependent Speciation Extinction [GeoSSE]
						free_rates          rates differ among all events within type
						equal_rates         rates equal among all events within type
						density_effect      equal_rates + local density-dependent extinction

	sirm                --                  Susceptible-Infected-Recovered-Migration [SIRM]
						free_rates          rates differ among all events within type
						equal_rates         rates equal among all events within type

Let's create a geographic state-dependent speciation-extinction (GeoSSE) model as a concrete example. GeoSSE models describe how species move and evolve among discrete regions through four event classes: within-region speciation, between-region speciation, dispersal, and local extinction. We'll create a GeoSSE model for a biogeographic system with three regions where all events within a class have equal rates, where rates are exponentially distributed with expected values of 1.0. The settings in the configuration file for this would be

.. code-block:: python

	'model_type'     : 'geosse',
	'model_variant'  : 'equal_rates',
	'num_char'       : 3,
	'rv_fn'          : {
			'w' : sp.stats.expon.rvs,
			'b' : sp.stats.expon.rvs,
			'd' : sp.stats.expon.rvs,
			'e' : sp.stats.expon.rvs
		},
	'rv_arg'         : {
			'w' : { 'scale' : 1.0 },
			'b' : { 'scale' : 1.0 },
			'd' : { 'scale' : 1.0 },
			'e' : { 'scale' : 1.0 }
		}




.. _Simulate:

Simulate
--------

:ref:`Simulate` instructs phyddle to simulate your training dataset. Currently, phyddle supports simulation by user-provided :ref:`Command_Line_Simulations` command line scripts and through the :ref:`Master_Simulations` plugin from BEAST.


.. _Command_Line_Simulations:

Command Line Simulations
^^^^^^^^^^^^^^^^^^^^^^^^
(to be written)

.. _Master_Simulations:

MASTER Simulations
^^^^^^^^^^^^^^^^^^
MASTER was designed primarily to simulate under Susceptible-Infected-Recovered compartment models from epidemiology. These models allow for lineage to evolve according to rates that depend on the state of the entire evolutionary system. For example, the rate of change for one species may depend on its state and the number of other species in that state or other states. See the Requirements section to see how phyddle expects MASTER and BEAST are configured for its use.

Results from simulations are stored based on the ``sim_dir`` and ``proj`` settings. ``sim_dir`` is the directory in phyddle that contains the "raw" simulated output across all projects, and is typically set to ``simulate``. ``proj`` defines the simulations for a single project. Each individual simulation is assigned a replicate index. You can simulate replicates in different "chunks" with the start (``start_idx``) and end (``end_idx``) index variables, which is especially useful for building up a training dataset for a project over multiple jobs, e.g. on a cluster.

Each replicate being simulated will run for some length of evolutionary time (``stop_time``) and may require some minimum (``min_num_taxa``) and/or maximum (``max_num_taxa``) number of lineages per simulation.

Assuming that ``sim_dir == workspace/simulate`` and ``proj == example``, the standard simulation output will follow this format

.. code-block:: shell

	workspace/simulate/example/sim.0.tre
	workspace/simulate/example/sim.0.dat.nex
	workspace/simulate/example/sim.0.param_col.csv
	workspace/simulate/example/sim.0.param_row.csv


The ``.tre`` file contains a Newick string. The ``.dat.nex`` contains a Nexus character matrix. These are reformatted as tensors to become the input training dataset. The ``.param_col.csv`` and ``.param_row.csv`` contain the simulating parameters in column and row format, with the row format files being converted to a tensor of training labels. 

In addition, MASTER will retain only the certain simulated taxa (populations) from the system, set using ``sample_population``. phyddle generates an ``xml`` file that specifies the MASTER simulation, a ``beast.log`` file that reports the text generated by BEAST during simulation, and a ``json`` file that reports metadata about the evolutionary history of the system. These files can be valuable for debugging and postprocessing, but they may become quite large, so the ``sim_logging`` setting will control whether they are retained, compressed, or deleted.

Note, that downstream steps in the pipeline, such as `Format`, only require that the appropriate files with the appropriate content exist to proceed. They can either be generated with the `Simulate` step within phyddle or completely outside of phyddle.

.. _Format:

Format
------

Raw simulated data must first boverted into a tensor format to interface with the neural network we'll later train and use for future estimateions. For most computational purposes, it is safe to think of a tensor as an n-dimensional array. It is essential that all individual datasets share a standard shape (e.g. numbers of rows and columns) to ensure the training dataset that contains estimateable data patterns. Learn more about phyddle tensors on the :ref:`Tensor Formats <Tensor_Formats>` page. Briefly, formatting in phyddle encodes two input tensors and one output tensor:

- One input tensor is the **phylogenetic-state tensor**. Loosely speaking, these tensors contain information about terminal taxa across columns and information about relevant branch lengths and states per taxon across rows. The phylogenetic-state tensors used by phyddle are based on the compact bijective ladderized vector (CBLV) format of Voznica et al. (2022).
- The second input is the **auxiliary data tensor**. This tensor contains summary statistics for the phylogeny and character data matrix and "known" parameters for the data generating process.
- The output tensor reports **labels** that are generally unknown data generating parameters to be estimated using the neural network.  Depending on the estimation task, all or only some model parameters might be treated as labels for training and estimateion. For example, when ``model_variant == 'free_rates'`` one might want to estimate every rate in the model, but estimate only one parameter per event-class when ``model_variant == 'equal_rates'``.

phyddle saves its formatted tensors to ``fmt_dir`` in a subdirectory called ``fmt_proj``. For example, if ``fmt_dir == workspace/format`` and ``fmt_proj == example`` then the tensors are stored in ``workspace/format/example``.

Format processes the tree, data matrix, and model parameters for each replicate. This is done in parallel, when the setting is enabled. Simulated data are processed using ``CBLV+S`` format if ``tree_type == 'serial'``. If ``tree_type = 'extant'`` then all non-extant taxa are pruned, saved as ``pruned.tre``, then encoded using CDV+S. The size of each tree ($n$) is then used to identify the largest value in ``tree_width_cats`` it can fit into. The phylogenetics-state tensors and auxiliary data tensors are then created. If ``save_phyenc_csv`` is set, then individual csv files are saved for each dataset, which is especially useful for formatting new empirical datasets into an accepted phyddle format. The ``param_pred`` setting identifies which parameters in the labels tensor you want to treat as downstream estimateion targets. The ``param_data`` setting identifies which of those parameters you want to treat as "known" auxiliary data.

Formatted tensors are then saved to disk either in simple comma-separated value format or in a compressed HDF5 format. For example, suppose we set ``fmt_dir == 'format``, ``proj == 'example'``, and ``tree_type == 'serial'``. If we set, it produces ``tensor_format == 'hdf5'``:

.. code-block:: shell

	workspace/format/example/sim.nt200.hdf5
	workspace/format/example/sim.nt500.hdf5

or if ``tensor_format == 'csv'``:

.. code-block:: shell

	workspace/format/example/sim.nt200.cdvs.data.csv
	workspace/format/example/sim.nt200.labels.csv
	workspace/format/example/sim.nt200.summ_stat.csv
	workspace/format/example/sim.nt500.cdvs.data.csv
	workspace/format/example/sim.nt500.labels.csv
	workspace/format/example/sim.nt500.summ_stat.csv

These files can then be processed by the `Train` step.


.. _Train:

Train
-----

`Train` builds a neural network that can be trained to make estimateions based on the tensors made by the `Format` step. This step also shuffles the replicate indices and splits the entire dataset into separate training, test, validation, and calibration subsets. The phylogenetic-state tensor is processed by convolutional and pooling layers, while the auxiliary data is processed by dense layers. All input layers are concatenated then pushed into three branches terminating in output layers to produce point estimates and upper and lower estimateion intervals. Lastly, the step runs the training procedure and stores its results, including the history and trained network, to file.

When data are read in, they are shuffled, with some set aside for test data (``prop_test``), validation data (``prop_validation``), and calibration data (``prop_calibration``), with the remaining data being used for ``training``. A network must be trained against a particular ``tree_width`` size (see above). The network also must target a particular estimateion interval (e.g. ``cpi_coverage == 0.95`` means 95% of test estimateions are expected contain the true simulating value) for two-sided conformalized quantile regression). Training runs for a number of intervals given by ``num_epoch`` using batch stochastic gradient descent, with batch sizes given by ``batch_size``. Parameter point estimates use a loss function (e.g. ``loss == 'mse'``; Tensorflow-supported string or function) while lower/upper estimateion intervals must use a pinball loss function (hard-coded). Different optimizers can be used to update network weight and bias parameters (e.g. ``optimizer == 'adam'``; Tensorflow-supported string or function).

Training is automatically parallelized using CPUs and GPUs, dependent on how Tensorflow was installed and system hardware. Output files are stored in the directory assigned to ``<lrn_dir>`` in the subdirectory ``<proj>``.


.. _Estimate:

Estimate
--------

`Estimate` loads a new dataset stored in ``<est_dir>/<est_proj>`` with filenames ``<est_prefix.tre>`` and ``<est_prefix>.dat.nex``. This step then loads a pretrained network and has it estimate new point estimates and calibrated prediction intervals (CPIs) based on other project settings. New estimations are then stored into the original ``<est_dir>/<est_proj>``.


.. _Plot:

Plot
----

`Plot` collects all results from the `Format`, `Train`, and `Estimate` steps to compile a set of useful figures, listed below. When results from `Estimate` are available, the step will integrate it into other figures to contextualize where that input dataset and estimateed labels fall with respect to the training dataset. Plots are stored within ``<plot_dir>`` in the ``<plot_proj>`` subdirectory. Colors for plot elements can be modified with ``plot_train_color``, ``plot_label_color``, ``plot_test_color``, ``plot_val_color``, ``plot_aux_color``, and ``plot_est_color`` using common color names or hex codes supported by Matplotlib.

- ``summary.pdf`` contains all figures in a single plot
- ``est_CI.pdf`` - simple plot of point estimates and calibrated estimateion intervals for estimateion
- ``histogram_aux.pdf`` - histograms of all values in the auxiliary dataset; red line for estimateed dataset
- ``pca_aux.pdf`` - pairwise PCA of all values in the auxiliary dataset; red dot for estimateed dataset
- ``history_.pdf`` - loss performance across epochs for test/validation datasets for entire network
- ``history_<stat_name>.pdf`` - loss, accuracy, error performance across epochs for test/validation datasets for particular statistics (point est., lower CPI, upper CPI)
- ``train_<label_name>.pdf`` - point estimates and calibrated estimateion intervals for training dataset
- ``test_<label_name>.pdf`` - point estimates and calibrated estimateion intervals for test dataset
- ``network_architecture.pdf`` - visualization of Tensorflow architecture


