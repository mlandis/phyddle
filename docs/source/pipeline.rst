.. _Pipeline:

Pipeline
========
..
    This guide provides phyddle users with an overview for how the pipeline
    toolkit works, where it stores files, and how to interpret files and
    figures. Learn how to configure phyddle analyses by reading the
    :ref:`Configuration` documentation. 

.. note:: 
    
    This section describes how a standard phyddle pipeline analysis is
    configured and how settings determine the behavior of a phyddle analysis.
    Visit :ref:`Configuration` to learn how to assign settings for a phyddle
    analysis. Visit :ref:`Glossary` to learn more about
    how phyddle defines different terms

.. image:: images/phyddle_pipeline.png
  :scale: 18%
  :align: right

A phyddle pipeline analysis has five steps: :ref:`Simulate`, :ref:`Format`,
:ref:`Train`, :ref:`Estimate`, and :ref:`Plot`. Standard analyses run all
steps, in order for a single batch of settings. That said, steps can be run
multiple times under different settings and orders, which is useful for
exploratory and advanced analyses. Visit :ref:`Tricks` to learn how to use
phyddle to its fullest potential.

All pipeline steps create output files. All pipeline (except :ref:`Simulate`)
also require input files corresponding to at least one other pipeline step.
A full phyddle analysis for a *project* will automatically generate the
input files for downstream pipeline steps and store them in a predictable
*project directory*.

Users may also elect to use phyddle for only some steps in their analysis, and
produce files for other steps by different means. For example, :ref:`Format`
expects to format and combine large numbers of simulated datasets into tensor
formats that can be used for supervised learning with neural networks.
These simulated files can either be generated through phyddle with
the :ref:`Simulate` step or outside of phyddle entirely.

Below is the project directory structure that a standard phyddle analysis
would use. In this section, we assume the project name is ``example``:

.. code-block:: shell

    Simulate 
    - input:   None
    - output:  ./workspace/example/simulate  # simulated datasets

    Format
    - input:   ./workspace/example/simulate  # simulated datasets
               ./workspace/example/empirical # empirical datasets
    - output:  ./workspace/example/format    # formatted datasets
  
    Train
    - input:   ./workspace/example/format    # simulated training dataset
    - output:  ./workspace/example/train     # trained network + train results
  
    Estimate
    - input:   ./workspace/example/format    # simulated test + empirical datasets
               ./workspace/example/train     # trained network
    - output:  ./workspace/example/estimate  # test + empirical results

    Plot
    - input:   ./workspace/example/format    # simulated training dataset
               ./workspace/example/train     # trained network and output
               ./workspace/example/estimate  # simulated + empirical estimates
    - output:  ./workspace/example/plot      # analysis figures


.. _Simulate:

Simulate
--------

:ref:`Simulate` instructs phyddle to simulate your training dataset. Any
simulator that can be called from command-line can be used to generate training
datasets with phyddle. This allows researchers to use their favorite simulator
with phyddle for phylogenetic modeling tasks.

As a worked example, suppose we have an R script called ``sim_bisse.R`` containing
the following code

.. code-block:: r

    #!/usr/bin/env Rscript
    library(castor)
    library(ape)

    # disable warnings
    options(warn = -1)

    # example command string to simulate for "sim.1" through "sim.10"
    # cd ~/projects/phyddle/workspace/example
    # Rscript sim_bisse.R ./simulate example 1 10

    # arguments
    args        = commandArgs(trailingOnly = TRUE)
    out_path    = args[1]
    out_prefix  = args[2]
    start_idx   = as.numeric(args[3])
    batch_size  = as.numeric(args[4])
    rep_idx     = start_idx:(start_idx+batch_size-1)
    num_rep     = length(rep_idx)
    get_mle     = FALSE

    # filesystem
    tmp_fn = paste0(out_path, "/", out_prefix, ".", rep_idx)  # sim path prefix
    phy_fn = paste0(tmp_fn, ".tre")               # newick file
    dat_fn = paste0(tmp_fn, ".dat.csv")           # csv of data
    lbl_fn = paste0(tmp_fn, ".labels.csv")        # csv of labels (e.g. params)

    # dataset setup
    num_states = 2
    tree_width = 500
    label_names = c( paste0("birth_",1:num_states), "death", "state_rate", "sample_frac")

    # simulate each replicate
    for (i in 1:num_rep) {

        # set RNG seed
        set.seed(rep_idx[i])

        # rejection sample
        num_taxa = 0
        while (num_taxa < 10) {

            # simulation conditions
            max_taxa = runif(1, 10, 5000)
            max_time = runif(1, 1, 100)
            sample_frac = 1.0
            if (max_taxa > tree_width) {
                sample_frac = tree_width / max_taxa
            }

            # simulate parameters
            Q = get_random_mk_transition_matrix(num_states, rate_model="ER", max_rate=0.1)
            birth = runif(num_states, 0, 1)
            death = min(birth) * runif(1, 0, 1.0)
            death = rep(death, num_states)
            parameters = list(
                birth_rates=birth,
                death_rates=death,
                transition_matrix_A=Q
            )

            # simulate tree/data
            res_sim = simulate_dsse(
                    Nstates=num_states,
                    parameters=parameters,
                    sampling_fractions=sample_frac,
                    max_extant_tips=max_taxa,
                    max_time=max_time,
                    include_labels=T,
                    no_full_extinction=T)

            # check if tree is valid
            num_taxa = length(res_sim$tree$tip.label)
        }

        # save tree
        tree_sim = res_sim$tree
        write.tree(tree_sim, file=phy_fn[i])

        # save data
        state_sim = res_sim$tip_states - 1
        df_state = data.frame(taxa=tree_sim$tip.label, data=state_sim)
        write.csv(df_state, file=dat_fn[i], row.names=F, quote=F)

        # save learned labels (e.g. estimated data-generating parameters)
        label_sim = c(birth[1], birth[2], death[1], Q[1,2], sample_frac)
        names(label_sim) = label_names
        df_label = data.frame(t(label_sim))
        write.csv(df_label, file=lbl_fn[i], row.names=F, quote=F)

    }


    # done!
    quit()
  

This particular script has a few important features. First, the simulator is
entirely responsible for simulating the dataset. Second, the script assumes it
will be provided runtime arguments (``args```) to generate filenames and to
determine how many simulated datasets will be generated when the script is run
(more details in next paragraph). Third, output for the Newick string is stored
into a ``.tre`` file, for the character matrix data into a ``.dat.csv`` file,
and for the training labels into a comma-separated ``.csv`` file.

Now that we understand the script, we need to configure phyddle to call it
properly. This is done by setting the ``sim_command`` argument equal to a
command string of the form ``MY_COMMAND [MY_COMMAND_ARGUMENTS]``. During
simulation, phyddle executes the command string against different filepath
locations. More specifically, phyddle will execute the command
``MY_COMMAND [MY_COMMAND_ARGUMENTS] [SIM_DIR] [SIM_PREFIX]``, where ``SIM_DIR``
is the path to the directory locating the individual simulated datasets, and 
``SIM_PREFIX`` is a common prefix shared by individual simulation files. As
part of the Simulate step, phyddle will execute the command string to generate
the complete simulated dataset of replicated training examples.

In this case, we assume that `sim_bisse.R` is an R script that is located in
the subdirectory `./workspace/example` and can be executed using the `Rscript` 
command. The correct `sim_command` value to run this script is:

.. code-block:: python

    'sim_command' : 'Rscript ./workspace/example/sim_bisse.R'

Assuming ``sim_dir = './workspace/example/simulate'``, ``sim_prefix = 'sim'``
``sim_batch_size = 10``, phyddle will execute the commands during simulation

.. code-block:: shell

    Rscript sim_one.R ../workspace/example/simulate/ sim 0 10
    Rscript sim_one.R ../workspace/example/simulate/ sim 10 10
    Rscript sim_one.R ../workspace/example/simulate/ sim 20 10
    ...

for every replication index between ``start_idx`` and ``end_idx`` in
increments of ``sim_batch_size``, where the R script itself is responsible
for generating the ``sim_batch_size`` replicates per batch. In fact,
executing ``Rscript sim_bisse.R ./workspace/example/simulate/ sim 1 10``
from terminal is an ideal way to validate that your custom simulator is
compatible with the phyddle requirements.


.. _Format:

Format
------

:ref:`Format` converts simulated and/or empirical data for a project into a
tensor format that phyddle uses to train neural networks in the :ref:`Train`
step. :ref:`Format` performs two main tasks:

1. Encode all individual raw datasets in the simulate and empirical project
   directory into individual tensor representations
2. Combines all the individual tensors into larger, singular tensors that can
   be processed by the neural network

For each example, :ref:`Format` encodes the raw data into two input
tensors and one output tensor:

- One input tensor is the **phylogenetic-state tensor**. Loosely speaking,
  these tensors contain information associated with clades across rows and
  information about relevant branch lengths and states per taxon across columns.
  The phylogenetic-state tensors used by phyddle are based on the compact
  bijective ladderized vector (**CBLV**) format of Voznica et al. (2022) and
  the compact diversity-reordered vector (**CDV**) format of
  Lambert et al. (2022) that incorporates tip states (**CBLV+S** and **CDV+S**)
  using the technique described in Thompson et al. (2022).
- The second input is the **auxiliary data tensor**. This tensor contains
  summary statistics for the phylogeny and character data matrix and "known"
  parameters for the data generating process.
- The output tensor reports **labels** that are generally unknown data
  generating parameters to be estimated using the neural network. Depending on
  the estimation task, all or only some model parameters might be treated as
  labels for training and estimation.

For most purposes within phyddle, it is safe to think of a tensor as an
n-dimensional array, such as a 1-d vector or a 2-d matrix. The tensor encoding
ensures training examples share a standard shape (e.g. numbers of rows and
columns) that helps the neural network to detect predictable data patterns.
Learn more about the formats of phyddle tensors on the
:ref:`Tensor Formats <Tensor_Formats>` page.

During tensor-encoding, :ref:`Format` processes the tree, data matrix, and
model parameters for each replicate. This is done in parallel, when the setting
``use_parallel`` is set to ``True``. Simulated data are processed using CBLV+S
format if ``tree_encode`` is set to ``'serial'``. If ``tree_encode`` is set to
``'extant'`` then all non-extant taxa are pruned, saved as ``pruned.tre``, then
encoded using CDV+S. Standard CBLV+S and CDV+S formats are used when
``brlen_encode`` is ``'height_only'``, while additional branch length
information is added as rows when ``brlen_encode`` is set to
``'height_brlen'``. Each tree is then encoded into a phylogenetic-state tensor
with a maximum of ``tree_width`` sampled taxa. Trees that contain more taxa are
downsampled to ``tree_width`` taxa. The number of taxa in each original dataset
is recorded in the summary statistics, allowing the trained network to 
make estimates on trees that are larger or smaller than th exact ``tree_width``
size. 

The phylogenetic-state tensors and auxiliary data tensors are then created. If
``save_phyenc_csv`` is set, then individual csv files are saved for each
dataset, which is especially useful for formatting new empirical datasets into
an accepted phyddle format. The ``param_est`` setting identifies which "unknown"
parameters in the labels tensor you want to treat as downstream estimation
targets. The ``param_data`` setting identifies which of those parameters you
want to treat as "known" auxiliary data. Lastly, Format creates a test dataset
containing proportion ``test_prop`` of examples, and a second training dataset
that contains all remaining examples.

Formatted tensors are then saved to disk either in simple comma-separated
value format or in a compressed HDF5 format. For example, suppose we set
``fmt_dir`` to ``'./workspace/format/example'``, ``fmt_prefix`` to ``'out'``,
and ``tree_encode`` to ``'serial'``. If we set ``tensor_format == 'hdf5'``,
it produces:

.. code-block:: shell

    workspace/example/format/out.empirical.hdf5
    workspace/example/format/out.test.hdf5
    workspace/example/format/out.train.hdf5

or if ``tensor_format == 'csv'``:

.. code-block:: shell

    workspace/example/format/out.empirical.aux_data.csv
    workspace/example/format/out.empirical.labels.csv
    workspace/example/format/out.empirical.phy_data.csv
    workspace/example/format/out.test.aux_data.csv
    workspace/example/format/out.test.labels.csv
    workspace/example/format/out.test.phy_data.csv
    workspace/example/format/out.train.aux_data.csv
    workspace/example/format/out.train.labels.csv
    workspace/example/format/out.train.phy_data.csv

:ref:`Format` behaves the same way for simulated vs. empirical datasets,
except in two key ways. First, simulated datasets will be split into datasets
used to train the neural network and test its accuracy (in proportions defined
by ``test_prop``), whereas empirical datasets are left whole. Second, simulated
datasets will contain labels for all data-generating parameters, meaning both 
the "unknown" parameters that we want to estimate and the "known" parameters
that contribute to the data-generating process, but could be measured in the 
real world. For example, the birth rate might be an "unknown" parameter we want 
to estimate, while the missing proportion of species is a "known" parameter 
that we can provide the network if we know e.g. only 10% of described
plant species are in the dataset.

When searching for empirical and simulated datasets, :ref:`Format` uses
``emp_dir`` and ``sim_dir`` to locate the datasets. The ``emp_prefix`` and
``sim_prefix`` settings are used to identify the datasets. :ref:`Format`
assumes that empirical datasets follow the naming pattern of
``<prefix>.<rep_idx>.<ext>`` described for :ref:`Simulate`. For example,
setting ``emp_dir`` to ``'./workspace/dnds/empirical'`` and ``emp_prefix``
to ``'mammal_gene'`` will cause :ref:`Format` to search for files with
these names:

.. code-block:: shell

    workspace/dnds/empirical/mammal_gene.1.tre
    workspace/dnds/empirical/mammal_gene.1.dat.csv
    workspace/dnds/empirical/mammal_gene.1.labels.csv  # if using known params
    workspace/dnds/empirical/mammal_gene.2.tre
    workspace/dnds/empirical/mammal_gene.2.dat.csv
    workspace/dnds/empirical/mammal_gene.2.labels.csv  # if using known params
    ...

Using the ``--no_emp`` or ``--no_sim`` flags will instruct :ref:`Format` to
skip processing the empirical and simulated datasets, respectively. In
addition, :ref:`Format` will report that it is skipping the empirical and
simulated datasets if they do not exist.

Once complete, the formatted files can then be processed by the
:ref:`Train` step and :ref:`Estimate` steps.


.. _Train:

Train
-----

:ref:`Train` builds a neural network and trains it to make model-based
estimates using the simulated training example tensors compiled by the
:ref:`Format` step.

The :ref:`Train` step performs six main tasks:

1. Load the input training example tensor.
2. Shuffle the input tensor and split it into training, test, validation, and calibration subsets.
3. Build and configure the neural network
4. Use supervised learning to train neural network to make accurate estimates (predictions)
5. Record network training performance to file
6. Save the trained network to file

Each network is trained for one set of prediction tasks for the exact model
as specified for the :ref:`Simulate` step. Each network is trained to
expect a specific set of :ref:`Format` settings (see above).
Important :ref:`Format` settings include ``tree_width``, ``num_char``,
``num_states``, ``char_encode``, ``tree_encode``, ``brlen_encode``,
``param_est``, and ``param_known``. 

When the training dataset is read in, its examples are randomly shuffled by
replicate index. It then sets aside some examples for a validation dataset
(``prop_val``) and others for a calibration dataset (``prop_cal``). Note, 
the :ref:`Format` step will have previously set aside some proportion of test 
number of examples (``prop_test``) to measure final network accuracy
during the later :ref:`Estimate` step. The ``prop_val`` and ``prop_cal``
are themselves proportions of the ``1.0 - prop_test`` training example
proportion.

phyddle uses `PyTorch <https://pytorch.org/>` to build and train the network.
The phylogenetic-state tensor is processed by convolutional and pooling layers,
while the auxiliary data is processed by dense layers. All input layers are
concatenated then pushed into three branches terminating in output layers
to produce point estimates and upper and lower estimation intervals. Here
is a simplified schematic of the network architecture:

.. code-block::

    Simplified network architecture:
                              
                         Phylo. Data                  Aux. Data
                            Input                       Input
                              |                           |
                .-------------+-------------.            |
               v              v              v            v
        Conv1D-plain   Conv1D-dilate   Conv1D-stride    Dense
           + Pool         + Pool          + Pool          |
               .              |              |            |
                `-------------+--------------+-----------'
                              |
                            Concat
                           + Dense
                              |     
                  .-----------+-----------.
                 v            v            v  
               Lower        Point        Upper
              quantile     estimate     quantile



Parameter point estimates use a loss function (e.g. ``loss`` set to ``'mse'``;
Tensorflow-supported string or function) while lower/upper quantile estimates
use a pinball loss function (hard-coded).

Calibrated prediction intervals (CPIs) are estimated using the conformalized
quantile regression technique of Romano et al. (2019). CPIs target a
particular estimation interval, e.g. set ``cpi_coverage`` to ``0.95`` so
95% of test estimations are expected contain the true simulating value.
More accurate CPIs can be obtained using two-sided conformalized quantile
regression by setting ``cpi_asymmetric`` to ``True``, though this often
requires larger numbers of calibration examples, determined through
``prop_cal``. 

The network is trained iteratively for ``num_epoch`` training cycles using
batch stochastic gradient descent, with batch sizes given by ``trn_batch_size``.
Different optimizers can be used to update network weight and bias
parameters (e.g. ``optimizer == 'adam'``; Tensorflow-supported string
or function). Network performance is also evaluated against validation data
set aside with ``prop_val`` that are not used for minimizing the loss function.

Number of layers and numbers of nodes per layer can be adjusted using
configuration settings. For example, setting ``phy_channel_plain`` to
``[64,96,128]`` will construct three convolutional layers with 64, 96, and 128
output channels, respectively.

Training is automatically parallelized using CPUs and GPUs, dependent on
how Tensorflow was installed and system hardware. Output files are stored
in the directory assigned to ``trn_dir``.


.. _Estimate:

Estimate
--------

:ref:`Estimate` loads the simulated and empirical test datasets created by
:ref:`Format` stored in ``fmt_dir`` with prefix ``fmt_prefix``. For example,
if ``fmt_dir == './workspace/format/example'``, ``fmt_prefix == 'out'``,
and ``tensor_format == 'hdf5'`` then :ref:`Estimate` will process the
following files, if they exist: 

.. code-block:: shell

    workspace/example/format/out.test.hdf5
    workspace/example/format/out.test.empirical.hdf5

This step then loads a pretrained network for a given ``tree_width`` and
uses it to estimate parameter values and calibrated prediction intervals
(CPIs) for both the empirical dataset and the (simulated) test dataset.
Estimates are then stored as separate files into the ``est_dir`` directory.

Using the ``--no_emp`` or ``--no_sim`` flags will instruct :ref:`Estimate` to
skip processing the empirical and simulated datasets, respectively. In
addition, :ref:`Estimate` will report that it is skipping the empirical and
simulated datasets if they do not exist.


.. _Plot:

Plot
----

:ref:`Plot` collects all results from the :ref:`Format`, :ref:`Train`, and
:ref:`Estimate` steps to compile a set of useful figures, listed below. When 
results from :ref:`Estimate` are available, the step will integrate it into
other figures to contextualize where that input dataset and estimated
labels fall with respect to the training dataset.

Plots are stored within ``plot_dir``.
Colors for plot elements can be modified with ``plot_train_color``,
``plot_label_color``, ``plot_test_color``, ``plot_val_color``,
``plot_aux_color``, and ``plot_est_color`` using hex codes or common color
names supported by `Matplotlib <https://matplotlib.org/stable/gallery/color/named_colors.html>`__.

- ``summary.pdf`` contains all figures in a single plot
- ``summary.csv`` records important results in plain text format
- ``density_<dataset_name>_aux_data.pdf`` - densities of all values in the auxiliary dataset;
  red line for empirical dataset; run for training and empirical datasets
- ``density_<dataset_name>_label.pdf`` - densities of all values in the auxiliary dataset;
  red line for empirical dataset; run for training and empirical datasets
- ``pca_<dataset_name>_contour_aux_data.pdf`` - pairwise PCA of all values in the auxiliary dataset;
  red dot for empirical dataset; run for training and empirical datasets
- ``pca_<dataset_name>_contour_label.pdf`` - pairwise PCA of all values in the auxiliary dataset;
  red dot for empirical dataset; run for training and empirical datasets
- ``train_history.pdf`` - loss performance across epochs for test/validation
  datasets for entire network
- ``estimate_<dataset_name>_<label_name>.pdf`` - point estimates and calibrated
  estimation intervals for test or training datasets
- ``empirical_estimate_<N>.pdf`` - simple plot of point estimates and
  calibrated prediction intervals for each empirical dataset
- ``network_architecture.pdf`` - visualization of Tensorflow architecture


.. _Example:

Example run
-----------

The output of phyddle pipeline analysis will resemble this:

.. code-block::

    ┏━━━━━━━━━━━━━━━━━━━━━━┓
    ┃   phyddle   v0.1.1   ┃
    ┣━━━━━━━━━━━━━━━━━━━━━━┫
    ┃                      ┃
    ┗━┳━▪ Simulating... ▪━━┛
      ┃
      ┗━━━▪ output: ./workspace/bisse_r/simulate
    
    ▪ Start time of 09:48:30
    ▪ Simulating raw data
    Simulating: 100%|███████████████████████| 100/100 [00:20<00:00,  4.80it/s]
    ▪ End time of 09:48:52 (+00:00:22)
    ... done!
    ┃                      ┃
    ┗━┳━▪ Formatting... ▪━━┛
      ┃
      ┣━━━▪ input:  ./workspace/bisse_r/simulate
      ┃             ./workspace/bisse_r/empirical
      ┗━━━▪ output: ./workspace/bisse_r/format
    
    ▪ Start time of 09:48:52
    ▪ Collecting simulated data
    ▪ Encoding simulated data as tensors
    Encoding: 100%|█████████████████████| 15030/15030 [03:14<00:00, 77.24it/s]
    Encoding found 15030 of 15030 valid examples.
    ▪ Splitting into train and test datasets
    ▪ Combining and writing simulated data as tensors
    Making train hdf5 dataset: 14279 examples for tree width = 500
    Combining: 100%|██████████████████| 14279/14279 [00:12<00:00, 1117.76it/s]
    Making test hdf5 dataset: 751 examples for tree width = 500
    Combining: 100%|██████████████████████| 751/751 [00:00<00:00, 1323.66it/s]
    ▪ Collecting empirical data
    ▪ Encoding empirical data as tensors
    Encoding: 100%|███████████████████████████| 10/10 [00:09<00:00,  1.01it/s]
    Encoding found 10 of 10 valid examples.
    ▪ Combining and writing empirical data as tensors
    Making empirical hdf5 dataset: 10 examples for tree width = 500
    Combining: 100%|████████████████████████| 10/10 [00:00<00:00, 1606.71it/s]
    ▪ End time of 09:52:38 (+00:03:46)
    ... done!
    ┃                      ┃
    ┗━┳━▪ Training...   ▪━━┛
      ┃
      ┣━━━▪ input:  ./workspace/bisse_r/format
      ┗━━━▪ output: ./workspace/bisse_r/train
    
    ▪ Start time of 09:52:40
    ▪ Loading input
    ▪ Building network
    ▪ Training network
    Training epoch 1 of 10: 100%|█████████████| 21/21 [00:27<00:00,  1.33s/it]
        Train        --   loss: 0.9831
        Validation   --   loss: 0.6960
    
    Training epoch 2 of 10: 100%|█████████████| 21/21 [00:31<00:00,  1.52s/it]
        Train        --   loss: 0.5950  abs: -0.3881  rel: -39.50%
        Validation   --   loss: 0.5356  abs: -0.1604  rel: -23.00%
    
    Training epoch 3 of 10: 100%|█████████████| 21/21 [00:33<00:00,  1.61s/it]
        Train        --   loss: 0.4686  abs: -0.1264  rel: -21.20%
        Validation   --   loss: 0.4611  abs: -0.0745  rel: -13.90%
    
    Training epoch 4 of 10: 100%|█████████████| 21/21 [00:32<00:00,  1.53s/it]
        Train        --   loss: 0.4031  abs: -0.0655  rel: -14.00%
        Validation   --   loss: 0.4136  abs: -0.0475  rel: -10.30%
    
    Training epoch 5 of 10: 100%|█████████████| 21/21 [00:31<00:00,  1.49s/it]
        Train        --   loss: 0.3696  abs: -0.0335  rel: -8.30%
        Validation   --   loss: 0.3914  abs: -0.0222  rel: -5.40%
    
    Training epoch 6 of 10: 100%|█████████████| 21/21 [00:31<00:00,  1.52s/it]
        Train        --   loss: 0.3357  abs: -0.0339  rel: -9.20%
        Validation   --   loss: 0.3509  abs: -0.0405  rel: -10.30%
    
    Training epoch 7 of 10: 100%|█████████████| 21/21 [00:31<00:00,  1.50s/it]
        Train        --   loss: 0.3217  abs: -0.0140  rel: -4.20%
        Validation   --   loss: 0.3359  abs: -0.0150  rel: -4.30%
    
    Training epoch 8 of 10: 100%|█████████████| 21/21 [00:31<00:00,  1.52s/it]
        Train        --   loss: 0.3030  abs: -0.0187  rel: -5.80%
        Validation   --   loss: 0.3291  abs: -0.0068  rel: -2.00%
    
    Training epoch 9 of 10: 100%|█████████████| 21/21 [00:33<00:00,  1.57s/it]
        Train        --   loss: 0.2963  abs: -0.0067  rel: -2.20%
        Validation   --   loss: 0.3149  abs: -0.0142  rel: -4.30%
    
    Training epoch 10 of 10: 100%|████████████| 21/21 [00:33<00:00,  1.59s/it]
        Train        --   loss: 0.2835  abs: -0.0128  rel: -4.30%
        Validation   --   loss: 0.3179  abs: +0.0030  rel: +1.00%
    
    ▪ Processing results
    ▪ Saving results
    ▪ End time of 09:58:14 (+00:05:34)
    ▪ ... done!
    ┃                      ┃
    ┗━┳━▪ Estimating... ▪━━┛
      ┃
      ┣━━━▪ input:  ./workspace/bisse_r/format
      ┃             ./workspace/bisse_r/train
      ┗━━━▪ output: ./workspace/bisse_r/estimate
    
    ▪ Start time of 09:58:14
    ▪ Loading simulated test input
    ▪ Making simulated test estimates
    ▪ Loading empirical input
    ▪ Making empirical estimates
    ▪ End time of 09:58:15 (+00:00:01)
    ... done!
    ┃                      ┃
    ┗━┳━▪ Plotting...   ▪━━┛
      ┃
      ┣━━━▪ input:  ./workspace/bisse_r/format
      ┃             ./workspace/bisse_r/train
      ┃             ./workspace/bisse_r/estimate
      ┗━━━▪ output: ./workspace/bisse_r/plot
    
    ▪ Start time of 10:01:09
    ▪ Loading input
    ▪ Generating individual plots
    ▪ Combining plots
    ▪ Making csv report
    ▪ End time of 10:01:40 (+00:00:31)
    ... done!
