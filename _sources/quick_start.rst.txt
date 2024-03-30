.. _Quick_Start:

Quick start
===========

Visit the :ref:`Installation` page to install phyddle. 

Download a copy of the phyddle repository to your workstation either as `.zip
file <https://github.com/mlandis/phyddle/archive/refs/heads/main.zip>`_ or by
cloning the `git repository <https://github.com/mlandis/phyddle.git>`_.

To run a phyddle analysis enter your new directory, e.g.:

.. code-block:: shell

  # enter phyddle directory  
  cd ~/projects/phyddle

Several example phyddle analyses are stored in the ``./workspace`` subdirectory.
Each project directory contains a :ref:`Configuration` file, a
:ref:`Simulation <Simulate>` script, and will contain directories for output
resulting from an analysis. For example, you can run an analysis for the
``bisse_r`` project using the commands:

.. code-block:: shell

  # enter project directory
  cd ./workspace/bisse_r
  
  # run phyddle with project config
  phyddle -c config.py

This will run an analysis using the default settings for all :ref:`Pipeline`
steps. As the first step, :ref:`Simulate` will execute the command string
stored under ``sim_command``  in the config file. For ``./workspace/bisse_r/config.py``
this command is 
	
.. code-block::

  # phyddle runs this R script to simulate
  Rscript ./sim_bisse.R
    
.. note::

	You must be able to run the simulation script from command line without
	administrative privileges. This means must install the required software
	for each project. The above script requires R and the packages ``ape``
	and ``castor`` to run.

Eventually, you will want to write your own config file. There are two easy
ways to create your own config file. One option is to copy and modify an
existing script. Another option is to create a new blank config with
``phyddle --make_cfg`` and then modify the new file

.. code-block:: shell

  # create a default config file
  phyddle --make_cfg
  
  # rename the config file
  mv config_default.py new_config.py
  
  # update the config file
  edit new_config.py

Now, let's assume ``new_config.py`` specifies a phyddle analysis with 1000 simulated 
training examples, using R for simulation. Run phyddle against the new config
with the following command:

.. code-block:: shell

  # run phyddle with new config file
  phyddle -c ./new_config.py

Provide phyddle with command-line options to customize how each pipeline step
is executed. Visit :ref:`Pipeline` and :ref:`Workspace` to learn more about
managing phyddle analyses.

In practice, you'll want to generate a larger training dataset with anywhere
from 10k to 1M examples, depending on the model. To add new examples to your
training set, for example:

.. code-block:: shell

  # enter project directory
  cd workspace/my_project
  
  # [S]imulate new training examples, stored in
  # ./simulate subdirectory
  phyddle -s S -c config.py --sim_more 14000

  # [F]ormat all raw_data examples as tensors,
  # stored in ./format subdirectory
  phyddle -s F -c config.py

  # [T]rain network with tensor_data, but override batch size,
  # stored in ./train subdirectory
  phyddle -s T -c config.py --trn_batch_size 256

  # [E]stimate parameters for biological dataset, with results
  # stored in ./estimate subdirectory; and then [P]lot
  # figures, storing them in ./plot subdirectory
  phyddle -s EP -c config.py


Visit :ref:`Configuration` to learn more about currently supported phyddle
settings. View supported command-line options with:

.. code-block:: shell

  # see help for phyddle
  phyddle --help

