.. _Quick_Start:

Quick start
===========

Visit the :ref:`Installation` page to install phyddle. 

Download a copy of the phyddle repository to your workstation either as `.zip
file <https://github.com/mlandis/phyddle/archive/refs/heads/main.zip>`_ or by
cloning the `git repository <https://github.com/mlandis/phyddle.git>`_.

To run a phyddle analysis enter your new directory, e.g.:

.. code-block:: shell

	cd ~/projects/phyddle

Several example phyddle analysis :ref:`Configuration` files are stored in the
``./configs`` subdirectory. You can run an analysis against a different
configuration files with the command:

.. code-block:: shell

	phyddle -c configs/config_analysis_name.py

There are two easy ways to create your own config file. One option is to 
download and modify a script from the `GitHub repository
<https://github.com/mlandis/phyddle/tree/main/scripts>`_ for phyddle. Another
option is to create a new blank config with ``phyddle --make_cfg``
and then modify it.

.. code-block:: shell

  phyddle --make_cfg
  mv config_default.py config_my_model.py
  edit config_my_model.py

Let's assume ``config.py`` specifies a phyddle analysis with 1000 simulated 
training examples, using R for simulation. 

.. code-block:: shell

  phyddle -c configs/config.py

Provide phyddle with command-line options to customize how each pipeline step
is executed. Visit :ref:`Pipeline` and :ref:`Workspace` to learn more about
managing phyddle analyses.

In practice, you'll want to generate a larger training dataset with anywhere
from 10k to 1M examples, depending on the model. To add new examples to your
training set, for example:

.. code-block:: shell

  # [S]imulate new training examples, stored in
  # workspace/my_project/simulate
  phyddle -s S -c configs/config.py --sim_more 14000

  # [F]ormat all raw_data examples as tensors,
  # stored in workspace/my_project/format
  phyddle -s F -c configs/config.py

  # [T]rain network with tensor_data, but override batch size,
  # stored in workspace/my_project/train
  phyddle -s T -c configs/config.py --trn_batch_size 256

  # [E]stimate parameters for biological dataset, with results
  # stored in workspace/my_project/estimate; and then [P]lot
  # figures, storing them in workspace/my_project/plot
  phyddle -s EP -c configs/config.py


Visit :ref:`Configuration` to learn more about currently supported phyddle
settings. View supported command-line options with:

.. code-block:: shell

  phyddle --help

