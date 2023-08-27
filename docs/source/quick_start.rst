.. _Quick_Start:

Quick start
===========

Visit the :ref:`Installation` page to install phyddle. 

To run a phyddle analysis enter the ``scripts`` directory:

.. code-block:: shell

	cd ~/projects/phyddle/scripts

Then create and run a pipeline under the settings you've specified in
``config.py``:

.. code-block:: shell

	phyddle -c config.py

There are two easy ways to create your own config file. One option is to 
download and modify a script from the `GitHub repository
<https://github.com/mlandis/phyddle/tree/main/scripts>`_ for phyddle. Another
option is to create a new blank config with ``phyddle --make_cfg config.py``
and then modify it.

.. code-block:: shell

   phyddle --make_cfg -c config.phy 
   vim config.py

Let's assume ``config.py`` specifies a phyddle analysis with 500 simulated 
training examples, using R for simulation. In practice, you'll want to generate
a larger training dataset with anywhere from 10k to 1M examples, depending on
the model.

Provide phyddle with command-line options to customize how each pipeline step
is executed. Visit :ref:`Pipeline` and :ref:`Workspace` to learn more about
managing phyddle analyses. To add new examples to your training set, for
example:

.. code-block:: shell

    # [S]imulate new training examples, stored in
    # workspace/simulate/my_project
    phyddle -s S -c config.py --sim_more 14500

    # [F]ormat all raw_data examples as tensors,
    # stored in workspace/format/my_project
    phyddle -s F -c config.py

    # [T]rain network with tensor_data, but override batch size,
    # stored in workspace/train/my_project
    phyddle -s T -c config.py --trn_batch_size 256

    # [E]stimate parameters for biological dataset, with results
    # stored in workspace/estimate/my_project; and then [P]lot
    # figures, storing them in workspace/plot/my_project
    phyddle -s EP -c config.py


Visit :ref:`Configuration` to learn more about currently supported phyddle
settings. View supported command-line options with:

.. code-block:: shell

	phyddle --help

