.. _Quick_Start:

Quick start
===========

Visit the :ref:`Installation` page to install phyddle. 

To run a phyddle analysis enter the ``scripts`` directory:

.. code-block:: shell

	cd ~/projects/phyddle/scripts

Then create and run a pipeline under the settings you've specified in ``config.py``:

.. code-block:: shell

	./run_phyddle.py -c config

This will run a phyddle analysis for a simple 3-region GeoSSE model with just 500 training examples. In practice, you'll want to generate a larger training dataset with anywhere from 10k to 1M examples, depending on the model.

Provide phyddle with command-line options to customize how each pipeline step is executed. Visit :ref:`Pipeline` and :ref:`Workspace` to learn more about managing phyddle analyses. To add new examples to your training set, for example:

.. code-block:: shell

    # [S]imulate new training examples, stored in
    # workspace/simulate/my_project
    ./run_phyddle.py -s S -c config --start_idx 500 --end_idx 15000

    # [F]ormat all raw_data examples as tensors,
    # stored in workspace/format/my_project
    ./run_phyddle.py -s F -c config --start_idx 0 --end_idx 15000

    # [T]rain network with tensor_data, but override batch size,
    # stored in workspace/train/my_project
    ./run_phyddle.py -s T -c config --batch_size 256

    # [E]stimate parameters for biological dataset, with results
    # stored in workspace/estimate/my_project; and then [P]lot
    # figures, storing them in workspace/plot/my_project
    ./run_phyddle.py -s EP -c config


Visit :ref:`Configuration` to learn more about currently supported phyddle settings. View supported command-line options with:

.. code-block:: shell

	./run_phyddle.py --help

