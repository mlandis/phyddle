Quick start
===========


To run a phyddle analysis enter the ``scripts`` directory:

.. code-block:: shell

	cd ~/projects/phyddle/scripts

Then create and run a pipeline under the settings you've specified in ``config.py``:

.. code-block:: shell

	./run_phyddle.py -c config

This will run a phyddle analysis for a simple 3-region GeoSSE model with just 500 training examples. In practice, you'll want to generate a larger training dataset with anywhere from 10k to 1M examples, depending on the model.

To add new examples to your training set

.. code-block:: shell

    # simulate new training examples, stored in
    # workspace/raw_data/my_project
    ./run_phyddle.py -s sim -c config --start_idx 500 --end_idx 15000

    # encode all raw_data examples as tensors,
    # stored in workspace/tensor_data/my_project
    ./run_phyddle.py -s fmt -c config --start_idx 0 --end_idx 15000

    # train network with tensor_data, but override batch size,
    # stored in workspace/network/my_project
    ./run_phyddle.py -s lrn -c config --batch_size 256

    # make prediction with example dataset, results stored in
    # workspace/predict/my_project
    ./run_phyddle.py -s prd -c config

    # generate figures, stored in workspace/plot/my_project
    ./run_phyddle.py -s plt -c config

Pipeline options are applied to all pipeline stages. See the full list of currently supported options with

.. code-block:: shell

	./run_phyddle.py --help

The files and directories for a small, example project can are found within the `workspace` subdirectories, e.g. `workspace/raw_data/example` or `workspace/plot/example`.
