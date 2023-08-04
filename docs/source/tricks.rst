.. _Tricks:

Tricks
======

Here are a few tricks for using phyddle on from a Unix-based terminal. These commands assume a standard phyddle workspace directory structure.

**Add new traing examples.**

To add new examples to your training set

.. code-block:: shell

    # simulate new training examples, stored in
    # workspace/raw_data/my_project
    ./run_phyddle.py -s sim -c config --start_idx 500 --end_idx 15000

    # encode all raw_data examples as tensors,
    # stored in workspace/tensor_data/my_project
    ./run_phyddle.py -s fmt -c config --start_idx 0 --end_idx 15000


**Print current number of training examples**

.. code-block::

   # (1) list all data files; (2) count lines in list
   ls workspace/raw_data/example/*.dat.nex | wc -l


**Print index of last simulated training example**

.. code-block::
    
   # (1) list all data files; (2) turn into list of indices;
   # (3) sort numerically; (4) get last (largest) value
   ls workspace/raw_data/example/*.dat.nex | cut -f2 -d. | sort -n | tail -n1

