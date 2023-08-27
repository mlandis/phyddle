.. _Tricks:

Tricks
======

Here are a few tricks for using phyddle using a Unix-based terminal. These
commands assume a standard phyddle workspace directory structure.


**Make a new config file**

.. code-block:: shell

  # Create and modify new config file
  phyddle --make_cfg -c my_new_config.py
  vim my_new_config.py


**Run a pipeline with modified command-line settings**

.. code-block:: shell
  
  # Run full pipeline while changing calibration and validation proportions 
  phyddle -c config.py -p my_project --cal_prop 0.10 --val_prop 0.10


**Re-run part of the pipeline with modified command-line settings**

.. code-block:: shell

  # Re-run pipeline Train, Estimate, and Plot steps with new training settings
  phyddle -c config.py -p my_project -s TEP --num_epoch 10 --trn_batch_size 64


**Redirect input/output across pipeline steps**

.. code-block:: shell
  
  # Run full pipeline 
  phyddle -c config.py -p my_project
  
  # Re-run Train, Estimate, Plot steps with new settings
  phyddle -c config.py -p my_project,T:new_results,E:new_results,P:new_results

  # ... or this way
  phyddle -c config.py -p new_results,S:my_project,F:my_project


**Simulate new traing examples**

.. code-block:: shell

  # Simulate training examples 0 to 999, storing results 
  # workspace/simulate/my_project
  phyddle -s S -c config.py -p my_project --start_idx 0 --end_idx 1000

  # Simulate 4000 more training examples, 0 to 4999
  phyddle -s S -c config.py -p my_project --sim_more 4000

  # Perform remaining Format, Train, Estimate, Plot steps
  phyddle -s FTEP -c config.py

  # ...or, to Simulate more and re-run all steps
  phyddle -c config.py -p my_project --sim_more 4000

