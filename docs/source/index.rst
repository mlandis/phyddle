phyddle
=======

.. note::

	**This beta version of phyddle is still under development.** Feedback is appreciated! michael.landis@wustl.edu

**phyddle** is software for fiddling around with phylogenetic models and deep learning. As a command-line tool and/or as a Python package, phyddle is designed for simulation-based supervised learning to train neural networks for phylogenetic model estimation tasks.

A standard phyddle analysis performs the following tasks for you:

.. image:: images/phyddle_pipeline.png
	:scale: 13%
	:align: right

* **Pipeline configuration** applies analysis settings provided through a config file and/or command line arguments.
* **Simulate** simulates a large training dataset under the model to be *Formatted* (parallelized, partly compressed).
* **Format** encodes the raw simulated data into tensor format for *Training*.
* **Train** shuffles and splits training data, builds a network, then trains and saves the network with the data for *Estimation*.
* **Estimate** produces model estimates for a new dataset with the trained network.
* **Plot** generates figures that summarize the training data (*Format*), the network and its training (*Train*), and any estimates for new datasets (*Estimate*).

In addition, phyddle is distributed with example scripts to simulate phylogenetic training datasets using `RevBayes <http://revbayes.com>`_, `R <https://www.r-project.org/>`_, and `MASTER <https://tgvaughan.github.io/MASTER/>`_.

To learn how to use phyddle, we recommend exploring the topics from top-to-bottom as listed on the left-hand side of this page.
Visit the :ref:`Quick_Start` and :ref:`Installation` pages to get started.

.. toctree::
    :hidden:

    quick_start
    tutorial
    installation
    configuration
    pipeline
    workspace
    formats
    tricks
    glossary
    updates
    about
