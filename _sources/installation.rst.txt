.. _Installation:

Installation
============

This page describes how to download and install phyddle and its dependencies.

Conda package
-------------

phyddle is available as a `conda package <https://anaconda.org/landismj/phyddle>`_,
and can be installed with the following commands:

.. code-block:: shell

    conda create -n phyddle -c bioconda -c landismj phyddle
    conda install -c bioconda -c landismj phyddle
    # ... install ...
    conda activate phyddle
    phyddle

Python package
--------------

phyddle is also available as a `PyPI package <https://pypi.org/project/phyddle/>`_ 
and can be installed using ``pip``.

.. code-block:: shell

  python3 -m pip install phyddle
  # ... install ...
  phyddle


GitHub repository
-----------------

To download the phyddle source code to your computer, you can either clone 
the `repository <https://github.com/mlandis/phyddle>`_

.. code-block:: shell

	git clone git@github.com:mlandis/phyddle.git       # using SSH
	git clone https://github.com/mlandis/phyddle.git   # using HTTPS
	gh repo clone mlandis/phyddle                      # using GitHub CLI

or you can `download <https://github.com/mlandis/phyddle/archive/refs/heads/main.zip>`_ 
and unzip the current version of the main branch

.. code-block:: shell

	wget https://github.com/mlandis/phyddle/archive/refs/heads/main.zip
	unzip main.zip

Once cloned, you can build phyddle into a local Python package

.. code-block:: shell

  cd ~/projects/phyddle
  pip install .
  # ... install ...
  phyddle


System configuration
--------------------

phyddle is regularly tested on Mac OS X 14.2.1 (Intel CPU) and Python
3.11.7 (installed with homebrew). phyddle is also intermittently tested 
on a 64-core Ubuntu LTS 22.04 server using Python 3.xx.xx (aptitude) and 
similar package versions. 

To install required Python packages

.. code-block:: shell

    python3 -m ensurepip --upgrade
    python3 -m pip install --upgrade pip
    python3 -m pip install dendropy graphviz h5py keras matplotlib numpy pandas Pillow pydot_ng pypdf scikit-learn scipy torch torchview tqdm

Last tested Python package versions are

.. code-block:: shell

  dendropy 4.5.2
  graphviz 0.20.1
  h5py 3.8.0
  keras 2.12.0
  matplotlib 3.7.1
  numpy 1.23.5
  pandas 2.0.0 
  Pillow 10.1.0
  pydot_ng 2.0.2
  pypdf 3.12.0
  scikit-learn 1.2.2
  scipy 1.11.4
  torch 2.0.0
  torchview 0.2.6
  tqdm 4.65.0

