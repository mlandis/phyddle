.. _installation:

Installation
============

This page describes how to download and install phyddle and its dependencies.


Download phyddle
----------------

To install phyddle on your computer, you can either clone the `repository <https://github.com/mlandis/phyddle>`_

.. code-block::

	$ git clone git@github.com:mlandis/phyddle.git       # SSH
	$ git clone https://github.com/mlandis/phyddle.git   # HTTPS
	$ gh repo clone mlandis/phyddle                      # GitHub CLI

or you can `download <https://github.com/mlandis/phyddle/archive/refs/heads/main.zip>`_ and unzip the current version of the main branch

.. code-block::

	$ wget https://github.com/mlandis/phyddle/archive/refs/heads/main.zip
	$ unzip main.zip


Build phyddle package
---------------------

Once cloned, you can build phyddle into a local Python package

.. code-block::

	$ cd ~/projects/phyddle   # enter local phyddle repo
	$ pip install .           # build local package


Alternatively, the beta version of phyddle can be installed through `TestPyPI <https://test.pypi.org/project/phyddle/>`_

.. code-block::

	$ python3 -m pip install --index-url https://test.pypi.org/simple/ --no-deps phyddle

When phyddle is public, it will be installed using

.. code-block::

	$ python3 -m pip install phyddle


Python requirements
-------------------

phyddle was last tested using with Mac OS X 11.6.4 (Intel CPU) using Python 3.11.3 (installed with homebrew) with the following versions of the required third-party package dependencies (installed with pip):

.. code-block::

	PIL 9.5.0
	argparse 1.1
	h5py 3.8.0
	joblib 1.2.0
	keras 2.12.0
	matplotlib 3.7.1
	numpy 1.23.5
	pandas 2.0.0
	pydot_ng 2.0.0
	pypdf 3.12.0
	scipy 1.10.1
	sklearn 1.2.2
	tensorflow 2.12.0
	tqdm 4.65.0

To install these packages:

.. code-block::
    python3 -m ensurepip --upgrade
    python3 -m pip install --upgrade pip
    python3 -m pip install argparse h5py joblib keras matplotlib numpy pandas Pillow pydot_ng pypdf scipy scikit-learn tensorflow tqdm

phyddle is also used with a 64-core Ubuntu LTS 22.04 server using Python 3.xx.xx (aptitude) and similar package versions. phyddle has yet not been tested using conda, Windows, M1 Macs, various GPUs, etc.


Simulator requirements
----------------------

phyddle currently relies on the BEAST plugin MASTER for simulation. The operating system must be able to call BEAST from anywhere in the filesystem through the ``beast`` command. This can be done by adding the BEAST executable to be covered by the ``$PATH`` shell variable. Creating a symbolic link (shortcut) to the BEAST binary ``beast`` with ``ln -s`` in ``~/.local/bin`` is one an easy way to make ``beast`` globally accessible on Mac OS X.

These instructions for Mac OS X install BEAST v2.7.3+ and MASTER v7.0.0+, then make the program accessible as a command through the terminal.

.. code-block::

    $ # install BEAST
      [ ... installation text ... ]
    $ cd /Applications/BEAST\ 2.7.3/bin              # enter directory for BEAST programs (binaries)
    $ ./packagemanager -add MASTER                   # install MASTER plugin for BEAST
    $ mkdir -p ~/.local/bin                          # directory is in PATH variable by default on Mac OS X
    $ cd ~/.local/bin                                # enter new directory
    $ ln -s /Applications/BEAST\ 2.7.3/bin/beast .   # create shortcut (symbolic link) to `beast` command
	$ ls -lart ~/.local/bin/beast                    # verify symbolic link points (->) to correct command
	lrwxr-xr-x  1 mlandis  staff  35 Feb 14 10:32 /Users/mlandis/.local/bin/beast -> /Applications/BEAST 2.7.3/bin/beast
    $ cd ~                                           # return to home directory
    $ source ~/.zshrc                                # refresh profile (and PATH variable); maybe ~/.zprofile??
	$ which beast                                    # verify beast can be called anywhere (e.g. from HOME)
	/Users/mlandis/.local/bin/beast
	$ beast -version
	BEAST v2.7.3
	---
	BEAST.base v2.7.3
	MASTER v7.0.0
	BEAST.app v2.7.3
	---
	Java version 17.0.5

