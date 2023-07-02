Usage
=====

.. _installation:

Installation
------------

To use phyddle, first install it using pip:

.. code-block:: console

   (.venv) $ pip install phyddle

Creating recipes
----------------

To retrieve a list of random ingredients,
you can use the ``phyddle.Utilities.write_to_file()`` function:

.. autofunction:: phyddle.Utilities.write_to_file

The ``kind`` parameter should be either ``"meat"``, ``"fish"``,
or ``"veggies"``. Otherwise, :py:func:`phyddle.Utilities.write_to_file`
will raise an exception.

For example:

>>> import phyddle
>>> phyddle.Utilities.write_to_file(fn)

