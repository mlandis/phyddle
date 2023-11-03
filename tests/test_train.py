"""
test_train
=============
Tests classes and methods for the Train step.

Authors:   Michael Landis and Ammon Thompson
Copyright: (c) 2022-2023, Michael Landis and Ammon Thompson
License:   MIT
"""

import phyddle.utilities as util
import phyddle.train as trn
import os

#-----------------------------------------------------------------------------#

def test_trn():

    # command line arguments
    cmd_args = ['--step', 'T',
                '--proj', 'test,F:valid',
                '--fmt_dir', 'tests/workspace/format',
                '--trn_dir', 'tests/workspace/train' ]

    # phyddle arguments
    my_args = util.load_config('scripts/config_R.py', arg_overwrite=True, args=cmd_args)

    # load trainer
    my_trn = trn.load(my_args)

    # run simulator
    my_trn.run()

    # verify output for sim.0.dat.nex
    # ...

    return

#-----------------------------------------------------------------------------#
