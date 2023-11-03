"""
test_simulate
=============
Tests classes and methods for the Simulate step.

Authors:   Michael Landis and Ammon Thompson
Copyright: (c) 2022-2023, Michael Landis and Ammon Thompson
License:   MIT
"""

import phyddle.utilities as util
import phyddle.simulate as sim
import os

#-----------------------------------------------------------------------------#

def test_sim():

    # command line arguments
    cmd_args = ['--step', 'S',
                '--proj', 'test',
                '--sim_dir', './tests/workspace/simulate',
                '--sim_command', 'Rscript scripts/sim/R/sim_one.R']

    # build arguments
    my_args = util.load_config('scripts/config_R.py', arg_overwrite=True, args=cmd_args)

    # load simulator
    my_sim = sim.load(my_args)

    # run simulator
    my_sim.run()

    # verify output for sim.0.dat.nex
    # ...

    return

#-----------------------------------------------------------------------------#
