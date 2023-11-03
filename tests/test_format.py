"""
test_format
=============
Tests classes and methods for the Format step.

Authors:   Michael Landis and Ammon Thompson
Copyright: (c) 2022-2023, Michael Landis and Ammon Thompson
License:   MIT
"""

import phyddle.utilities as util
import phyddle.format as fmt

#-----------------------------------------------------------------------------#

def test_fmt():

    # command line arguments
    cmd_args = ['--step', 'F',
                '--proj', 'test,S:valid',
                '--sim_dir', 'tests/workspace/simulate',
                '--fmt_dir', 'tests/workspace/format',
                '--use_parallel', 'F']

    # phyddle arguments
    my_args = util.load_config('scripts/config_R.py', arg_overwrite=True, args=cmd_args)

    # load simulator
    my_fmt = fmt.load(my_args)

    # run simulator
    my_fmt.run()

    # verify output for sim.0.dat.nex
    # ...

    return

#-----------------------------------------------------------------------------#
