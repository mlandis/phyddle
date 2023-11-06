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
import numpy as np

#-----------------------------------------------------------------------------#

def test_sim():
    
    # set seed
    util.set_seed(0)
    
    # filesystem
    base_dir = './tests/workspace/simulate'
    test_dir = base_dir + '/test'
    valid_dir = base_dir + '/valid'

    # command line arguments
    cmd_args = ['--step', 'S',
                '--proj', 'test',
                '--sim_dir', base_dir,
                '--sim_command', 'Rscript scripts/sim/R/sim_one.R',
                '--end_idx', '100',
                '--use_parallel', 'F']

    # build arguments
    my_args = util.load_config('scripts/config_R.py', arg_overwrite=True, args=cmd_args)

    # load simulator
    my_sim = sim.load(my_args)

    # run simulator
    my_sim.run()

    # confirm test and valid tree match
    phy_test = util.read_tree(test_dir + '/sim.0.tre')
    phy_valid = util.read_tree(valid_dir + '/sim.0.tre')
    assert( phy_test.length() == phy_valid.length() )
    assert( str(phy_test) == str(phy_valid) )

    # confirm test and valid data match
    dat_valid = util.convert_csv_to_array(valid_dir + '/sim.0.dat.nex', 'integer', 2)
    dat_test = util.convert_csv_to_array(test_dir + '/sim.0.dat.nex', 'integer', 2)
    assert(np.array_equal(dat_valid, dat_test))

    # confirm test and valid params match
    param_valid = util.convert_csv_to_array(valid_dir + '/sim.0.param_row.csv', 'numeric', 2)
    param_test = util.convert_csv_to_array(test_dir + '/sim.0.param_row.csv', 'numeric', 2)
    assert(np.array_equal(param_valid, param_test))

    # success
    return

#-----------------------------------------------------------------------------#
