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
import numpy as np
import h5py

#-----------------------------------------------------------------------------#

def test_fmt():

    # set seed
    util.set_seed(0)

    # filesystem
    base_dir = './tests/workspace'
    sim_dir = base_dir + '/simulate'
    fmt_dir = base_dir + '/format'
    test_dir = fmt_dir + '/test'
    valid_dir = fmt_dir + '/valid'

    # command line arguments
    cmd_args = ['--step', 'F',
                '--proj', 'test,S:valid',
                '--sim_dir', sim_dir,
                '--fmt_dir', fmt_dir,
                '--prop_test','0.10',
                '--prop_val','0.10',
                '--use_parallel', 'F']

    # phyddle arguments
    my_args = util.load_config('scripts/config_R.py', arg_overwrite=True, args=cmd_args)

    # load simulator
    my_fmt = fmt.load(my_args)

    # run simulator
    my_fmt.run()

    # confirm test and valid tree match
    dat_test  = h5py.File(test_dir + '/test.nt500.hdf5', 'r')
    dat_valid = h5py.File(valid_dir + '/test.nt500.hdf5', 'r')

    # collect test format output arrays
    aux_data_names_test  = dat_test['aux_data_names'][:]
    label_names_test     = dat_test['label_names'][:]
    phy_data_test        = dat_test['phy_data'][:]
    aux_data_test        = dat_test['aux_data'][:]
    labels_test          = dat_test['labels'][:]

    # collect valid format output arrays
    aux_data_names_valid = dat_valid['aux_data_names'][:]
    label_names_valid    = dat_valid['label_names'][:]
    phy_data_valid       = dat_valid['phy_data'][:]
    aux_data_valid       = dat_valid['aux_data'][:]
    labels_valid         = dat_valid['labels'][:]

    # close files
    dat_test.close()
    dat_valid.close()

    # verify all test and valid format output match
    assert(np.array_equal(aux_data_names_test, aux_data_names_valid))
    assert(np.array_equal(label_names_test, label_names_valid))
    assert(np.array_equal(phy_data_test, phy_data_valid))
    assert(np.array_equal(aux_data_test, aux_data_valid))
    assert(np.array_equal(labels_test, labels_valid))

    # success
    return

#-----------------------------------------------------------------------------#
