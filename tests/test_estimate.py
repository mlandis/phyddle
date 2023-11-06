"""
test_estimate
=============
Tests classes and methods for the Estimate step.

Authors:   Michael Landis and Ammon Thompson
Copyright: (c) 2022-2023, Michael Landis and Ammon Thompson
License:   MIT
"""

import phyddle.utilities as util
import phyddle.format as fmt
import phyddle.estimate as est
import pandas as pd
import numpy as np

#-----------------------------------------------------------------------------#

def test_est():

    # set seed
    util.set_seed(0)

    # filesystem
    base_dir = './tests/workspace'
    sim_dir = base_dir + '/simulate'
    fmt_dir = base_dir + '/format'
    trn_dir = base_dir + '/train'
    est_dir = base_dir + '/estimate'
    test_dir = est_dir + '/test'
    valid_dir = est_dir + '/valid'
    est_prefix = 'new.0'

	# command line arguments
    cmd_args = ['--step', 'E',
                '--proj', 'test,S:valid,F:valid,T:valid',
                '--sim_dir', sim_dir,
                '--fmt_dir', fmt_dir,
                '--trn_dir', trn_dir,
                '--est_dir', est_dir,
                '--est_prefix', est_prefix,
                '--use_parallel', 'F']

	# phyddle arguments
    my_args = util.load_config('scripts/config_R.py', arg_overwrite=True, args=cmd_args)

    # make formatted dataset
    est_prefix_path = f'tests/workspace/estimate/test/{est_prefix}'
    my_fmt = fmt.load(my_args)
    my_fmt.encode_one(tmp_fn=est_prefix_path, idx=-1, save_phyenc_csv=True)

    # load estimator
    my_est = est.load(my_args)

    # run simulator
    my_est.run()

    # load test output for Estimate
    est_lbl_test_fn = test_dir + f'/{est_prefix}.test_est.labels.csv'
    est_lbl_test = pd.read_csv(est_lbl_test_fn, header=0).to_numpy()

    # load valid output for Estimate
    est_lbl_valid_fn = valid_dir + f'/{est_prefix}.test_est.labels.csv'
    est_lbl_valid = pd.read_csv(est_lbl_valid_fn, header=0).to_numpy()

    # compare test and valid estimate labels
    assert( np.array_equal(est_lbl_test, est_lbl_valid) )

    return

#-----------------------------------------------------------------------------#
