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

#-----------------------------------------------------------------------------#

def test_est():

	# command line arguments
	cmd_args = ['--step', 'E',
                '--proj', 'test,F:valid,T:valid',
                '--sim_dir', 'tests/workspace/simulate',
                '--fmt_dir', 'tests/workspace/format',
                '--trn_dir', 'tests/workspace/train',
                '--est_dir', 'tests/workspace/estimate' ]

	# phyddle arguments
	my_args = util.load_config('scripts/config_R.py', arg_overwrite=True, args=cmd_args)

    # make formatted dataset
	est_prefix = 'tests/workspace/estimate/test/new.0'
	my_fmt = fmt.load(my_args)
	my_fmt.encode_one(tmp_fn=est_prefix, idx=-1, save_phyenc_csv=True)

    # load estimator
	my_est = est.load(my_args)

    # run simulator
	my_est.run()

	# verify output for sim.0.dat.nex
	# ...

	return

#-----------------------------------------------------------------------------#
