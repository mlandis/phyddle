"""
test_plot
=============
Tests classes and methods for the Plot step.

Authors:   Michael Landis and Ammon Thompson
Copyright: (c) 2022-2023, Michael Landis and Ammon Thompson
License:   MIT
"""

import phyddle.utilities as util
import phyddle.plot as plt

#-----------------------------------------------------------------------------#

def test_plot():

	# command line arguments
	cmd_args = ['--step', 'P',
                '--proj', 'test,S:valid,F:valid,T:valid,E:valid',
                '--sim_dir', 'tests/workspace/simulate',
                '--fmt_dir', 'tests/workspace/format',
                '--trn_dir', 'tests/workspace/train',
                '--est_dir', 'tests/workspace/estimate',
                '--plt_dir', 'tests/workspace/plot',
                '--use_parallel', 'F']

	# phyddle arguments
	my_args = util.load_config('scripts/config_R.py', arg_overwrite=True, args=cmd_args)

    # load estimator
	my_plt = plt.load(my_args)

    # run simulator
	my_plt.run()

	# verify output for sim.0.dat.nex
	# ...

	return

#-----------------------------------------------------------------------------#
