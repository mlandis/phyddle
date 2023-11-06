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
import os

#-----------------------------------------------------------------------------#

def test_plot():
    
    # set seed
    util.set_seed(0)

    # filesystem
    base_dir = './tests/workspace'
    sim_dir = base_dir + '/simulate'
    fmt_dir = base_dir + '/format'
    trn_dir = base_dir + '/train'
    est_dir = base_dir + '/estimate'
    plt_dir = base_dir + '/plot'
    test_dir = plt_dir + '/test'
    valid_dir = plt_dir + '/valid'

	# command line arguments
    cmd_args = ['--step', 'P',
                '--proj', 'test,S:valid,F:valid,T:valid,E:valid',
                '--sim_dir', sim_dir,
                '--fmt_dir', fmt_dir,
                '--trn_dir', trn_dir,
                '--est_dir', est_dir,
                '--plt_dir', plt_dir,
                '--use_parallel', 'F']

	# phyddle arguments
    my_args = util.load_config('scripts/config_R.py', arg_overwrite=True, args=cmd_args)

    # load estimator
    my_plt = plt.load(my_args)

    # run simulator
    my_plt.run()

	# verify output
    out_files = [
        'fig_nt500.density_aux_data.pdf',
        'fig_nt500.density_label.pdf',
        'fig_nt500.network_architecture.pdf',
        'fig_nt500.pca_contour_aux_data.pdf',
        'fig_nt500.pca_contour_labels.pdf',
        'fig_nt500.summary.pdf',
        'fig_nt500.train_history_param_value.pdf',
        'fig_nt500.train_history.pdf',
        'fig_nt500.train_history_param_upper.pdf',
        'fig_nt500.train_history_param_lower.pdf'
    ]
    
    # verify all test output files exist
    for fn in out_files:
        assert( os.path.exists(test_dir + '/' + fn) )

    # verify same set of test and valid output files
    valid_files = os.listdir(valid_dir)
    for fn in valid_files:
        assert( os.path.exists(test_dir + '/' + fn) )
        # compare file size is perfect match

    return

#-----------------------------------------------------------------------------#
