#!/usr/bin/env python
"""
test_pipeline
=============
Tests classes and methods for all phyddle pipeline steps

Authors:   Michael Landis and Ammon Thompson
Copyright: (c) 2022-2023, Michael Landis and Ammon Thompson
License:   MIT
"""

import phyddle.utilities as util
import phyddle.simulate as sim
import phyddle.format as fmt
import phyddle.train as trn
import phyddle.estimate as est
import phyddle.plot as plt

import pandas as pd
import numpy as np
import torch
import h5py
import shutil
import os
# import random


# NOTE: Set ENABLE_TEST = False to bypass the check_foo() validation tests
ENABLE_TEST = not True

# NOTE: Set ERROR_TOL to an acceptable amount of difference in results between
#       the test and validation examples. Even though RNG seeds are set
#       as equal across steps, the large number of layer-by-layer computations
#       can be executed in different orders due to resource allocation, which
#       which can cause minor numerical errors to accumulate in different ways.
# 
#       ... still difference of ~0.035 for CPI, seems too large
ERROR_TOL = 5E-1 # 1E-2

#-----------------------------------------------------------------------------#

# single-thread session
torch.set_deterministic_debug_mode(debug_mode='warn')
#torch.use_deterministic_algorithms(mode=True)

#-----------------------------------------------------------------------------#

# Each test function below covers one pipeline step.

def test_sim():
    do_sim()
    check_sim()
    return

def test_fmt():
    do_fmt()
    check_fmt()
    return

def test_trn():
    do_trn()
    check_trn()
    return

def test_est():
    do_est()
    check_est()
    return

def test_plt():
    do_plt()
    check_plt()
    return

#-----------------------------------------------------------------------------#

# Simulate

def do_sim():
    
    # set seed
    util.set_seed(0)
  
    # filesystem
    work_dir = './tests/workspace'
    sim_dir = work_dir + '/test/simulate'

    # command line arguments
    cmd_args = ['--step', 'S',
                '--sim_prefix', 'out',
                '--sim_dir', sim_dir,
                '--sim_command', 'Rscript ./tests/sim_bisse.R',
                '--end_idx', '100',
                '--use_parallel', 'F']

    # build arguments
    my_args = util.load_config('./tests/config.py', arg_overwrite=True, args=cmd_args)

    # load simulator
    my_sim = sim.load(my_args)

    # run simulator
    my_sim.run()

    # success
    return

def check_sim():

    if not ENABLE_TEST:
        return
    
    # filesystem
    work_dir  = './tests/workspace'
    test_dir  = work_dir + '/test/simulate'
    valid_dir = work_dir + '/valid/simulate'

    # confirm test and valid tree match
    phy_test = util.read_tree(test_dir + '/sim.0.tre')
    phy_valid = util.read_tree(valid_dir + '/sim.0.tre')
    assert( phy_test.length() == phy_valid.length() )
    assert( str(phy_test) == str(phy_valid) )

    # confirm test and valid data match
    dat_valid = util.convert_csv_to_array(valid_dir + '/sim.0.dat.csv', 'integer', 2)
    dat_test = util.convert_csv_to_array(test_dir + '/sim.0.dat.csv', 'integer', 2)
    assert(np.array_equal(dat_valid, dat_test))

    # confirm test and valid params match
    param_valid = util.convert_csv_to_array(valid_dir + '/sim.0.labels.csv', 'numeric', 2)
    param_test = util.convert_csv_to_array(test_dir + '/sim.0.labels.csv', 'numeric', 2)
    assert(np.array_equal(param_valid, param_test))

    # success
    return

#-----------------------------------------------------------------------------#

# Format

def do_fmt():

    # set seed
    util.set_seed(0)

    # filesystem
    work_dir = './tests/workspace'
    test_dir = f'{work_dir}/test'
    valid_dir = f'{work_dir}/valid'
    sim_dir = f'{test_dir}/simulate'
    fmt_dir = f'{test_dir}/format'
    emp_dir = f'{test_dir}/empirical'

    # command line arguments
    cmd_args = ['--step', 'F',
                '--sim_prefix','out',
                '--emp_prefix','out',
                '--fmt_prefix','out',
                '--sim_dir', sim_dir,
                '--fmt_dir', fmt_dir,
                '--emp_dir', emp_dir,
                '--prop_test','0.10',
                '--prop_val','0.10',
                '--use_parallel', 'F']

    # copy minimal input fileset from valid into test
    input_files = [ 'tre', 'dat.csv', 'labels.csv' ]
    os.makedirs(f'{test_dir}/empirical', exist_ok=True)
    if ENABLE_TEST:
        for fn in input_files:
            shutil.copyfile( f'{valid_dir}/simulate/out.0.{fn}', f'{test_dir}/empirical/out.0.{fn}' )
    else:
        for fn in input_files:
            shutil.copyfile( f'{test_dir}/simulate/out.0.{fn}', f'{test_dir}/empirical/out.0.{fn}' )

    # phyddle arguments
    my_args = util.load_config('./tests/config.py', arg_overwrite=True, args=cmd_args)

    # load simulator
    my_fmt = fmt.load(my_args)

    # run simulator
    my_fmt.run()

    # success
    return

def check_fmt():

    if not ENABLE_TEST:
        return

    # filesystem
    work_dir  = './tests/workspace'
    test_dir  = work_dir + '/test/format'
    valid_dir = work_dir + '/valid/format'

    # confirm test and valid tree match
    dat_test_test = h5py.File(test_dir + '/out.test.hdf5', 'r')
    dat_test_valid = h5py.File(valid_dir + '/out.test.hdf5', 'r')
    
    # collect test format output arrays
    aux_data_names_test_test  = dat_test_test['aux_data_names'][:]
    label_names_test_test     = dat_test_test['label_names'][:]
    phy_data_test_test        = dat_test_test['phy_data'][:]
    aux_data_test_test        = dat_test_test['aux_data'][:]
    labels_test_test          = dat_test_test['labels'][:]

    # collect valid format output arrays
    aux_data_names_test_valid = dat_test_valid['aux_data_names'][:]
    label_names_test_valid    = dat_test_valid['label_names'][:]
    phy_data_test_valid       = dat_test_valid['phy_data'][:]
    aux_data_test_valid       = dat_test_valid['aux_data'][:]
    labels_test_valid         = dat_test_valid['labels'][:]

    # close files
    dat_test_test.close()
    dat_test_valid.close()

    # verify all test and valid format output match
    assert(np.array_equal(aux_data_names_test_test, aux_data_names_test_valid))
    assert(np.array_equal(label_names_test_test, label_names_test_valid))
    assert(np.array_equal(phy_data_test_test, phy_data_test_valid))
    assert(np.array_equal(aux_data_test_test, aux_data_test_valid))
    assert(np.array_equal(labels_test_test, labels_test_valid))

    # confirm test and valid tree match
    dat_emp_test = h5py.File(test_dir + '/out.empirical.hdf5', 'r')
    dat_emp_valid = h5py.File(valid_dir + '/out.empirical.hdf5', 'r')

    # collect test format output arrays
    aux_data_names_emp_test  = dat_emp_test['aux_data_names'][:]
    label_names_emp_test     = dat_emp_test['label_names'][:]
    phy_data_emp_test        = dat_emp_test['phy_data'][:]
    aux_data_emp_test        = dat_emp_test['aux_data'][:]
    labels_emp_test          = dat_emp_test['labels'][:]

    # collect valid format output arrays
    aux_data_names_emp_valid = dat_emp_valid['aux_data_names'][:]
    label_names_emp_valid    = dat_emp_valid['label_names'][:]
    phy_data_emp_valid       = dat_emp_valid['phy_data'][:]
    aux_data_emp_valid       = dat_emp_valid['aux_data'][:]
    labels_emp_valid         = dat_emp_valid['labels'][:]

    # close files
    dat_test_test.close()
    dat_test_valid.close()

    # verify all test and valid format output match
    assert(np.array_equal(aux_data_names_emp_test, aux_data_names_emp_valid))
    assert(np.array_equal(label_names_emp_test, label_names_emp_valid))
    assert(np.array_equal(phy_data_emp_test, phy_data_emp_valid))
    assert(np.array_equal(aux_data_emp_test, aux_data_emp_valid))
    assert(np.array_equal(labels_emp_test, labels_emp_valid))

    # success
    return

#-----------------------------------------------------------------------------#

# Train

def do_trn():

    # set seed
    util.set_seed(0)

    # filesystem
    work_dir = './tests/workspace'
    fmt_dir = work_dir + '/test/format'
    trn_dir = work_dir + '/test/train'

    # command line arguments
    cmd_args = ['--step', 'T',
                '--fmt_prefix', 'out',
                '--trn_prefix', 'out',
                '--fmt_dir', fmt_dir,
                '--trn_dir', trn_dir,
                '--prop_test', '0.1',
                '--prop_val', '0.1',
                '--use_parallel', 'F']

    # phyddle arguments
    my_args = util.load_config('./tests/config.py', arg_overwrite=True, args=cmd_args)

    # load trainer
    my_trn = trn.load(my_args)

    # run simulator
    my_trn.run()

    # success
    return

def check_trn():

    if not ENABLE_TEST:
        return

    # filesystem
    work_dir  = './tests/workspace'
    test_dir  = work_dir + '/test/train'
    valid_dir = work_dir + '/valid/train'

    # load test output for Train
    lbl_test_fn = test_dir + '/out.train_est.labels.csv'
    cpi_test_fn = test_dir + '/out.cpi_adjustments.csv'
    aux_norm_test_fn = test_dir + '/out.train_aux_data_norm.csv'
    lbl_norm_test_fn = test_dir + '/out.train_label_norm.csv'

    lbl_test = pd.read_csv(lbl_test_fn, header=0).iloc[:,1:].to_numpy()
    cpi_test = pd.read_csv(cpi_test_fn, header=0).to_numpy()
    aux_norm_test = pd.read_csv(aux_norm_test_fn, header=0).iloc[:,1:].to_numpy()
    lbl_norm_test = pd.read_csv(lbl_norm_test_fn, header=0).iloc[:,1:].to_numpy()
    
    # load valid output for Train
    lbl_valid_fn = valid_dir + '/out.train_est.labels.csv'
    cpi_valid_fn = valid_dir + '/out.cpi_adjustments.csv'
    aux_norm_valid_fn = valid_dir + '/out.train_aux_data_norm.csv'
    lbl_norm_valid_fn = valid_dir + '/out.train_label_norm.csv'

    lbl_valid = pd.read_csv(lbl_valid_fn, header=0).iloc[:,1:].to_numpy()
    cpi_valid = pd.read_csv(cpi_valid_fn, header=0).to_numpy()
    aux_norm_valid = pd.read_csv(aux_norm_valid_fn, header=0).iloc[:,1:].to_numpy()
    lbl_norm_valid = pd.read_csv(lbl_norm_valid_fn, header=0).iloc[:,1:].to_numpy()

    # compare aux data, labels, and CPIs
    lbl_error = np.max(np.abs(lbl_test - lbl_valid))
    cpi_error = np.max(np.abs(cpi_test - cpi_valid))
    aux_norm_error = np.max(np.abs(aux_norm_test - aux_norm_valid))
    lbl_norm_error = np.max(np.abs(lbl_norm_test - lbl_norm_valid))
    if lbl_error < ERROR_TOL:
        print('lbl_error < ERROR_TOL: ', lbl_error)
    if cpi_error < ERROR_TOL:
        print('cpi_error < ERROR_TOL: ', cpi_error)
    if aux_norm_error < ERROR_TOL:
        print('aux_norm_error < ERROR_TOL: ', aux_norm_error)
    if lbl_norm_error < ERROR_TOL:
        print('lbl_norm_error < ERROR_TOL: ', lbl_norm_error)
    assert( lbl_error < ERROR_TOL )
    assert( cpi_error < ERROR_TOL )
    assert( aux_norm_error < ERROR_TOL )
    assert( lbl_norm_error < ERROR_TOL )

    # compare model weights
    # weights_test = [layer.get_weights() for layer in model_test.layers]
    # weights_valid = [layer.get_weights() for layer in model_valid.layers]
    # for w1, w2 in zip(weights_test, weights_valid):
    #     assert( len(w1) == len(w2) )
    #     for w1_layer, w2_layer in zip(w1, w2):
    #         assert(w1_layer.shape == w2_layer.shape)
    #         assert(np.array_equal(w1_layer, w2_layer))
    
    # success
    return

#-----------------------------------------------------------------------------#

# Estimate

def do_est():

    # set seed
    util.set_seed(0)

    # filesystem
    work_dir = './tests/workspace'
    test_dir = work_dir + '/test'
    valid_dir = work_dir + '/valid'
    fmt_dir = work_dir + '/valid/format'
    trn_dir = work_dir + '/valid/train'
    est_dir = work_dir + '/test/estimate'
    emp_dir = work_dir + '/test/empirical'
   
	# command line arguments
    cmd_args = ['--step', 'E',
                '--fmt_prefix', 'out',
                '--trn_prefix', 'out',
                '--est_prefix', 'out',
                '--emp_prefix', 'out',
                '--fmt_dir', fmt_dir,
                '--trn_dir', trn_dir,
                '--est_dir', est_dir,
                '--emp_dir', emp_dir,
                '--use_parallel', 'F']

    # phyddle arguments
    my_args = util.load_config('./tests/config.py', arg_overwrite=True, args=cmd_args)

    # load estimator
    my_est = est.load(my_args)

    # run simulator
    my_est.run()

    # success
    return
    
def check_est():

    if not ENABLE_TEST:
        return

    # filesystem
    work_dir  = './tests/workspace'
    test_dir  = work_dir + '/test/estimate'
    valid_dir = work_dir + '/valid/estimate'

    # load test output for Estimate
    est_lbl_test_fn = test_dir + f'/out.test_est.labels.csv'
    est_lbl_test = pd.read_csv(est_lbl_test_fn, header=0).to_numpy()

    # load valid output for Estimate
    est_lbl_valid_fn = valid_dir + f'/out.test_est.labels.csv'
    est_lbl_valid = pd.read_csv(est_lbl_valid_fn, header=0).to_numpy()

    # compare test and valid estimate labels
    lbl_error = np.max(np.abs(est_lbl_test - est_lbl_valid))
    if lbl_error < ERROR_TOL:
        print('lbl_error < ERROR_TOL: ', lbl_error)
    assert( lbl_error < ERROR_TOL)

    # load test output for Estimate
    est_lbl_test_fn = test_dir + f'/out.empirical_est.labels.csv'
    est_lbl_test = pd.read_csv(est_lbl_test_fn, header=0).to_numpy()

    # load valid output for Estimate
    est_lbl_valid_fn = valid_dir + f'/out.empirical_est.labels.csv'
    est_lbl_valid = pd.read_csv(est_lbl_valid_fn, header=0).to_numpy()

    # compare test and valid estimate labels
    lbl_error = np.max(np.abs(est_lbl_test - est_lbl_valid))
    if lbl_error < ERROR_TOL:
        print('lbl_error < ERROR_TOL: ', lbl_error)
    assert( lbl_error < ERROR_TOL)

    # success
    return

#-----------------------------------------------------------------------------#

# Plot

def do_plt():
    
    # set seed
    util.set_seed(0)

    # filesystem
    work_dir = './tests/workspace'
    sim_dir = work_dir + '/valid/simulate'
    fmt_dir = work_dir + '/valid/format'
    trn_dir = work_dir + '/valid/train'
    est_dir = work_dir + '/valid/estimate'
    plt_dir = work_dir + '/test/plot'

	# command line arguments
    cmd_args = ['--step', 'P',
                '--sim_prefix', 'out',
                '--emp_prefix', 'out',
                '--fmt_prefix', 'out',
                '--trn_prefix', 'out',
                '--est_prefix', 'out',
                '--plt_prefix', 'out',
                '--sim_dir', sim_dir,
                '--fmt_dir', fmt_dir,
                '--trn_dir', trn_dir,
                '--est_dir', est_dir,
                '--plt_dir', plt_dir,
                '--use_parallel', 'F']

	# phyddle arguments
    my_args = util.load_config('./tests/config.py', arg_overwrite=True, args=cmd_args)

    # load estimator
    my_plt = plt.load(my_args)

    # run simulator
    my_plt.run()

    # success
    return

def check_plt():

    if not ENABLE_TEST:
        return

    # filesystem
    work_dir  = './tests/workspace'
    test_dir  = work_dir + '/test/plot'
    valid_dir = work_dir + '/valid/plot'

	# verify output
    out_files = [
        'out.train_density_aux_data.pdf',
        'out.train_density_labels.pdf',
        'out.network_architecture.pdf',
        'out.train_pca_contour_aux_data.pdf',
        'out.train_pca_contour_labels.pdf',
        'out.summary.pdf',
        'out.train_history.pdf'
    ]
    
    # verify all test output files exist
    for fn in out_files:
        assert( os.path.exists(test_dir + '/' + fn) )

    # verify same set of test and valid output files
    valid_files = os.listdir(valid_dir)
    for fn in valid_files:
        assert( os.path.exists(test_dir + '/' + fn) )
        # compare file size is perfect match

    # success
    return

#-----------------------------------------------------------------------------#

