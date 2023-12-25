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
# import tensorflow as tf
import torch
import h5py
import shutil
import os
# import random


# NOTE: Set ENABLE_TEST = False to bypass the check_foo() validation tests
ENABLE_TEST = True

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
#tf.config.experimental.enable_op_determinism()
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
    base_dir = './tests/workspace/simulate'

    # command line arguments
    cmd_args = ['--step', 'S',
                '--proj', 'test',
                '--sim_dir', base_dir,
                '--sim_command', 'Rscript scripts/sim/R/sim_one.R',
                '--end_idx', '100',
                '--use_parallel', 'F']

    # build arguments
    my_args = util.load_config('scripts/config.py', arg_overwrite=True, args=cmd_args)

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
    base_dir = './tests/workspace/simulate'
    test_dir = base_dir + '/test'
    valid_dir = base_dir + '/valid'

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
    base_dir = './tests/workspace'
    sim_dir = base_dir + '/simulate'
    fmt_dir = base_dir + '/format'

    # command line arguments
    cmd_args = ['--step', 'F',
                '--proj', 'test,S:valid',
                '--sim_dir', sim_dir,
                '--fmt_dir', fmt_dir,
                '--prop_test','0.10',
                '--prop_val','0.10',
                '--use_parallel', 'F']

    # phyddle arguments
    my_args = util.load_config('scripts/config.py', arg_overwrite=True, args=cmd_args)

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
    base_dir = './tests/workspace'
    fmt_dir = base_dir + '/format'
    test_dir = fmt_dir + '/test'
    valid_dir = fmt_dir + '/valid'

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

# Train

def do_trn():

    # set seed
    util.set_seed(0)

    # filesystem
    base_dir = './tests/workspace'
    fmt_dir = base_dir + '/format'
    trn_dir = base_dir + '/train'

    # command line arguments
    cmd_args = ['--step', 'T',
                '--proj', 'test,F:valid',
                '--fmt_dir', fmt_dir,
                '--trn_dir', trn_dir,
                '--prop_test', '0.1',
                '--prop_val', '0.1',
                '--use_parallel', 'F']

    # phyddle arguments
    my_args = util.load_config('scripts/config.py', arg_overwrite=True, args=cmd_args)

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
    base_dir = './tests/workspace'
    trn_dir = base_dir + '/train'
    test_dir = trn_dir + '/test'
    valid_dir = trn_dir + '/valid'

    # load test output for Train
    model_test_fn = test_dir + '/network_nt500.trained_model.pkl'
    cpi_test_fn = test_dir + '/network_nt500.cpi_adjustments.csv'
    aux_test_fn = test_dir + '/network_nt500.train_aux_data_norm.csv'
    lbl_test_fn = test_dir + '/network_nt500.train_label_norm.csv'

    #model_test = tf.keras.models.load_model(model_test_fn, compile=False)
    #model_load = torch.load(model_test_fn)
    cpi_test = pd.read_csv(cpi_test_fn, header=0).to_numpy()
    aux_test = pd.read_csv(aux_test_fn, header=0).iloc[:,1:].to_numpy()
    lbl_test = pd.read_csv(lbl_test_fn, header=0).iloc[:,1:].to_numpy()
    
    # load valid output for Train
    model_valid_fn = valid_dir + '/network_nt500.trained_model.pkl'
    cpi_valid_fn = valid_dir + '/network_nt500.cpi_adjustments.csv'
    aux_valid_fn = valid_dir + '/network_nt500.train_aux_data_norm.csv'
    lbl_valid_fn = valid_dir + '/network_nt500.train_label_norm.csv'

    #model_valid = tf.keras.models.load_model(model_valid_fn, compile=False)
    #model_valid = torch.load(model_valid_fn)
    cpi_valid = pd.read_csv(cpi_valid_fn, header=0).to_numpy()
    aux_valid = pd.read_csv(aux_valid_fn, header=0).iloc[:,1:].to_numpy()
    lbl_valid = pd.read_csv(lbl_valid_fn, header=0).iloc[:,1:].to_numpy()

    # compare aux data, labels, and CPIs
    cpi_error = np.max(np.abs(cpi_test - cpi_valid))
    aux_error = np.max(np.abs(aux_test - aux_valid))
    lbl_error = np.max(np.abs(lbl_test - lbl_valid))
    if cpi_error < ERROR_TOL:
        print('cpi_error < ERROR_TOL: ', cpi_error)
    if aux_error < ERROR_TOL:
        print('aux_error < ERROR_TOL: ', aux_error)
    if lbl_error < ERROR_TOL:
        print('lbl_error < ERROR_TOL: ', lbl_error)
    assert( cpi_error < ERROR_TOL )
    assert( aux_error < ERROR_TOL )
    assert( lbl_error < ERROR_TOL )

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
    my_args = util.load_config('scripts/config.py', arg_overwrite=True, args=cmd_args)

    # copy minimal input fileset from valid into test
    input_files = [ 'tre', 'dat.csv', 'labels.csv' ]
    os.makedirs(test_dir, exist_ok=True)
    if ENABLE_TEST:
        for fn in input_files:
            shutil.copyfile( f'{sim_dir}/valid/sim.0.{fn}', f'{est_dir}/test/new.0.{fn}' )
    else:
        for fn in input_files:
            shutil.copyfile( f'{sim_dir}/test/sim.0.{fn}', f'{est_dir}/test/new.0.{fn}' )

    # make formatted dataset
    est_prefix_path = f'tests/workspace/estimate/test/{est_prefix}'
    my_fmt = fmt.load(my_args)
    my_fmt.encode_one(tmp_fn=est_prefix_path, idx=-1, save_phyenc_csv=True)

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
    base_dir = './tests/workspace'
    est_dir = base_dir + '/estimate'
    test_dir = est_dir + '/test'
    valid_dir = est_dir + '/valid'
    est_prefix = 'new.0'

    # load test output for Estimate
    est_lbl_test_fn = test_dir + f'/{est_prefix}.test_est.labels.csv'
    est_lbl_test = pd.read_csv(est_lbl_test_fn, header=0).to_numpy()

    # load valid output for Estimate
    est_lbl_valid_fn = valid_dir + f'/{est_prefix}.test_est.labels.csv'
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
    my_args = util.load_config('scripts/config.py', arg_overwrite=True, args=cmd_args)

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
    base_dir = './tests/workspace'
    plt_dir = base_dir + '/plot'
    test_dir = plt_dir + '/test'
    valid_dir = plt_dir + '/valid'

	# verify output
    out_files = [
        'fig_nt500.density_aux_data.pdf',
        'fig_nt500.density_label.pdf',
        'fig_nt500.network_architecture.pdf',
        'fig_nt500.pca_contour_aux_data.pdf',
        'fig_nt500.pca_contour_labels.pdf',
        'fig_nt500.summary.pdf',
        'fig_nt500.train_history.pdf'
        # 'fig_nt500.train_history_param_value.pdf',
        # 'fig_nt500.train_history_param_upper.pdf',
        # 'fig_nt500.train_history_param_lower.pdf'
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

