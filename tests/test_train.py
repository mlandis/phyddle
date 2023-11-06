"""
test_train
=============
Tests classes and methods for the Train step.

Authors:   Michael Landis and Ammon Thompson
Copyright: (c) 2022-2023, Michael Landis and Ammon Thompson
License:   MIT
"""

import phyddle.utilities as util
import phyddle.train as trn
import tensorflow as tf
import numpy as np
import pandas as pd

#-----------------------------------------------------------------------------#

def test_trn():

    # set seed
    util.set_seed(0)

    # filesystem
    base_dir = './tests/workspace'
    fmt_dir = base_dir + '/format'
    trn_dir = base_dir + '/train'
    test_dir = trn_dir + '/test'
    valid_dir = trn_dir + '/valid'

    # command line arguments
    cmd_args = ['--step', 'T',
                '--proj', 'test,F:valid',
                '--fmt_dir', fmt_dir,
                '--trn_dir', trn_dir,
                '--prop_test', '0.1',
                '--prop_val', '0.1',
                '--use_parallel', 'F']

    # phyddle arguments
    my_args = util.load_config('scripts/config_R.py', arg_overwrite=True, args=cmd_args)

    # load trainer
    my_trn = trn.load(my_args)

    # run simulator
    my_trn.run()

    # load test output for Train
    model_test_fn = test_dir + '/network_nt500_trained_model'
    cpi_test_fn = test_dir + '/network_nt500.cpi_adjustments.csv'
    aux_test_fn = test_dir + '/network_nt500.train_aux_data_norm.csv'
    lbl_test_fn = test_dir + '/network_nt500.train_label_norm.csv'

    model_test = tf.keras.models.load_model(model_test_fn, compile=False)
    cpi_test = pd.read_csv(cpi_test_fn, header=0).to_numpy()
    aux_test = pd.read_csv(aux_test_fn, header=0).to_numpy()
    lbl_test = pd.read_csv(lbl_test_fn, header=0).to_numpy()

    # load valid output for Train
    model_valid_fn = valid_dir + '/network_nt500_trained_model'
    cpi_valid_fn = valid_dir + '/network_nt500.cpi_adjustments.csv'
    aux_valid_fn = valid_dir + '/network_nt500.train_aux_data_norm.csv'
    lbl_valid_fn = valid_dir + '/network_nt500.train_label_norm.csv'

    model_valid = tf.keras.models.load_model(model_valid_fn, compile=False)
    cpi_valid = pd.read_csv(cpi_valid_fn, header=0).to_numpy()
    aux_valid = pd.read_csv(aux_valid_fn, header=0).to_numpy()
    lbl_valid = pd.read_csv(lbl_valid_fn, header=0).to_numpy()

    # compare aux data, labels, and CPIs
    assert( (cpi_test == cpi_valid).all() )
    assert( (aux_test == aux_valid).all() )
    assert( (lbl_test == lbl_valid).all() )

    # compare model weights
    weights_test = [layer.get_weights() for layer in model_test.layers]
    weights_valid = [layer.get_weights() for layer in model_valid.layers]
    for w1, w2 in zip(weights_test, weights_valid):
        assert( len(w1) == len(w2) )
        for w1_layer, w2_layer in zip(w1, w2):
            assert(w1_layer.shape == w2_layer.shape)
            assert(np.array_equal(w1_layer, w2_layer))
    
    # success
    return

#-----------------------------------------------------------------------------#
