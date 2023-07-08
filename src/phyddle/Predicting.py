#!/usr/bin/env python
"""
Predicting
==========
Defines classes and methods for the Predicting step, which loads a pre-trained model
and uses it to make predictions (e.g. parameter estimates) for e.g. a new empirical
dataset.

Authors:   Michael Landis, Ammon Thompson
Copyright: (c) 2023, Michael Landis
License:   MIT
"""

# standard imports
import os

# external imports
import numpy as np
import pandas as pd
import tensorflow as tf
#from keras import *

# phyddle imports
from phyddle import Utilities
#from Formatting import encode_phy_tensor

#-----------------------------------------------------------------------------------------------------------------#

def load(args):
    #sim_method = args['learn_method']
    predict_method = 'default'
    if predict_method == 'default':
        return Predictor(args)
    else:
        return None

#-----------------------------------------------------------------------------------------------------------------#

class Predictor:
    def __init__(self, args):
        self.set_args(args)
        self.prepare_files()
        return
    
    def set_args(self, args):
        self.args              = args
        self.project_name      = args['proj']
        self.net_dir           = args['net_dir']
        self.pred_dir          = args['pred_dir']
        self.pred_prefix       = args['pred_prefix']
        #self.num_char_row      = args['num_char']
        self.batch_size        = args['batch_size']
        self.num_epochs        = args['num_epochs']
        self.tree_width        = args['tree_width']
        self.tree_type         = args['tree_type']
        self.char_encode_type  = args['char_encode_type']
        self.tree_encode_type  = args['tree_encode_type']
        self.num_char          = args['num_char']
        self.num_states        = args['num_states']
        
        return
    
    def prepare_files(self):

        # main directories
        self.network_dir            = f'{self.net_dir}/{self.project_name}'
        self.predict_dir            = f'{self.pred_dir}/{self.project_name}'

        # main job filenames
        self.model_prefix           = f'sim_batchsize{self.batch_size}_numepoch{self.num_epochs}_nt{self.tree_width}'
        self.model_sav_fn           = f'{self.network_dir}/{self.model_prefix}.hdf5'
        self.model_trn_lbl_norm_fn  = f'{self.network_dir}/{self.model_prefix}.train_label_norm.csv'
        self.model_trn_ss_norm_fn   = f'{self.network_dir}/{self.model_prefix}.train_summ_stat_norm.csv'
        #self.model_cpi_func_fn      = f'{self.network_dir}/{self.model_prefix}.cpi_func.obj'
        self.model_cpi_fn           = f'{self.network_dir}/{self.model_prefix}.cpi_adjustments.csv'

        # save predictions to file
        self.model_pred_fn          = f'{self.predict_dir}/{self.pred_prefix}.{self.model_prefix}.pred_labels.csv'

        # test summ stats
        self.pred_summ_stat_fn      = f'{self.predict_dir}/{self.pred_prefix}.summ_stat.csv'
        self.pred_known_param_fn    = f'{self.predict_dir}/{self.pred_prefix}.known_param.csv'

        # test phy vector
        if self.tree_type == 'extant':
            self.pred_phyvec_fn     = f'{self.predict_dir}/{self.pred_prefix}.cdvs.csv'    
        #    self.num_tree_row       = 3
        elif self.tree_type == 'serial':
            self.pred_phyvec_fn     = f'{self.predict_dir}/{self.pred_prefix}.cblvs.csv'
        #    self.num_tree_row       = 4   
        else:
            raise NotImplementedError

        self.num_tree_row = Utilities.get_num_tree_row(self.tree_type, self.tree_encode_type)
        self.num_char_row = Utilities.get_num_char_row(self.char_encode_type, self.num_char, self.num_states)
        self.num_data_row = self.num_tree_row + self.num_char_row
        
        return

    def run(self):
        os.makedirs(self.predict_dir, exist_ok=True)
        self.load_input()
        self.make_results()
        
    def load_input(self):

        # denormalization factors for new summ stats
        train_stats_norm        = pd.read_csv(self.model_trn_ss_norm_fn, sep=',', index_col=False)
        self.train_stats_means  = train_stats_norm['mean'].T.to_numpy().flatten()
        self.train_stats_sd     = train_stats_norm['sd'].T.to_numpy().flatten()

        # denormalization factors for labels
        train_labels_norm       = pd.read_csv(self.model_trn_lbl_norm_fn, sep=',', index_col=False)
        self.train_labels_means = train_labels_norm['mean'].T.to_numpy().flatten()
        self.train_labels_sd    = train_labels_norm['sd'].T.to_numpy().flatten()

        # get param_names from training labels
        self.param_names        = train_labels_norm['name'].to_list()
        self.stat_names         = train_stats_norm['name'].to_list()

        # read & reshape new test data
        self.pred_data_tensor   = pd.read_csv(self.pred_phyvec_fn, header=None, sep=',', index_col=False).to_numpy()
        #self.pred_data_tensor.shape  = ( 1, -1, (self.num_tree_row+self.num_char_row) )
        self.pred_data_tensor   = self.pred_data_tensor.reshape( (1, -1, (self.num_tree_row+self.num_char_row)) )
        
        # read & normalize new aux data
        self.pred_summ_stats     = pd.read_csv(self.pred_summ_stat_fn, sep=',', index_col=False).to_numpy().flatten()
        try:
            self.pred_known_params   = pd.read_csv(self.pred_known_param_fn, sep=',', index_col=False).to_numpy().flatten()
            self.pred_auxdata_tensor = np.concatenate( [self.pred_summ_stats, self.pred_known_params] )
        except FileNotFoundError:
            self.pred_auxdata_tensor = self.pred_summ_stats
            
        self.pred_auxdata_tensor.shape = ( 1, -1 )

        #print(self.pred_auxdata_tensor)

        self.norm_pred_stats          = Utilities.normalize(self.pred_auxdata_tensor, (self.train_stats_means, self.train_stats_sd))
        self.denormalized_pred_stats  = Utilities.denormalize(self.norm_pred_stats, self.train_stats_means, self.train_stats_sd)


        # read in CQR interval adjustments
        self.cpi_adjustments = pd.read_csv(self.model_cpi_fn, sep=',', index_col=False).to_numpy()
        # # CPI functions
        # with open(self.model_cpi_func_fn, 'rb') as f:
        #     self.cpi_func = dill.load(f)

        return


    def make_results(self):

        # load model
        self.mymodel = tf.keras.models.load_model(self.model_sav_fn, compile=False)

        # get predictions
        self.norm_preds                = self.mymodel.predict([self.pred_data_tensor, self.norm_pred_stats])
        self.norm_preds                = np.array( self.norm_preds )
        self.norm_preds[1,:,:]         = self.norm_preds[1,:,:] - self.cpi_adjustments[0,:]
        self.norm_preds[2,:,:]         = self.norm_preds[2,:,:] + self.cpi_adjustments[1,:]
        self.denormalized_pred_labels  = Utilities.denormalize(self.norm_preds, self.train_labels_means, self.train_labels_sd)
        #print(self.denormalized_pred_labels)
        self.denormalized_pred_labels[ self.denormalized_pred_labels > 300. ] = 300.
        self.pred_labels               = np.exp( self.denormalized_pred_labels )
        #print(np.exp( self.denormalized_pred_labels ))
       
        # output predictions
        self.df_pred_all_labels = Utilities.make_param_VLU_mtx(self.pred_labels, self.param_names)
        self.df_pred_all_labels.to_csv(self.model_pred_fn, index=False, sep=',')

        return
    
