
#import cnn_utilities as cn
import pandas as pd
import numpy as np
#import dill
#import os
#import csv
#import json

import tensorflow as tf
from keras import *
#from keras import layers

import Utilities

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
        self.num_char_row      = args['num_char']
        self.batch_size        = args['batch_size']
        self.num_epochs        = args['num_epochs']
        self.tree_size         = args['tree_size']
        self.tree_type         = args['tree_type']
        
        return
    
    def prepare_files(self):

        # main directories
        self.network_dir            = f'{self.net_dir}/{self.project_name}'
        self.predict_dir            = f'{self.pred_dir}/{self.project_name}'

        # main job filenames
        self.model_prefix           = f'sim_batchsize{self.batch_size}_numepoch{self.num_epochs}_nt{self.tree_size}'
        self.model_sav_fn           = f'{self.network_dir}/{self.model_prefix}.hdf5'
        self.model_trn_lbl_norm_fn  = f'{self.network_dir}/{self.model_prefix}.train_label_norm.csv'
        self.model_trn_ss_norm_fn   = f'{self.network_dir}/{self.model_prefix}.train_summ_stat_norm.csv'
        self.model_cpi_func_fn      = f'{self.network_dir}/{self.model_prefix}.cpi_func.obj'
        self.model_cqr_fn           = f'{self.network_dir}/{self.model_prefix}.cqr_interval_adjustments.csv'

        # save predictions to file
        self.model_pred_fn          = f'{self.predict_dir}/{self.pred_prefix}.{self.model_prefix}.pred_labels.csv'

        # test summ stats
        self.pred_summ_stat_fn      = f'{self.predict_dir}/{self.pred_prefix}.summ_stat.csv'

        # test phy vector
        if self.tree_type == 'extant':
            self.pred_phyvec_fn     = f'{self.predict_dir}/{self.pred_prefix}.cdvs.csv'    
            self.num_tree_row       = 3
        elif self.tree_type == 'serial':
            self.pred_phyvec_fn     = f'{self.predict_dir}/{self.pred_prefix}.cblvs.csv'
            self.num_tree_row       = 4   
        else:
            raise NotImplementedError
        
        return

    def run(self):
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
        
        # read & normalize new summary stats
        self.pred_stats_tensor       = pd.read_csv(self.pred_summ_stat_fn, sep=',', index_col=False).to_numpy().flatten()
        self.pred_stats_tensor.shape = ( 1, -1 )

        self.norm_pred_stats          = Utilities.normalize(self.pred_stats_tensor, (self.train_stats_means, self.train_stats_sd))
        self.denormalized_pred_stats  = Utilities.denormalize(self.norm_pred_stats, self.train_stats_means, self.train_stats_sd)


        # read in CQR interval adjustments
        self.cqr_interval_adjustments = pd.read_csv(self.model_cqr_fn, sep=',', index_col=False).to_numpy()
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
        self.norm_preds[1,:,:]         = self.norm_preds[1,:,:] - self.cqr_interval_adjustments[0,:]
        self.norm_preds[2,:,:]         = self.norm_preds[2,:,:] + self.cqr_interval_adjustments[1,:]
        self.denormalized_pred_labels  = Utilities.denormalize(self.norm_preds, self.train_labels_means, self.train_labels_sd)
        self.pred_labels               = np.exp( self.denormalized_pred_labels )
       
        # output predictions
        self.df_pred_all_labels = Utilities.make_param_VLU_mtx(self.pred_labels, self.param_names)
        self.df_pred_all_labels.to_csv(self.model_pred_fn, index=False, sep=',')

        return
    
