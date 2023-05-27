
#import cnn_utilities as cn
import pandas as pd
import numpy as np
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
        self.job_name          = args['job_name']
        self.net_dir           = args['net_dir']
        self.pred_dir          = args['pred_dir']
        self.pred_prefix       = args['pred_prefix']
        self.num_char_row      = args['num_char']
        self.tree_size         = args['tree_size']
        self.tree_type         = args['tree_type']
        
        return
    
    def prepare_files(self):

        # main directories
        self.network_dir    = self.net_dir + '/' + self.job_name
        self.prediction_dir = self.pred_dir + '/' + self.job_name

        # main job filenames
        self.model_prefix           = f'sim_batchsize{self.batch_size}_numepoch{self.num_epochs}_nt{self.tree_size}'
        self.model_sav_fn           = f'{self.network_dir}/{self.model_prefix}.hdf5'
        self.model_trn_lbl_norm_fn  = f'{self.network_dir}/{self.model_prefix}.train_label_norm.csv'
        self.model_trn_ss_norm_fn   = f'{self.network_dir}/{self.model_prefix}.train_summ_stat_norm.csv'

        # test summ stats
        self.pred_summ_stat_fn      = f'{self.prediction_dir}/{self.pred_prefix}.summ_stat.csv'

        # test phy vector
        if self.tree_type == 'extant':
            self.pred_phyvec_fn     = f'{self.prediction_dir}/{self.pred_prefix}.cdvs.csv'    
            self.num_tree_row       = 1
        elif self.tree_type == 'serial':
            self.pred_phyvec_fn     = f'{self.prediction_dir}/{self.pred_prefix}.cblvs.csv'
            self.num_tree_row       = 2   
        else:
            raise NotImplementedError
        return

    def run(self):
        self.load_input()
        self.make_results()
        
    def load_input(self):

        # denormalization factors for new summ stats
        train_stats_norm        = pd.read_csv(self.model_trn_ss_norm_fn, sep=',', header=1, index_col=False)
        self.train_stats_means  = train_stats_norm['mean'].T
        self.train_stats_sd     = train_stats_norm['sd'].T

        # denormalization factors for labels
        train_labels_norm       = pd.read_csv(self.model_trn_lbl_norm_fn, sep=',', header=1, index_col=False)
        self.train_labels_means = train_labels_norm['mean'].T
        self.train_labels_sd    = train_labels_norm['sd'].T

        # get param_names from training labels
        self.param_names        = train_labels_norm['name'].to_list()
        self.stat_names         = train_stats_norm['name'].to_list()

        self.num_params         = len(self.param_names) #pred_data.shape[1]
        self.num_stats          = len(self.pred_stats) #.shape[1]

        # read & reshape new test data
        self.pred_data          = pd.read_csv(self.pred_phyvec_fn, header=None, on_bad_lines='skip').to_numpy()
        self.pred_data.shape    = ( 1, -1, (self.num_tree_row+self.num_char_row) )
        
        # read & normalize new summary stats
        pred_stats              = pd.read_csv(self.pred_summ_stat_fn, header=None, on_bad_lines='skip').to_numpy()
        self.norm_pred_stats    = Utilities.normalize(pred_stats, (self.train_stats_means, self.train_stats_sd))
        
        return


    def make_results(self):

        # load model
        self.mymodel              = tf.keras.models.load_model(self.model_sav_fn)

        # get predictions
        norm_preds                = self.mymodel.predict([self.test_data_tensor, self.test_stats_tensor])
        denormalized_pred_labels  = Utilities.denormalize(norm_preds, self.train_label_means, self.train_label_sd)
        self.pred_labels          = np.exp( denormalized_pred_labels )
        print(self.pred_labels)

        # get confidence intervals
        # TBD
        
        return
    
