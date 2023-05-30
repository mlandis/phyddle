
#import cnn_utilities as cn
import pandas as pd
import numpy as np
import dill
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
        self.batch_size        = args['batch_size']
        self.num_epochs        = args['num_epochs']
        self.tree_size         = args['tree_size']
        self.tree_type         = args['tree_type']
        
        return
    
    def prepare_files(self):

        # main directories
        self.network_dir    = self.net_dir + '/' + self.job_name

        # main job filenames
        self.model_prefix           = f'sim_batchsize{self.batch_size}_numepoch{self.num_epochs}_nt{self.tree_size}'
        self.model_sav_fn           = f'{self.network_dir}/{self.model_prefix}.hdf5'
        self.model_trn_lbl_norm_fn  = f'{self.network_dir}/{self.model_prefix}.train_label_norm.csv'
        self.model_trn_ss_norm_fn   = f'{self.network_dir}/{self.model_prefix}.train_summ_stat_norm.csv'
        self.model_cpi_func_fn      = f'{self.network_dir}/{self.model_prefix}.cpi_func.obj'

        # save predictions to file
        self.model_pred_fn          = f'{self.network_dir}/{self.pred_prefix}.{self.model_prefix}.pred_labels.csv'

        # test summ stats
        self.pred_summ_stat_fn      = f'{self.pred_dir}/{self.pred_prefix}.summ_stat.csv'

        # test phy vector
        if self.tree_type == 'extant':
            self.pred_phyvec_fn     = f'{self.pred_dir}/{self.pred_prefix}.cdvs.csv'    
            self.num_tree_row       = 1
        elif self.tree_type == 'serial':
            self.pred_phyvec_fn     = f'{self.pred_dir}/{self.pred_prefix}.cblvs.csv'
            self.num_tree_row       = 2   
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

        # sizes
        #self.num_params         = len(self.param_names)
        #self.num_stats          = len(self.stat_names)

        # read & reshape new test data
        self.pred_data_tensor        = pd.read_csv(self.pred_phyvec_fn, header=None, sep=',', index_col=False).to_numpy()
        self.pred_data_tensor.shape  = ( 1, -1, (self.num_tree_row+self.num_char_row) )
        
        # read & normalize new summary stats
        self.pred_stats_tensor       = pd.read_csv(self.pred_summ_stat_fn, sep=',', index_col=False).to_numpy().flatten()
        self.pred_stats_tensor.shape = ( 1, -1 )
        #self.ln_pred_stats_tensor    = np.log(self.pred_stats_tensor)
        #print(self.pred_data_tensor)
        #print(self.pred_stats_tensor)
        #print(self.train_stats_means)
        #print(self.train_stats_sd)
        self.norm_pred_stats          = Utilities.normalize(self.pred_stats_tensor, (self.train_stats_means, self.train_stats_sd))
        self.denormalized_pred_stats  = Utilities.denormalize(self.norm_pred_stats, self.train_stats_means, self.train_stats_sd)
        
        # CPI functions
        with open(self.model_cpi_func_fn, 'rb') as f:
            self.cpi_func = dill.load(f)

        return


    def make_results(self):

        # load model
        self.mymodel              = tf.keras.models.load_model(self.model_sav_fn)

        # get predictions
        self.norm_preds                = self.mymodel.predict([self.pred_data_tensor, self.pred_stats_tensor])
        self.denormalized_pred_labels  = Utilities.denormalize(self.norm_preds, self.train_labels_means, self.train_labels_sd)
        self.pred_labels               = np.exp( self.denormalized_pred_labels )

        # get CIs
        #self.cpi_vals = {}
        self.pred_lower_CPI = []
        self.pred_upper_CPI = []
        for i,k in enumerate(self.param_names):
            # MJL: we pass the predicted parameter vector into the function
            #      each cpi_func is trained for a different parameter
            #self.cpi_vals[k] = {}
            #lower_val = self.cpi_func[k]['lower']( self.pred_labels[i] )
            #upper_val = self.cpi_func[k]['upper']( self.pred_labels[i] )
            if k in self.cpi_func:
                #print(self.norm_preds.shape)
                #print(self.norm_pred_stats.shape)
                x_pred = self.denormalized_pred_labels[:,[i]]  # denormalized_pred_labels????
                x_stat = self.denormalized_pred_stats[:,0:2]
                self.pred_cpi_val = np.hstack( [x_pred, x_stat] )

                lower_val = self.cpi_func[k]['lower']( self.pred_cpi_val )
                upper_val = self.cpi_func[k]['upper']( self.pred_cpi_val )

                self.pred_lower_CPI.append( np.exp( lower_val[0] ) )
                self.pred_upper_CPI.append( np.exp( upper_val[0] ) )
            else:
                self.pred_lower_CPI.append(0.)
                self.pred_upper_CPI.append(10.)
                #self.cpi_vals[k]['upper'] = self.cpi_func[k]['upper']( self.pred_labels[i] )
        #print(self.cpi_vals)
        #print(self.pred_stats_tensor[:,0:3])
        # save predictions
        #self.df_pred_labels = pd.DataFrame(self.pred_labels, columns=self.param_names, index=None)
        #self.df_pred_labels.to_csv(self.model_pred_fn, index=False, sep=',')
        #print(lower_val)
        self.df_pred_all_labels = pd.DataFrame()
        self.df_pred_all_labels['name'] = self.param_names
        self.df_pred_all_labels['estimate'] = self.pred_labels.flatten()
        self.df_pred_all_labels['lower_CPI'] = self.pred_lower_CPI
        self.df_pred_all_labels['upper_CPI'] = self.pred_upper_CPI
        self.df_pred_all_labels.to_csv(self.model_pred_fn, index=False, sep=',')

        return
    
