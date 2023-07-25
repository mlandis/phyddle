#!/usr/bin/env python
"""
estimate
========
Defines classes and methods for the Estimate step, which loads a pre-trained
model and uses it to make e.g. parameter estimates for e.g. a
new empirical dataset.

Authors:   Michael Landis and Ammon Thompson
Copyright: (c) 2022-2023, Michael Landis and Ammon Thompson
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
from phyddle import utilities
#from Formatting import encode_phy_tensor

#------------------------------------------------------------------------------#

def load(args):
    """
    Load function that takes in arguments dictionary and returns a Estimator object.

    Parameters:
    args (dict): A dictionary containing the arguments.

    Returns:
    Estimator: A Estimator object if est_method is 'default', None otherwise.
    """
    #sim_method = args['trn_objective']
    est_method = 'default'
    if est_method == 'default':
        return Estimator(args)
    else:
        return None

#------------------------------------------------------------------------------#

class Estimator:
    """A class for performing estimations."""

    def __init__(self, args):
        """Initialize the Estimator object.
        
        Args:
            args (dict): A dictionary containing the arguments.

        Returns:
            None
        """
        self.set_args(args)
        self.prepare_files()
        self.logger = utilities.Logger(args)
        return
    
    def set_args(self, args):
        """Set the arguments for the Estimator object.
        
        Args:
            args (dict): A dictionary containing the arguments.

        Returns:
            None
        """
        self.args              = args
        self.proj              = args['proj']
        self.verbose           = args['verbose']
        self.trn_dir           = args['trn_dir']
        self.est_dir           = args['est_dir']
        self.est_prefix        = args['est_prefix']
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
        """Prepare the necessary files for estimation.

        Returns:
            None
        """
        # main directories
        self.trn_proj_dir           = f'{self.trn_dir}/{self.proj}'
        self.est_proj_dir           = f'{self.est_dir}/{self.proj}'

        # main job filenames
        self.model_prefix           = f'sim_batchsize{self.batch_size}_numepoch{self.num_epochs}_nt{self.tree_width}'
        self.model_sav_fn           = f'{self.trn_proj_dir}/{self.model_prefix}.hdf5'
        self.model_trn_lbl_norm_fn  = f'{self.trn_proj_dir}/{self.model_prefix}.train_label_norm.csv'
        self.model_trn_ss_norm_fn   = f'{self.trn_proj_dir}/{self.model_prefix}.train_summ_stat_norm.csv'
        self.model_cpi_fn           = f'{self.trn_proj_dir}/{self.model_prefix}.cpi_adjustments.csv'

        # save estimates to file
        self.model_est_fn          = f'{self.est_proj_dir}/{self.est_prefix}.{self.model_prefix}.est_labels.csv'

        # test summ stats
        self.est_summ_stat_fn       = f'{self.est_proj_dir}/{self.est_prefix}.summ_stat.csv'
        self.est_known_param_fn     = f'{self.est_proj_dir}/{self.est_prefix}.known_param.csv'

        # test phy vector
        if self.tree_type == 'extant':
            self.est_phyvec_fn      = f'{self.est_proj_dir}/{self.est_prefix}.cdvs.csv'    
        elif self.tree_type == 'serial':
            self.est_phyvec_fn      = f'{self.est_proj_dir}/{self.est_prefix}.cblvs.csv'
        else:
            raise NotImplementedError(f'{self.tree_type} not recognized tree type')

        self.num_tree_row = utilities.get_num_tree_row(self.tree_type, self.tree_encode_type)
        self.num_char_row = utilities.get_num_char_row(self.char_encode_type, self.num_char, self.num_states)
        self.num_data_row = self.num_tree_row + self.num_char_row
        
        return

    def run(self):
        """
        Runs the estimation process.

        Args:
            None

        Returns:
            None
        """
        
        utilities.print_step_header('est', self.proj, [self.est_dir, self.trn_dir], self.est_dir, verbose=self.verbose)        

        os.makedirs(self.est_proj_dir, exist_ok=True)

        utilities.print_str('▪ loading input ...', verbose=self.verbose)
        self.load_input()

        utilities.print_str('▪ making estimation ...', verbose=self.verbose)
        self.make_results()

        utilities.print_str('... done!', verbose=self.verbose)
        
    def load_input(self):
        """
        Loads the input data for estimation and performs necessary preprocessing.

        Args:
            None

        Returns:
            None
        """
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
        self.est_data_tensor    = pd.read_csv(self.est_phyvec_fn, header=None, sep=',', index_col=False).to_numpy()
        #self.est_data_tensor.shape  = ( 1, -1, (self.num_tree_row+self.num_char_row) )
        self.est_data_tensor    = self.est_data_tensor.reshape( (1, -1, (self.num_tree_row+self.num_char_row)) )
        
        # read & normalize new aux data
        self.est_summ_stats     = pd.read_csv(self.est_summ_stat_fn, sep=',', index_col=False).to_numpy().flatten()
        try:
            self.est_known_params   = pd.read_csv(self.est_known_param_fn, sep=',', index_col=False).to_numpy().flatten()
            self.est_auxdata_tensor = np.concatenate( [self.est_summ_stats, self.est_known_params] )
        except FileNotFoundError:
            self.est_auxdata_tensor = self.est_summ_stats
            
        self.est_auxdata_tensor.shape = ( 1, -1 )

        #print(self.est_auxdata_tensor)

        self.norm_est_stats          = utilities.normalize(self.est_auxdata_tensor, (self.train_stats_means, self.train_stats_sd))
        self.denormalized_est_stats  = utilities.denormalize(self.norm_est_stats, self.train_stats_means, self.train_stats_sd)


        # read in CQR interval adjustments
        self.cpi_adjustments = pd.read_csv(self.model_cpi_fn, sep=',', index_col=False).to_numpy()
        # # CPI functions
        # with open(self.model_cpi_func_fn, 'rb') as f:
        #     self.cpi_func = dill.load(f)

        return


    def make_results(self):
        """
        Load a trained model, generate estimates, denormalize the estimates,
        apply adjustments, and output them to a CSV file.

        Returns:
            None
        """
        # load model
        self.mymodel = tf.keras.models.load_model(self.model_sav_fn, compile=False)

        # get estimates
        self.norm_est                = self.mymodel.predict([self.est_data_tensor, self.norm_est_stats])
        self.norm_est                = np.array( self.norm_est )
        self.norm_est[1,:,:]         = self.norm_est[1,:,:] - self.cpi_adjustments[0,:]
        self.norm_est[2,:,:]         = self.norm_est[2,:,:] + self.cpi_adjustments[1,:]
        self.denormalized_est_labels  = utilities.denormalize(self.norm_est, self.train_labels_means, self.train_labels_sd)
        #print(self.denormalized_est_labels)
        self.denormalized_est_labels[ self.denormalized_est_labels > 300. ] = 300.
        self.est_labels               = np.exp( self.denormalized_est_labels )
        #print(np.exp( self.denormalized_est_labels ))
       
        # output estimates
        self.df_est_all_labels = utilities.make_param_VLU_mtx(self.est_labels, self.param_names)
        self.df_est_all_labels.to_csv(self.model_est_fn, index=False, sep=',')

        return
    
