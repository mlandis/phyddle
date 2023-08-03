#!/usr/bin/env python
"""
estimate
========
Defines classes and methods for the Estimate step, which loads a pre-trained
network and uses it to generate new estimates, e.g. estimate model parmaeters
for a new empirical dataset.

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

# phyddle imports
from phyddle import utilities as util

#------------------------------------------------------------------------------#

def load(args):
    """
    Load an Estimator object.

    This function creates an instance of the Estimator class, initialized using
    phyddle settings stored in args (dict).

    Args:
        args (dict): Contains phyddle settings.
    """

    # load object
    est_method = 'default'
    if est_method == 'default':
        return Estimator(args)
    else:
        return NotImplementedError

#------------------------------------------------------------------------------#

class Estimator:
    """
    Class for making neural network estimates (i.e. label predictions) from new
    (e.g. empirical) phylogenetic datasets. This class requires a trained
    network from Train and input processed by Format. Output is written to
    file and can be visualized using Plot.
    """

    def __init__(self, args):
        """
        Initializes a new Simulator object.

        Args:
            args (dict): Contains phyddle settings.
        """
        # initialize with phyddle settings
        self.set_args(args)
        # construct filepaths
        self.prepare_filepaths()
        # get size of CPV+S tensors
        self.num_tree_row = util.get_num_tree_row(self.tree_encode,
                                                  self.brlen_encode)
        self.num_char_row = util.get_num_char_row(self.char_encode,
                                                  self.num_char,
                                                  self.num_states)
        self.num_data_row = self.num_tree_row + self.num_char_row
        # create logger to track runtime info
        self.logger = util.Logger(args)
        # done
        return
    
    def set_args(self, args):
        """
        Assigns phyddle settings as Estimator attributes.

        Args:
            args (dict): Contains phyddle settings.
        """
        # estimator arguments
        self.args          = args
        self.verbose       = args['verbose']
        self.trn_dir       = args['trn_dir']
        self.est_dir       = args['est_dir']
        self.trn_proj      = args['trn_proj']
        self.est_proj      = args['est_proj']
        self.est_prefix    = args['est_prefix']
        self.batch_size    = args['batch_size']
        self.num_char      = args['num_char']
        self.num_states    = args['num_states']
        self.num_epochs    = args['num_epochs']
        self.tree_width    = args['tree_width']
        self.tree_encode   = args['tree_encode']
        self.char_encode   = args['char_encode']
        self.brlen_encode  = args['brlen_encode']
        return
    
    def prepare_filepaths(self):
        """
        Prepare filepaths for the project.

        This script generates all the filepaths for input and output based off
        of Trainer attributes. The Format and Train directories are input and
        the Estimate directory is used for both input and output.

        Returns: None
        """
        # main directories
        self.trn_proj_dir           = f'{self.trn_dir}/{self.trn_proj}'
        self.est_proj_dir           = f'{self.est_dir}/{self.est_proj}'

        # prefixes
        self.model_prefix           = f'train_batchsize{self.batch_size}_numepoch{self.num_epochs}_nt{self.tree_width}'
        self.trn_prefix_dir         = f'{self.trn_proj_dir}/{self.model_prefix}'
        self.est_prefix_dir         = f'{self.est_proj_dir}/{self.est_prefix}'

        # model files
        self.model_sav_fn           = f'{self.trn_prefix_dir}.hdf5'
        self.train_labels_norm_fn   = f'{self.trn_prefix_dir}.train_label_norm.csv'
        self.train_aux_data_norm_fn = f'{self.trn_prefix_dir}.train_aux_data_norm.csv'
        self.model_cpi_fn           = f'{self.trn_prefix_dir}.cpi_adjustments.csv'

        # save estimates to file
        self.model_est_fn           = f'{self.est_prefix_dir}.{self.model_prefix}.est_labels.csv'

        # test summ stats
        self.est_aux_data_fn        = f'{self.est_prefix_dir}.summ_stat.csv'
        self.est_known_param_fn     = f'{self.est_prefix_dir}.known_param.csv'

        # test phy vector
        self.est_phy_data_fn        = f'{self.est_prefix_dir}.phy_data.csv'    
        #if self.tree_encode == 'extant':
        #    self.est_phy_data_fn      = f'{self.est_prefix_dir}.phy_data.csv'    
        #elif self.tree_encode == 'serial':
        #    self.est_phy_data_fn      = f'{self.est_prefix_dir}.phy_data.csv'
        #else:
        #    error_msg = f'{self.tree_encode} not recognized tree type'
        #    raise NotImplementedError(error_msg)

        return

    def run(self):
        """
        Executes all simulations.

        This method prints status updates, creates the target directory for new
        simulations, then runs all simulation jobs.
        
        Simulation jobs are numbered by the replicate-index list (self.rep_idx). 
        Each job is executed by calling self.sim_one(idx) where idx is a unique
        value in self.rep_idx.
 
        When self.use_parallel is True then all jobs are run in parallel via
        multiprocessing.Pool. When self.use_parallel is false, jobs are run
        serially with one CPU.
        """
        verbose = self.verbose

        # print header
        util.print_step_header('est', [self.est_proj_dir, self.trn_proj_dir],
                               self.est_proj_dir, verbose)
        
        # prepare workspace
        os.makedirs(self.est_proj_dir, exist_ok=True)

        # load input
        util.print_str('▪ Loading input ...', verbose)
        self.load_input()

        # make estimates
        util.print_str('▪ Making estimates ...', verbose)
        self.make_results()

        # done
        util.print_str('... done!', verbose)
        
    def load_input(self):
        """
        Load input data for estimation.

        This function loads input from Train and Estimate. From Train, it
        imports the trained network, scaling factors for the aux. data and
        labels. It also loads the phy. data and aux. data tensors stored in the
        Estimate job directory.
        
        The script re-normalizes the new estimation to match the scale/location
        used for simulated training examples to train the network.
        """
        # read & reshape new phylo-state data
        self.est_data_tensor = pd.read_csv(self.est_phy_data_fn, header=None, sep=',', index_col=False).to_numpy()
        self.est_data_tensor = self.est_data_tensor.reshape((1, -1, self.num_data_row))

        # denormalization factors for new aux data
        train_aux_data_norm = pd.read_csv(self.train_aux_data_norm_fn, sep=',', index_col=False)
        self.train_aux_data_means = train_aux_data_norm['mean'].T.to_numpy().flatten()
        self.train_aux_data_sd = train_aux_data_norm['sd'].T.to_numpy().flatten()

        # denormalization factors for labels
        train_labels_norm = pd.read_csv(self.train_labels_norm_fn, sep=',', index_col=False)
        self.train_labels_means = train_labels_norm['mean'].T.to_numpy().flatten()
        self.train_labels_sd = train_labels_norm['sd'].T.to_numpy().flatten()

        # get param_names from training labels
        self.param_names = train_labels_norm['name'].to_list()
        # self.aux_data_names     = train_aux_data_norm['name'].to_list()
        
        # read & normalize new aux data (when files exist)
        self.est_aux_data = pd.read_csv(self.est_aux_data_fn, sep=',', index_col=False).to_numpy().flatten()
        try:
            self.est_known_params = pd.read_csv(self.est_known_param_fn, sep=',', index_col=False).to_numpy().flatten()
            self.est_auxdata_tensor = np.concatenate([self.est_aux_data, self.est_known_params])
        except FileNotFoundError:
            self.est_auxdata_tensor = self.est_aux_data

        # reshape and rescale new aux data
        self.est_auxdata_tensor.shape = (1, -1)
        self.norm_est_aux_data = util.normalize(self.est_auxdata_tensor,
                                                (self.train_aux_data_means, self.train_aux_data_sd))
        # self.denormalized_est_aux_data  = util.denormalize(self.norm_est_aux_data, self.train_aux_data_means, self.train_aux_data_sd)

        # read in CQR interval adjustments
        self.cpi_adjustments = pd.read_csv(self.model_cpi_fn, sep=',', index_col=False).to_numpy()
        
        return


    def make_results(self):
        """
        Makes all results for the Estimate step.

        This function loads a trained model from the Train stem, then uses it
        to perform the estimation task. For example, the step might estimate all
        model parameter values and adjusted lower and upper CPI bounds. This step 
        """
        # load model
        self.mymodel = tf.keras.models.load_model(self.model_sav_fn, compile=False)
        # get estimates
        self.norm_est = self.mymodel.predict([self.est_data_tensor, self.norm_est_aux_data])
        # point estimates
        self.norm_est = np.array( self.norm_est )
        # adjust lower CPI
        self.norm_est[1,:,:] = self.norm_est[1,:,:] - self.cpi_adjustments[0,:]
        # adjust upper CPI
        self.norm_est[2,:,:] = self.norm_est[2,:,:] + self.cpi_adjustments[1,:]
        # denormalize results
        self.denormalized_est_labels = util.denormalize(self.norm_est,
                                                        self.train_labels_means,
                                                        self.train_labels_sd)
        # avoid overflow
        self.denormalized_est_labels[self.denormalized_est_labels > 300.] = 300.
        # revert from log to linear scalle
        self.est_labels = np.exp( self.denormalized_est_labels )
        # convert parameters to param table
        self.df_est_all_labels = util.make_param_VLU_mtx(self.est_labels,
                                                         self.param_names)
        # save results to file
        self.df_est_all_labels.to_csv(self.model_est_fn, index=False, sep=',')
        # done
        return
    
#------------------------------------------------------------------------------#