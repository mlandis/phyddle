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
import h5py

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
        self.args = args
        step_args = util.make_step_args('E', args)
        for k,v in step_args.items():
            setattr(self, k, v)

        # self.args          = args
        # self.verbose       = args['verbose']
        # self.trn_dir       = args['trn_dir']
        # self.est_dir       = args['est_dir']
        # self.trn_proj      = args['trn_proj']
        # self.est_proj      = args['est_proj']
        # self.est_prefix    = args['est_prefix']
        # self.batch_size    = args['trn_batch_size']
        # self.num_char      = args['num_char']
        # self.num_states    = args['num_states']
        # self.num_epochs    = args['num_epochs']
        # self.tree_width    = args['tree_width']
        # self.tree_encode   = args['tree_encode']
        # self.char_encode   = args['char_encode']
        # self.brlen_encode  = args['brlen_encode']
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
        self.fmt_proj_dir           = f'{self.fmt_dir}/{self.fmt_proj}'

        # prefixes
        test_prefix            = f'test.nt{self.tree_width}'
        network_prefix           = f'network_nt{self.tree_width}'
        #network_prefix           = f'train_batchsize{self.trn_batch_size}_numepoch{self.num_epochs}_nt{self.tree_width}'
        self.trn_prefix_dir         = f'{self.trn_proj_dir}/{network_prefix}'
        self.est_prefix_dir         = f'{self.est_proj_dir}/{self.est_prefix}'
        self.fmt_prefix_dir         = f'{self.fmt_proj_dir}/{test_prefix}'

        # model files
        self.model_sav_fn           = f'{self.trn_prefix_dir}.hdf5'
        self.train_labels_norm_fn   = f'{self.trn_prefix_dir}.train_label_norm.csv'
        self.train_aux_data_norm_fn = f'{self.trn_prefix_dir}.train_aux_data_norm.csv'
        self.model_cpi_fn           = f'{self.trn_prefix_dir}.cpi_adjustments.csv'

        # simulated test datasets for csv or hdf5
        self.test_phy_data_fn      = f'{self.fmt_prefix_dir}.phy_data.csv'
        self.test_aux_data_fn      = f'{self.fmt_prefix_dir}.aux_data.csv'
        self.test_labels_fn        = f'{self.fmt_prefix_dir}.labels.csv'
        self.test_hdf5_fn          = f'{self.fmt_prefix_dir}.hdf5'

        # empirical test dataset
        self.est_summ_stat_fn       = f'{self.est_prefix_dir}.summ_stat.csv'
        self.est_known_param_fn     = f'{self.est_prefix_dir}.known_param.csv'
        self.est_aux_data_fn        = f'{self.est_prefix_dir}.aux_data.csv'
        self.est_phy_data_fn        = f'{self.est_prefix_dir}.phy_data.csv'    
        
        # test outputs
        self.out_emp_label_est_fn   = f'{self.est_prefix_dir}.emp_est.labels.csv'
        self.out_test_label_est_fn   = f'{self.est_prefix_dir}.test_est.labels.csv'
        self.out_test_label_true_fn  = f'{self.est_prefix_dir}.test_true.labels.csv'
    
        # check if empirical dataset exists
        self.emp_input_exists = True
        for fn in [ self.est_summ_stat_fn,
                    self.est_phy_data_fn ]:
            if not os.path.exists(fn):
                self.emp_input_exists = False

        # done
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

        # start time
        start_time,start_time_str = util.get_time()
        util.print_str(f'▪ Start time of {start_time_str}', verbose)

        # load input
        util.print_str('▪ Loading input', verbose)
        self.load_input()

        # make estimates
        util.print_str('▪ Making estimates', verbose)
        self.make_results()
        
        # end time
        end_time,end_time_str = util.get_time()
        run_time = util.get_time_diff(start_time, end_time)
        # util.print_str(f'▪ End time:     {end_time_str}', verbose)
        util.print_str(f'▪ End time of {end_time_str} (+{run_time})', verbose)

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
        self.emp_phy_data = None
        if self.emp_input_exists:
            self.emp_phy_data = pd.read_csv(self.est_phy_data_fn,
                                            header=None, sep=',',
                                            index_col=False).to_numpy()
            self.emp_phy_data = self.emp_phy_data.reshape((1, -1, self.num_data_row))

        # denormalization factors for new aux data
        train_aux_data_norm = pd.read_csv(self.train_aux_data_norm_fn, sep=',', index_col=False)
        train_aux_data_means = train_aux_data_norm['mean'].T.to_numpy().flatten()
        train_aux_data_sd = train_aux_data_norm['sd'].T.to_numpy().flatten()
        self.train_aux_data_mean_sd = (train_aux_data_means, train_aux_data_sd)

        # denormalization factors for labels
        train_labels_norm = pd.read_csv(self.train_labels_norm_fn, sep=',', index_col=False)
        train_labels_means = train_labels_norm['mean'].T.to_numpy().flatten()
        train_labels_sd = train_labels_norm['sd'].T.to_numpy().flatten()
        self.train_labels_mean_sd = (train_labels_means, train_labels_sd)

        # get param_names from training labels
        self.aux_data_names = train_aux_data_norm['name'].to_list()
        self.label_names = train_labels_norm['name'].to_list()
        
        # read & normalize new aux data (when files exist)
        self.emp_aux_data = None
        
        if self.emp_input_exists:
            self.est_summ_stat = pd.read_csv(self.est_summ_stat_fn, sep=',',
                                            index_col=False).to_numpy().flatten()
            try:
                self.emp_known_params = pd.read_csv(self.est_known_param_fn,
                                                    sep=',', index_col=False).to_numpy().flatten()
                self.emp_aux_data = np.concatenate([self.est_summ_stat,
                                                    self.emp_known_params])
            except FileNotFoundError:
                self.emp_aux_data = self.est_summ_stat

            # reshape and rescale new aux data
            self.emp_aux_data.shape = (1, -1)
            self.norm_emp_aux_data = util.normalize(self.emp_aux_data,
                                                    self.train_aux_data_mean_sd)
        
        # read in CQR interval adjustments
        self.cpi_adjustments = pd.read_csv(self.model_cpi_fn, sep=',', index_col=False).to_numpy()
        

        # TEST DATA
        # load all the test dataset
        if self.tensor_format == 'csv':
            test_phy_data = pd.read_csv(self.test_phy_data_fn, header=None,
                                        on_bad_lines='skip').to_numpy()
            test_aux_data = pd.read_csv(self.test_aux_data_fn, header=None,
                                        on_bad_lines='skip').to_numpy()
            test_labels   = pd.read_csv(self.test_labels_fn, header=None,
                                        on_bad_lines='skip').to_numpy()
            # self.aux_data_names = test_aux_data[0,:]
            # self.label_names    = test_labels[0,:]
            test_aux_data       = test_aux_data[1:,:].astype('float64')
            test_labels         = test_labels[1:,:].astype('float64')   

        elif self.tensor_format == 'hdf5':
            hdf5_file = h5py.File(self.test_hdf5_fn, 'r')
            # self.aux_data_names = [ s.decode() for s in hdf5_file['aux_data_names'][0,:] ]
            # self.label_names    = [ s.decode() for s in hdf5_file['label_names'][0,:] ]
            test_phy_data       = pd.DataFrame(hdf5_file['phy_data']).to_numpy()
            test_aux_data       = pd.DataFrame(hdf5_file['aux_data']).to_numpy()
            test_labels         = pd.DataFrame(hdf5_file['labels']).to_numpy()
            hdf5_file.close()

        num_sample = test_phy_data.shape[0]
        test_phy_data.shape = (num_sample, -1, self.num_data_row)

        # create phylogenetic data tensors
        self.test_phy_data = test_phy_data
        # self.test_aux_data = test_aux_data
        self.norm_test_aux_data = util.normalize(test_aux_data,
                                                 self.train_aux_data_mean_sd)
        # self.norm_test_labels = util.normalize(test_labels,
        #                                        self.train_labels_mean_sd)
        self.test_label_true = test_labels

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

        # empirical dataset (if it exists)
        if self.emp_input_exists:

            # get estimates
            norm_emp_label_est = self.mymodel.predict([self.emp_phy_data,
                                                       self.norm_emp_aux_data])
            
            # point estimates & CPIs for emp. labels
            norm_emp_label_est        = np.array(norm_emp_label_est)
            norm_emp_label_est[1,:,:] = norm_emp_label_est[1,:,:] - self.cpi_adjustments[0,:]
            norm_emp_label_est[2,:,:] = norm_emp_label_est[2,:,:] + self.cpi_adjustments[1,:]
            
            # detransform results
            # log_emp_label_est = util.denormalize(norm_emp_label_est,
            #                                      self.train_labels_mean_sd[0],
            #                                      self.train_labels_mean_sd[1])
            # log_emp_label_est[log_emp_label_est > 300.] = 300.
            # emp_label_est = np.exp(log_emp_label_est)
            emp_label_est = util.denormalize(norm_emp_label_est,
                                             self.train_labels_mean_sd,
                                             exp=True)

            # save empirial label estimates
            df_emp_label_est = util.make_param_VLU_mtx(emp_label_est, self.label_names)
            df_emp_label_est.to_csv(self.out_emp_label_est_fn, index=False, sep=',')
            
            # save empirical auxiliary dataset
            df_emp_aux_data = pd.DataFrame(self.emp_aux_data, columns=self.aux_data_names)
            df_emp_aux_data.to_csv(self.est_aux_data_fn, index=False, sep=',')


        # test dataset
        norm_test_label_est = self.mymodel.predict([self.test_phy_data,
                                                    self.norm_test_aux_data])
        
        # point estimates & CPIs for test labels
        norm_test_label_est = np.array(norm_test_label_est)
        norm_test_label_est[1,:,:] = norm_test_label_est[1,:,:] - self.cpi_adjustments[0,:]
        norm_test_label_est[2,:,:] = norm_test_label_est[2,:,:] + self.cpi_adjustments[1,:]
        
        # detransform test label estimates
        # log_test_label_est = util.denormalize(norm_test_label_est,
        #                                       self.train_labels_mean_sd[0],
        #                                       self.train_labels_mean_sd[1])
        # log_test_label_est[log_test_label_est > 300.] = 300.
        # test_label_est = np.exp(log_test_label_est)
        test_label_est = util.denormalize(norm_test_label_est,
                                          self.train_labels_mean_sd,
                                          exp=True)

        #save test estimates
        df_test_label_est = util.make_param_VLU_mtx(test_label_est, self.label_names)
        df_test_label_est.to_csv(self.out_test_label_est_fn, index=False, sep=',')
        df_test_label_true = pd.DataFrame(self.test_label_true, columns=self.label_names)
        df_test_label_true.to_csv(self.out_test_label_true_fn, index=False, sep=',')
        
        # done
        return
    
#------------------------------------------------------------------------------#


        # # scatter of estimate vs true for test data
        # self.normalized_test_ests        = self.mymodel.predict([self.test_phy_data_tensor, self.test_sux_data_tensor])
        # self.normalized_test_ests        = np.array(self.normalized_test_ests)
        # self.denormalized_test_ests      = util.denormalize(self.normalized_test_ests, self.train_label_means, self.train_label_sd)
        # self.denormalized_test_ests      = np.exp(self.denormalized_test_ests)
        # self.denormalized_test_labels    = util.denormalize(self.norm_test_labels, self.train_label_means, self.train_label_sd)
        # self.denormalized_test_labels    = np.exp(self.denormalized_test_labels)
        
        # # test predictions with calibrated CQR CIs
        # self.denorm_test_ests_calib        = self.normalized_test_ests
        # self.denorm_test_ests_calib[1,:,:] = self.denorm_test_ests_calib[1,:,:] - self.cpi_adjustments[0,:]
        # self.denorm_test_ests_calib[2,:,:] = self.denorm_test_ests_calib[2,:,:] + self.cpi_adjustments[1,:]
        # self.denorm_test_ests_calib        = util.denormalize(self.denorm_test_ests_calib, self.train_label_means, self.train_label_sd)
        # self.denorm_test_ests_calib        = np.exp(self.denorm_test_ests_calib)

        #  # test scatterplot results (Value, Lower, Upper)
        # df_test_est_nocalib  = util.make_param_VLU_mtx(self.denormalized_test_ests[0:max_idx,:], self.label_names )
        # df_test_est_calib    = util.make_param_VLU_mtx(self.denorm_test_ests_calib[0:max_idx,:], self.label_names )
        
        # # save train/test labels
        # df_test_labels   = pd.DataFrame( self.denormalized_test_labels[0:max_idx,:], columns=self.label_names )

        # # convert to csv and save
        # df_test_est_nocalib.to_csv(self.test_est_nocalib_fn, index=False, sep=',')
        # df_test_est_calib.to_csv(self.test_est_calib_fn, index=False, sep=',')
        # df_test_labels.to_csv(self.test_labels_fn, index=False, sep=',')
        