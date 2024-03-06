#!/usr/bin/env python
"""
estimate
========
Defines classes and methods for the Estimate step, which loads a pre-trained
network and uses it to generate new estimates, e.g. estimate model parameters
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
import h5py
import torch

# phyddle imports
from phyddle import utilities as util

##################################################


def load(args):
    """Load an Estimator object.

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

##################################################


class Estimator:
    """
    Class for making neural network estimates (i.e. label predictions) from new
    (e.g. empirical) phylogenetic datasets. This class requires a trained
    network from Train and input processed by Format. Output is written to
    file and can be visualized using Plot.
    """

    def __init__(self, args):
        """Initializes a new Simulator object.

        Args:
            args (dict): Contains phyddle settings.
            
        """
        
        # settings
        self.verbose            = bool(args['verbose'])
        self.emp_analysis       = bool(args['emp_analysis'])
        
        # filesystem
        self.trn_prefix         = str(args['trn_prefix'])
        self.fmt_prefix         = str(args['fmt_prefix'])
        self.est_prefix         = str(args['est_prefix'])
        self.trn_dir            = str(args['trn_dir'])
        self.fmt_dir            = str(args['fmt_dir'])
        self.est_dir            = str(args['est_dir'])
        self.log_dir            = str(args['log_dir'])
        
        # prefixes
        self.fmt_prefix_dir         = f'{self.fmt_dir}/{self.fmt_prefix}.test'
        self.est_prefix_dir         = f'{self.est_dir}/{self.est_prefix}.test'
        if self.emp_analysis:
            self.fmt_prefix_dir         = f'{self.fmt_dir}/{self.fmt_prefix}.empirical'
            self.est_prefix_dir         = f'{self.est_dir}/{self.est_prefix}.empirical'
        self.trn_prefix_dir         = f'{self.trn_dir}/{self.trn_prefix}'
        
        # dimensions
        self.tree_encode        = str(args['tree_encode'])
        self.char_encode        = str(args['char_encode'])
        self.brlen_encode       = str(args['brlen_encode'])
        self.tensor_format      = str(args['tensor_format'])
        self.num_char           = int(args['num_char'])
        self.num_states         = int(args['num_states'])
        self.log_offset         = float(args['log_offset'])

        # model files
        self.model_arch_fn          = f'{self.trn_prefix_dir}.trained_model.pkl'
        self.train_labels_norm_fn   = f'{self.trn_prefix_dir}.train_label_norm.csv'
        self.train_aux_data_norm_fn = f'{self.trn_prefix_dir}.train_aux_data_norm.csv'
        self.model_cpi_fn           = f'{self.trn_prefix_dir}.cpi_adjustments.csv'

        # simulated test datasets for csv or hdf5
        self.test_phy_data_fn       = f'{self.fmt_prefix_dir}.phy_data.csv'
        self.test_aux_data_fn       = f'{self.fmt_prefix_dir}.aux_data.csv'
        self.test_labels_fn         = f'{self.fmt_prefix_dir}.labels.csv'
        self.test_hdf5_fn           = f'{self.fmt_prefix_dir}.hdf5'

        # test outputs
        self.out_label_est_fn       = f'{self.est_prefix_dir}_est.labels.csv'
        self.out_label_true_fn      = f'{self.est_prefix_dir}_true.labels.csv'
        
        # get size of CPV+S tensors
        self.num_tree_col = util.get_num_tree_col(self.tree_encode,
                                                  self.brlen_encode)
        self.num_char_col = util.get_num_char_col(self.char_encode,
                                                  self.num_char,
                                                  self.num_states)
        self.num_data_col = self.num_tree_col + self.num_char_col
        
        # create logger to track runtime info
        self.logger = util.Logger(args)

        # initialized later
        self.train_labels_mean_sd = None       # init in load_input()
        self.label_names          = None       # init in load_input()
        self.test_phy_data        = None       # init in load_input()
        self.norm_test_aux_data   = None       # init in load_input()
        self.test_label_true      = None       # init in load_input()
        self.cpi_adjustments      = None       # init in load_input()
        self.mymodel              = None       # init in make_results()
        
        # done
        return
    
    def run(self):
        """Executes all estimation tasks.

        This method prints status updates, creates the target directory for new
        estimates, then runs all estimation jobs.

        Estimation tasks are performed against all entries in the test
        dataset and against a single dataset (typically assumed to be the
        empirical dataset).

        Estimation will load the trained network, predict point estimates
        and calibrated prediction intervals (CPIs), and save results to file.
        
        """
        verbose = self.verbose

        # print header
        util.print_step_header('est',
                               [self.fmt_dir, self.trn_dir],
                                self.est_dir, verbose)
        
        # prepare workspace
        os.makedirs(self.est_dir, exist_ok=True)

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
        util.print_str(f'▪ End time of {end_time_str} (+{run_time})', verbose)

        # done
        util.print_str('... done!', verbose)
        return
        
    def load_input(self):
        """Load input data for estimation.

        This function loads input from Train and Estimate. From Train, it
        imports the trained network, scaling factors for the aux. data and
        labels. It also loads the phy. data and aux. data tensors stored in the
        Estimate job directory.
        
        The script re-normalizes the new estimation to match the scale/location
        used for simulated training examples to train the network.

        """
        
        # INPUT FROM TRAIN STEP
        # Load factors for aux. data and label standardization;
        # Load factors for calibration step for prediction intervals

        # denormalization factors for new aux data
        train_aux_data_norm = pd.read_csv(self.train_aux_data_norm_fn, sep=',', index_col=False)
        train_aux_data_means = train_aux_data_norm['mean'].T.to_numpy().flatten()
        train_aux_data_sd = train_aux_data_norm['sd'].T.to_numpy().flatten()
        train_aux_data_mean_sd = (train_aux_data_means, train_aux_data_sd)

        # denormalization factors for labels
        train_labels_norm = pd.read_csv(self.train_labels_norm_fn, sep=',', index_col=False)
        train_labels_means = train_labels_norm['mean'].T.to_numpy().flatten()
        train_labels_sd = train_labels_norm['sd'].T.to_numpy().flatten()
        self.train_labels_mean_sd = (train_labels_means, train_labels_sd)

        # get param_names from training labels
        # self.aux_data_names = train_aux_data_norm['name'].to_list()
        self.label_names = train_labels_norm['name'].to_list()
        
        # read in CQR interval adjustments
        self.cpi_adjustments = pd.read_csv(self.model_cpi_fn, sep=',', index_col=False).to_numpy()
        
        # INPUT TEST DATA FROM FORMAT STEP
        # load all the test dataset
        test_phy_data = None
        test_aux_data = None
        test_labels = None
        if self.tensor_format == 'csv':
            test_phy_data = pd.read_csv(self.test_phy_data_fn, header=None,
                                        on_bad_lines='skip').to_numpy()
            test_aux_data = pd.read_csv(self.test_aux_data_fn, header=None,
                                        on_bad_lines='skip').to_numpy()
            if not self.emp_analysis:
                test_labels   = pd.read_csv(self.test_labels_fn, header=None,
                                            on_bad_lines='skip').to_numpy()
            test_aux_data = test_aux_data[1:,:].astype('float64')
            test_labels   = test_labels[1:,:].astype('float64')

        elif self.tensor_format == 'hdf5':
            hdf5_file           = h5py.File(self.test_hdf5_fn, 'r')
            test_phy_data       = pd.DataFrame(hdf5_file['phy_data']).to_numpy()
            test_aux_data       = pd.DataFrame(hdf5_file['aux_data']).to_numpy()
            if not self.emp_analysis:
                test_labels         = pd.DataFrame(hdf5_file['labels']).to_numpy()
            hdf5_file.close()

        # add assert statement to guarantee that num examples matches across
        # different test dataset components: phy. data, aux. data, labels
        assert test_phy_data.shape[0] == test_aux_data.shape[0]
        if not self.emp_analysis:
            assert test_phy_data.shape[0] == test_labels.shape[0]
        
        # reshape phylogenetic state tensor
        num_sample = test_phy_data.shape[0]
        test_phy_data.shape = (num_sample, -1, self.num_data_col)
        test_phy_data = np.transpose(test_phy_data, axes=[0,2,1]).astype('float32')
        self.test_phy_data = test_phy_data

        # test dataset normalization
        test_aux_data = np.log(test_aux_data + self.log_offset)
        self.norm_test_aux_data = util.normalize(test_aux_data,
                                                 train_aux_data_mean_sd)
        
        if not self.emp_analysis:
            self.test_label_true = test_labels
            
        # done
        return

    def make_results(self):
        """Makes all results for the Estimate step.

        This function loads a trained model from the Train stem, then uses it
        to perform the estimation task. For example, the step might estimate all
        model parameter values and adjusted lower and upper CPI bounds.

        """
        
        # load model
        self.mymodel = torch.load(self.model_arch_fn)

        # test dataset
        norm_test_label_est = self.mymodel(torch.Tensor(self.test_phy_data),
                                           torch.Tensor(self.norm_test_aux_data))

        # point estimates & CPIs for test labels
        norm_test_label_est        = torch.stack(norm_test_label_est).detach().numpy()
        # norm_test_label_est        = np.array(norm_test_label_est)
        norm_test_label_est[1,:,:] = norm_test_label_est[1,:,:] - self.cpi_adjustments[0,:]
        norm_test_label_est[2,:,:] = norm_test_label_est[2,:,:] + self.cpi_adjustments[1,:]
        
        # denormalize test label estimates
        test_label_est = util.denormalize(norm_test_label_est,
                                          self.train_labels_mean_sd,
                                          exp=True) - self.log_offset

        # save test estimates
        df_test_label_est = util.make_param_VLU_mtx(test_label_est, self.label_names)
        df_test_label_est.to_csv(self.out_label_est_fn, index=False, sep=',', float_format=util.PANDAS_FLOAT_FMT_STR)
        if not self.emp_analysis:
            df_test_label_true = pd.DataFrame(self.test_label_true, columns=self.label_names)
            df_test_label_true.to_csv(self.out_label_true_fn, index=False, sep=',', float_format=util.PANDAS_FLOAT_FMT_STR)
        
        # done
        return
    
##################################################
