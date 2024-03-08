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
        
        # dimensions
        self.tree_encode        = str(args['tree_encode'])
        self.char_encode        = str(args['char_encode'])
        self.brlen_encode       = str(args['brlen_encode'])
        self.tensor_format      = str(args['tensor_format'])
        self.num_char           = int(args['num_char'])
        self.num_states         = int(args['num_states'])
        self.log_offset         = float(args['log_offset'])
        
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
        self.train_aux_data_mean_sd  = None       # init in load_train_input()
        self.train_labels_mean_sd    = None       # init in load_train_input()
        self.label_names             = None       # init in load_train_input()
        self.cpi_adjustments         = None       # init in load_train_input()
        self.phy_data                = None       # init in load_format_input()
        self.aux_data                = None       # init in load_format_input()
        self.labels                  = None       # init in load_format_input()
        self.mymodel                 = None       # init in make_results()
        
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

        # load Train input
        self.load_train_input()

        found_sim = False
        if self.has_valid_dataset(mode='sim'):
            found_sim = True

            # load input
            # todo: load input based on mode
            util.print_str('▪ Loading simulated test input', verbose)
            self.load_format_input(mode='sim')
    
            # make estimates
            # todo: make estimates based on mode
            util.print_str('▪ Making simulated test estimates', verbose)
            self.make_results(mode='sim')

        found_emp = False
        if self.has_valid_dataset(mode='emp'):
            found_emp = True
            # load input
            # todo: load input based on mode
            util.print_str('▪ Loading empirical input', verbose)
            self.load_format_input(mode='emp')
    
            # make estimates
            # todo: make estimates based on mode
            util.print_str('▪ Making empirical estimates', verbose)
            self.make_results(mode='emp')
        
        if not found_sim and not found_emp:
            util.print_str('No simulated test or empirical datasets found. '
                           'Check config settings.', verbose)
            return
        
        # end time
        end_time,end_time_str = util.get_time()
        run_time = util.get_time_diff(start_time, end_time)
        util.print_str(f'▪ End time of {end_time_str} (+{run_time})', verbose)

        # done
        util.print_str('... done!', verbose)
        return

    def has_valid_dataset(self, mode='sim'):
        """Determines if empirical analysis is being performed.
        
        Args:
            mode (str): 'sim' or 'emp' for simulated or empirical analysis.
            
        Returns:
            bool: True if empirical analysis is being performed.
        """

        assert mode in ['sim', 'emp']
        
        # check if empirical directory exists
        if not os.path.exists(self.fmt_dir):
            return False

        # check if empirical directory contains files
        fn = ''
        if mode == 'sim':
            fn = f'{self.fmt_dir}/{self.fmt_prefix}.test.hdf5'
        elif mode == 'emp':
            fn = f'{self.fmt_dir}/{self.fmt_prefix}.empirical.hdf5'

        if os.path.exists(fn):
            return True
        else:
            return False
    
    def load_train_input(self):
        """Load input data for estimation.

        This function loads input from Train and Estimate. From Train, it
        imports the trained network, scaling factors for the aux. data and
        labels. It also loads the phy. data and aux. data tensors stored in the
        Estimate job directory.
        
        The script re-normalizes the new estimation to match the scale/location
        used for simulated training examples to train the network.
            
        """
        # filesystem
        path_prefix = f'{self.trn_dir}/{self.trn_prefix}'
        train_aux_data_norm_fn = f'{path_prefix}.train_aux_data_norm.csv'
        train_labels_norm_fn = f'{path_prefix}.train_label_norm.csv'
        model_cpi_fn = f'{path_prefix}.cpi_adjustments.csv'

        # denormalization factors for new aux data
        train_aux_data_norm = pd.read_csv(train_aux_data_norm_fn, sep=',', index_col=False)
        train_aux_data_means = train_aux_data_norm['mean'].T.to_numpy().flatten()
        train_aux_data_sd = train_aux_data_norm['sd'].T.to_numpy().flatten()
        self.train_aux_data_mean_sd = (train_aux_data_means, train_aux_data_sd)

        # denormalization factors for labels
        train_labels_norm = pd.read_csv(train_labels_norm_fn, sep=',', index_col=False)
        train_labels_means = train_labels_norm['mean'].T.to_numpy().flatten()
        train_labels_sd = train_labels_norm['sd'].T.to_numpy().flatten()
        self.train_labels_mean_sd = (train_labels_means, train_labels_sd)

        # get param_names from training labels
        self.label_names = train_labels_norm['name'].to_list()
        
        # read in CQR interval adjustments
        self.cpi_adjustments = pd.read_csv(model_cpi_fn, sep=',', index_col=False).to_numpy()
        
        # done
        return

    def load_format_input(self, mode='sim'):
        """Load input data for estimation.

        This function loads the phy. data and aux. data tensors stored in the
        Format job directory.
        
        Args:
            mode (str): 'sim' or 'emp' for simulated or empirical analysis.
            
        """

        assert mode in ['sim', 'emp']
        
        path_prefix = ''
        if mode == 'sim':
            path_prefix = f'{self.fmt_dir}/{self.fmt_prefix}.test'
        elif mode == 'emp':
            path_prefix = f'{self.fmt_dir}/{self.fmt_prefix}.empirical'
        
        # simulated test datasets for csv or hdf5
        phy_data_fn = f'{path_prefix}.phy_data.csv'
        aux_data_fn = f'{path_prefix}.aux_data.csv'
        labels_fn = f'{path_prefix}.labels.csv'
        hdf5_fn = f'{path_prefix}.hdf5'
        
        # load all the test dataset
        phy_data = None
        aux_data = None
        labels = None
        if self.tensor_format == 'csv':
            phy_data = pd.read_csv(phy_data_fn, header=None,
                                        on_bad_lines='skip').to_numpy()
            aux_data = pd.read_csv(aux_data_fn, header=None,
                                        on_bad_lines='skip').to_numpy()
            if mode == 'sim':
                labels = pd.read_csv(labels_fn, header=None,
                                            on_bad_lines='skip').to_numpy()
            aux_data = aux_data[1:,:].astype('float64')
            labels = labels[1:,:].astype('float64')

        elif self.tensor_format == 'hdf5':
            hdf5_file = h5py.File(hdf5_fn, 'r')
            phy_data = pd.DataFrame(hdf5_file['phy_data']).to_numpy()
            aux_data = pd.DataFrame(hdf5_file['aux_data']).to_numpy()
            if mode == 'sim':
                labels = pd.DataFrame(hdf5_file['labels']).to_numpy()
            hdf5_file.close()
        
        # get number of samples
        num_sample = phy_data.shape[0]

        # reshape phylogenetic state tensor
        phy_data.shape = (num_sample, -1, self.num_data_col)
        phy_data = np.transpose(phy_data, axes=[0,2,1]).astype('float32')
        self.phy_data = phy_data

        # test dataset normalization
        assert aux_data.shape[0] == num_sample
        aux_data = np.log(aux_data + self.log_offset)
        self.aux_data = util.normalize(aux_data, self.train_aux_data_mean_sd)

        if mode == 'sim':
            assert labels.shape[0] == num_sample
            self.labels = labels

    def make_results(self, mode='sim'):
        """Makes all results for the Estimate step.

        This function loads a trained model from the Train stem, then uses it
        to perform the estimation task. For example, the step might estimate all
        model parameter values and adjusted lower and upper CPI bounds.

        Args:
            mode (str): 'sim' or 'emp' for simulated or empirical analysis.

        """

        # filesystem
        path_prefix = ''
        if mode == 'sim':
            path_prefix = f'{self.est_dir}/{self.est_prefix}.test'
        if mode == 'emp':
            path_prefix = f'{self.est_dir}/{self.est_prefix}.empirical'
            
        model_arch_fn = f'{self.trn_dir}/{self.trn_prefix}.trained_model.pkl'
        out_label_est_fn = f'{path_prefix}_est.labels.csv'
        out_label_true_fn = f'{path_prefix}_true.labels.csv'
    
        # load model
        self.mymodel = torch.load(model_arch_fn)

        # test dataset
        label_est = self.mymodel(torch.Tensor(self.phy_data),
                                      torch.Tensor(self.aux_data))

        # point estimates & CPIs for test labels
        label_est = torch.stack(label_est).detach().numpy()
        label_est[1,:,:] = label_est[1,:,:] - self.cpi_adjustments[0,:]
        label_est[2,:,:] = label_est[2,:,:] + self.cpi_adjustments[1,:]
        
        # denormalize test label estimates
        denorm_label_est = util.denormalize(label_est,
                                            self.train_labels_mean_sd,
                                            exp=True) - self.log_offset

        # save test estimates
        df_test_label_est = util.make_param_VLU_mtx(denorm_label_est, self.label_names)
        df_test_label_est.to_csv(out_label_est_fn, index=False, sep=',', float_format=util.PANDAS_FLOAT_FMT_STR)
        if mode == 'sim':
            df_test_label_true = pd.DataFrame(self.labels, columns=self.label_names)
            df_test_label_true.to_csv(out_label_true_fn, index=False, sep=',', float_format=util.PANDAS_FLOAT_FMT_STR)
        
        # done
        return
    
##################################################
