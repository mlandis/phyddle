#!/usr/bin/env python
"""
plot
====
Defines classes and methods for the Plot step. Requires loading output files
from the Simulate, Train, and Estimate steps. Generates figures for previous
pipeline steps, then compiles them into a standard pdf report.

Authors:   Michael Landis and Ammon Thompson
Copyright: (c) 2022-2023, Michael Landis and Ammon Thompson
License:   MIT
"""

# standard imports
import os

# external imports
import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import scipy as sp
import torch
import torchview
from PIL import Image
from pypdf import PdfMerger
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# phyddle imports
from phyddle import utilities as util

##################################################


def load(args):
    """Load a Plotter object.

    This function creates an instance of the Plotter class, initialized using
    phyddle settings stored in args (dict).

    Args:
        args (dict): Contains phyddle settings.

    """
    
    # load object
    plot_method = 'default'
    if plot_method == 'default':
        return Plotter(args)
    else:
        return NotImplementedError

##################################################


class Plotter:
    """
    Class for generating figures from output produced by the Format, Train, and
    Estimate steps. Those outputs are processed and visualized in different ways
    e.g. scatterplot, PCA, KDE, etc. then saved as individual pdfs and one
    combined pdf.
    """
    def __init__(self, args):
        """Initializes a new Plotter object.

        Args:
           
             args (dict): Contains phyddle settings.
        """

        # settings
        self.verbose            = bool(args['verbose'])
        
        # filesystem
        self.sim_prefix            = str(args['sim_prefix'])
        self.emp_prefix            = str(args['emp_prefix'])
        self.fmt_prefix            = str(args['fmt_prefix'])
        self.trn_prefix            = str(args['trn_prefix'])
        self.est_prefix            = str(args['est_prefix'])
        self.plt_prefix            = str(args['plt_prefix'])
        self.fmt_dir               = str(args['fmt_dir'])
        self.emp_dir               = str(args['emp_dir'])
        self.trn_dir               = str(args['trn_dir'])
        self.est_dir               = str(args['est_dir'])
        self.plt_dir               = str(args['plt_dir'])
        self.log_dir               = str(args['log_dir'])

        # dataset info
        self.tensor_format         = str(args['tensor_format'])
        self.param_est             = dict(args['param_est'])
        self.plot_aux_color        = str(args['plot_aux_color'])
        self.plot_label_color      = str(args['plot_label_color'])
        self.plot_train_color      = str(args['plot_train_color'])
        self.plot_test_color       = str(args['plot_test_color'])
        self.plot_val_color        = str(args['plot_val_color'])
        self.plot_emp_color        = str(args['plot_emp_color'])
        self.log_offset            = float(args['log_offset'])
        self.cpi_coverage          = float(args['cpi_coverage'])
        self.plot_num_scatter      = int(args['plot_num_scatter'])
        self.plot_min_emp          = int(args['plot_min_emp'])
        self.plot_num_emp          = int(args['plot_num_emp'])
        self.plot_pca_noise             = float(args['plot_pca_noise'])

        # prefixes
        fmt_proj_prefix             = f'{self.fmt_dir}/{self.fmt_prefix}'
        trn_proj_prefix             = f'{self.trn_dir}/{self.trn_prefix}'
        est_proj_prefix             = f'{self.est_dir}/{self.est_prefix}'
        plt_proj_prefix             = f'{self.plt_dir}/{self.plt_prefix}'

        # train dataset, main dataset
        self.train_hdf5_fn         = f'{fmt_proj_prefix}.train.hdf5'
        self.train_aux_data_fn     = f'{fmt_proj_prefix}.train.aux_data.csv'
        self.train_labels_fn       = f'{fmt_proj_prefix}.train.labels.csv'

        # train dataset tensors
        self.train_est_real_fn     = f'{trn_proj_prefix}.train_est.labels_real.csv'
        self.train_true_real_fn    = f'{trn_proj_prefix}.train_true.labels_real.csv'
        self.train_est_cat_fn      = f'{trn_proj_prefix}.train_est.labels_cat.csv'
        self.train_true_cat_fn     = f'{trn_proj_prefix}.train_true.labels_cat.csv'

        # test dataset tensors
        self.test_est_real_fn      = f'{est_proj_prefix}.test_est.labels_real.csv'
        self.test_true_real_fn     = f'{est_proj_prefix}.test_true.labels_real.csv'
        self.test_est_cat_fn       = f'{est_proj_prefix}.test_est.labels_cat.csv'
        self.test_true_cat_fn      = f'{est_proj_prefix}.test_true.labels_cat.csv'
        
        # empirical dataset tensors
        self.emp_hdf5_fn           = f'{fmt_proj_prefix}.empirical.hdf5'
        self.emp_aux_data_fn       = f'{fmt_proj_prefix}.empirical.aux_data.csv'
        self.emp_est_real_fn       = f'{est_proj_prefix}.empirical_est.labels_real.csv'
        self.emp_est_cat_fn        = f'{est_proj_prefix}.empirical_est.labels_cat.csv'

        # network
        self.model_arch_fn         = f'{trn_proj_prefix}.trained_model.pkl'
        self.history_fn            = f'{trn_proj_prefix}.train_history.csv'

        # new empirical plot
        self.save_cpi_est_fn       = f'{plt_proj_prefix}.empirical_estimate'

        # PCA plotting output
        self.save_train_pca_aux_data_fn  = f'{plt_proj_prefix}.train_pca_aux_data.pdf'
        self.save_train_pca_labels_fn    = f'{plt_proj_prefix}.train_pca_labels_real.pdf'
        # self.save_emp_pca_aux_data_fn    = f'{plt_proj_prefix}.empirical_pca_contour_aux_data.pdf'
        # self.save_emp_pca_labels_fn      = f'{plt_proj_prefix}.empirical_pca_contour_labels_real.pdf'

        # density plotting output
        self.save_train_density_aux_fn   = f'{plt_proj_prefix}.train_density_aux_data.pdf'
        self.save_train_density_label_fn = f'{plt_proj_prefix}.train_density_labels_real.pdf'
        self.save_emp_density_aux_fn     = f'{plt_proj_prefix}.empirical_density_aux_data.pdf'
        self.save_emp_density_label_fn   = f'{plt_proj_prefix}.empirical_density_labels_real.pdf'

        # scatter plotting output
        self.save_train_est_fn     = f'{plt_proj_prefix}.train_estimate'
        self.save_test_est_fn      = f'{plt_proj_prefix}.test_estimate'

        # network plotting output
        self.save_network_fn       = f'{plt_proj_prefix}.network_architecture.pdf'
        self.save_history_fn       = f'{plt_proj_prefix}.train_history'

        # summary plotting output
        self.save_summary_fn       = f'{plt_proj_prefix}.summary.pdf'
        self.save_report_fn        = f'{plt_proj_prefix}.summary.csv'
        
        # cat vs. real parameter names
        self.param_name_real = [ k for k,v in self.param_est.items() if v == 'real' ]
        self.param_name_cat = [ k for k,v in self.param_est.items() if v == 'cat' ]
        
        # initialized later
        self.train_aux_data = None         # init with load_input()
        self.train_labels_real = None      # init with load_input()
        self.train_labels_cat = None       # init with load_input()
        self.emp_aux_data = None           # init with load_input()
        self.aux_data_names = None         # init with load_input()
        self.model = None                  # init with load_input()

        # datasets to load
        self.train_est_real = None         # init with load_input()
        self.train_true_real = None        # init with load_input()
        self.train_est_cat = None          # init with load_input()
        self.train_true_cat = None         # init with load_input()
        self.test_est_real = None         # init with load_input()
        self.test_true_real = None        # init with load_input()
        self.test_est_cat = None          # init with load_input()
        self.test_true_cat = None         # init with load_input()
        self.emp_est_real = None         # init with load_input()
        self.emp_est_cat = None          # init with load_input()

        # what datasets do we have?
        self.has_train_real = False
        self.has_train_cat = False
        self.has_test_real = False
        self.has_test_cat = False
        self.has_emp_real = False
        self.has_emp_cat = False
        
        # analysis info
        self.history_table = None          # init with load_input()
        self.num_empirical = int(0)        # init with load_input()
        self.sim_test_valid = False
        self.sim_train_valid = False

        return

    def run(self):
        """Generates all plots.

        This method creates the target directory for new plots, generates
        a set of standard summary plots, along with a combined report.

        """

        verbose = self.verbose

        # print header
        util.print_step_header('plt',
                               [self.fmt_dir, self.trn_dir, self.est_dir],
                               self.plt_dir,
                               [self.fmt_prefix, self.trn_prefix, self.est_prefix],
                               self.plt_prefix, verbose)

        # prepare workspace
        os.makedirs(self.plt_dir, exist_ok=True)

        # start time
        start_time,start_time_str = util.get_time()
        util.print_str(f'▪ Start time of {start_time_str}', verbose)

        # load input
        util.print_str('▪ Loading input', verbose)
        self.load_input()

        # generate individual plots
        util.print_str('▪ Generating individual plots', verbose)
        self.generate_plots()

        # combining all plots
        util.print_str('▪ Combining plots', verbose)
        self.combine_plots()

        # generating output summary
        util.print_str('▪ Making csv report', verbose)
        self.make_report()

        # end time
        end_time,end_time_str = util.get_time()
        run_time = util.get_time_diff(start_time, end_time)
        util.print_str(f'▪ End time of {end_time_str} (+{run_time})', verbose)

        # done
        util.print_str('... done!', verbose=self.verbose)
        return

    def load_input(self):
        """Load input data for plotting.

        This function loads input from Format, Train, and Estimate. We load the
        simulated training examples from Format. From Train, we load the
        network, the training history, and the test/train estimates/labels. For
        Estimate, we load the estimates and aux. data, if they exist.

        """

        # load input from Format step
        train_labels = None
        if self.tensor_format == 'csv':
            # csv tensor format
            self.train_aux_data = util.read_csv_as_pandas( self.train_aux_data_fn )
            train_labels = util.read_csv_as_pandas( self.train_labels_fn )
            
        elif self.tensor_format == 'hdf5':
            # hdf5 tensor format
            hdf5_file = h5py.File(self.train_hdf5_fn, 'r')
            train_aux_data_names = [ s.decode() for s in hdf5_file['aux_data_names'][0,:] ]
            self.train_aux_data = pd.DataFrame( hdf5_file['aux_data'][:,:], columns=train_aux_data_names )
            train_label_names = [ s.decode() for s in hdf5_file['label_names'][0,:] ]
            train_labels = pd.DataFrame( hdf5_file['labels'][:,:], columns=train_label_names)
            hdf5_file.close()

        # split training labels from format into real/cat
        self.train_labels_real = train_labels[ self.param_name_real ]
        self.train_labels_cat = train_labels[ self.param_name_cat ]

        # aux data column names
        self.aux_data_names = self.train_aux_data.columns.to_list()

        # trained model
        self.model = torch.load(self.model_arch_fn)
        self.model = self.model.to('cpu')
        
        # training true/estimated labels
        self.train_est_real  = util.read_csv_as_pandas(self.train_est_real_fn)
        self.train_true_real = util.read_csv_as_pandas(self.train_true_real_fn)
        self.train_est_cat   = util.read_csv_as_pandas(self.train_est_cat_fn)
        self.train_true_cat  = util.read_csv_as_pandas(self.train_true_cat_fn)
        
        # test true/estimated labels
        self.test_est_real   = util.read_csv_as_pandas(self.test_est_real_fn)
        self.test_true_real  = util.read_csv_as_pandas(self.test_true_real_fn)
        self.test_est_cat    = util.read_csv_as_pandas(self.test_est_cat_fn)
        self.test_true_cat   = util.read_csv_as_pandas(self.test_true_cat_fn)
        
        # empirical estimated labels
        self.emp_est_real = util.read_csv_as_pandas(self.emp_est_real_fn)
        self.emp_est_cat  = util.read_csv_as_pandas(self.emp_est_cat_fn)
        
        # check what datasets we have
        if self.test_est_real is not None and self.test_true_real is not None:
            self.has_test_real = True
        if self.test_est_cat is not None and self.test_true_cat is not None:
            self.has_test_cat = True
        if self.train_est_real is not None and self.train_true_real is not None:
            self.has_train_real = True
        if self.train_est_cat is not None and self.train_true_cat is not None:
            self.has_train_cat = True
        if self.emp_est_real is not None:
            self.has_emp_real = True
        if self.emp_est_cat is not None:
            self.has_emp_cat = True
        
        # training history for network
        self.history_table   = util.read_csv_as_pandas(self.history_fn)

        # load empirical aux. data, if they exist
        self.emp_aux_data = None
        if self.tensor_format == 'csv':
            self.emp_aux_data = pd.read_csv(self.emp_aux_data_fn)
            
        elif self.tensor_format == 'hdf5' and \
                os.path.isfile(self.emp_hdf5_fn):
            hdf5_file = h5py.File(self.emp_hdf5_fn, 'r')
            emp_aux_data_names = [ s.decode() for s in hdf5_file['aux_data_names'][0,:] ]
            self.emp_aux_data = pd.DataFrame( hdf5_file['aux_data'][:,:], columns=emp_aux_data_names )
            hdf5_file.close()
            
        if self.emp_aux_data is not None:
            self.num_empirical = self.emp_aux_data.shape[0]

        # done
        return
    
##################################################

    def generate_plots(self):
        """Generates all plots.
        
        This function generates the following plots:
        - history: network metrics across training epochs
        - predictions: truth vs. estimated values for trained network
        - histograms: histograms of training examples
        - PCA: PCA for training examples
        - new results: point est. + CPI for new dataset
        - network: neural network architecture

        Plots are generated based on args, which initialize filenames,
        datasets, and colors.

        """
        
        # Densities for aux. data and labels
        self.make_plot_stat_density('train', 'aux_data')
        if self.has_train_real:
            self.make_plot_stat_density('train', 'labels')

        if self.num_empirical >= self.plot_min_emp:
            self.make_plot_stat_density('empirical', 'aux_data')
        if self.num_empirical >= self.plot_min_emp and self.has_emp_real:
            self.make_plot_stat_density('empirical', 'labels')

        # PCA-contours for aux. data and labels
        aux_pca_model = self.make_plot_pca_hexbin('train', 'aux_data')
        lbl_pca_model = None
        if self.has_train_real:
            lbl_pca_model = self.make_plot_pca_hexbin('train', 'labels')
        # if self.num_empirical >= self.min_num_emp_density:
        #     self.make_plot_pca_contour('empirical', 'aux_data', pca_model=aux_pca_model)
        # if self.num_empirical >= self.min_num_emp_density and self.has_emp_real:
        #     self.make_plot_pca_contour('empirical', 'labels', pca_model=lbl_pca_model)

        # scatter accuracy
        if self.has_train_real:
            self.make_plot_scatter_accuracy('train')
        if self.has_test_real:
            self.make_plot_scatter_accuracy('test')

        # confusion matrix
        if self.has_train_cat:
            self.make_plot_confusion_matrix('train')
        if self.has_test_cat:
            self.make_plot_confusion_matrix('test')

        # point estimates and CPIs in empirical dataset
        if self.has_emp_real:
            self.make_plot_emp_ci()

        # bar plot for categorical in empirical dataset
        # if self.has_emp_cat:
            # self.make_plot_emp_cat()

        # training history stats
        self.make_plot_train_history()

        # network architecture
        self.make_plot_network_architecture()

        # done
        return
    

##################################################

    def make_plot_stat_density(self, dataset_name, dataset_type):
        """Calls plot_stat_density with arguments."""
        assert dataset_name in ['train', 'empirical']
        assert dataset_type in ['aux_data', 'labels']

        train_aux_data = None
        train_labels_real = None
        emp_aux_data = None
        emp_est_real = None
        if self.train_aux_data is not None:
            train_aux_data = self.train_aux_data.copy()
        if self.train_true_real is not None:
            train_labels_real = self.train_labels_real.copy()
        if self.emp_aux_data is not None:
            emp_aux_data = self.emp_aux_data.copy()
        if self.emp_est_real is not None:
            emp_est_real = self.emp_est_real.copy()
        
        if dataset_name == 'train' and dataset_type == 'aux_data':
            self.plot_stat_density(save_fn=self.save_train_density_aux_fn,
                                   dist_values=train_aux_data,
                                   point_values=emp_aux_data,
                                   color=self.plot_aux_color,
                                   title='training aux. data')
            
        elif dataset_name == 'train' and dataset_type == 'labels':
            self.plot_stat_density(save_fn=self.save_train_density_label_fn,
                                   dist_values=train_labels_real,
                                   point_values=emp_est_real,
                                   color=self.plot_label_color,
                                   title='training labels')
            
        elif dataset_name == 'empirical' and dataset_type == 'aux_data':
            self.plot_stat_density(save_fn=self.save_emp_density_aux_fn,
                                   dist_values=emp_aux_data,
                                   point_values=None,
                                   color=self.plot_aux_color,
                                   title='empirical aux. data')
        
        elif dataset_name == 'empirical' and dataset_type == 'labels':
            self.plot_stat_density(save_fn=self.save_emp_density_label_fn,
                                   dist_values=emp_est_real,
                                   point_values=None,
                                   color=self.plot_label_color,
                                   title='empirical labels')
        
        # done
        return
    
    def make_plot_scatter_accuracy(self, dataset_name):
        """Calls plot_scatter_accuracy with arguments."""
        assert dataset_name in ['train', 'test']

        # n_max = self.plot_num_scatter
        if dataset_name == 'train':
            # plot train scatter
            # n = min(n_max, self.train_est_real.shape[0])
            self.plot_scatter_accuracy(ests=self.train_est_real.copy(),
                                       labels=self.train_true_real.copy(),
                                       prefix=self.save_train_est_fn,
                                       color=self.plot_train_color,
                                       title='Train')
        elif dataset_name == 'test':
            # plot test scatter
            # n = min(n_max, self.test_est_real.shape[0])
            self.plot_scatter_accuracy(ests=self.test_est_real.copy(),
                                       labels=self.test_true_real.copy(),
                                       prefix=self.save_test_est_fn,
                                       color=self.plot_test_color,
                                       title='Test')
        # done
        return

    def make_plot_confusion_matrix(self, dataset_name):
        """Calls plot_confusion_matrix with arguments."""
        assert dataset_name in ['train', 'test']

        if dataset_name == 'train':
            self.plot_confusion_matrix(ests=self.train_est_cat.copy(),
                                      labels=self.train_true_cat.copy(),
                                      prefix=self.save_train_est_fn,
                                      color=self.plot_train_color,
                                      title='Train')
        elif dataset_name == 'test':
            self.plot_confusion_matrix(ests=self.test_est_cat.copy(),
                                      labels=self.test_true_cat.copy(),
                                      prefix=self.save_test_est_fn,
                                      color=self.plot_test_color,
                                      title='Test')
        # done
        return

    def plot_confusion_matrix(self, ests, labels, prefix, color, title):
        """Plots confusion matrix.

        This function plots the confusion matrix for categorical labels.

        Args:
            ests (numpy.array): Estimated categorical labels.
            labels (numpy.array): True categorical labels.
            prefix (str): Filename prefix to save plot.
            color (str): Color of histograms.
            title (str): Plot title.

        """
        
        # loop over cat. parameters
        for p in self.param_name_cat:
            # get true/est values
            est_cats_p = [ x for x in ests.columns if p in x ]
            lbls_p = labels[p].copy()
            ests_p = ests[est_cats_p].copy()
            
            # make confusion matrix
            num_cat = ests_p.shape[1]
            conf_mtx = np.zeros( (num_cat, num_cat) )
            for i in range(num_cat):
                for j in range(num_cat):
                    true_match = (lbls_p == i)
                    est_match = (ests_p.idxmax(axis=1) == est_cats_p[j])
                    conf_mtx[i,j] = np.sum(true_match & est_match)
            conf_mtx = conf_mtx / np.sum(conf_mtx, axis=1)[:,None]
            conf_mtx = np.transpose(conf_mtx)
        
            # get stats
            # true_pos = np.diag(conf_mtx)
            # false_pos = np.sum(conf_mtx, axis=0) - true_pos
            # true_pos_rate = true_pos / np.sum(conf_mtx, axis=0) * 100
            # false_pos_rate = false_pos / np.sum(conf_mtx, axis=0) * 100
            # s_tpr = [ '{:.2E}'.format(x) for x in true_pos_rate ]
            # s_fpr = [ '{:.2E}'.format(x) for x in false_pos_rate ]
            # s_tpr = [ '{:.1f}%'.format(x) for x in true_pos_rate ]
            # s_fpr = [ '{:.1f}%'.format(x) for x in false_pos_rate ]
        
            # plot confusion matrix
            fig, ax = plt.subplots(figsize=(6, 6))
            fig.tight_layout()
            cm = LinearSegmentedColormap.from_list(
                "Custom", ['white', color], N=20)
            cax = ax.matshow(conf_mtx, cmap=cm, vmin=0.0, vmax=1.0)
            for (i,j), z in np.ndenumerate(conf_mtx):
                text_color = 'black'
                if z > 0.5:
                    text_color = 'white'
                ax.text(j, i, '{:0.2f}'.format(z), ha='center', va='center', color=text_color)
            ax.xaxis.set_ticks_position('bottom')
            cbar = plt.colorbar(cax, fraction=0.046, pad=0.04)
            # plt.text(x=0,y=0,s=f'False Positive Rate: {s_fpr}', ha='right', va='top', fontsize=10)
            # plt.text(x=0,y=0,s=f'True Positive Rate: {s_tpr}', ha='right', va='bottom', fontsize=10)
            plt.xlabel(f'{p} truth')
            plt.ylabel(f'{p} estimate')
            plt.title(f'{title} estimates: {p}')
            plt.savefig(fname=f'{prefix}_{p}.pdf', format='pdf', dpi=300, bbox_inches='tight')
            plt.clf()
            plt.close()
            
        return

    def make_plot_pca_hexbin(self, dataset_name, dataset_type, pca_model=None):
        """Calls plot_pca_hexbin with arguments."""
        assert dataset_name in ['train', 'empirical']
        assert dataset_type in ['aux_data', 'labels']

        train_aux_data = None
        train_labels_real = None
        emp_aux_data = None
        emp_est_real = None
        if self.train_aux_data is not None:
            train_aux_data = self.train_aux_data.copy()
        if self.train_true_real is not None:
            train_labels_real = self.train_labels_real.copy()
        if self.emp_aux_data is not None:
            emp_aux_data = self.emp_aux_data.copy()
        if self.emp_est_real is not None:
            emp_est_real = self.emp_est_real.copy()
            
        mdl = None
        num_comp = 4
        if dataset_name == 'train' and dataset_type == 'aux_data':
            mdl = self.plot_pca_hexbin(save_fn=self.save_train_pca_aux_data_fn,
                                        dist_values=train_aux_data,
                                        point_values=emp_aux_data,
                                        pca_model=pca_model,
                                        num_comp=num_comp,
                                        color=self.plot_aux_color,
                                        title='training aux. data')
            
        elif dataset_name == 'train' and dataset_type == 'labels':
            if train_labels_real.shape[1] <= 1:
                return None
            mdl = self.plot_pca_hexbin(save_fn=self.save_train_pca_labels_fn,
                                        dist_values=train_labels_real,
                                        point_values=emp_est_real,
                                        pca_model=pca_model,
                                        num_comp=num_comp,
                                        color=self.plot_label_color,
                                        title='training labels')

        # elif dataset_name == 'empirical' and dataset_type == 'aux_data':
        #     mdl = self.plot_pca_hexbin(save_fn=self.save_emp_pca_aux_data_fn,
        #                                 dist_values=emp_aux_data,
        #                                 point_values=None,
        #                                 pca_model=pca_model,
        #                                 num_comp=num_comp,
        #                                 color=self.plot_aux_color,
        #                                 title='empirical aux. data')
        # 
        # elif dataset_name == 'empirical' and dataset_type == 'labels':
        #     if emp_est_real.shape[1] <= 1:
        #         return None
        #     mdl = self.plot_pca_hexbin(save_fn=self.save_emp_pca_labels_fn,
        #                                 dist_values=emp_est_real,
        #                                 point_values=None,
        #                                 pca_model=pca_model,
        #                                 num_comp=num_comp,
        #                                 color=self.plot_label_color,
        #                                 title='empirical labels')
        
        # todo: fix PCA num columns/labels
        mdl = None
        return mdl

    def make_plot_emp_ci(self):
        """Calls plot_est_CI with arguments."""
        max_num = np.min([self.plot_num_emp, self.num_empirical])
        for i in range(max_num):
            save_fn = f'{self.save_cpi_est_fn}_real_{i}.pdf'
            title = f'Estimate: {self.est_prefix}.empirical.{i}'
            self.plot_emp_ci(save_fn=save_fn,
                             est_label=self.emp_est_real.iloc[[i]].copy(),
                             title=title,
                             color=self.plot_emp_color)
        return

    def make_plot_emp_cat(self):
        """Calls plot_emp_cat with arguments."""
        max_num = np.min([self.plot_num_emp, self.num_empirical])
        for i in range(max_num):
            save_fn = f'{self.save_cpi_est_fn}_cat_{i}.pdf'
            title = f'Estimate: {self.est_prefix}.empirical.{i}'
            self.plot_emp_cat(save_fn=save_fn,
                              est_label=self.emp_est_cat.iloc[[i]].copy(),
                              title=title,
                              color=self.plot_emp_color)
        
        return

    def make_plot_train_history(self):
        """Calls plot_train_history with arguments."""
        self.plot_train_history(self.history_table.copy(),
                                prefix=self.save_history_fn,
                                train_color=self.plot_train_color,
                                val_color=self.plot_val_color)
        return

    def make_plot_network_architecture(self):
        """Calls torchview.draw_graph with arguments."""
        n_fake = 10
        phy_dat_fake = torch.zeros( [n_fake] + list(self.model.phy_dat_shape),
                                    dtype=torch.float32 )
        aux_dat_fake = torch.zeros( [n_fake] + list(self.model.aux_dat_shape),
                                    dtype=torch.float32 )
        lbl_fake = self.model(phy_dat_fake, aux_dat_fake)
        
        # save as png
        torchview.draw_graph(self.model,
                             input_data=[phy_dat_fake, aux_dat_fake],
                             filename=self.save_network_fn,
                             save_graph=True)
        
        # convert from png to pdf
        image = Image.open(self.save_network_fn + '.png')
        image = image.convert('RGB')
        
        # save as pdf
        image.save(self.save_network_fn, dpi=(300, 300), size=(3000,2100))
        
        return
    
##################################################

    def plot_stat_density(self, save_fn, dist_values, point_values=None,
                           title='', ncol_plot=3, color='blue'):
        """Plots histograms.

        This function plots the histograms (KDEs) for simulated training
        examples, e.g. aux. data or labels. The function will also plot
        values from the new dataset if it is available (est_values != None).

        Args:
            save_fn (str): Filename to save plot.
            dist_values (numpy.array): Simulated values from training examples.
            point_values (numpy.array): Estimated values from new dataset.
            title (str): Plot title.
            ncol_plot (int): Number of columns in plot
            color (str): Color of histograms

        """
        
        # data dimensions
        col_names = sorted(dist_values.columns )
        num_aux = len(col_names)
        nrow = int( np.ceil(num_aux/ncol_plot) )

        # figure dimensions
        fig_width = 9
        fig_height = 1 + int(np.ceil(2*nrow))

        # basic figure structure
        fig, axes = plt.subplots(ncols=ncol_plot, nrows=nrow, squeeze=False,
                                 figsize=(fig_width, fig_height))
        fig.tight_layout()

        # fill in plot
        i = 0
        for i_row, ax_row in enumerate(axes):
            for j_col,ax in enumerate(ax_row):
                if i >= num_aux:
                    axes[i_row,j_col].axis('off')
                    continue
                  
                # input data
                p = col_names[i]
                x = sorted(dist_values[p])
                if np.var(x) == 0.0:
                    x = sp.stats.norm.rvs(size=len(x), loc=x, scale=x[0]*1e-3)
                
                mn = np.min(x)
                mx = np.max(x)
                xs = np.linspace(mn, mx, 300)
                
                kde = sp.stats.gaussian_kde(x)
                ys = kde.pdf(xs)
                ax.plot(xs, ys, label="PDF", color=color)
                
                # plot quantiles
                left, middle, right = np.percentile(x, [2.5, 50, 97.5])
                ax.vlines(middle, 0, np.interp(middle, xs, ys), color=color, ls=':')
                ax.fill_between(xs, 0, kde(xs), where=(left <= xs) & (xs <= right), facecolor=color, alpha=0.2)

                if middle-mn < mx-middle:
                    ha = 'right'
                    x_pos = 0.99
                else:
                    ha = 'left'
                    x_pos = 0.01
                y_pos = 0.98
                dy_pos = 0.10
                aux_median_str = "M={:.2f}".format(middle)
                aux_lower_str = "L={:.2f}".format(left)
                aux_upper_str = "U={:.2f}".format(right)
                
                ax.annotate(aux_median_str, xy=(x_pos, y_pos-0*dy_pos), xycoords='axes fraction', fontsize=10, horizontalalignment=ha, verticalalignment='top')
                ax.annotate(aux_lower_str, xy=(x_pos, y_pos-1*dy_pos), xycoords='axes fraction', fontsize=10, horizontalalignment=ha, verticalalignment='top')
                ax.annotate(aux_upper_str, xy=(x_pos, y_pos-2*dy_pos), xycoords='axes fraction', fontsize=10, horizontalalignment=ha, verticalalignment='top')
                
                p_point = p
                if point_values is not None and p not in point_values:
                    p_point = f'{p}_value'
                    
                if point_values is not None and p_point in point_values:
                    x_data = point_values[p_point][0]
                    y_data = kde(x_data)
                    q_true = np.sum(x < x_data) / len(x)
                    ax.vlines(x_data, 0, y_data, color='red')
                    lbl_est_val_str = "X={:.2f}".format(x_data)
                    lbl_est_quant_str = f'Q={int(q_true*100)}%'
                    ax.annotate(lbl_est_val_str, xy=(x_pos, y_pos-3*dy_pos), xycoords='axes fraction', fontsize=10, horizontalalignment=ha, verticalalignment='top', color='red')
                    ax.annotate(lbl_est_quant_str, xy=(x_pos, y_pos-4*dy_pos), xycoords='axes fraction', fontsize=10, horizontalalignment=ha, verticalalignment='top', color='red')
                    
                # cosmetics
                ax.title.set_text(col_names[i])
                ax.yaxis.set_visible(False)
                i = i + 1

        # add labels to superplot axes
        fig.supxlabel('Data')
        fig.supylabel('Density')

        fig.suptitle(f'Density: {title}')
        fig.tight_layout(rect=[0, 0.03, 1, 0.98])
        plt.savefig(fname=save_fn, format='pdf', dpi=300, bbox_inches='tight')
        plt.clf()
        plt.close()
        
        # done
        return
    
    def plot_pca_hexbin(self, save_fn, dist_values, point_values=None,
                         pca_model=None, num_comp=4, color='blue', title=''):
        """
        Plots PCA Hexbin Plot.

        This function plots the PCA for simulated training aux. data examples.
        The function plots a grid of pairs of principal components. It will also
        plot values from the new dataset, when variable (est_values != None).

        Args:
            save_fn (str): Filename to save plot.
            dist_values (numpy.array): Simulated values from training examples.
            point_values (numpy.array): Estimated values from new dataset.
            pca_model (PCA): Fitted PCA model to use (default None)
            num_comp (int): Number of components to plot (default 4)
            color (str): Color of histograms
            title (str): Plot title

        """
        
        # figure size
        fig_width = 8
        fig_height = 8
        
        # drop lower/upper values
        # dist_values_names = [ p.replace('_value','') for p in dist_values.columns if p.endswith('_value') ]
        # dist_values_names = [ p for p in dist_values_names if not p.endswith('_lower') and not p.endswith('_upper') ]
        # dist_values.columns = [ p.replace('_value','') for p in dist_values.columns ]
        # dist_values = dist_values[ dist_values_names ]
        # if point_values is not None:
        #     point_values_names = [ p.replace('_value','') for p in point_values.columns if '_value' in p ]
        #     point_values.columns = [ p.replace('_value','') for p in point_values.columns ]
        #     point_values = point_values[ point_values_names ]

        # reduce num components if needed
        num_comp = min(dist_values.shape[1], num_comp)

        # rescale input data
        # dist_values = np.log(dist_values + self.log_offset)
        scaler = StandardScaler()
        x = scaler.fit_transform(dist_values)
        if self.plot_pca_noise != 0.0:
            x = x + sp.stats.norm.rvs(size=x.shape, loc=0, scale=self.plot_pca_noise)
        
        # apply PCA to sim_values
        if pca_model is None:
            pca_model = PCA(n_components=num_comp)
            pca = pca_model.fit_transform(x)
        else:
            pca = pca_model.transform(x)
            
        pca_var = pca_model.explained_variance_ratio_
        pca_coef = np.transpose(pca_model.components_)
        plot_pca_loadings = False

        # project est_values on to PCA space
        pca_est = None
        if point_values is not None:
            point_values.columns = [ p.replace('_value','') for p in point_values.columns ]
            point_values = point_values[ dist_values.columns ]
            # point_values = np.log(point_values + self.log_offset)
            point_values = scaler.transform(point_values)
            if self.plot_pca_noise != 0.0:
                point_values = point_values + sp.stats.norm.rvs(size=point_values.shape, loc=0, scale=self.plot_pca_noise)
            pca_est = pca_model.transform(point_values)
        
        # figure dimensions
        fig, axs = plt.subplots(num_comp-1, num_comp-1, squeeze=False,
                                sharex=False, sharey=False,
                                figsize=(fig_width, fig_height))

        # use this to turn off subplots
        # axes[i_row,j_col].axis('off')

        # color map
        cmap0 = mpl.colors.LinearSegmentedColormap.from_list('white2col',
                                                             ['white', color])

        # generate PCA subplots
        vmin = np.min(pca)
        vmax = np.max(pca)
        for i in range(0, num_comp-1):
            
            # disable x,y axes for all plots, later turned on for some
            for j in range(i+1, num_comp-1):
                axs[i,j].axis('off')

            for j in range(0, i+1):
                # contours
                x = pca[:,i+1]
                y = pca[:,j]
                xmin, xmax = np.min(x), np.max(x)
                ymin, ymax = np.min(y), np.max(y)
                xscale = (xmax - xmin)
                yscale = (ymax - ymin)

                # Perform the kernel density estimate
                xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]

                #hb = axs[i,j].hexbin(x, y, gridsize=(19,13), bins=10,
                #                cmap=cmap0, linewidths=0.1,) # vmin=0) #, vmax=vmax)
                #axs[i,j].set_visible(False)
                hb = axs[i,j].hexbin(x, y, 
                                #C=np.ones_like(y, dtype=np.float64) / hb.get_array().max(),
                                #reduce_C_function=np.sum,
                                gridsize=(19,13), bins=10,
                                cmap=cmap0, linewidths=0.1,) # vmin=0) #, vmax=vmax)
                
                axs[i,j].set_visible(True)
                    
                if pca_est is not None:
                    axs[i,j].scatter(pca_est[:,i+1], pca_est[:,j],
                                     alpha=1.0, color='white',
                                     edgecolor='black', s=80, zorder=2.1)
                    axs[i,j].scatter(pca_est[:,i+1], pca_est[:,j],
                                     alpha=1.0, color='red',
                                     edgecolor='white', s=40, zorder=2.1)
                # axes
                if j == 0:
                    idx = str(i+2)
                    var = int(100*round(pca_var[i+1], ndigits=2))
                    ylabel = f'PC{idx} ({var}%)'
                    axs[i,j].set_ylabel(ylabel, fontsize=12)
                if i == (num_comp-2):
                    idx = str(j+1)
                    var = int(100*round(pca_var[j], ndigits=2))
                    xlabel = f'PC{idx} ({var}%)'
                    axs[i,j].set_xlabel(xlabel, fontsize=12)

                # cb = fig.colorbar(hb, ax=axs[i,j], label='freq.')
                    
                axs[i,j].set_xlim(xmin, xmax)
                axs[i,j].set_ylim(ymin, ymax)
                
        fig.suptitle(f'PCA: {title}')
        fig.tight_layout(rect=[0, 0.03, 1, 0.98])
        plt.savefig(save_fn, format='pdf', dpi=300, bbox_inches='tight')
        plt.clf()
        plt.close()
        
        # done
        return pca_model

    def plot_scatter_accuracy(self, ests, labels, prefix,
                              color="blue", axis_labels=("truth", "estimate"),
                              title='', plot_log=False):
        """Plots accuracy of estimates and CPIs for labels.

        This function generates a scatterplot for true vs. estimated labels
        from the trained network. Points are point estimates. Bars are
        CPIs.

        Args:
            ests (numpy.array): Simulated values from training examples.
            labels (numpy.array): Estimated values from new dataset.
            prefix (str): Filename prefix to save plot.
            axis_labels (list): Labels for x and y axes.
            color (str): Color of histograms.
            title (str): Title for the plot.
            color (str): Color of histograms.
            plot_log (bool): Plot y-axis on log scale? Default True.

        """
        # figure size
        fig_width = 6
        fig_height = 6

        # create figure
        plt.figure(figsize=(fig_width,fig_height))

        max_num = np.min([self.plot_num_scatter, ests.shape[0]])
        
        # plot parameters
        for i,p in enumerate(labels.columns):

            # labels
            x_label = f'{p} {axis_labels[0]}'
            y_label = f'{p} {axis_labels[1]}'

            # estimates (x) and true values (y)
            lbl_est   = ests[f'{p}_value'][:].to_numpy()
            lbl_lower = ests[f'{p}_lower'][:].to_numpy()
            lbl_upper = ests[f'{p}_upper'][:].to_numpy()
            lbl_true  = labels[p][:].to_numpy()

            # only_positive = np.all(lbl_true >= 0.)
            # if only_positive and plot_log:
            #     lbl_est = np.log(lbl_est)
            #     lbl_lower = np.log(lbl_lower)
            #     lbl_upper = np.log(lbl_upper)
            #     lbl_true = np.log(lbl_true)
            #     x_label = f'ln {p} {axis_labels[0]}'
            #     y_label = f'ln {p} {axis_labels[1]}'
                
            # accuracy stats
            stat_mae = np.mean( np.abs(lbl_est - lbl_true) )
            stat_mape = 100 * np.mean( np.abs(lbl_est - lbl_true) / lbl_true )
            stat_mse = np.mean( np.power(lbl_est - lbl_true, 2) )
            stat_rmse = np.sqrt( stat_mse )
            
            # coverage stats
            stat_cover = np.logical_and(lbl_lower < lbl_true, lbl_upper > lbl_true )
            stat_not_cover = np.logical_not(stat_cover)
            f_stat_cover = sum(stat_cover) / len(stat_cover) * 100
            f_stat_cover_target = self.cpi_coverage * 100

            # linear regression slope
            # if only_positive:
            #     reg = LinearRegression().fit( np.log(lbl_est.reshape(-1, 1)), np.log(lbl_true.reshape(-1, 1)))
            #     stat_slope = reg.coef_[0][0]
            #     stat_intercept = reg.intercept_[0]
            # else:
            reg = LinearRegression().fit( lbl_est.reshape(-1, 1), lbl_true.reshape(-1, 1))
            stat_slope = reg.coef_[0][0]
            stat_intercept = reg.intercept_[0]
            
            # convert to strings
            s_mae  = '{:.2E}'.format(stat_mae)
            s_mse  = '{:.2E}'.format(stat_mse)
            s_rmse = '{:.2E}'.format(stat_rmse)
            s_mape = '{:.1f}%'.format(stat_mape)
            s_slope = '{:.2E}'.format(stat_slope)
            s_intercept  = '{:.2E}'.format(stat_intercept)
            s_cover = '{:.1f}%'.format(f_stat_cover)
            s_cover_target = '{:.1f}%'.format(f_stat_cover_target)
            
            # covered points
            alpha = 0.5
            # downsample
            lbl_true = lbl_true[0:max_num]
            lbl_est = lbl_est[0:max_num]
            lbl_lower = lbl_lower[0:max_num]
            lbl_upper = lbl_upper[0:max_num]
            stat_cover = stat_cover[0:max_num]
            stat_not_cover = stat_not_cover[0:max_num]
            plt.scatter(lbl_true[stat_cover], lbl_est[stat_cover],
                        alpha=alpha, c=color, zorder=3, s=3)
            # covered bars
            plt.plot([lbl_true[stat_cover], lbl_true[stat_cover]],
                     [lbl_lower[stat_cover], lbl_upper[stat_cover]],
                     color=color, alpha=alpha, linestyle="-", marker='_',
                     linewidth=0.5, zorder=2 )
            
            # not covered points
            plt.scatter(lbl_true[stat_not_cover], lbl_est[stat_not_cover],
                        alpha=alpha, c='#aaaaaa', zorder=5, s=3)
            # not covered bars
            plt.plot([lbl_true[stat_not_cover], lbl_true[stat_not_cover]],
                     [lbl_lower[stat_not_cover], lbl_upper[stat_not_cover]],
                     color='#aaaaaa', alpha=alpha, linestyle="-", marker='_',
                     linewidth=0.5, zorder=4 )
            
            # regression line
            # plt.axline((0, stat_intercept), slope=(stat_slope, 0), color=color,
            #            alpha=1.0, zorder=0, linestyle='dotted')
            plt.axline((stat_intercept, 0), slope=1./stat_slope, color=color,
                       alpha=1.0, zorder=0, linestyle='dotted')

            # 1:1 line
            plt.axline((0,0), slope=1, color=color, alpha=1.0, zorder=0)
            plt.gca().set_aspect('equal')

            # set axes
            xlim = plt.xlim()
            ylim = plt.ylim()
            minlim = min(xlim[0], ylim[0])
            maxlim = max(xlim[1], ylim[1])
            plt.xlim([minlim, maxlim])
            plt.ylim([minlim, maxlim])
            
            # write text
            dx = 0.03
            stat_str = [f'MAE: {s_mae}', f'MAPE: {s_mape}', f'MSE: {s_mse}',
                        f'RMSE: {s_rmse}', f'Intercept: {s_intercept}',
                        f'Slope: {s_slope}', f'Coverage: {s_cover}',
                        f'Coverage target: {s_cover_target}']
            
            for j,s in enumerate(stat_str):
                plt.annotate(s, xy=(0.01,0.99-j*dx), xycoords='axes fraction',
                             fontsize=10, horizontalalignment='left',
                             verticalalignment='top', color='black')

            # cosmetics
            plt.title(f'{title} estimates: {p}')
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            # if plot_log:
            #     plt.xscale('log')
            #     plt.yscale('log')

            # save
            save_fn = f'{prefix}_{p}.pdf'
            plt.savefig(save_fn, format='pdf', dpi=300, bbox_inches='tight')
            plt.clf()
            plt.close()

        # done
        return
    
    def plot_emp_ci(self, save_fn, est_label, title='Estimates', color='black',
                    plot_log=True):
        """Plots point estimates and CPIs.

        This function plots the point estimates and calibrated prediction
        intervals for the new dataset, if it exists.

        Args:
            save_fn (str): Filename to save plot.
            est_label (numpy.array): Estimated values from new dataset.
            title (str): Title for the plot.
            color (str): Color of histograms
            plot_log (bool): Plot y-axis on log scale? Default True.

        """

        # abort if no labels from Estimate found
        if est_label is None:
            return

        # figure size
        fig_width = 5
        fig_height = 5

        # data dimensions
        label_names = [ f.replace('_value','') for f in est_label.columns.to_list() if '_value' in f ]
        num_label = len(label_names)
        
        # set up plot
        fig = plt.figure(figsize=(fig_width,fig_height))
        
        # use log-scale for y-axis?
        # if plot_log:
        #     plt.yscale('log')

        # plot each estimated label
        for i,lbl in enumerate(label_names):
            # col_data = est_label[col]
            y_value = est_label[lbl+'_value'].iloc[0]
            y_lower = est_label[lbl+'_lower'].iloc[0]
            y_upper = est_label[lbl+'_upper'].iloc[0]

            # if plot_log:
            #     y_value = np.max([1e-4, y_value])
            #     y_lower = np.max([1e-4, y_lower])
            #     y_upper = np.max([1e-4, y_upper])

            s_value = '{:.2E}'.format(y_value)
            s_lower = '{:.2E}'.format(y_lower)
            s_upper = '{:.2E}'.format(y_upper)
            
            # plot CI
            plt.plot([i,i], [y_lower, y_upper],
                     color=color, linestyle="-",
                     marker='_', linewidth=1.5)
            
            # plot values as text
            for y_,s_ in zip([y_value,y_lower,y_upper],
                             [s_value, s_lower, s_upper]):
                plt.text(x=i+0.10, y=y_, s=s_,
                         color='black', va='center', size=8)

            # plot point estimate
            plt.scatter(i, y_value, color='white',
                        edgecolors=color, s=60, zorder=3)
            plt.scatter(i, y_value, color='red',
                        edgecolors='white', s=30, zorder=3)
            
        # plot values as text
        plt.title(title)
        plt.xticks(np.arange(num_label), label_names)
        plt.xlim( -0.5, num_label )
        # plt.ylim( )
        plt.xticks(rotation=90, fontsize=10)
        plt.savefig(save_fn, format='pdf', dpi=300, bbox_inches='tight')
        plt.clf()
        plt.close()
        
        # done
        return

    def plot_emp_cat(self, save_fn, est_label, title='Estimates', color='black'):
        """Plots point estimates and CPIs.
    
        This function plots the point estimates and calibrated prediction
        intervals for the new dataset, if it exists.
    
        Args:
            save_fn (str): Filename to save plot.
            est_label (numpy.array): Estimated values from new dataset.
            title (str): Title for the plot.
            color (str): Color of histograms
            plot_log (bool): Plot y-axis on log scale? Default True.
    
        """
    
        # abort if no labels from Estimate found
        if est_label is None:
            return
    
        # data dimensions
        label_names = self.param_name_cat
        num_labels = len(label_names)

        # figure size
        fig_width = 6
        plot_height = 2
        fig_height = num_labels * plot_height
        # set up plot
        fig, axs = plt.subplots(nrows=num_labels, ncols=1, squeeze=False,
                                 figsize=(fig_width, fig_height))
        fig.tight_layout(pad=3)
        
        # fill in plot
        i = 0
        for i in range(num_labels):
            
            # get estimated value
            p = label_names[i]
            est_cats_p = [ x for x in est_label.columns if p in x ]
            ests_p = est_label[est_cats_p].copy()

            # bar plot
            state_labels = [ x.split('_')[-1] for x in est_cats_p ]
            axs[i][0].bar(state_labels, ests_p.iloc[0], color=color, alpha=0.5,
                          edgecolor=color)
            for k,v in enumerate(ests_p.iloc[0]):
                axs[i][0].text(k, v, f'{v:.2f}', ha='center', va='bottom',
                               color='black')
                
            # cosmetics
            axs[i][0].xaxis.set_ticks_position('bottom')
            # axs[i][0].set_xticks(np.arange(len(est_cats_p)))
            # axs[i][0].set_xticklabels(est_cats_p,
            #                           rotation=90, ha='right')
            axs[i][0].set_xlabel(p)
            axs[i][0].set_ylim(0,1)
            # axs[i][0].title.set_text(f'{p}')
            
        # plot values as text
        fig.suptitle(title)
        # plt.xticks(np.arange(num_labels), label_names)
        # plt.xlim( -0.5, num_labels )
        # plt.ylim( )
        plt.savefig(save_fn, format='pdf', dpi=300, bbox_inches='tight')
        plt.clf()
        plt.close()
    
        # done
        return
    
    def plot_train_history(self, history, prefix, train_color='blue',
                           val_color='red'):
        """Plot training history for network.

        This function plots trained network performance metrics as a time-series
        across training epochs. Typically, it will compare performance between
        training vs. validation examples.

        Args:
            history (DataFrame): Training performance metrics
            prefix (str): Used to construct filename
            train_color (str): Color for training example metrics
            val_color (str): Color for validation example metrics

        """

        # get data names/dimensions
        epochs        = sorted(np.unique(history['epoch']))
        dataset_names = sorted(np.unique(history['dataset']))
        metric_names  = sorted(np.unique(history['metric']))
        # num_datasets  = len(dataset_names)
        num_metrics   = len(metric_names)
 
        # figure dimensions
        fig_width = 6
        fig_height = int(np.ceil(1.5*num_metrics))

        # figure colors
        colors = { 'train': train_color,
                   'validation': val_color }

        # plot for all parameters
        fig, axs = plt.subplots(nrows=num_metrics, ncols=1, sharex=True,
                                figsize=(fig_width, fig_height))
        
        # plot for all metrics
        for j,v2 in enumerate(metric_names):

            # plot training example metrics
            legend_handles = []
            legend_labels = []
            for k,v3 in enumerate(dataset_names):
                df = history.loc[ (history.metric == v2) &
                                  (history.dataset == v3) ]

                lines_train, = axs[j].plot(epochs, df.value,
                                             color=colors[v3],
                                             label=v2)
                axs[j].scatter(epochs, df.value,
                                 color=colors[v3],
                                 label=v2,
                                 zorder=3)
                
                axs[j].set(ylabel=metric_names[j])
                
                legend_handles.append( lines_train )
                legend_labels.append( v3.capitalize() )

            # plot legend
            if j == 0:
                axs[j].legend(handles=legend_handles,
                              labels=legend_labels,
                              loc='upper right' )
        
        fig.supxlabel('Epochs')
        fig.supylabel('Metrics')
        fig.suptitle('Training history')
        fig.tight_layout()

        # save figure
        save_fn = f'{prefix}.pdf'

        # print(save_fn)
        plt.savefig(save_fn, format='pdf', dpi=300, bbox_inches='tight')
        plt.clf()
        plt.close()
        
        # done
        return

##################################################

    def combine_plots(self):
        """Combine all plots.
        
        This function collects all pdfs in the plot project directory, orders
        them into meaningful groups, then plots a merged report.
    
        """
    
        # collect and sort file names
        files_unsorted = os.listdir(self.plt_dir)
        files_unsorted.sort()
        files = []
    
        for f in files_unsorted:
            tok = f.split('.')
            has_prefix = self.plt_prefix == tok[0]
            has_all_not = 'summary' not in tok[1:-1]
            has_pdf = 'pdf' == tok[-1]
            if all([has_pdf, has_prefix, has_all_not]):
                files.append(f)
    
        # get files for different categories
        files_emp_real   = self.filter_files(files, 'empirical_estimate_real')
        files_emp_cat    = self.filter_files(files, 'empirical_estimate_cat')
        files_pca        = self.filter_files(files, 'pca')
        files_density    = self.filter_files(files, 'density')
        files_train      = self.filter_files(files, 'train_estimate')
        files_test       = self.filter_files(files, 'test_estimate')
        files_arch       = self.filter_files(files, 'architecture')
        files_history    = self.filter_files(files, 'train_history')
    
        # construct ordered list of files
        files_ordered = files_emp_real + files_emp_cat + files_pca + files_density + \
                        files_train + files_test + files_history + files_arch
    
        # combine pdfs
        merger = PdfMerger()
        for f in files_ordered:
            merger.append( f'{self.plt_dir}/{f}' )
    
        # write combined pdf
        merger.write(self.save_summary_fn)
    
        # done
        return
    
    @staticmethod
    def filter_files(files, pattern):
        ret = []
        for f in files:
            if pattern in '.'.join(f.split('.')[-2:]):
                ret.append(f)
        return ret
    
    ##################################################
    
    def make_report(self):
        """Makes CSV of main results."""
    
        df = pd.DataFrame(columns=['id1','id2','metric','variable','value'])
    
        # TODO: dataset sizes
        # - num examples in train/test/val/cal
        # - tree width
        # - num char col
        # - num brlen col
        # - num total col
    
        # prediction stats
        test_train = []
        if self.has_train_real:
            test_train.append(('train', self.train_true_real, self.train_est_real))
        if self.has_test_real:
            test_train.append(('test', self.test_true_real, self.test_est_real))
        if self.has_train_cat:
            pass
        if self.has_test_cat:
            pass
        if self.has_emp_real:
            pass
        if self.has_emp_cat:
            pass
        
        for name, lbl, est in test_train:
            for col in lbl:
                # get stats
                mae = np.mean(np.abs(lbl[col] - est[col+'_value']))
                mse = np.mean((lbl[col] - est[col+'_value'])**2)
                mape = np.mean(np.abs((lbl[col] - est[col+'_value']) / lbl[col]))
                cov = np.mean(np.logical_and(est[col+'_lower'] < lbl[col],
                                             est[col+'_upper'] > lbl[col]))
                ci_width = est[col+'_upper'] - est[col+'_lower']
                rel_ci_width = np.divide(ci_width, est[col+'_value'])
                # store stats
                df.loc[len(df)] = [ name, 'true', 'mean', col, np.mean(lbl[col]) ]
                df.loc[len(df)] = [ name, 'true', 'var', col, np.var(lbl[col]) ]
                df.loc[len(df)] = [ name, 'true', 'lower95', col, np.quantile(lbl[col], 0.025) ]
                df.loc[len(df)] = [ name, 'true', 'upper95', col, np.quantile(lbl[col], 0.975) ]
                df.loc[len(df)] = [ name, 'est', 'mean', col, np.mean(est[col+'_value']) ]
                df.loc[len(df)] = [ name, 'est', 'var', col, np.var(est[col+'_value']) ]
                df.loc[len(df)] = [ name, 'est', 'lower95', col, np.quantile(est[col+'_value'], 0.025) ]
                df.loc[len(df)] = [ name, 'est', 'upper95', col, np.quantile(est[col+'_value'], 0.975) ]
                df.loc[len(df)] = [ name, 'est', 'mae', col, mae ]
                df.loc[len(df)] = [ name, 'est', 'mse', col, mse ]
                df.loc[len(df)] = [ name, 'est', 'mape', col, mape ]
                df.loc[len(df)] = [ name, 'est', 'coverage', col, cov ]
                df.loc[len(df)] = [ name, 'est', 'mean_CI_width', col, np.mean(ci_width) ]
                df.loc[len(df)] = [ name, 'est', 'mean_rel_CI_width', col, np.mean(rel_ci_width) ]
    
        # TODO: auxiliary data
        # - similar stuff as prediction for aux data
    
        # TODO: empirical estimate
        # - values against empirical datasets
        # - quantile against training/test datasets
    
        # TODO: training stats
        # - best epoch
    
        # save results
        df.to_csv(self.save_report_fn, index=False, float_format=util.PANDAS_FLOAT_FMT_STR)
    
        return

##################################################
