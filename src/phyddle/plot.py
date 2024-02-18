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

#------------------------------------------------------------------------------#

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

#------------------------------------------------------------------------------#

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

        # initialize with phyddle settings
        self.set_args(args)
        self.prepare_filepaths()
        return

    def set_args(self, args):
        """Assigns phyddle settings as Plotter attributes.

        Args:
            args (dict): Contains phyddle settings.
        """

        self.args = args
        step_args = util.make_step_args('P', args)
        for k,v in step_args.items():
            setattr(self, k, v)
        return

    def prepare_filepaths(self):
        """Prepare filepaths for the project.

        This script generates all the filepaths for input and output based off
        of Trainer attributes. The Format, Train, and Estimate directories
        are input and the Plot directory is output.

        """

        # directories
        # self.fmt_proj_dir = f'{self.work_dir}/{self.fmt_proj}/{self.fmt_dir}'
        # self.trn_proj_dir = f'{self.work_dir}/{self.trn_proj}/{self.trn_dir}'
        # self.est_proj_dir = f'{self.work_dir}/{self.est_proj}/{self.est_dir}'
        # self.plt_proj_dir = f'{self.work_dir}/{self.plt_proj}/{self.plt_dir}'
        self.fmt_proj_dir = f'{self.fmt_dir}'
        self.trn_proj_dir = f'{self.trn_dir}'
        self.est_proj_dir = f'{self.est_dir}'
        self.plt_proj_dir = f'{self.plt_dir}'

        # prefixes
        network_prefix          = f'network_nt{self.tree_width}'
        self.plot_prefix        = f'fig_nt{self.tree_width}'
        fmt_proj_prefix         = f'{self.fmt_proj_dir}/train.nt{self.tree_width}'
        trn_proj_prefix         = f'{self.trn_proj_dir}/{network_prefix}'
        est_proj_prefix         = f'{self.est_proj_dir}/{self.est_prefix}'
        plt_proj_prefix         = f'{self.plt_proj_dir}/{self.plot_prefix}'
        
        # tensors
        self.train_aux_data_fn  = f'{fmt_proj_prefix}.aux_data.csv'
        self.train_labels_fn    = f'{fmt_proj_prefix}.labels.csv'
        self.input_hdf5_fn      = f'{fmt_proj_prefix}.hdf5'

        # network
        self.model_arch_fn      = f'{trn_proj_prefix}.trained_model.pkl'
        self.history_fn         = f'{trn_proj_prefix}.train_history.csv'
        self.train_est_fn       = f'{trn_proj_prefix}.train_est.labels.csv'
        self.train_labels_fn    = f'{trn_proj_prefix}.train_true.labels.csv'
        
        # estimates
        self.est_aux_data_fn    = f'{est_proj_prefix}.aux_data.csv'
        #self.est_known_param_fn = f'{est_proj_prefix}.known_param.csv'
        self.est_lbl_fn         = f'{est_proj_prefix}.emp_est.labels.csv'
        self.test_est_fn        = f'{est_proj_prefix}.test_est.labels.csv'
        self.test_labels_fn     = f'{est_proj_prefix}.test_true.labels.csv'
        
        # plotting output
        self.save_density_aux_fn   = f'{plt_proj_prefix}.density_aux_data.pdf'
        self.save_density_label_fn = f'{plt_proj_prefix}.density_label.pdf'
        self.save_train_est_fn     = f'{plt_proj_prefix}.estimate_train'
        self.save_test_est_fn      = f'{plt_proj_prefix}.estimate_test'
        self.save_pca_aux_data_fn  = f'{plt_proj_prefix}.pca_contour_aux_data.pdf'
        self.save_pca_labels_fn    = f'{plt_proj_prefix}.pca_contour_labels.pdf'
        self.save_cpi_est_fn       = f'{plt_proj_prefix}.estimate_new.pdf'
        self.save_network_fn       = f'{plt_proj_prefix}.network_architecture.pdf'
        self.save_history_fn       = f'{plt_proj_prefix}.train_history'
        self.save_summary_fn       = f'{plt_proj_prefix}.summary.pdf'
        self.save_report_fn        = f'{plt_proj_prefix}.summary.csv'

        return

    def run(self):
        """Generates all plots.

        This method creates the target directory for new plots, generates
        a set of standard summary plots, along with a combined report.

        """

        verbose = self.verbose

        # print header
        util.print_step_header('plt', [self.fmt_proj_dir, self.trn_proj_dir, self.est_proj_dir],
                               self.plt_proj_dir, verbose)

        # prepare workspace
        os.makedirs(self.plt_proj_dir, exist_ok=True)

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

        #done
        util.print_str('... done!', verbose=self.verbose)
        return

    def load_input(self):
        """Load input data for plotting.

        This function loads input from Format, Train, and Estimate. We load the
        simulated training examples from Format. From Train, we load the
        network, the training history, and the test/train estimates/labels. For
        Estimate, we load the estimates and aux. data, if they exist.

        """

        ### load input from Format step
        if self.tensor_format == 'csv':
            # csv tensor format
            self.train_aux_data = pd.read_csv( self.train_aux_data_fn )
            self.train_labels = pd.read_csv( self.train_labels_fn )
        elif self.tensor_format == 'hdf5':
            # hdf5 tensor format
            hdf5_file = h5py.File(self.input_hdf5_fn, 'r')
            train_aux_data_names = [ s.decode() for s in hdf5_file['aux_data_names'][0,:] ]
            train_label_names = [ s.decode() for s in hdf5_file['label_names'][0,:] ]
            self.train_aux_data = pd.DataFrame( hdf5_file['aux_data'][:,:], columns=train_aux_data_names )
            self.train_labels = pd.DataFrame( hdf5_file['labels'][:,:], columns=train_label_names)
            hdf5_file.close()

        # label and aux data column names
        self.param_names = self.train_labels.columns.to_list()
        self.aux_data_names = self.train_aux_data.columns.to_list()

        # trained model
        # self.model = tf.keras.models.load_model(self.model_arch_fn, compile=False)
        self.model = torch.load(self.model_arch_fn)
        
        # training estimates/labels
        self.train_ests   = pd.read_csv(self.train_est_fn)
        self.train_labels = pd.read_csv(self.train_labels_fn)
        
        # test estimates/labels
        self.test_ests    = pd.read_csv(self.test_est_fn)
        self.test_labels  = pd.read_csv(self.test_labels_fn)
        
        # training history for network
        # TODO: Need to get training history from torch
        self.history_table = pd.read_csv(self.history_fn)
        # self.history_dict = json.load(open(self.history_json_fn, 'r'))

        # load new aux data from Estimate
        self.est_aux_data = None
        new_aux_found = os.path.isfile(self.est_aux_data_fn)
        if new_aux_found:
            self.est_aux_data = pd.read_csv(self.est_aux_data_fn)
        # new_param_found = os.path.isfile(self.est_known_param_fn)
        # if new_param_found and new_summ_stat_found:
        #     self.est_known_params = pd.read_csv(self.est_known_param_fn)
        #     self.est_aux_data = pd.concat( [self.est_aux_data, self.est_known_params], axis=1)
        #     self.est_aux_data = self.est_aux_data[self.train_aux_data.columns]
        
        # load new estimates from Estimate
        self.est_lbl_loaded = os.path.isfile(self.est_lbl_fn)
        if self.est_lbl_loaded:
            _data = pd.read_csv(self.est_lbl_fn)
            _value = _data[ [x for x in _data.columns if '_value' in x] ]
            _lower = _data[ [x for x in _data.columns if '_lower' in x] ]
            _upper = _data[ [x for x in _data.columns if '_upper' in x] ]
            _value.columns = [ x.replace('_value','') for x in _value.columns ]
            _lower.columns = [ x.replace('_lower','') for x in _lower.columns ]
            _upper.columns = [ x.replace('_upper','') for x in _upper.columns ]
            self.est_lbl_df = pd.concat([_value, _lower, _upper])
            self.est_lbl_df.index = ['value','lower','upper']
            self.est_lbl_value = pd.DataFrame(_value, columns=self.param_names)
        else:
            self.est_lbl_value = None
            self.est_lbl_df = None

        # done
        return
    
#------------------------------------------------------------------------------#

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
        
        # training aux. data densities
        self.make_plot_stat_density('aux_data')
        
        # training labels histogram
        self.make_plot_stat_density('labels')

        # PCA-contour of training aux. data
        self.make_plot_pca_contour('aux_data')

        # PCA-contour of training aux. data
        self.make_plot_pca_contour('labels')

        # train scatter accuracy
        self.make_plot_scatter_accuracy('train')

        # # test scatter accuracy
        self.make_plot_scatter_accuracy('test')

        # when available, point estimates and CPIs for new dataset
        self.make_plot_est_CI()
        
        # training history stats
        self.make_plot_train_history()

        # network architecture
        self.make_plot_network_architecture()

        #done
        return
    
    def combine_plots(self):
        """Combine all plots.
        
        This function collects all pdfs in the plot project directory, orders
        them into meaningful groups, then plots a merged report.

        """

        # collect and sort file names
        files_unsorted = os.listdir(self.plt_proj_dir)
        files_unsorted.sort()
        files = []
        for f in files_unsorted:
            has_pdf = '.pdf' in f
            has_net = self.plot_prefix in f
            has_all_not = 'all_results' not in f
            if all([has_pdf, has_net, has_all_not]):
                files.append(f)

        # get files for different categories
        files_CPI        = self.filter_files(files, 'estimate_new')
        files_pca        = self.filter_files(files, 'pca_contour')
        files_density    = self.filter_files(files, 'density')
        files_train      = self.filter_files(files, 'estimate_train')
        files_test       = self.filter_files(files, 'estimate_test')
        files_arch       = self.filter_files(files, 'architecture')
        files_history    = self.filter_files(files, 'train_history')

        # construct ordered list of files
        files_ordered = files_CPI + files_pca + files_density + \
                        files_train + files_test + files_history + files_arch
        
        # combine pdfs
        merger = PdfMerger()
        for f in files_ordered:
            merger.append( f'{self.plt_proj_dir}/{f}' )

        # write combined pdf
        merger.write(self.save_summary_fn)

        # done
        return

    def filter_files(self, files, filter):
        ret = []
        for f in files:
            if filter in '.'.join(f.split('.')[-2:]):
                ret.append(f)
        return ret
#------------------------------------------------------------------------------#

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
        test_train = [ ('train', self.train_labels, self.train_ests),
              ('test', self.test_labels, self.test_ests) ]
        
        for name, lbl, est in test_train:
            for col in lbl:
                # get stats
                mae = np.mean(np.abs(lbl[col] - est[col+'_value']))
                mse = np.mean((lbl[col] - est[col+'_value'])**2)
                mape = np.mean(np.abs((lbl[col] - est[col+'_value']) / lbl[col]))
                cov = np.mean(np.logical_and(est[col+'_lower'] < lbl[col],
                                             est[col+'_upper'] > lbl[col]))
                CI_width = est[col+'_upper'] - est[col+'_lower']
                rel_CI_width = np.divide(CI_width, est[col+'_value'])
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
                df.loc[len(df)] = [ name, 'est', 'mean_CI_width', col, np.mean(CI_width) ]
                df.loc[len(df)] = [ name, 'est', 'mean_rel_CI_width', col, np.mean(rel_CI_width) ]

        # TODO: auxiliary data
        # - similar stuff as prediction for aux data

        # TODO: empirical estimate
        # - values against empirical datasets
        # - quantile against training/test datasets

        # TODO: training stats
        # - best epoch
        # - 
                
        # save results
        self.df_report = df
        self.df_report.to_csv(self.save_report_fn, index=False, float_format=util.PANDAS_FLOAT_FMT_STR)

        return

#------------------------------------------------------------------------------#

    def make_plot_stat_density(self, type):
        """Calls plot_stat_density with arguments."""
        assert(type in ['aux_data', 'labels'])
        if type == 'aux_data':
            self.plot_stat_density(save_fn=self.save_density_aux_fn,
                                   sim_values=self.train_aux_data,
                                   est_values=self.est_aux_data,
                                   color=self.plot_aux_color,
                                   title='Aux. data')
        elif type == 'labels':
            self.plot_stat_density(save_fn=self.save_density_label_fn,
                                   sim_values=self.train_labels,
                                   est_values=self.est_lbl_value,
                                   color=self.plot_label_color,
                                   title='Labels' )
                
        return
    
    def make_plot_scatter_accuracy(self, type):
        """Calls plot_scatter_accuracy with arguments."""
        assert(type in ['train', 'test'])

        n_max = 250
        if type == 'train':
            # plot train scatter
            n = min(n_max, self.train_ests.shape[0])
            self.plot_scatter_accuracy(ests=self.train_ests.iloc[0:n],
                                       labels=self.train_labels.iloc[0:n],
                                       prefix=self.save_train_est_fn,
                                       color=self.plot_train_color,
                                       title='Train')
        elif type == 'test':
            # plot test scatter
            n = min(n_max, self.test_ests.shape[0])
            self.plot_scatter_accuracy(ests=self.test_ests.iloc[0:n],
                                       labels=self.test_labels.iloc[0:n],
                                       prefix=self.save_test_est_fn,
                                       color=self.plot_test_color,
                                       title='Test')
        # done
        return

    def make_plot_pca(self):
        """Calls plot_PCA with arguments."""
        self.plot_pca(save_fn=self.save_pca_aux_fn,
                      sim_values=self.train_aux_data,
                      est_values=self.est_aux_data,
                      color=self.plot_aux_color)
        return
    
    def make_plot_pca_contour(self, type):
        """Calls plot_pca_contour with arguments."""
        if type == 'aux_data':
            self.plot_pca_contour(save_fn=self.save_pca_aux_data_fn,
                                  sim_values=self.train_aux_data,
                                  est_values=self.est_aux_data,
                                  color=self.plot_aux_color,
                                  title='Aux. data')
        elif type == 'labels':
            self.plot_pca_contour(save_fn=self.save_pca_labels_fn,
                                  sim_values=self.train_labels,
                                  est_values=self.est_lbl_value,
                                  color=self.plot_label_color,
                                  title='Labels')
        return

    def make_plot_est_CI(self):
        """Calls plot_est_CI with arguments."""
        self.plot_est_CI(save_fn=self.save_cpi_est_fn,
                         est_label=self.est_lbl_df,
                         title=f'Estimate: {self.est_dir}/{self.est_prefix}',
                         color=self.plot_est_color)
        return

    def make_plot_train_history(self):
        """Calls plot_train_history with arguments."""
        self.plot_train_history(self.history_table,
                                prefix=self.save_history_fn,
                                train_color=self.plot_train_color,
                                val_color=self.plot_val_color)
        return

    def make_plot_network_architecture(self):
        """Calls torchview.draw_graph with arguments."""
        
        phy_dat_fake = torch.empty( self.model.phy_dat_shape, dtype=torch.float32 )[None,:,:]
        aux_dat_fake = torch.empty( self.model.aux_dat_shape, dtype=torch.float32 )[None,:]
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
    
#------------------------------------------------------------------------------#        

    def plot_stat_density(self, save_fn, sim_values, est_values=None,
                           title='', ncol_plot=3, color='blue'):
        """Plots histograms.

        This function plots the histograms (KDEs) for simulated training
        examples, e.g. aux. data or labels. The function will also plot
        values from the new dataset if it is available (est_values != None).

        Args:
            save_fn (str): Filename to save plot.
            sim_values (numpy.array): Simulated values from training examples.
            est_values (numpy.array): Estimated values from new dataset.
            title (str): Plot title.
            ncol_plot (int): Number of columns in plot
            color (str): Color of histograms

        """
        
        # data dimensions
        col_names = sorted( sim_values.columns )
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
                x = sorted(sim_values[p])
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
                
                if est_values is not None and p in est_values:
                    x_data = est_values[p][0]
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

        # done
        return
    
    def plot_pca_contour(self, save_fn, sim_values, est_values=None,
                         num_comp=4, color='blue', title=''):
        """
        Plots PCA Contour Plot.

        This function plots the PCA for simulated training aux. data examples.
        The function plots a grid of pairs of principal components. It will also
        plot values from the new dataset, when vailable (est_values != None).

        Args:
            save_fn (str): Filename to save plot.
            sim_values (numpy.array): Simulated values from training examples.
            est_values (numpy.array): Estimated values from new dataset.
            num_comp (int): Number of components to plot (default 4)
            f_show (float): Proportion of scatter points to show in PCs
            color (str): Color of histograms

        """
        
        # figure size
        fig_width = 8
        fig_height = 8 

        # reduce num components if needed
        num_comp = min(sim_values.shape[1], num_comp)

        # rescale input data
        sim_values = np.log(sim_values + self.log_offset)
        scaler = StandardScaler()
        x = scaler.fit_transform(sim_values)
        
        # apply PCA to sim_values
        pca_model = PCA(n_components=num_comp)
        pca = pca_model.fit_transform(x)
        pca_var = pca_model.explained_variance_ratio_
        pca_coef = np.transpose(pca_model.components_)
        plot_pca_loadings = False

        # project est_values on to PCA space
        if est_values is not None:
            est_values = np.log(est_values + self.log_offset)
            est_values = scaler.transform(est_values)
            pca_est = pca_model.transform(est_values)
        
        # figure dimennsions
        fig, axs = plt.subplots(num_comp-1, num_comp-1,
                                sharex=True, sharey=True,
                                figsize=(fig_width, fig_height))

        # use this to turn off subplots
        #axes[i_row,j_col].axis('off')

        # color map
        cmap0 = mpl.colors.LinearSegmentedColormap.from_list('white2col',
                                                             ['white', color])

        # generate PCA subplots
        for i in range(0, num_comp-1):
            
            # disable x,y axes for all plots, later turned on for some
            for j in range(i+1, num_comp-1):
                axs[i,j].axis('off')

            for j in range(0, i+1):
                # countours
                x = pca[:,i+1]
                y = pca[:,j]
                xmin, xmax = np.min(x), np.max(x)
                ymin, ymax = np.min(y), np.max(y)
                xscale = (xmax - xmin)
                yscale = (ymax - ymin)

                # Peform the kernel density estimate
                xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
                positions = np.vstack([xx.ravel(), yy.ravel()])
                values = np.vstack([x, y])
                kernel = sp.stats.gaussian_kde(values)
                f = np.reshape(kernel(positions).T, xx.shape)
                n_levels = 6
                extent = (-4,4,-4,4)
                cfset = axs[i,j].contourf(xx, yy, f, levels=n_levels,
                                          extend='both', cmap=cmap0)
                cset = axs[i,j].contour(xx, yy, f, levels=n_levels, 
                                        extend='both', linewidths=0.5,
                                        colors='k')
                axs[i,j].clabel(cset, inline=1, fontsize=6)

                # loads
                if plot_pca_loadings:
                    nn = pca_coef.shape[0]
                    for k in range(nn):
                        axs[i,j].arrow(x=0,y=0,
                                    dx=pca_coef[k,i]*yscale/2,
                                    dy=pca_coef[k,j]*xscale/2,
                                    color='black',alpha=0.75, width=0.05)
                        axs[i,j].text(pca_coef[k,i]*yscale,
                                    pca_coef[k,j]*xscale,
                                    self.train_aux_data_names[k],
                                    color='black', fontsize=8,
                                    ha='center',va='center')
                    
                if est_values is not None:    
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
                    idx=str(j+1)
                    var=int(100*round(pca_var[j], ndigits=2))
                    xlabel = f'PC{idx} ({var}%)'
                    axs[i,j].set_xlabel(xlabel, fontsize=12)
                
        plt.tight_layout()
        fig.suptitle(f'PCA: {title}')
        fig.tight_layout(rect=[0, 0.03, 1, 0.98])
        plt.savefig(save_fn, format='pdf', dpi=300, bbox_inches='tight')
        plt.clf()

        #done
        return

    def plot_scatter_accuracy(self, ests, labels, prefix,
                              color="blue", axis_labels = ["truth", "truth"],
                              title = '', plot_log=False):
        """Plots accuracy of estimates and CPIs for labels.

        This function generates a scatterplot for true vs. estimated labels
        from the trained network. Points are point estimates. Bars are
        CPIs.

        Args:
            save_fn (str): Filename to save plot.
            est_label (numpy.array): Estimated values from new dataset.
            title (str): Title for the plot.
            color (str): Color of histograms
            plot_log (bool): Plot y-axis on log scale? Default True.

        """
        # figure size
        fig_width = 6
        fig_height = 6

        # create figure
        plt.figure(figsize=(fig_width,fig_height))

        # plot parameters
        for i,p in enumerate(self.param_names):

            # labels
            x_label = f'{p} {axis_labels[0]}'
            y_label = f'{p} {axis_labels[1]}'

            # estimates (x) and true values (y)
            lbl_est = ests[f'{p}_value'][:].to_numpy()
            lbl_lower = ests[f'{p}_lower'][:].to_numpy()
            lbl_upper = ests[f'{p}_upper'][:].to_numpy()
            lbl_true = labels[p][:].to_numpy()

            only_positive = np.all(lbl_true >= 0.)
            if only_positive and plot_log:
                lbl_est = np.log(lbl_est)
                lbl_lower = np.log(lbl_lower)
                lbl_upper = np.log(lbl_upper)
                lbl_true = np.log(lbl_true)
                x_label = f'ln {p} {axis_labels[0]}'
                y_label = f'ln {p} {axis_labels[1]}'
                

            # accuracy stats
            stat_mae = np.mean( np.abs(lbl_est - lbl_true) )
            stat_mape = 100 * np.mean( np.abs(lbl_est - lbl_true) / lbl_true )
            stat_mse = np.mean( np.power(lbl_est - lbl_true, 2) )
            stat_rmse = np.sqrt( stat_mse )
            
            # coverage stats
            stat_cover = np.logical_and(lbl_lower < lbl_true, lbl_upper > lbl_true )
            stat_not_cover = np.logical_not(stat_cover)
            f_stat_cover = sum(stat_cover) / len(stat_cover) * 100

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
            
            alpha = 0.5 # 50. / len(y_cover)

            # covered points
            plt.scatter(lbl_true[stat_cover], lbl_est[stat_cover],
                        alpha=alpha, c=color, zorder=3, s=3)
            # covered bars
            plt.plot([lbl_true[stat_cover], lbl_true[stat_cover]],
                     [lbl_lower[stat_cover], lbl_upper[stat_cover]],
                     color=color, alpha=alpha, linestyle="-", marker='_',
                     linewidth=0.5, zorder=2 )



            # not covered points
            plt.scatter(lbl_true[stat_not_cover], lbl_est[stat_not_cover], 
                        alpha=alpha, c='red', zorder=5, s=3)
            # not covered bars
            plt.plot([lbl_true[stat_not_cover], lbl_true[stat_not_cover]],
                     [lbl_lower[stat_not_cover], lbl_upper[stat_not_cover]],
                     color='red', alpha=alpha, linestyle="-", marker='_',
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
                        f'Slope: {s_slope}', f'Coverage: {s_cover}' ]
            
            for j,s in enumerate(stat_str):
                plt.annotate(s, xy=(0.01,0.99-j*dx),
                         xycoords='axes fraction', fontsize=10,
                         horizontalalignment='left', verticalalignment='top',
                         color='black')

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

        # done    
        return
    
    def plot_est_CI(self, save_fn, est_label, title='Estimates', color='black',
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
        label_names = est_label.columns
        num_label = len(label_names)
        
        # set up plot
        plt.figure(figsize=(fig_width,fig_height))      
        
        # use log-scale for y-axis?
        if plot_log:
            plt.yscale('log')

        # plot each estimated label
        for i,col in enumerate(label_names):
            col_data = est_label[col]
            y_value = col_data.loc['value']
            y_lower = col_data.loc['lower']
            y_upper = col_data.loc['upper']

            if plot_log:
                y_value = np.max([1e-4, y_value])
                y_lower = np.max([1e-4, y_lower])
                y_upper = np.max([1e-4, y_upper])

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
        plt.ylim( )
        plt.savefig(save_fn, format='pdf', dpi=300, bbox_inches='tight')
        plt.clf()
        
        #done
        return
    
    def plot_train_history(self, history, prefix, train_color='blue',
                          val_color='red'):
        """Plot training history for network.

        This function plots trained network performance metrics as a time-series
        across training epochs. Typically, it will compare performance between
        trainiing vs. validation examples.

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
                df = history.loc[ (history.metric==v2) & \
                                  (history.dataset==v3) ]

                lines_train, = axs[j].plot(epochs, df.value,
                                             color=colors[v3],
                                             label = v2)
                axs[j].scatter(epochs, df.value,
                                 color=colors[v3],
                                 label = v2,
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
        
        # done
        return

#------------------------------------------------------------------------------#