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
import json
import os

# external imports
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import tensorflow as tf
from pypdf import PdfMerger
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# phyddle imports
from phyddle import utilities as util

#------------------------------------------------------------------------------#

def load(args):
    """
    Load a Plotter object.

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
        """
        Initializes a new Plotter object.

        Args:
            args (dict): Contains phyddle settings.
        """
        # initialize with phyddle settings
        self.set_args(args)
        self.prepare_filepaths()
        return

    def set_args(self, args):
        """
        Assigns phyddle settings as Plotter attributes.

        Args:
            args (dict): Contains phyddle settings.
        """
        self.args              = args
        self.verbose           = args['verbose']
        self.fmt_dir           = args['fmt_dir']
        self.trn_dir           = args['trn_dir']
        self.plt_dir           = args['plt_dir']
        self.est_dir           = args['est_dir'] if 'est_dir' in args else ''
        self.est_prefix        = args['est_prefix'] if 'est_prefix' in args else ''
        self.fmt_proj          = args['fmt_proj']
        self.trn_proj          = args['trn_proj']
        self.plt_proj          = args['plt_proj']
        self.est_proj          = args['est_proj']
        self.batch_size        = args['trn_batch_size']
        self.num_epochs        = args['num_epochs']
        self.tree_width        = args['tree_width']
        self.tensor_format     = args['tensor_format']
        self.train_color       = args['plot_train_color']
        self.test_color        = args['plot_test_color']
        self.validation_color  = args['plot_val_color']
        self.aux_color         = args['plot_aux_color']
        self.label_color       = args['plot_label_color']
        self.est_color         = args['plot_est_color']
        return

    def prepare_filepaths(self):
        """
        Prepare filepaths for the project.

        This script generates all the filepaths for input and output based off
        of Trainer attributes. The Format, Train, and Estimate directories
        are input and the Plot directory is output.

        Returns: None
        """
        # directories
        self.trn_proj_dir       = f'{self.trn_dir}/{self.trn_proj}'
        self.fmt_proj_dir       = f'{self.fmt_dir}/{self.fmt_proj}'
        self.est_proj_dir       = f'{self.est_dir}/{self.est_proj}'
        self.plt_proj_dir       = f'{self.plt_dir}/{self.plt_proj}'

        # prefixes
        self.network_prefix     = f'train_batchsize{self.batch_size}_numepoch{self.num_epochs}_nt{self.tree_width}'
        self.fmt_proj_prefix    = f'{self.fmt_proj_dir}/sim.nt{self.tree_width}'
        self.trn_proj_prefix    = f'{self.trn_proj_dir}/{self.network_prefix}'
        self.est_proj_prefix    = f'{self.est_proj_dir}/{self.est_prefix}'
        self.plt_proj_prefix    = f'{self.plt_proj_dir}/{self.network_prefix}'
        
        # tensors
        self.input_aux_data_fn     = f'{self.fmt_proj_prefix}.summ_stat.csv'
        self.input_labels_fn    = f'{self.fmt_proj_prefix}.labels.csv'
        self.input_hdf5_fn      = f'{self.fmt_proj_prefix}.hdf5'

        # network
        self.train_est_fn       = f'{self.trn_proj_prefix}.train_est.csv'
        self.train_labels_fn    = f'{self.trn_proj_prefix}.train_labels.csv'
        self.test_est_fn        = f'{self.trn_proj_prefix}.test_est.csv'
        self.test_labels_fn     = f'{self.trn_proj_prefix}.test_labels.csv'
        self.network_fn         = f'{self.trn_proj_prefix}.hdf5'
        self.history_json_fn    = f'{self.trn_proj_prefix}.train_history.json'

        # estimates
        self.est_summ_stat_fn   = f'{self.est_proj_prefix}.summ_stat.csv'
        self.est_lbl_fn         = f'{self.est_proj_prefix}.{self.network_prefix}.est_labels.csv'
        self.est_known_param_fn = f'{self.est_proj_prefix}.known_param.csv'
        
        # plotting output
        self.save_hist_aux_fn   = f'{self.plt_proj_prefix}.density_aux.pdf'
        self.save_hist_label_fn = f'{self.plt_proj_prefix}.density_label.pdf'
        self.save_pca_aux_fn    = f'{self.plt_proj_prefix}.pca_aux.pdf'
        self.save_cpi_est_fn    = f'{self.plt_proj_prefix}.est_CPI.pdf'
        self.save_network_fn    = f'{self.plt_proj_prefix}.network_architecture.pdf'
        self.save_summary_fn    = f'{self.plt_proj_prefix}.summary.pdf'

        return

    def run(self):
        """
        Generates all plots.

        This method creates the target directory for new plots, generates
        a set of standard summary plots, along with a combined report.
        """
        verbose = self.verbose

        # print header
        util.print_step_header('plt', [self.fmt_proj_dir, self.trn_proj_dir, self.est_proj_dir],
                               self.plt_proj_dir, verbose)

        # prepare workspace
        os.makedirs(self.plt_proj_dir, exist_ok=True)

        # load input
        util.print_str('▪ Loading input ...', verbose)
        self.load_input()

        # generate individual plots
        util.print_str('▪ Generating individual plots ...', verbose)
        self.generate_plots()

        # combining all plots
        util.print_str('▪ Combining plots ...', verbose)
        self.combine_plots()

        #done
        util.print_str('... done!', verbose=self.verbose)
        return

    def load_input(self):
        """
        Load input data for plotting.

        This function loads input from Format, Train, and Estimate. We load the
        simulated training examples from Format. From Train, we load the
        network, the training history, and the test/train estimates/labels. For
        Estimate, we load the estimates and aux. data, if they exist.
        """

        ### load input from Format step
        if self.tensor_format == 'csv':
            # csv tensor format
            self.input_aux_data = pd.read_csv( self.input_aux_data_fn )
            self.input_labels = pd.read_csv( self.input_labels_fn )
        elif self.tensor_format == 'hdf5':
            # hdf5 tensor format
            hdf5_file = h5py.File(self.input_hdf5_fn, 'r')
            self.input_aux_data_names = [ s.decode() for s in hdf5_file['aux_data_names'][0,:] ]
            self.input_label_names = [ s.decode() for s in hdf5_file['label_names'][0,:] ]
            self.input_aux_data = pd.DataFrame( hdf5_file['aux_data'][:,:], columns=self.input_aux_data_names )
            self.input_labels = pd.DataFrame( hdf5_file['labels'][:,:], columns=self.input_label_names )
            hdf5_file.close()

        ### load input from Train step
        self.model = tf.keras.models.load_model(self.network_fn, compile=False)
        # training estimates/labels
        self.train_ests  = pd.read_csv(self.train_est_fn)
        self.train_labels = pd.read_csv(self.train_labels_fn)
        # test estimates/labels
        self.test_ests   = pd.read_csv(self.test_est_fn)
        self.test_labels  = pd.read_csv(self.test_labels_fn)
        # parameter names from training labels
        self.param_names  = self.train_ests.columns.to_list()
        self.param_names  = [ '_'.join(s.split('_')[:-1]) for s in self.param_names  ]
        self.param_names  = np.unique(self.param_names)
        # training history for network
        self.history_dict = json.load(open(self.history_json_fn, 'r'))

        ### load input from Estimate step
        # read in aux. data from new dataset, if it exists
        self.est_aux_loaded = os.path.isfile(self.est_summ_stat_fn) 
        if self.est_aux_loaded:
            self.est_aux_data = pd.read_csv(self.est_summ_stat_fn)
        self.est_known_param_loaded = os.path.isfile(self.est_known_param_fn)
        if self.est_known_param_loaded:
            self.est_known_params = pd.read_csv(self.est_known_param_fn) #, sep=',', index_col=False).to_numpy().flatten()
            self.est_aux_data = pd.concat( [self.est_aux_data, self.est_known_params], axis=1)
            self.est_aux_data = self.est_aux_data[self.input_aux_data.columns]
        if not self.est_aux_loaded and not self.est_known_param_loaded:
            self.est_aux_data = None

        # read in estimates from new dataset, if it exists
        self.est_lbl_loaded = os.path.isfile(self.est_lbl_fn)
        if self.est_lbl_loaded:
            self.est_lbl_data = pd.read_csv(self.est_lbl_fn)
            self.est_lbl_value = self.est_lbl_data[ [col for col in self.est_lbl_data.columns if 'value' in col] ]
            self.est_lbl_lower = self.est_lbl_data[ [col for col in self.est_lbl_data.columns if 'lower' in col] ]
            self.est_lbl_upper = self.est_lbl_data[ [col for col in self.est_lbl_data.columns if 'upper' in col] ]
            self.est_lbl_value.columns = [ col.rstrip('_value') for col in self.est_lbl_value.columns ]
            self.est_lbl_lower.columns = [ col.rstrip('_lower') for col in self.est_lbl_lower.columns ]
            self.est_lbl_upper.columns = [ col.rstrip('_upper') for col in self.est_lbl_upper.columns ]
            self.est_lbl_df = pd.concat( [self.est_lbl_value, self.est_lbl_lower, self.est_lbl_upper] )
        else:
            self.est_lbl_data = None
            self.est_lbl_value = None
            self.est_lbl_lower = None
            self.est_lbl_upper = None
            self.est_lbl_df = None

        # done
        return
    
#------------------------------------------------------------------------------#

    def generate_plots(self):
        """
        Generates all plots.
        
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
        
        # PCA of training aux. data
        self.make_plot_pca()

        # train scatter accuracy
        self.make_plot_scatter_accuracy('train')

        # test scatter accuracy
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
        """
        Combine all plots.
        
        This function collects all pdfs in the plot project directory, orders
        them into meaningful groups, then plots a merged report.
        """

        # collect and sort file names
        files_unsorted = os.listdir(self.plt_proj_dir)
        files_unsorted.sort()
        files = []
        for f in files_unsorted:
            has_pdf = '.pdf' in f
            has_net = self.network_prefix in f
            has_all_not = 'all_results' not in f
            if all([has_pdf, has_net, has_all_not]):
                files.append(f)

        # get files for different categories
        files_CPI        = self.filter_files(files, 'CPI')
        files_pca        = self.filter_files(files, 'pca')
        files_density    = self.filter_files(files, 'density')
        files_train      = self.filter_files(files, 'train')
        files_test       = self.filter_files(files, 'test')
        files_arch       = self.filter_files(files, 'architecture')
        files_history    = self.filter_files(files, 'history')

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

    def make_plot_stat_density(self, type):
        """Calls plot_stat_density with arguments."""
        assert(type in ['aux_data', 'labels'])
        if type == 'aux_data':
            self.plot_stat_density(save_fn=self.save_hist_aux_fn,
                                   sim_values=self.input_aux_data,
                                   est_values=self.est_aux_data,
                                   color=self.aux_color,
                                   title='Aux. data')
        elif type == 'labels':
            self.plot_stat_density(save_fn=self.save_hist_label_fn,
                                   sim_values=self.input_labels,
                                   est_values=self.est_lbl_value,
                                   color=self.label_color,
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
                                       param_names=self.param_names,
                                       prefix=f'{self.network_prefix}.train',
                                       color=self.train_color,
                                       plot_dir=self.plt_proj_dir,
                                       title='Train')
        elif type == 'test':
            # plot test scatter
            n = min(n_max, self.test_ests.shape[0])
            self.plot_scatter_accuracy(ests=self.test_ests.iloc[0:n],
                                       labels=self.test_labels.iloc[0:n],
                                       param_names=self.param_names,
                                       prefix=f'{self.network_prefix}.test',
                                       color=self.test_color,
                                       plot_dir=self.plt_proj_dir,
                                       title='Test')
        # done
        return

    def make_plot_pca(self):
        """Calls plot_PCA with arguments."""
        self.plot_pca(save_fn=self.save_pca_aux_fn,
                      sim_values=self.input_aux_data,
                      est_values=self.est_aux_data,
                      color=self.aux_color)
        return

    def make_plot_est_CI(self):
        """Calls plot_est_CI with arguments."""
        self.plot_est_CI(save_fn=self.save_cpi_est_fn,
                         est_label=self.est_lbl_df,
                         title=f'Estimate: {self.est_dir}/{self.est_prefix}',
                         color=self.est_color)
        return

    def make_plot_train_history(self):
        prefix = f'{self.plt_proj_dir}/{self.network_prefix}.history'
        self.plot_train_history(self.history_dict,
                                prefix=prefix,
                                train_color=self.train_color,
                                val_color=self.validation_color)
        return

    def make_plot_network_architecture(self):
        """Calls tf.keras.utils.plot_model with arguments."""
        tf.keras.utils.plot_model(self.model,
                                  to_file=self.save_network_fn,
                                  show_shapes=True)
        return
    
#------------------------------------------------------------------------------#        

    def plot_stat_density(self, save_fn, sim_values, est_values=None,
                           title='', ncol_plot=3, color='blue'):
        """
        Plots histograms.

        This function plots the histograms (KDEs) for simulated training
        examples, e.g. aux. data or labels. The function will also plot
        values from the new dataset if it is available (est_values != None).

        Arguments:
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

        fig.suptitle(f'Histogram: {title}')
        fig.tight_layout(rect=[0, 0.03, 1, 0.98])
        plt.savefig(fname=save_fn, format='pdf', dpi=300, bbox_inches='tight')
        plt.clf()

        # done
        return

    def plot_pca(self, save_fn, sim_values, est_values=None, num_comp=4, f_show=0.10, color='blue'):
        """
        Plots PCA.

        This function plots the PCA for simulated training aux. data examples.
        The function plots a grid of pairs of principal components. It will also
        plot values from the new dataset, when vailable (est_values != None).

        Arguments:
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

        # rescale input data
        scaler = StandardScaler()
        x = scaler.fit_transform(sim_values)
        
        # thin dataset
        nrow_keep = int(x.shape[0] * f_show)
        alpha = np.min( [1, 100 / nrow_keep] )
        
        # apply PCA to sim_values
        pca_model = PCA(n_components=num_comp)
        pca = pca_model.fit_transform(x)
        pca_var = pca_model.explained_variance_ratio_
        
        # project est_values on to PCA space
        if self.est_aux_loaded:
            est_values = scaler.transform(est_values)
            pca_est = pca_model.transform(est_values)
        
        # figure dimennsions
        fig, axs = plt.subplots(num_comp-1, num_comp-1, sharex=True, sharey=True, figsize=(fig_width, fig_height))

        # use this to turn off subplots
        #axes[i_row,j_col].axis('off')

        # generate PCA subplots
        for i in range(0, num_comp-1):
            for j in range(i+1, num_comp-1):
                axs[i,j].axis('off')
            for j in range(0, i+1):
                # scatter plots
                axs[i,j].scatter( pca[0:nrow_keep,i+1], pca[0:nrow_keep,j], alpha=alpha, marker='x', color=color )
                if self.est_aux_loaded:    
                    axs[i,j].scatter(pca_est[0:nrow_keep,i+1], pca_est[0:nrow_keep,j],
                                     alpha=1.0, color='white', edgecolor='black', s=80)
                    axs[i,j].scatter(pca_est[0:nrow_keep,i+1], pca_est[0:nrow_keep,j],
                                     alpha=1.0, color='red', edgecolor='white', s=40)
                # axes
                if j == 0:
                    ylabel = 'PC{idx} ({var}%)'.format( idx=str(i+2), var=int(100*round(pca_var[i+1], ndigits=2)) )
                    axs[i,j].set_ylabel(ylabel, fontsize=12)
                if i == (num_comp-2):
                    xlabel = 'PC{idx} ({var}%)'.format( idx=str(j+1), var=int(100*round(pca_var[j], ndigits=2)) )
                    axs[i,j].set_xlabel(xlabel, fontsize=12)
                
        plt.tight_layout()
        fig.suptitle('PCA: aux. data')
        fig.tight_layout(rect=[0, 0.03, 1, 0.98])
        plt.savefig(save_fn, format='pdf', dpi=300, bbox_inches='tight')
        plt.clf()

        #done
        return
    
    def plot_scatter_accuracy(self, ests, labels, param_names, plot_dir, prefix,
                              color="blue", axis_labels = ["estimate", "truth"],
                              title = '', plot_log=False):
        """
        Plots accuracy of estimates and CPIs for labels.

        This function generates a scatterplot for true vs. estimated labels
        from the trained network. Points are point estimates. Bars are
        CPIs.

        Arguments:
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
        for i,p in enumerate(param_names):

            # estimates (y) and true values (x)
            y_value = ests[f'{p}_value'][:].to_numpy()
            y_lower = ests[f'{p}_lower'][:].to_numpy()
            y_upper = ests[f'{p}_upper'][:].to_numpy()
            x_value = labels[p][:].to_numpy()

            # accuracy stats
            y_mae = np.mean( np.abs(y_value - x_value) )
            y_mape = 100 * np.mean( np.abs(x_value - y_value) / x_value )
            y_mse = np.mean( np.power(y_value - x_value, 2) )
            y_rmse = np.sqrt( y_mse )
            
            # convert to strings
            s_mae  = '{:.2E}'.format(y_mae)
            s_mse  = '{:.2E}'.format(y_mse)
            s_rmse = '{:.2E}'.format(y_rmse)
            s_mape = '{:.1f}%'.format(y_mape)

            # coverage stats
            y_cover = np.logical_and(y_lower < x_value, x_value < y_upper )
            y_not_cover = np.logical_not(y_cover)
            f_cover = sum(y_cover) / len(y_cover) * 100
            s_cover = '{:.1f}%'.format(f_cover)
            
            # covered estimates
            alpha = 0.5 # 50. / len(y_cover)
            plt.scatter(x_value[y_cover], y_value[y_cover],
                        alpha=alpha, c=color, zorder=3)
            plt.plot([x_value[y_cover], x_value[y_cover]],
                     [y_lower[y_cover], y_upper[y_cover]],
                     color=color, alpha=alpha, linestyle="-", marker='_', linewidth=0.5, zorder=2 )

            # not covered estimates
            plt.scatter(x_value[y_not_cover], y_value[y_not_cover],
                        alpha=alpha, c='red', zorder=5)
            plt.plot([x_value[y_not_cover], x_value[y_not_cover]],
                     [y_lower[y_not_cover], y_upper[y_not_cover]],
                     color='red', alpha=alpha, linestyle="-", marker='_', linewidth=0.5, zorder=4 )
            
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
            plt.annotate(f'MAE: {s_mae}',        xy=(0.01,0.99-0*dx), xycoords='axes fraction', fontsize=10, horizontalalignment='left', verticalalignment='top', color='black')
            plt.annotate(f'MAPE: {s_mape}',      xy=(0.01,0.99-1*dx), xycoords='axes fraction', fontsize=10, horizontalalignment='left', verticalalignment='top', color='black')
            plt.annotate(f'MSE: {s_mse}',        xy=(0.01,0.99-2*dx), xycoords='axes fraction', fontsize=10, horizontalalignment='left', verticalalignment='top', color='black')
            plt.annotate(f'RMSE: {s_rmse}',      xy=(0.01,0.99-3*dx), xycoords='axes fraction', fontsize=10, horizontalalignment='left', verticalalignment='top', color='black')
            plt.annotate(f'Coverage: {s_cover}', xy=(0.01,0.99-4*dx), xycoords='axes fraction', fontsize=10, horizontalalignment='left', verticalalignment='top', color='black')

            # cosmetics
            plt.title(f'{title} estimates: {p}')
            plt.xlabel(f'{p} {axis_labels[0]}')
            plt.ylabel(f'{p} {axis_labels[1]}')
            if plot_log:
                plt.xscale('log')         
                plt.yscale('log')         

            # save
            save_fn = f'{plot_dir}/{prefix}_{p}.pdf'
            plt.savefig(save_fn, format='pdf', dpi=300, bbox_inches='tight')
            plt.clf()

        # done    
        return
    
    def plot_est_CI(self, save_fn, est_label, title='Estimates', color='black', plot_log=True):
        """
        Plots point estimates and CPIs.

        This function plots the point estimates and calibrated prediction
        intervals for the new dataset, if it exists.

        Arguments:
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
            y_value = col_data[0].iloc[0]
            y_lower = col_data[0].iloc[1]
            y_upper = col_data[0].iloc[2]
            s_value = '{:.2E}'.format(y_value)
            s_lower = '{:.2E}'.format(y_lower)
            s_upper = '{:.2E}'.format(y_upper)
            
            # plot CI
            plt.plot([i,i], [y_lower, y_upper],
                     color=color, linestyle="-", marker='_', linewidth=1.5)
            
            # plot values as text
            for y_,s_ in zip( [y_value,y_lower,y_upper], [s_value, s_lower, s_upper] ):
                plt.text( x=i+0.10, y=y_, s=s_, color='black', va='center', size=8  )

            # plot point estimate
            plt.scatter(i, y_value, color='white', edgecolors=color, s=60, zorder=3)
            plt.scatter(i, y_value, color='red', edgecolors='white', s=30, zorder=3)
            
        # plot values as text
        plt.title(title)
        plt.xticks(np.arange(num_label), label_names)
        plt.xlim( -0.5, num_label )
        plt.savefig(save_fn, format='pdf', dpi=300, bbox_inches='tight')
        plt.clf()
        
        #done
        return
    
    def plot_train_history(self, history, prefix, train_color='blue',
                          val_color='red'):
        """
        Plot training history for network.

        This function plots trained network performance metrics as a time-series
        across training epochs. Typically, it will compare performance between
        trainiing vs. validation examples.

        Arguments:
            history (str): Training performance metrics
            prefix (str): Used to construct filename
            train_color (str): Color for training example metrics
            val_color (str): Color for validation example metrics
        """
        
        # get data names/dimensions
        epochs       = range(1, len(history['loss']) + 1)
        train_keys   = [ x for x in history.keys() if 'val_' not in x ]
        #val_keys     = [ 'val_'+x for x in train_keys ]
        label_names  = [ '_'.join( x.split('_')[0:-1] ) for x in train_keys ]
        label_names  = sorted( np.unique(label_names) )
        #num_labels   = len(label_names)

        # get metric names
        metric_names = [ x.split('_')[-1] for x in train_keys ]
        metric_names = np.unique(metric_names)
        metric_names = [ 'loss' ] + [ x for x in metric_names if x != 'loss' ]
        num_metrics  = len(metric_names)

        # figure dimensions
        fig_width = 6
        fig_height = int(np.ceil(2*num_metrics))

        # plot for all parameters
        for i,v1 in enumerate(label_names):
            fig, axs = plt.subplots(nrows=num_metrics, ncols=1, sharex=True,
                                    figsize=(fig_width, fig_height))
            idx = 0
            # plot for all metrics
            for j,v2 in enumerate(metric_names):
                # get val name from train name 
                if v1 == '':
                    k_train = v2
                else:
                    k_train = f'{v1}_{v2}'
                k_val = 'val_' + k_train

                # plot training example metrics
                legend_handles = []
                legend_labels = []
                if k_train in history:
                    lines_train, = axs[idx].plot(epochs, history[k_train], color=train_color, label = k_train)
                    axs[idx].scatter(epochs, history[k_train], color=train_color, label = k_train, zorder=3)
                    axs[idx].set(ylabel=metric_names[j])
                    legend_handles.append( lines_train )
                    legend_labels.append( 'Train' )

                # plot validation example metrics
                if k_val in history:
                    lines_val, = axs[idx].plot(epochs, history[k_val], color=val_color, label = k_val)
                    axs[idx].scatter(epochs, history[k_val], color=val_color, label = k_val, zorder=3)
                    legend_handles.append( lines_val )
                    legend_labels.append( 'Validation' )

                # plot legend
                if k_train in history or k_val in history:
                    if idx == 0:
                        axs[idx].legend( handles=legend_handles, labels=legend_labels, loc='upper right' )
                    idx += 1

            # turn off unused rows            
            for j in range(num_metrics):
                if j >= idx:
                    axs[j].axis('off')

            # aesthetics
            title_metric = label_names[i]
            if title_metric == '':
                title_metric = 'entire network'
            fig.supxlabel('Epochs')
            fig.supylabel('Metrics')
            fig.suptitle('Training history: ' + title_metric)
            fig.tight_layout()

            # save figure
            save_fn = f'{prefix}'
            if label_names[i] != '':
                save_fn += f'_{label_names[i]}'
            save_fn += '.pdf'
            plt.savefig(save_fn, format='pdf', dpi=300, bbox_inches='tight')
            plt.clf()
        
        # done
        return
#------------------------------------------------------------------------------#