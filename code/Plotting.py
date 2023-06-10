# takes inputs and outputs and produces report of figures

# PCA test vs. training data
# training conformal prediction intervals
# CNN training diagnostics
# training/validation accuracy
# network architecture
# loadings?
# parameter point estimates and CIs

import pandas as pd
import numpy as np
import os
import json
import h5py
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
#from sklearn import metrics

import Utilities

from PyPDF2 import PdfMerger

class Plotter:

    def __init__(self, args):
        self.set_args(args)
        self.prepare_files()
        return

    def set_args(self, args):
        # simulator arguments
        self.args              = args
        self.job_name          = args['job_name']
        self.network_dir       = args['net_dir']
        self.plot_dir          = args['plt_dir']
        self.tensor_dir        = args['fmt_dir']
        self.batch_size        = args['batch_size']
        self.num_epochs        = args['num_epochs']
        self.tree_size         = args['tree_size']
        self.tensor_format     = args['tensor_format']
        self.pred_dir          = args['pred_dir'] if 'pred_dir' in args else ''
        self.pred_prefix       = args['pred_prefix'] if 'pred_prefix' in args else ''
        return

    def prepare_files(self):
        self.network_prefix     = f'sim_batchsize{self.batch_size}_numepoch{self.num_epochs}_nt{self.tree_size}'

        self.net_job_dir        = f'{self.network_dir}/{self.job_name}'
        self.fmt_job_dir        = f'{self.tensor_dir}/{self.job_name}'
        self.plt_job_dir        = f'{self.plot_dir}/{self.job_name}'
        self.train_pred_fn      = f'{self.net_job_dir}/{self.network_prefix}.train_pred.csv'
        self.train_labels_fn    = f'{self.net_job_dir}/{self.network_prefix}.train_labels.csv'
        self.test_pred_fn       = f'{self.net_job_dir}/{self.network_prefix}.test_pred.csv'
        self.test_labels_fn     = f'{self.net_job_dir}/{self.network_prefix}.test_labels.csv'
        self.input_stats_fn     = f'{self.fmt_job_dir}/sim.nt{self.tree_size}.summ_stat.csv'
        self.input_labels_fn    = f'{self.fmt_job_dir}/sim.nt{self.tree_size}.labels.csv'
        self.input_hdf5_fn      = f'{self.fmt_job_dir}/sim.nt{self.tree_size}.hdf5'
        self.history_json_fn    = f'{self.net_job_dir}/{self.network_prefix}.train_history.json'
        self.save_hist_aux_fn   = f'{self.plt_job_dir}/{self.network_prefix}.histogram_aux.pdf'
        self.save_hist_label_fn = f'{self.plt_job_dir}/{self.network_prefix}.histogram_label.pdf'
        self.save_pca_aux_fn    = f'{self.plt_job_dir}/{self.network_prefix}.pca_aux.pdf'
        self.pred_aux_fn        = f'{self.pred_dir}/{self.pred_prefix}.summ_stat.csv'
        self.pred_lbl_fn        = f'{self.net_job_dir}/{self.pred_prefix}.{self.network_prefix}.pred_labels.csv'
        self.save_cqr_test_fn   = f'{self.plt_job_dir}/{self.network_prefix}.train_est_CI.pdf'
        self.save_cqr_pred_fn   = f'{self.plt_job_dir}/{self.network_prefix}.pred_est_CI.pdf'
        
        
        return

    def load_data(self):
        
        self.train_preds  = pd.read_csv(self.train_pred_fn)
        self.train_labels = pd.read_csv(self.train_labels_fn)
        self.test_preds   = pd.read_csv(self.test_pred_fn)
        self.test_labels  = pd.read_csv(self.test_labels_fn)
        self.train_preds  = self.train_preds #.loc[:, self.train_preds.columns.str.contains('_value')] 
        self.test_preds   = self.test_preds #.loc[:, self.test_preds.columns.str.contains('_value')] a

        self.param_names  = self.train_preds.columns.to_list()
        self.param_names  = [ '_'.join(s.split('_')[:-1]) for s in self.param_names  ]
        self.param_names  = np.unique(self.param_names)
        
        if self.tensor_format == 'csv':
            self.input_stats = pd.read_csv( self.input_stats_fn )
            self.input_labels = pd.read_csv( self.input_labels_fn )
        elif self.tensor_format == 'hdf5':
            hdf5_file = h5py.File(self.input_hdf5_fn, 'r')
            self.input_stat_names = [ s.decode() for s in hdf5_file['summ_stat_names'][0,:] ]
            self.input_label_names = [ s.decode() for s in hdf5_file['label_names'][0,:] ]
            self.input_stats = pd.DataFrame( hdf5_file['summ_stat'][:,:], columns=self.input_stat_names )
            self.input_labels = pd.DataFrame( hdf5_file['labels'][:,:], columns=self.input_label_names )
            hdf5_file.close()

        #self.test_preds_df = Utilities.make_param_VLU_mtx( self.test_preds, self.param_names )
        #self.train_preds_df = Utilities.make_param_VLU_mtx( self.train_preds, self.param_names )


        #print(self.test_preds_df.iloc[0])
        #print(self.test_preds_df.shape)
        self.max_pred_display     = 250
        self.train_preds_max_idx  = min( self.max_pred_display, self.train_preds.shape[0] )
        self.train_labels_max_idx = min( self.max_pred_display, self.train_labels.shape[0] )
        self.test_preds_max_idx   = min( self.max_pred_display, self.test_preds.shape[0] )
        self.test_labels_max_idx  = min( self.max_pred_display, self.test_labels.shape[0] )

        self.history_dict = json.load(open(self.history_json_fn, 'r'))

        # read in prediction aux dataset, if it exists
        self.pred_aux_loaded = os.path.isfile(self.pred_aux_fn) 
        if self.pred_aux_loaded:
            self.pred_aux_data = pd.read_csv(self.pred_aux_fn)
        else:
            self.pred_aux_data = None
        
        # read in predicted label dataset, if it exists
        self.pred_lbl_loaded = os.path.isfile(self.pred_lbl_fn)
        if self.pred_lbl_loaded:
            self.pred_lbl_data = pd.read_csv(self.pred_lbl_fn)
            self.pred_lbl_value = self.pred_lbl_data[ [col for col in self.pred_lbl_data.columns if 'value' in col] ]
            self.pred_lbl_lower = self.pred_lbl_data[ [col for col in self.pred_lbl_data.columns if 'lower' in col] ]
            self.pred_lbl_upper = self.pred_lbl_data[ [col for col in self.pred_lbl_data.columns if 'upper' in col] ]
            self.pred_lbl_value.columns = [ col.rstrip('_value') for col in self.pred_lbl_value.columns ]
            self.pred_lbl_lower.columns = [ col.rstrip('_lower') for col in self.pred_lbl_lower.columns ]
            self.pred_lbl_upper.columns = [ col.rstrip('_upper') for col in self.pred_lbl_upper.columns ]
            self.pred_lbl_df = pd.concat( [self.pred_lbl_value, self.pred_lbl_lower, self.pred_lbl_upper] )
            #print(self.pred_lbl_value)
            #self.pred_lbl_df = pd.DataFrame( [ self.pred_lbl_value, self.pred_lbl_value, self.pred_lbl_value ], columns=['value','lower','upper'] )
            #print(self.pred_lbl_df)

        else:
            self.pred_lbl_data = None
            self.pred_lbl_value = None
            self.pred_lbl_lower = None
            self.pred_lbl_upper = None
            self.pred_lbl_df = None

        return


    def run(self):

        # load data
        self.load_data()

        # training stats
        Utilities.make_history_plot(self.history_dict, prefix=self.network_prefix+'_history', plot_dir=self.plt_job_dir)

        # train prediction scatter plots
        self.plot_preds_labels(\
            preds=self.train_preds.iloc[0:self.train_preds_max_idx],
            labels=self.train_labels.iloc[0:self.train_labels_max_idx],
            param_names=self.param_names,
            prefix=self.network_prefix+'_train',
            color="blue",
            plot_dir=self.plt_job_dir,
            title='Train predictions')

        # test predicition scatter plots
        self.plot_preds_labels(\
            preds=self.test_preds.iloc[0:self.test_preds_max_idx],
            labels=self.test_labels.iloc[0:self.test_labels_max_idx],
            param_names=self.param_names,
            prefix=self.network_prefix+'_test',
            color="purple",
            plot_dir=self.plt_job_dir,
            title='Test predictions')


        # save histograms
        self.plot_aux_hist(save_fn=self.save_hist_aux_fn, sim_values=self.input_stats, pred_values=self.pred_aux_data, color='blue') #, title='Auxiliary data')
        self.plot_aux_hist(save_fn=self.save_hist_label_fn, sim_values=self.input_labels, pred_values=self.pred_lbl_value, color='orange') #, title='Labels' )
        
        # save PCA
        self.plot_pca(save_fn=self.save_pca_aux_fn, sim_stat=self.input_stats, pred_stat=self.pred_aux_data)
        
        # save point est. and CI for test dataset (if it exists)
        self.plot_pred_est_CI(save_fn=self.save_cqr_pred_fn, pred_label=self.pred_lbl_df)
        
        # collect and sort file names
        files = os.listdir(self.plt_job_dir)
        files.sort()
        files = [ f for f in files if '.pdf' in f and self.network_prefix in f and 'all_results' not in f ]

        files_history    = list(filter(lambda x: 'history' in x, files))
        files_train      = list(filter(lambda x: 'train' in x, files))
        files_test       = list(filter(lambda x: 'test' in x, files))
        files_histogram  = list(filter(lambda x: 'histogram' in x, files))
        files_pca        = list(filter(lambda x: 'pca' in x, files))
        files_CI         = list(filter(lambda x: 'CI' in x, files))
        files_ordered    = files_history + files_train + files_test + files_histogram + files_pca + files_CI

        # combine pdfs
        merger = PdfMerger()
        for f in files_ordered:
            merger.append(self.plt_job_dir + '/' + f)

        merger.write(self.plt_job_dir+'/'+self.network_prefix+'_'+'all_results.pdf')
        return


    def plot_aux_hist(self, save_fn, sim_values, pred_values=None, ncol_plot=3, color='blue'):
        
        col_names = sorted( sim_values.columns )
        num_aux = len(col_names)
        nrow = int( np.ceil(num_aux/ncol_plot) )

        # figure dimensions
        fig_width = 9
        fig_height = int(np.ceil(2*nrow))

        # basic figure structure
        fig, axes = plt.subplots(ncols=ncol_plot, nrows=nrow, figsize=(fig_width, fig_height))
        fig.tight_layout(h_pad=2.25)
        fig.subplots_adjust(bottom=0.05)

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
                
                if pred_values is not None and p in pred_values:
                    x_data = pred_values[p][0]
                    y_data = kde(x_data)
                    q_true = np.sum(x < x_data) / len(x)
                    ax.vlines(x_data, 0, y_data, color='red')
                    lbl_pred_val_str = "X={:.2f}".format(x_data)
                    lbl_pred_quant_str = f'Q={int(q_true*100)}%'
                    ax.annotate(lbl_pred_val_str, xy=(x_pos, y_pos-3*dy_pos), xycoords='axes fraction', fontsize=10, horizontalalignment=ha, verticalalignment='top', color='red')
                    ax.annotate(lbl_pred_quant_str, xy=(x_pos, y_pos-4*dy_pos), xycoords='axes fraction', fontsize=10, horizontalalignment=ha, verticalalignment='top', color='red')
                    
                # cosmetics
                ax.title.set_text(col_names[i])
                ax.yaxis.set_visible(False)
                i = i + 1

        # add labels to superplot axes
        fig.supxlabel('Data')
        fig.supylabel('Density')

        #fig.suptitle(title)
        #plt.margins(x=0.05, y=0.05)
        plt.savefig(fname=save_fn)
        plt.clf()

    

    def plot_pca(self, save_fn, sim_stat, pred_stat=None, num_comp=4, f_show=1.0, color='blue'):

        #x = sim_stat #StandardScaler().fit_transform(df)
        x = sim_stat # pd.DataFrame(sim_stat, columns=sim_stat.columns)
        nrow_keep = int(x.shape[0] * f_show)
        alpha = 100 / nrow_keep
        pca_model = PCA(n_components=num_comp)
        pca = pca_model.fit_transform(x)
        if self.pred_aux_loaded:
            pca_pred = pca_model.transform(pred_stat)
        
        pca_var = pca_model.explained_variance_ratio_
        fig, axs = plt.subplots(num_comp-1, num_comp-1, sharex=True, sharey=True)
        #tick_spacing = 2
        for i in range(0, num_comp-1):
            for j in range(0, i+1):
                axs[i,j].scatter( pca[0:nrow_keep,i+1], pca[0:nrow_keep,j], alpha=alpha, marker='x', color=color )
                if self.pred_aux_loaded:    
                    axs[i,j].scatter( pca_pred[0:nrow_keep,i+1], pca_pred[0:nrow_keep,j], alpha=1.0, color='white', edgecolor='black', s=50)
                    axs[i,j].scatter( pca_pred[0:nrow_keep,i+1], pca_pred[0:nrow_keep,j], alpha=1.0, color='red', edgecolor='white', s=20 )
                if j == 0:
                    ylabel = 'PC{idx} ({var}%)'.format( idx=str(i+2), var=int(100*round(pca_var[i+1], ndigits=2)) )
                    axs[i,j].set_ylabel(ylabel, fontsize=12)
                if i == (num_comp-2):
                    xlabel = 'PC{idx} ({var}%)'.format( idx=str(j+1), var=int(100*round(pca_var[j], ndigits=2)) )
                    axs[i,j].set_xlabel(xlabel, fontsize=12)
                #axs[i,j].xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
                #axs[i,j].yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
                
        plt.tight_layout()
        plt.savefig(save_fn, format='pdf')
        plt.clf()
        return


    def plot_pred_est_CI(self, save_fn, pred_label, plot_log=True):
        if pred_label is None:
            return
                
        label_names = pred_label.columns
        num_label = len(label_names)
        
        if plot_log:
            plt.yscale('log')

        for i,col in enumerate(label_names):
            col_data = pred_label[col]
            y_value = col_data[0].iloc[0]
            y_lower = col_data[0].iloc[1]
            y_upper = col_data[0].iloc[2]
            s_value = '{:.2E}'.format(y_value)
            s_lower = '{:.2E}'.format(y_lower)
            s_upper = '{:.2E}'.format(y_upper)
            
            # plot CI
            plt.plot( [i,i], [y_lower, y_upper], color='black', linestyle="-", marker='_', linewidth=0.9 )
            # plot values as text
            for y_,s_ in zip( [y_value,y_lower,y_upper], [s_value, s_lower, s_upper] ):
                plt.text( x=i+0.05, y=y_, s=s_, color='black', va='center', size=7  )
            # plot point estimate
            plt.scatter(i, y_value, color='white', edgecolors='black', s=50, zorder=3)
            plt.scatter(i, y_value, color='red', edgecolors='white', s=30, zorder=3)
            
        # plot values as text
        plt.xticks(np.arange(num_label), label_names)
        plt.xlim( -0.5, num_label )
        plt.savefig(save_fn, format='pdf')
        plt.clf()
        return


    def plot_preds_labels(self, preds, labels, param_names, plot_dir, prefix, color="blue", axis_labels = ["prediction", "truth"], title = '', plot_log=False):
   
        for i,p in enumerate(param_names):
            # preds/labels
            y_value = preds[f'{p}_value'][:].to_numpy()
            y_lower = preds[f'{p}_lower'][:].to_numpy()
            y_upper = preds[f'{p}_upper'][:].to_numpy()
            x_value = labels[p][:].to_numpy()
            
            # coverage stats
            y_cover = np.logical_and(y_lower < x_value, x_value < y_upper )
            y_not_cover = np.logical_not(y_cover)
            f_cover = sum(y_cover) / len(y_cover) * 100
            s_cover = '{:.1f}'.format(f_cover)
            
            # covered predictions
            alpha = 0.5 # 50. / len(y_cover)
            plt.scatter(x_value[y_cover], y_value[y_cover],
                        alpha=alpha, c=color, zorder=3)
            plt.plot([x_value[y_cover], x_value[y_cover]],
                     [y_lower[y_cover], y_upper[y_cover]],
                     color=color, alpha=alpha, linestyle="-", marker='_', linewidth=0.5, zorder=2 )

            # not covered predictions
            plt.scatter(x_value[y_not_cover], y_value[y_not_cover],
                        alpha=alpha, c='red', zorder=5)
            plt.plot([x_value[y_not_cover], x_value[y_not_cover]],
                     [y_lower[y_not_cover], y_upper[y_not_cover]],
                     color='red', alpha=alpha, linestyle="-", marker='_', linewidth=0.5, zorder=4 )
            
            # 1:1 line
            plt.axline((np.min(x_value),np.min(x_value)), slope=1, color=color, alpha=1.0, zorder=0)
            
            plt.annotate(f'Coverage: {s_cover}%', xy=(0.01,0.99), xycoords='axes fraction', fontsize=10, horizontalalignment='left', verticalalignment='top', color='black')

            # cosmetics
            plt.title(title)
            plt.xlabel(f'{p} {axis_labels[0]}')
            plt.ylabel(f'{p} {axis_labels[1]}')
            if plot_log:
                plt.xscale('log')         
                plt.yscale('log')         

            # save
            save_fn = f'{plot_dir}/{prefix}_{p}.pdf'
            plt.savefig(save_fn, format='pdf')
            plt.clf()
        return