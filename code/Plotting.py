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
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
#from tensorflow.keras.utils import plot_model
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
        self.proj              = args['proj']
        self.network_dir       = args['net_dir']
        self.plot_dir          = args['plt_dir']
        self.pred_dir          = args['pred_dir']
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

        self.net_job_dir        = f'{self.network_dir}/{self.proj}'
        self.fmt_job_dir        = f'{self.tensor_dir}/{self.proj}'
        self.plt_job_dir        = f'{self.plot_dir}/{self.proj}'
        self.pred_job_dir       = f'{self.pred_dir}/{self.proj}'

        # tensors
        self.input_stats_fn     = f'{self.fmt_job_dir}/sim.nt{self.tree_size}.summ_stat.csv'
        self.input_labels_fn    = f'{self.fmt_job_dir}/sim.nt{self.tree_size}.labels.csv'
        self.input_hdf5_fn      = f'{self.fmt_job_dir}/sim.nt{self.tree_size}.hdf5'

        # network
        self.train_pred_fn      = f'{self.net_job_dir}/{self.network_prefix}.train_pred.csv'
        self.train_labels_fn    = f'{self.net_job_dir}/{self.network_prefix}.train_labels.csv'
        self.test_pred_fn       = f'{self.net_job_dir}/{self.network_prefix}.test_pred.csv'
        self.test_labels_fn     = f'{self.net_job_dir}/{self.network_prefix}.test_labels.csv'
        self.network_fn         = f'{self.net_job_dir}/{self.network_prefix}.hdf5'
        self.history_json_fn    = f'{self.net_job_dir}/{self.network_prefix}.train_history.json'

        # predictions
        self.pred_aux_fn        = f'{self.pred_job_dir}/{self.pred_prefix}.summ_stat.csv'
        self.pred_lbl_fn        = f'{self.pred_job_dir}/{self.pred_prefix}.{self.network_prefix}.pred_labels.csv'
        
        # plotting output
        self.save_hist_aux_fn   = f'{self.plt_job_dir}/{self.network_prefix}.histogram_aux.pdf'
        self.save_hist_label_fn = f'{self.plt_job_dir}/{self.network_prefix}.histogram_label.pdf'
        self.save_pca_aux_fn    = f'{self.plt_job_dir}/{self.network_prefix}.pca_aux.pdf'
        self.save_cqr_test_fn   = f'{self.plt_job_dir}/{self.network_prefix}.train_est_CI.pdf'
        self.save_cqr_pred_fn   = f'{self.plt_job_dir}/{self.network_prefix}.pred_est_CI.pdf'
        self.save_network_fn    = f'{self.plt_job_dir}/{self.network_prefix}.network_architecture.pdf'
        self.save_summary_fn    = f'{self.plt_job_dir}/{self.network_prefix}.summary.pdf'
        
        self.train_color        = 'blue'
        self.test_color         = 'purple'
        self.validation_color   = 'red'
        self.aux_color          = 'green'
        self.label_color        = 'orange'
        self.pred_color         = 'black'

        return

    def load_data(self):
        
        self.model        = tf.keras.models.load_model(self.network_fn, compile=False)

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
        self.make_history_plot(self.history_dict, prefix=self.network_prefix+'_history', plot_dir=self.plt_job_dir, train_color=self.train_color, val_color=self.validation_color)

        # train prediction scatter plots
        self.plot_preds_labels(\
            preds=self.train_preds.iloc[0:self.train_preds_max_idx],
            labels=self.train_labels.iloc[0:self.train_labels_max_idx],
            param_names=self.param_names,
            prefix=self.network_prefix+'_train',
            color=self.train_color,
            plot_dir=self.plt_job_dir,
            title='Train')

        # test predicition scatter plots
        self.plot_preds_labels(\
            preds=self.test_preds.iloc[0:self.test_preds_max_idx],
            labels=self.test_labels.iloc[0:self.test_labels_max_idx],
            param_names=self.param_names,
            prefix=self.network_prefix+'_test',
            color=self.test_color,
            plot_dir=self.plt_job_dir,
            title='Test')


        # save histograms
        self.plot_sim_histogram(save_fn=self.save_hist_aux_fn, sim_values=self.input_stats, pred_values=self.pred_aux_data, color=self.aux_color, title='Aux. data')
        self.plot_sim_histogram(save_fn=self.save_hist_label_fn, sim_values=self.input_labels, pred_values=self.pred_lbl_value, color=self.label_color, title='Labels' )
        
        # save PCA
        self.plot_pca(save_fn=self.save_pca_aux_fn, sim_stat=self.input_stats, pred_stat=self.pred_aux_data, color=self.aux_color)
        
        # save point est. and CI for test dataset (if it exists)
        self.plot_pred_est_CI(save_fn=self.save_cqr_pred_fn, pred_label=self.pred_lbl_df, title=f'Prediction: {self.pred_dir}/{self.pred_prefix}', color=self.pred_color)
        
        # save network
        tf.keras.utils.plot_model(self.model, to_file=self.save_network_fn, show_shapes=True)

        # collect and sort file names
        files = os.listdir(self.plt_job_dir)
        files.sort()
        files = [ f for f in files if '.pdf' in f and self.network_prefix in f and 'all_results' not in f ]

        files_CI         = list(filter(lambda x: 'CI' in x, files))
        files_pca        = list(filter(lambda x: 'pca' in x, files))
        files_histogram  = list(filter(lambda x: 'histogram' in x, files))
        files_train      = list(filter(lambda x: 'train' in x, files))
        files_test       = list(filter(lambda x: 'test' in x, files))
        files_arch       = list(filter(lambda x: 'architecture' in x, files))
        files_history    = list(filter(lambda x: 'history' in x, files))
        files_ordered    = files_CI + files_pca + files_histogram + files_train + files_test + files_history + files_arch

        # combine pdfs
        merger = PdfMerger()
        for f in files_ordered:
            merger.append(self.plt_job_dir + '/' + f)

        merger.write(self.save_summary_fn)
        return


    def plot_sim_histogram(self, save_fn, sim_values, pred_values=None, title='', ncol_plot=3, color='blue'):
        
        col_names = sorted( sim_values.columns )
        num_aux = len(col_names)
        nrow = int( np.ceil(num_aux/ncol_plot) )

        # figure dimensions
        fig_width = 9
        fig_height = 1 + int(np.ceil(2*nrow))

        # basic figure structure
        fig, axes = plt.subplots(ncols=ncol_plot, nrows=nrow, squeeze=False, figsize=(fig_width, fig_height))
        fig.tight_layout() #h_pad=2.25)
        #fig.subplots_adjust(bottom=0.05)

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

        fig.suptitle(f'Histogram: {title}')
        fig.tight_layout(rect=[0, 0.03, 1, 0.98])
        plt.savefig(fname=save_fn, format='pdf', dpi=300, bbox_inches='tight')
        plt.clf()

    

    def plot_pca(self, save_fn, sim_stat, pred_stat=None, num_comp=4, f_show=0.05, color='blue'):

        #x = sim_stat #StandardScaler().fit_transform(df)
        x = sim_stat # pd.DataFrame(sim_stat, columns=sim_stat.columns)
        nrow_keep = int(x.shape[0] * f_show)
        alpha = np.min( [1, 100 / nrow_keep] )
        
        pca_model = PCA(n_components=num_comp)
        pca = pca_model.fit_transform(x)
        if self.pred_aux_loaded:
            pca_pred = pca_model.transform(pred_stat)
        
        fig_width = 8
        fig_height = 8 

        pca_var = pca_model.explained_variance_ratio_
        fig, axs = plt.subplots(num_comp-1, num_comp-1, sharex=True, sharey=True, figsize=(fig_width, fig_height))
        #tick_spacing = 2

        # use this to turn off subplots
        #axes[i_row,j_col].axis('off')

        for i in range(0, num_comp-1):
            for j in range(i+1, num_comp-1):
                axs[i,j].axis('off')
            for j in range(0, i+1):
                axs[i,j].scatter( pca[0:nrow_keep,i+1], pca[0:nrow_keep,j], alpha=alpha, marker='x', color=color )
                if self.pred_aux_loaded:    
                    axs[i,j].scatter(pca_pred[0:nrow_keep,i+1], pca_pred[0:nrow_keep,j],
                                     alpha=1.0, color='white', edgecolor='black', s=80)
                    axs[i,j].scatter(pca_pred[0:nrow_keep,i+1], pca_pred[0:nrow_keep,j],
                                     alpha=1.0, color='red', edgecolor='white', s=40)
                if j == 0:
                    ylabel = 'PC{idx} ({var}%)'.format( idx=str(i+2), var=int(100*round(pca_var[i+1], ndigits=2)) )
                    axs[i,j].set_ylabel(ylabel, fontsize=12)
                if i == (num_comp-2):
                    xlabel = 'PC{idx} ({var}%)'.format( idx=str(j+1), var=int(100*round(pca_var[j], ndigits=2)) )
                    axs[i,j].set_xlabel(xlabel, fontsize=12)
                #axs[i,j].xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
                #axs[i,j].yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
                
        plt.tight_layout()
        fig.suptitle('PCA: aux. data')
        fig.tight_layout(rect=[0, 0.03, 1, 0.98])
        plt.savefig(save_fn, format='pdf', dpi=300, bbox_inches='tight')
        plt.clf()
        return


    def plot_pred_est_CI(self, save_fn, pred_label, title='Prediction', color='black', plot_log=True):
        if pred_label is None:
            return

        plt.figure(figsize=(5,5))      
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
        return


    def plot_preds_labels(self, preds, labels, param_names, plot_dir, prefix, color="blue", axis_labels = ["prediction", "truth"], title = '', plot_log=False):
   

        plt.figure(figsize=(6,6))
        for i,p in enumerate(param_names):
            # preds/labels
            y_value = preds[f'{p}_value'][:].to_numpy()
            y_lower = preds[f'{p}_lower'][:].to_numpy()
            y_upper = preds[f'{p}_upper'][:].to_numpy()
            x_value = labels[p][:].to_numpy()

            # accuracy stats
            y_mae = np.mean( np.abs(y_value - x_value) )
            y_mape = 100 * np.mean( np.abs(x_value - y_value) / x_value )
            y_mse = np.mean( np.power(y_value - x_value, 2) )
            y_rmse = np.sqrt( y_mse )
            
            s_mae  = '{:.2E}'.format(y_mae)
            s_mse  = '{:.2E}'.format(y_mse)
            s_rmse = '{:.2E}'.format(y_rmse)
            s_mape = '{:.1f}%'.format(y_mape)

            # coverage stats
            y_cover = np.logical_and(y_lower < x_value, x_value < y_upper )
            y_not_cover = np.logical_not(y_cover)
            f_cover = sum(y_cover) / len(y_cover) * 100
            s_cover = '{:.1f}%'.format(f_cover)
            
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
            plt.axline((0,0), slope=1, color=color, alpha=1.0, zorder=0)
            plt.gca().set_aspect('equal')
            #minlim = np.min( np.concatenate([x_value, y_lower]) )
            #maxlim = np.max( np.concatenate([x_value, y_upper]) )
            #adjlim = 0.20 #* (maxlim - minlim)
            #if minlim < 0:
            #    minlim -= 0.1 * (maxlim - minlim)
            #    maxlim += 0.1 * (maxlim - minlim)
            #else:
            #    minlim /= (1.1)
            #   maxlim *= (1.1)

            xlim = plt.xlim()
            ylim = plt.ylim()
            minlim = min(xlim[0], ylim[0])
            maxlim = max(xlim[1], ylim[1])
            plt.xlim([minlim, maxlim])
            plt.ylim([minlim, maxlim])
            #plt.xlim( 0, maxlim )
            #plt.ylim( 0, maxlim )
            
            dx = 0.03
            plt.annotate(f'MAE: {s_mae}',        xy=(0.01,0.99-0*dx), xycoords='axes fraction', fontsize=10, horizontalalignment='left', verticalalignment='top', color='black')
            plt.annotate(f'MAPE: {s_mape}',      xy=(0.01,0.99-1*dx), xycoords='axes fraction', fontsize=10, horizontalalignment='left', verticalalignment='top', color='black')
            plt.annotate(f'MSE: {s_mse}',        xy=(0.01,0.99-2*dx), xycoords='axes fraction', fontsize=10, horizontalalignment='left', verticalalignment='top', color='black')
            plt.annotate(f'RMSE: {s_rmse}',      xy=(0.01,0.99-3*dx), xycoords='axes fraction', fontsize=10, horizontalalignment='left', verticalalignment='top', color='black')
            plt.annotate(f'Coverage: {s_cover}', xy=(0.01,0.99-4*dx), xycoords='axes fraction', fontsize=10, horizontalalignment='left', verticalalignment='top', color='black')

            # cosmetics
            plt.title(f'{title} predictions: {p}')
            plt.xlabel(f'{p} {axis_labels[0]}')
            plt.ylabel(f'{p} {axis_labels[1]}')
            if plot_log:
                plt.xscale('log')         
                plt.yscale('log')         

            # save
            save_fn = f'{plot_dir}/{prefix}_{p}.pdf'
            plt.savefig(save_fn, format='pdf', dpi=300, bbox_inches='tight')
            plt.clf()
        return
    

    def make_history_plot(self, history, prefix, plot_dir, train_color='blue', val_color='red'):

        epochs      = range(1, len(history['loss']) + 1)
        #print(history.keys())
        train_keys  = [ x for x in history.keys() if 'val_' not in x ]
        val_keys    = [ 'val_'+x for x in train_keys ]

        #print(train_keys)
        #print(val_keys)
        label_names = [ '_'.join( x.split('_')[0:-1] ) for x in train_keys ]
        #label_names = [ x for x in label_names if x != '' ]
        label_names = sorted( np.unique(label_names) )
        num_labels = len(label_names)

        metric_names = [ x.split('_')[-1] for x in train_keys ]
        metric_names = np.unique(metric_names)
        metric_names = [ 'loss' ] + [ x for x in metric_names if x != 'loss' ]
        num_metrics = len(metric_names)

        fig_width = 6
        fig_height = int(np.ceil(2*num_metrics))

        for i,v1 in enumerate(label_names):

            fig, axs = plt.subplots(nrows=num_metrics, ncols=1, sharex=True, figsize=(fig_width, fig_height))
            #used_idx = [ True ] * num_metrics
            idx = 0

            for j,v2 in enumerate(metric_names):

                if v1 == '':
                    k_train = v2
                else:
                    k_train = f'{v1}_{v2}'
                k_val = 'val_' + k_train

                legend_handles = []
                legend_labels = []
                if k_train in history:
                    lines_train, = axs[idx].plot(epochs, history[k_train], color=train_color, label = k_train)
                    axs[idx].scatter(epochs, history[k_train], color=train_color, label = k_train, zorder=3)
                    axs[idx].set(ylabel=metric_names[j])
                    legend_handles.append( lines_train )
                    legend_labels.append( 'Train' )

                if k_val in history:
                    lines_val, = axs[idx].plot(epochs, history[k_val], color=val_color, label = k_val)
                    axs[idx].scatter(epochs, history[k_val], color=val_color, label = k_val, zorder=3)
                    legend_handles.append( lines_val )
                    legend_labels.append( 'Validation' )

                if k_train in history or k_val in history:
                    if idx == 0:
                        axs[idx].legend( handles=legend_handles, labels=legend_labels, loc='upper right' )
                    idx += 1

            #print(v1, used_idx)
            # turn off unused rows            
            for j in range(num_metrics):
                if j >= idx:
                    axs[j].axis('off')

            title_metric = label_names[i]
            if title_metric == '':
                title_metric = 'entire network'
            fig.supxlabel('Epochs')
            fig.supylabel('Metrics')
            fig.suptitle('Training history: ' + title_metric)
            fig.tight_layout()

            save_fn = plot_dir + '/' + prefix + '_' + label_names[i] + '.pdf'

            plt.savefig(save_fn, format='pdf', dpi=300, bbox_inches='tight')
            plt.clf()
        return