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
        self.network_prefix  = f'sim_batchsize{self.batch_size}_numepoch{self.num_epochs}_nt{self.tree_size}'

        self.net_job_dir     = f'{self.network_dir}/{self.job_name}'
        self.fmt_job_dir     = f'{self.tensor_dir}/{self.job_name}'
        self.plt_job_dir     = f'{self.plot_dir}/{self.job_name}'
        self.train_pred_fn   = f'{self.net_job_dir}/{self.network_prefix}.train_pred.csv'
        self.train_labels_fn = f'{self.net_job_dir}/{self.network_prefix}.train_labels.csv'
        self.test_pred_fn    = f'{self.net_job_dir}/{self.network_prefix}.test_pred.csv'
        self.test_labels_fn  = f'{self.net_job_dir}/{self.network_prefix}.test_labels.csv'
        self.input_stats_fn  = f'{self.fmt_job_dir}/sim.nt{self.tree_size}.summ_stat.csv'
        self.input_labels_fn = f'{self.fmt_job_dir}/sim.nt{self.tree_size}.labels.csv'
        self.input_hdf5_fn   = f'{self.fmt_job_dir}/sim.nt{self.tree_size}.hdf5'
        self.history_json_fn = f'{self.net_job_dir}/{self.network_prefix}.train_history.json'

        #self.pred_job_dir    = f'{self.}'
        
        return

    def load_data(self):
        #genfromtxt('my_file.csv', delimiter=',')
        self.train_preds  = pd.read_csv(self.train_pred_fn)
        self.train_labels = pd.read_csv(self.train_labels_fn)
        self.test_preds   = pd.read_csv(self.test_pred_fn)
        self.test_labels  = pd.read_csv(self.test_labels_fn)
        self.train_preds  = self.train_preds.loc[:, self.train_preds.columns.str.contains('_value')] 
        self.test_preds   = self.test_preds.loc[:, self.test_preds.columns.str.contains('_value')] 

        self.param_names  = self.train_preds.columns.to_list()
        
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


        self.train_preds_max_idx = min( 1000, self.train_preds.shape[0] )
        self.train_labels_max_idx = min( 1000, self.train_labels.shape[0] )
        self.test_preds_max_idx = min( 1000, self.test_preds.shape[0] )
        self.test_labels_max_idx = min( 1000, self.test_labels.shape[0] )

        self.history_dict = json.load(open(self.history_json_fn, 'r'))

        self.save_hist_fn = f'{self.plt_job_dir}/{self.network_prefix}.sim_histogram.pdf'
        self.save_pca_fn = f'{self.plt_job_dir}/{self.network_prefix}.sim_pca.pdf'

        # read in prediction data set, if it exists
        self.pred_data_loaded = False
        self.pred_aux_data = None
        if self.pred_dir != '' and self.pred_prefix != '':
            self.pred_data_loaded = True
            self.pred_aux_fn = f'{self.pred_dir}/{self.pred_prefix}.summ_stat.csv'
            self.pred_aux_data = pd.read_csv(self.pred_aux_fn)


        return

    def run(self):

        # load data
        self.load_data()

        # training stats
        Utilities.make_history_plot(self.history_dict, prefix=self.network_prefix+'_history', plot_dir=self.plt_job_dir)

        # train prediction scatter plots
        Utilities.plot_preds_labels(\
            preds=self.train_preds.iloc[0:self.train_preds_max_idx].to_numpy(),
            labels=self.train_labels.iloc[0:self.train_labels_max_idx].to_numpy(),
            param_names=self.param_names,
            prefix=self.network_prefix+'_train',
            color="blue",
            plot_dir=self.plt_job_dir,
            title='Train predictions')

        # test predicition scatter plots
        Utilities.plot_preds_labels(\
            preds=self.test_preds.iloc[0:self.test_preds_max_idx].to_numpy(),
            labels=self.test_labels.iloc[0:self.test_labels_max_idx].to_numpy(),
            param_names=self.param_names,
            prefix=self.network_prefix+'_test',
            color="purple",
            plot_dir=self.plt_job_dir,
            title='Test predictions')


        # save PCA and summ_stat histograms
        df_all = pd.concat( [self.input_stats, self.input_labels], axis=1 )
        df_all = df_all.T.drop_duplicates().T


        #Utilities.plot_ss_param_hist(df=df_all, save_fn=self.save_hist_fn)
        #Utilities.plot_pca(df=self.input_stats, save_fn=self.save_pca_fn, f_show=1.0, num_comp=4)

        self.plot_aux_hist(save_fn=self.save_hist_fn, x_sim=df_all, x_pred=self.pred_aux_data, )
        self.plot_pca(save_fn=self.save_pca_fn, x_sim=self.input_stats, x_pred=self.pred_aux_data)

        #self, df, save_fn, num_comp=4, f_show=1.0
        # collect and sort file names
        files = os.listdir(self.plt_job_dir)
        files.sort()
        files = [ f for f in files if '.pdf' in f and self.network_prefix in f and 'all_results' not in f ]

        files_history = list(filter(lambda x: 'history' in x, files))
        files_train = list(filter(lambda x: 'train' in x, files))
        files_test = list(filter(lambda x: 'test' in x, files))
        files_histogram = list(filter(lambda x: 'histogram' in x, files))
        files_pca = list(filter(lambda x: 'pca' in x, files))
        files_ordered = files_history + files_train + files_test + files_histogram + files_pca

        # combine pdfs
        merger = PdfMerger()
        for f in files_ordered:
            merger.append(self.plt_job_dir + '/' + f)

        merger.write(self.plt_job_dir+'/'+self.network_prefix+'_'+'all_results.pdf')
        return

    #Utilities.plot_ss_param_hist(df=df_all, save_fn=self.plt_job_dir + '/' + self.network_prefix + '.sim_histogram.pdf')
    def plot_aux_hist(self, save_fn, x_sim, x_pred=None, ncol=4):
        
        # data dimension
        params_true = sorted( list(x_pred.columns) )
        params_sim = sorted( [ p for p in x_sim.columns if p not in params_true ] )
        params = params_true + params_sim
        num_params = len(params)
        nrow = int( np.ceil(num_params/ncol) )

        # figure dimensions
        fig_width = 12
        fig_height = int(np.ceil(2*nrow))

        #titles = [ f'parameter_{i}' for i in range(ncell) ]
        #true_vals = sp.stats.norm.rvs(size=num_params, scale=0.3)
        fig, axes = plt.subplots(ncols=ncol, nrows=nrow, figsize=(fig_width, fig_height))
        fig.tight_layout(w_pad=1.5, h_pad=2.25)
        fig.subplots_adjust(bottom=0.05, left=0.10)

        aux_color = 'blue'
        param_color = 'orange'
        i = 0
        
        #print(true_vals)
        #print(p)
        
        for i_row, ax_row in enumerate(axes):
            for j_col,ax in enumerate(ax_row):
                if i >= num_params:
                    axes[i_row,j_col].axis('off')
                    continue
                  
                # input data
                p = params[i]
                if p in x_pred:
                    kde_color = aux_color
                else:
                    kde_color = param_color

                x = sorted( x_sim[p] )
                mn = np.min(x)
                mx = np.max(x)
                xs = np.linspace(mn, mx, 300)
                kde = sp.stats.gaussian_kde(x)
                ys = kde.pdf(xs)
                ax.plot(xs, ys, label="PDF", color=kde_color)
                # plot quantiles
                left, middle, right = np.percentile(x, [2.5, 50, 97.5])
                ax.vlines(middle, 0, np.interp(middle, xs, ys), color=kde_color, ls=':')
                #ax.text(0.5, 0.25, f'{q_true}%', transform=ax.gca().transAxes)
                ax.fill_between(xs, 0, kde(xs), where=(left <= xs) & (xs <= right), facecolor=kde_color, alpha=0.2)
                if p in x_pred:
                    x_data = x_pred[p][0]
                    q_true = np.sum( x < x_data ) / len(x)
                    #print(x_data)
                    y_data = kde(x_data)
                    ax.vlines(x_data, 0, y_data, color='red')
                    ax.annotate(f'{int(q_true*100)}%', xy=(1, 1), xycoords='axes fraction', fontsize=8, horizontalalignment='right', verticalalignment='bottom')
                    
                # cosmetics
                ax.title.set_text(params[i])
                i = i + 1

        #plt.grid(False)
        fig.supxlabel('Data')
        fig.supylabel('Density')

        plt.savefig(fname=save_fn)

    def plot_pca(self, save_fn, x_sim, x_pred=None, num_comp=4, f_show=1.0):

        sim_color = 'blue'
        pred_color = 'orange'
        x = x_sim #StandardScaler().fit_transform(df)
        x = pd.DataFrame(x, columns=x.columns)
        nrow_keep = int(x.shape[0] * f_show)
        alpha = 100 / nrow_keep
        pca_model = PCA(n_components=num_comp)
        pca = pca_model.fit_transform(x)
        if self.pred_data_loaded:
            pca_pred = pca_model.transform(x_pred)
        
        pca_var = pca_model.explained_variance_ratio_
        fig, axs = plt.subplots(num_comp-1, num_comp-1, sharex=True, sharey=True)
        tick_spacing = 2
        for i in range(0, num_comp-1):
            for j in range(0, i+1):
                axs[i,j].scatter( pca[0:nrow_keep,i+1], pca[0:nrow_keep,j], alpha=alpha, marker='x', color=sim_color )
                if self.pred_data_loaded:
                    axs[i,j].scatter( pca_pred[0:nrow_keep,i+1], pca_pred[0:nrow_keep,j], alpha=1.0, color=pred_color, edgecolor='black' )
                #axs[i,j].xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
                #axs[i,j].yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
                if j == 0:
                    ylabel = 'PC{idx} ({var}%)'.format( idx=str(i+2), var=int(100*round(pca_var[i+1], ndigits=2)) )
                    axs[i,j].set_ylabel(ylabel, fontsize=12)
                if i == (num_comp-2):
                    xlabel = 'PC{idx} ({var}%)'.format( idx=str(j+1), var=int(100*round(pca_var[j], ndigits=2)) )
                    axs[i,j].set_xlabel(xlabel, fontsize=12)
        plt.tight_layout()
        plt.savefig(save_fn, format='pdf')
        plt.clf()