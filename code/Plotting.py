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
        #self.network_prefix    = args['network_prefix']
        self.batch_size        = args['batch_size']
        self.num_epochs        = args['num_epochs']
        self.tree_size         = args['tree_size']
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
        self.history_json_fn = f'{self.net_job_dir}/{self.network_prefix}.train_history.json'

        
        return

    def load_data(self):
        #genfromtxt('my_file.csv', delimiter=',')
        self.train_preds  = pd.read_csv(self.train_pred_fn)
        self.train_labels = pd.read_csv(self.train_labels_fn)
        self.test_preds   = pd.read_csv(self.test_pred_fn)
        self.test_labels  = pd.read_csv(self.test_labels_fn)
        
        self.input_stats  = pd.read_csv( self.input_stats_fn )
        self.input_labels = pd.read_csv( self.input_labels_fn )

        self.param_names  = self.train_preds.columns.to_list()

        self.train_preds_max_idx = min( 1000, self.train_preds.shape[0] )
        self.train_labels_max_idx = min( 1000, self.train_labels.shape[0] )
        self.test_preds_max_idx = min( 1000, self.test_preds.shape[0] )
        self.test_labels_max_idx = min( 1000, self.test_labels.shape[0] )

        self.history_dict = json.load(open(self.history_json_fn, 'r'))

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
        Utilities.plot_ss_param_hist(df=df_all, save_fn=self.plt_job_dir + '/' + self.network_prefix + '.sim_histogram.pdf')
        Utilities.plot_pca(df=df_all, save_fn=self.plt_job_dir + '/' + self.network_prefix + '.sim_pca.pdf')

    
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