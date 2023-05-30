
#import cnn_utilities as cn
import pandas as pd
import numpy as np
import os
import csv
import json
#import pickle
import dill # richer serialization than pickle

from PyPDF2 import PdfMerger

#import tensorflow as tf
from keras import *
from keras import layers

import Utilities

class Learner:
    def __init__(self, args):
        self.set_args(args)
        self.prepare_files()
        return
    
    def set_args(self, args):
        self.args              = args
        self.num_test          = 0
        self.num_validation    = 0
        self.prop_test         = 0
        self.prop_test         = 0
        self.job_name          = args['job_name']
        self.tree_size         = args['tree_size']
        self.tree_type         = args['tree_type']
        if self.tree_type == 'extant':
            self.num_tree_row = 1
        elif self.tree_type == 'serial':
            self.num_tree_row = 2
        else:
            raise NotImplementedError
        self.num_char_row      = args['num_char']
        #self.predict_idx       = args['predict_idx']
        self.fmt_dir           = args['fmt_dir']
        self.plt_dir           = args['plt_dir']
        self.net_dir           = args['net_dir']
        self.batch_size        = args['batch_size']
        self.num_epochs        = args['num_epochs']    
        if 'num_test' in args and 'num_validation' in args:
            self.num_test          = args['num_test']
            self.num_validation    = args['num_validation']
        elif 'prop_test' in args and 'prop_validation' in args:
            self.prop_test         = args['prop_test']
            self.prop_validation   = args['prop_validation']
        self.loss              = args['loss']
        self.optimizer         = args['optimizer']
        self.metrics           = args['metrics']
        return
    
    def prepare_files(self):

        # main directories
        self.input_dir   = self.fmt_dir + '/' + self.job_name
        self.plot_dir    = self.plt_dir + '/' + self.job_name
        self.network_dir = self.net_dir + '/' + self.job_name

        # create new job directories
        os.makedirs(self.plot_dir, exist_ok=True)
        os.makedirs(self.network_dir, exist_ok=True)

        # main job filenames
        self.model_prefix       = f'sim_batchsize{self.batch_size}_numepoch{self.num_epochs}_nt{self.tree_size}'
        self.model_csv_fn       = f'{self.network_dir}/{self.model_prefix}.csv'
        self.model_sav_fn       = f'{self.network_dir}/{self.model_prefix}.hdf5'
        self.model_trn_lbl_norm_fn  = f'{self.network_dir}/{self.model_prefix}.train_label_norm.csv'
        self.model_trn_ss_norm_fn   = f'{self.network_dir}/{self.model_prefix}.train_summ_stat_norm.csv'
        self.model_hist_fn      = f'{self.network_dir}/{self.model_prefix}.train_history.json'
        self.model_cpi_func_fn  = f'{self.network_dir}/{self.model_prefix}.cpi_func.obj'
        self.train_pred_fn      = f'{self.network_dir}/{self.model_prefix}.train_pred.csv'
        self.train_labels_fn    = f'{self.network_dir}/{self.model_prefix}.train_labels.csv'
        self.test_pred_fn       = f'{self.network_dir}/{self.model_prefix}.test_pred.csv'
        self.test_labels_fn     = f'{self.network_dir}/{self.model_prefix}.test_labels.csv'
        self.input_stats_fn     = f'{self.input_dir}/sim.nt{self.tree_size}.summ_stat.csv'
        self.input_labels_fn    = f'{self.input_dir}/sim.nt{self.tree_size}.labels.csv'
        if self.tree_type == 'extant':
            self.input_data_fn  = f'{self.input_dir}/sim.nt{self.tree_size}.cdvs.data.csv'
        elif self.tree_type == 'serial':
            self.input_data_fn  = f'{self.input_dir}/sim.nt{self.tree_size}.cblvs.data.csv' 
        
        return

    def run(self):
        self.load_input()
        self.build_network()
        self.train()
        self.make_results()
        self.save_results()
    
    def load_input(self):
        raise NotImplementedError
    def build_network(self):
        raise NotImplementedError
    def train(self):
        raise NotImplementedError
    def make_results(self):
        raise NotImplementedError
    def save_results(self):
        raise NotImplementedError
    
class CnnLearner(Learner):
    def __init__(self, args):
        super().__init__(args)
        return
    
    def load_input(self):

        # read data
        full_data   = pd.read_csv(self.input_data_fn, header=None, on_bad_lines='skip').to_numpy()
        full_stats  = pd.read_csv(self.input_stats_fn, header=None, on_bad_lines='skip').to_numpy()
        full_labels = pd.read_csv(self.input_labels_fn, header=None, on_bad_lines='skip').to_numpy()

        # get summary stats
        self.stat_names = full_stats[0,:]
        full_stats = full_stats[1:,:].astype('float64')

        # get parameters and labels
        #if len(self.predict_idx) > 0:
        #    full_labels = full_labels[:, self.predict_idx]
        self.param_names = full_labels[0,:]
        full_labels = full_labels[1:,:].astype('float64')   

        # data dimensions
        # num_chars  = 3  # MJL: better way to get this? either from tensor or from an info file?
        num_sample = full_data.shape[0]
        #self.num_chars = full_data.shape[1]
        self.num_params = full_labels.shape[1]
        self.num_stats = full_stats.shape[1]

        # take logs of labels (rates)
        # for variance stabilization for heteroskedastic (variance grows with mean)
        full_labels = np.log(full_labels)

        # randomize data to ensure iid of batches
        # do not want to use exact same datapoints when iteratively improving
        # training/validation accuracy
        randomized_idx = np.random.permutation(full_data.shape[0])
        full_data      = full_data[randomized_idx,:]
        full_stats     = full_stats[randomized_idx,:]
        full_labels    = full_labels[randomized_idx,:]

        # reshape full_data
        # depends on CBLV/CDV and num_states
        full_data.shape = ( num_sample, -1, (self.num_tree_row+self.num_char_row) )
        # print(self.num_char_row)
        # print(self.num_tree_row)
        # print(full_data.shape)

        # split dataset into training, test, and validation parts
        if self.num_test != 0 and self.num_validation != 0:
            num_val = self.num_validation
            num_test = self.num_test
        elif self.prop_test != 0 and self.prop_validation != 0:
            num_val = int(np.floor(num_sample * self.prop_validation))
            num_test = int(np.floor(num_sample * self.prop_test))

        # print(num_val)
        # print(num_test)

        # create input subsets
        train_idx = np.arange( num_test+num_val, num_sample )
        val_idx   = np.arange( num_test, num_test+num_val )
        test_idx  = np.arange( 0, num_test )

        # print(train_idx)
        # print(val_idx)
        # print(test_idx)

        ## MJL need to save train_stats_means, train_stats_sd
        # these will then be used to normalize new test data for predictions

        # normalize summary stats
        self.norm_train_stats, self.train_stats_means, self.train_stats_sd = Utilities.normalize( full_stats[train_idx,:] )
        self.norm_val_stats  = Utilities.normalize(full_stats[val_idx,:], (self.train_stats_means, self.train_stats_sd))
        self.norm_test_stats = Utilities.normalize(full_stats[test_idx,:], (self.train_stats_means, self.train_stats_sd))

        # (option for diff schemes) try normalizing against 0 to 1
        self.norm_train_labels, self.train_label_means, self.train_label_sd = Utilities.normalize( full_labels[train_idx,:] )
        self.norm_val_labels  = Utilities.normalize(full_labels[val_idx,:], (self.train_label_means, self.train_label_sd))
        self.norm_test_labels = Utilities.normalize(full_labels[test_idx,:], (self.train_label_means, self.train_label_sd))

        # print(self.param_names)
        # print(self.norm_train_labels)
        # print(type(self.norm_train_labels))
        # print(full_stats[0,:])
        # df = pd.DataFrame( [self.stat_names, self.train_stats_means, self.train_stats_sd] ).T # columns=['name', 'mean', 'sd'] )
        # df.columns = ['name', 'mean', 'sd']


        # create data tensors
        self.train_data_tensor = full_data[train_idx,:]
        self.val_data_tensor   = full_data[val_idx,:]
        self.test_data_tensor  = full_data[test_idx,:]

        # summary stats
        self.train_stats_tensor = self.norm_train_stats #full_stats[train_idx,:]
        self.val_stats_tensor   = self.norm_val_stats #full_stats[val_idx,:]
        self.test_stats_tensor  = self.norm_test_stats #full_stats[test_idx,:]
        self.unnormalized_train_stats = full_stats[train_idx,:]

        return
    
    def build_network(self):
        # Build CNN
        input_data_tensor = Input(shape = self.train_data_tensor.shape[1:3])

        # convolutional layers
        # e.g. you expect to see 64 patterns, width of 3, stride (skip-size) of 1, padding zeroes so all windows are 'same'
        w_conv = layers.Conv1D(64, 3, activation = 'relu', padding = 'same', name='in_conv_std')(input_data_tensor)
        w_conv = layers.Conv1D(96, 5, activation = 'relu', padding = 'same')(w_conv)
        w_conv = layers.Conv1D(128, 7, activation = 'relu', padding = 'same')(w_conv)
        w_conv_global_avg = layers.GlobalAveragePooling1D(name = 'w_conv_global_avg')(w_conv)

        # stride layers (skip sizes during slide
        w_stride = layers.Conv1D(64, 7, strides = 3, activation = 'relu', padding = 'same', name='in_conv_stride')(input_data_tensor)
        w_stride = layers.Conv1D(96, 9, strides = 6, activation = 'relu', padding = 'same')(w_stride)
        w_stride_global_avg = layers.GlobalAveragePooling1D(name = 'w_stride_global_avg')(w_stride)

        # dilation layers (spacing among grid elements)
        w_dilated = layers.Conv1D(32, 3, dilation_rate = 2, activation = 'relu', padding = 'same', name='in_conv_dilation')(input_data_tensor)
        w_dilated = layers.Conv1D(64, 5, dilation_rate = 4, activation = 'relu', padding = "same")(w_dilated)
        w_dilated_global_avg = layers.GlobalAveragePooling1D(name = 'w_dilated_global_avg')(w_dilated)

        # summary stats
        input_stats_tensor = Input(shape = self.train_stats_tensor.shape[1:2])
        w_stats_ffnn = layers.Dense(128, activation = 'relu', kernel_initializer = 'VarianceScaling', name='in_ffnn_stat')(input_stats_tensor)
        w_stats_ffnn = layers.Dense(64, activation = 'relu', kernel_initializer = 'VarianceScaling')(w_stats_ffnn)
        w_stats_ffnn = layers.Dense(32, activation = 'relu', kernel_initializer = 'VarianceScaling')(w_stats_ffnn)

        # concatenate all above -> deep fully connected network
        concatenated_wxyz = layers.Concatenate(axis = 1, name = 'all_concatenated')([w_conv_global_avg,
                                                                                    w_stride_global_avg,
                                                                                    w_dilated_global_avg,
                                                                                    w_stats_ffnn])

        # VarianceScaling for kernel initializer (look up?? )
        wxyz = layers.Dense(128, activation = 'relu', kernel_initializer = 'VarianceScaling')(concatenated_wxyz)
        #wxyz = layers.Dense(96, activation = 'relu', kernel_initializer = 'VarianceScaling')(wxyz)
        wxyz = layers.Dense(64, activation = 'relu', kernel_initializer = 'VarianceScaling')(wxyz)
        wxyz = layers.Dense(32, activation = 'relu', kernel_initializer = 'VarianceScaling')(wxyz)

        output_params = layers.Dense(self.num_params, activation = 'linear', name = "params")(wxyz)

        # instantiate MODEL
        self.mymodel = Model(inputs = [input_data_tensor, input_stats_tensor], 
                        outputs = output_params)

    def train(self):
        
        self.mymodel.compile(optimizer = self.optimizer, 
                        loss = self.loss, 
                        metrics = self.metrics)
        
        self.history = self.mymodel.fit(\
            x = [self.train_data_tensor, self.train_stats_tensor], 
            y = self.norm_train_labels,
            epochs = self.num_epochs,
            batch_size = self.batch_size, 
            validation_data = ([self.val_data_tensor, self.val_stats_tensor], self.norm_val_labels))
        
        self.history_dict = self.history.history
        #print(self.history2)
        
        return

    def make_results(self):

        # evaluate ???
        self.mymodel.evaluate([self.test_data_tensor, self.test_stats_tensor], self.norm_test_labels)

        # scatter plot training prediction to truth
        max_idx = 1000

        self.normalized_train_preds       = self.mymodel.predict([self.train_data_tensor, self.train_stats_tensor])
        #self.normalized_train_preds_thin  = normalized_train_preds[0:max_idx,:]
        self.denormalized_train_preds     = Utilities.denormalize(self.normalized_train_preds, self.train_label_means, self.train_label_sd)
        self.denormalized_train_preds     = np.exp(self.denormalized_train_preds)
        self.denormalized_train_labels    = Utilities.denormalize(self.norm_train_labels[0:max_idx,:], self.train_label_means, self.train_label_sd)
        #denormalized_train_labels         = Utilities.denormalize(self.norm_train_labels[0:max_idx,:], self.train_label_means, self.train_label_sd)
        self.denormalized_train_labels    = np.exp(self.denormalized_train_labels)

        # scatter plot test prediction to truth
        self.normalized_test_preds        = self.mymodel.predict([self.test_data_tensor, self.test_stats_tensor])
        self.denormalized_test_preds      = Utilities.denormalize(self.normalized_test_preds, self.train_label_means, self.train_label_sd)
        self.denormalized_test_preds      = np.exp(self.denormalized_test_preds)
        self.denormalized_test_labels     = Utilities.denormalize(self.norm_test_labels, self.train_label_means, self.train_label_sd)
        self.denormalized_test_labels     = np.exp(self.denormalized_test_labels)
        
        # conformalized prediction interval functions
        self.cpi_func = {} # 'lower':[], 'upper':[] }
        #print(self.train_preds)
        #print(self.denormalized_train_labels)
        for i,p in enumerate(self.param_names):
        #print(i,p)
        #x_cpi = self.unnormalized_train_stats[:,0:3]
        #x_cpi = self.unnormalized_train_stats[:,0:3]
        #print(p)
            self.cpi_func[p] = {}
            x_pred_cpi = self.normalized_train_preds[:,i].reshape(-1,1)
            x_stat_cpi = self.norm_train_stats[:,0:2]
            x_true_cpi = self.norm_train_labels[:,i].reshape(-1,1)

            #print(x_pred_cpi)
            #print(x_stat_cpi)
            #print(x_true_cpi)
            #print(x_pred_cpi.shape)
            #print(x_stat_cpi.shape)
            #print(x_true_cpi.shape)
            #print(x_cpi[:10,])
            #print(y_cpi.shape)
            #print(y_cpi[:10])
            self.lower_cpi, self.upper_cpi = Utilities.get_CPI2(x_pred_cpi, x_stat_cpi, x_true_cpi, frac=0.1, inner_quantile=0.95, num_grid_points=4)
            self.cpi_func[p]['lower'] = self.lower_cpi
            self.cpi_func[p]['upper'] = self.upper_cpi


        return
    

    def save_results(self):

        max_idx = 1000
        
        # save model to file
        self.mymodel.save(self.model_sav_fn)

        # save summ_stat names, means, sd for new test dataset normalization
        df_stats = pd.DataFrame( [self.stat_names, self.train_stats_means, self.train_stats_sd] ).T # columns=['name', 'mean', 'sd'] )
        df_stats.columns = ['name', 'mean', 'sd']
        df_stats.to_csv(self.model_trn_ss_norm_fn, index=False, sep=',')
 
        # save label names, means, sd for new test dataset normalization
        df_labels = pd.DataFrame( [self.param_names, self.train_label_means, self.train_label_sd] ).T # columns=['name', 'mean', 'sd'] )
        df_labels.columns = ['name', 'mean', 'sd']
        df_labels.to_csv(self.model_trn_lbl_norm_fn, index=False, sep=',')

        # save train prediction scatter data
        df_train_pred   = pd.DataFrame( self.denormalized_train_preds[0:max_idx,:], columns=self.param_names )
        df_train_labels = pd.DataFrame( self.denormalized_train_labels[0:max_idx,:], columns=self.param_names )
        df_test_pred    = pd.DataFrame( self.denormalized_test_preds[0:max_idx,:], columns=self.param_names )
        df_test_labels  = pd.DataFrame( self.denormalized_test_labels[0:max_idx,:], columns=self.param_names )
        df_train_pred.to_csv(self.train_pred_fn, index=False, sep=',')
        df_train_labels.to_csv(self.train_labels_fn, index=False, sep=',')
        df_test_pred.to_csv(self.test_pred_fn, index=False, sep=',')
        df_test_labels.to_csv(self.test_labels_fn, index=False, sep=',')
        
        #, self.denormalized_train_labels[0:1000,:] ], )
        json.dump(self.history_dict, open(self.model_hist_fn, 'w'))

        # pickle CPI
        cpi_file_obj = open(self.model_cpi_func_fn, 'wb')
        dill.dump(self.cpi_func, cpi_file_obj)
        cpi_file_obj.close()
        
        # make history plots
        ## Utilities.make_history_plot(self.history2, prefix=self.model_prefix+'_train', plot_dir=self.plot_dir)

        # # train prediction scatter plots
        # Utilities.plot_preds_labels(preds=self.train_preds[0:1000,:],
        #                      labels=self.denormalized_train_labels[0:1000,:],
        #                      param_names=self.param_names,
        #                      prefix=self.model_prefix+'_train',
        #                      plot_dir=self.plot_dir,
        #                      title='Train predictions')

        # # test predicition scatter plots
        # Utilities.plot_preds_labels(preds=self.test_preds[0:1000,:],
        #                      labels=self.denormalized_test_labels[0:1000,:],
        #                      param_names=self.param_names,
        #                      prefix=self.model_prefix+'_test',
        #                      plot_dir=self.plot_dir,
        #                      title='Test predictions')

        # # save PCA and summ_stat histograms
        # df_stats = pd.read_csv( self.input_stats_fn )
        # df_param = pd.read_csv( self.input_labels_fn )
        # df_all = pd.concat( [df_stats, df_param], axis=1 )
        # df_all = df_all.T.drop_duplicates().T
        # Utilities.plot_ss_param_hist(df=df_all, save_fn=self.plot_dir + '/' + self.model_prefix + '.summ_stat_param_hist.pdf')
        # Utilities.plot_pca(df=df_all, save_fn=self.plot_dir + '/' + self.model_prefix + '.summ_stat_param_pca.pdf')

        # # combine pdfs
        # merger = PdfMerger()
        # files = os.listdir(self.plot_dir)
        # files.sort()
        # for f in files:
        #     if '.pdf' in f and self.model_prefix in f and 'all_results.pdf' not in f:
        #         merger.append(self.plot_dir + '/' + f)

        # merger.write(self.plot_dir+'/'+self.model_prefix+'_'+'all_results.pdf')
        return
