
#import cnn_utilities as cn
import pandas as pd
import numpy as np
import os
import csv

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
        self.model_prefix    = f'sim_batchsize{self.batch_size}_numepoch{self.num_epochs}_nt{self.tree_size}'
        self.model_csv_fn    = f'{self.network_dir}/{self.model_prefix}.csv'
        self.model_sav_fn    = f'{self.network_dir}/{self.model_prefix}.hdf5'
        self.input_stats_fn  = f'{self.input_dir}/sim.nt{self.tree_size}.summ_stat.csv'
        self.input_labels_fn = f'{self.input_dir}/sim.nt{self.tree_size}.labels.csv'
        if self.tree_type == 'extant':
            self.input_data_fn   = f'{self.input_dir}/sim.nt{self.tree_size}.cdvs.data.csv'
        elif self.tree_type == 'serial':
            self.input_data_fn   = f'{self.input_dir}/sim.nt{self.tree_size}.cblvs.data.csv' 

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
        print(self.num_char_row)
        print(self.num_tree_row)
        print(full_data.shape)

        # split dataset into training, test, and validation parts
        if self.num_test != 0 and self.num_validation != 0:
            num_val = self.num_validation
            num_test = self.num_test
        elif self.prop_test != 0 and self.prop_validation != 0:
            num_val = int(np.floor(num_sample * self.prop_validation))
            num_test = int(np.floor(num_sample * self.prop_test))

        print(num_val)
        print(num_test)

        # create input subsets
        train_idx = np.arange( num_test+num_val, num_sample )
        val_idx   = np.arange( num_test, num_test+num_val )
        test_idx  = np.arange( 0, num_test )

        print(train_idx)
        print(val_idx)
        print(test_idx)

        # normalize summary stats
        self.norm_train_stats, self.train_stats_means, self.train_stats_sd = Utilities.normalize( full_stats[train_idx,:] )
        self.norm_val_stats  = Utilities.normalize(full_stats[val_idx,:], (self.train_stats_means, self.train_stats_sd))
        self.norm_test_stats = Utilities.normalize(full_stats[test_idx,:], (self.train_stats_means, self.train_stats_sd))

        # (option for diff schemes) try normalizing against 0 to 1
        self.norm_train_labels, self.train_label_means, self.train_label_sd = Utilities.normalize( full_labels[train_idx,:] )
        self.norm_val_labels  = Utilities.normalize(full_labels[val_idx,:], (self.train_label_means, self.train_label_sd))
        self.norm_test_labels = Utilities.normalize(full_labels[test_idx,:], (self.train_label_means, self.train_label_sd))

        # create data tensors
        self.train_data_tensor = full_data[train_idx,:]
        self.val_data_tensor   = full_data[val_idx,:]
        self.test_data_tensor  = full_data[test_idx,:]

        # summary stats
        self.train_stats_tensor = full_stats[train_idx,:]
        self.val_stats_tensor   = full_stats[val_idx,:]
        self.test_stats_tensor  = full_stats[test_idx,:]
        
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
        
        return

    def make_results(self):

        # evaluate ???
        self.mymodel.evaluate([self.test_data_tensor, self.test_stats_tensor], self.norm_test_labels)

        # scatter plot training prediction to truth
        max_idx = 1000
        
        normalized_train_preds_thin = self.mymodel.predict([self.train_data_tensor[0:max_idx,:,:], self.train_stats_tensor[0:max_idx,:]])
        train_preds = Utilities.denormalize(normalized_train_preds_thin, self.train_label_means, self.train_label_sd)
        self.train_preds = np.exp(train_preds)

        denormalized_train_labels = Utilities.denormalize(self.norm_train_labels[0:max_idx,:], self.train_label_means, self.train_label_sd)
        self.denormalized_train_labels = np.exp(denormalized_train_labels)

        # scatter plot test prediction to truth
        normalized_test_preds = self.mymodel.predict([self.test_data_tensor, self.test_stats_tensor])
        test_preds = Utilities.denormalize(normalized_test_preds, self.train_label_means, self.train_label_sd)
        self.test_preds = np.exp(test_preds)

        denormalized_test_labels = Utilities.denormalize(self.norm_test_labels, self.train_label_means, self.train_label_sd)
        self.denormalized_test_labels = np.exp(denormalized_test_labels)
        
        return
    

    def save_results(self):

        # make history plots
        Utilities.make_history_plot(self.history, prefix=self.model_prefix+'_train', plot_dir=self.plot_dir)
        #cn.make_history_plot(self.history, plot_dir=self.plot_dir)

        
        # make scatter plots
        Utilities.plot_preds_labels(preds=self.train_preds,
                             labels=self.denormalized_train_labels,
                             param_names=self.param_names,
                             prefix=self.model_prefix+'_train',
                             plot_dir=self.plot_dir,
                             title='Train predictions')

        # summarize results
        Utilities.plot_preds_labels(preds=self.test_preds[0:1000,:],
                             labels=self.denormalized_test_labels[0:1000,:],
                             param_names=self.param_names,
                             prefix=self.model_prefix+'_test',
                             plot_dir=self.plot_dir,
                             title='Test predictions')

        # SAVE MODEL to FILE
        all_means = self.train_label_means #np.append(train_label_means, train_aux_priors_means)
        all_sd = self.train_label_sd #np.append(train_label_sd, train_aux_priors_sd)
        with open(self.model_csv_fn, 'w') as file:
            the_writer = csv.writer(file)
            the_writer.writerow(np.append( 'mean_sd', self.param_names ))
            the_writer.writerow(np.append( 'mean', all_means))
            the_writer.writerow(np.append( 'sd', all_sd))

        self.mymodel.save(self.model_sav_fn)

        # merge pdfs
        merger = PdfMerger()
        files = os.listdir(self.plot_dir)
        files.sort()
        for f in files:
            if '.pdf' in f and self.model_prefix in f and 'all_results.pdf' not in f:
                merger.append(self.plot_dir + '/' + f)

        merger.write(self.plot_dir+'/'+self.model_prefix+'_'+'all_results.pdf')
        return
