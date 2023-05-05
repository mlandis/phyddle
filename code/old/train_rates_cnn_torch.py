#!/usr/local/bin/python3

import csv
import os
import shutil
import importlib as im
import numpy as np
import scipy as sp
import tensorflow as tf
import cnn_utilities as cn
import sklearn
import matplotlib as plt
import eli5
import pydot, pydot_ng, plot_model

from eli5.sklearn import PermutationImportance  # check for important input features?
from phyddle_util import *
from scipy.stats import kde
from keras import *
from keras import layers

import torch

from PyPDF2 import PdfMerger

# analysis settings
settings = {}
settings['max_taxa'] = 200
settings['num_test'] = 20
settings['num_validation'] = 20
settings['num_epoch'] = 20
settings['batch_size'] = 32
settings['prefix'] = 'sim'
settings['model_name'] = 'bd1'

settings = init_cnn_settings(settings)

train_model  = settings['model_name']
train_prefix = settings['prefix']
num_test     = settings['num_test']
num_val      = settings['num_validation']
num_epochs   = settings['num_epoch']
batch_size   = settings['batch_size']
max_taxa     = settings['max_taxa']

# IO
#model_dir   = '../model/' + train_model
train_dir   = '../tensor_data/' + train_model
plot_dir    = '../plot/' + train_model
network_dir = '../network/' + train_model

os.makedirs(plot_dir, exist_ok=True)
os.makedirs(network_dir, exist_ok=True)

model_prefix    = train_prefix + '_batchsize' + str(batch_size) + '_numepoch' + str(num_epochs) + '_nt' + str(max_taxa)
model_csv_fn    = network_dir + '/' + model_prefix + '.csv' 
model_sav_fn    = network_dir + '/' + model_prefix + '.hdf5'
train_data_fn   = train_dir + '/' + train_prefix + '.nt' + str(max_taxa) + '.cdvs.data.csv'
train_stats_fn   = train_dir + '/' + train_prefix + '.nt' + str(max_taxa) + '.summ_stat.csv'
train_labels_fn = train_dir + '/' + train_prefix + '.nt' + str(max_taxa) + '.labels.csv'

# load data
full_data = pd.read_csv(train_data_fn, header=None, on_bad_lines='skip').to_numpy()
full_stats = pd.read_csv(train_stats_fn, header=None, on_bad_lines='skip').to_numpy()
full_labels = pd.read_csv(train_labels_fn, header=None, on_bad_lines='skip').to_numpy()

stat_names = full_stats[0,:]
full_stats = full_stats[1:,:].astype('float64')

full_labels = full_labels[:, [0, 3, 6, 18]] # geosse
#full_labels = full_labels[:, [0,3]] # birth + death
param_names = full_labels[0,:]
full_labels = full_labels[1:,:].astype('float64')

# data dimensions
num_chars  = 3 # another way to get this??
num_sample = full_data.shape[0]
num_params = full_labels.shape[1]
num_stats = full_stats.shape[1]

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
full_data.shape = (num_sample,-1,1+num_chars)

# create input subsets
train_idx = np.arange( num_test+num_val, num_sample )
val_idx   = np.arange( num_test, num_test+num_val )
test_idx  = np.arange( 0, num_test )

# create & normalize label tensors
#labels = full_labels[:,:] # 2nd column was 5:12 previously to exclude ancestral stem location

# normalize summary stats
norm_train_stats, train_stats_means, train_stats_sd = cn.normalize( full_stats[train_idx,:] )
norm_val_stats  = cn.normalize(full_stats[val_idx,:], (train_stats_means, train_stats_sd))
norm_test_stats = cn.normalize(full_stats[test_idx,:], (train_stats_means, train_stats_sd))

# (option for diff schemes) try normalizing against 0 to 1
norm_train_labels, train_label_means, train_label_sd = cn.normalize( full_labels[train_idx,:] )
norm_val_labels  = cn.normalize(full_labels[val_idx,:], (train_label_means, train_label_sd))
norm_test_labels = cn.normalize(full_labels[test_idx,:], (train_label_means, train_label_sd))

# potentially create new tensor that pulls info from both data and labels (Ammon advises put only in data)
# create data tensors
train_data_tensor = full_data[train_idx,:]
val_data_tensor   = full_data[val_idx,:]
test_data_tensor  = full_data[test_idx,:]

# summary stats
train_stats_tensor = full_stats[train_idx,:]
val_stats_tensor   = full_stats[val_idx,:]
test_stats_tensor  = full_stats[test_idx,:]

# initializer
#initializer = tf.keras.initializers.GlorotUniform()

# Build CNN
input_data_tensor = Input(shape = train_data_tensor.shape[1:3])

# double-check this stuff
# 64 patterns you expect to see, width of 3, stride (skip-size) of 1, padding zeroes so all windows are 'same'
# convolutional layers
w_conv = layers.Conv1D(64, 3, activation = 'relu', padding = 'same', name='in_conv_std')(input_data_tensor)
#w_conv = layers.Conv1D(64, 5, activation = 'relu', padding = 'same')(w_conv)
w_conv = layers.Conv1D(96, 5, activation = 'relu', padding = 'same')(w_conv)
w_conv = layers.Conv1D(128, 7, activation = 'relu', padding = 'same')(w_conv)
#w_conv = layers.Conv1D(256, 7, activation = 'relu', padding = 'same')(w_conv)
w_conv_global_avg = layers.GlobalAveragePooling1D(name = 'w_conv_global_avg')(w_conv)

# stride layers
w_stride = layers.Conv1D(64, 7, strides = 3, activation = 'relu', padding = 'same', name='in_conv_stride')(input_data_tensor)
w_stride = layers.Conv1D(96, 9, strides = 6, activation = 'relu', padding = 'same')(w_stride)
w_stride_global_avg = layers.GlobalAveragePooling1D(name = 'w_stride_global_avg')(w_stride)

# dilation layers
w_dilated = layers.Conv1D(32, 3, dilation_rate = 2, activation = 'relu', padding = 'same', name='in_conv_dilation')(input_data_tensor)
w_dilated = layers.Conv1D(64, 5, dilation_rate = 4, activation = 'relu', padding = "same")(w_dilated)
#w_dilated = layers.Conv1D(128, 7, dilation_rate = 8, activation = 'relu', padding = 'same')(w_dilated)
w_dilated_global_avg = layers.GlobalAveragePooling1D(name = 'w_dilated_global_avg')(w_dilated)


# summary stats
input_stats_tensor = Input(shape = train_stats_tensor.shape[1:2])
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

output_params = layers.Dense(num_params, activation = 'linear', name = "params")(wxyz)


# model initializer
# init = tf.initializers.GlorotUniform()
# var = tf.Variable(init(shape=shape))
# or a oneliner with a little confusing brackets
# var = tf.Variable(tf.initializers.GlorotUniform()(shape=shape))

# instantiate MODEL
mymodel = Model(inputs = [input_data_tensor, input_stats_tensor], 
                outputs = output_params)

my_loss = "mse"
mymodel.compile(optimizer = 'adam', 
                loss = my_loss, 
                metrics = ['mae', 'acc', 'mape'])

history = mymodel.fit([train_data_tensor, train_stats_tensor], 
                      norm_train_labels,
                      epochs = num_epochs,
                      batch_size = batch_size, 
                      validation_data = ([val_data_tensor, val_stats_tensor], norm_val_labels))
# validation_splitq
# use_multiprocessing

# make history plots
cn.make_history_plot(history, plot_dir=plot_dir)

# evaluate ???
mymodel.evaluate([test_data_tensor, test_stats_tensor], norm_test_labels)

# scatter plot training prediction to truth
max_idx = 1000
normalized_train_preds = mymodel.predict([train_data_tensor[0:max_idx,:,:], train_stats_tensor[0:max_idx,:]])

# reverse normalization
denormalized_train_labels = cn.denormalize(norm_train_labels[0:max_idx,:], train_label_means, train_label_sd)
denormalized_train_labels = np.exp(denormalized_train_labels)
train_preds = cn.denormalize(normalized_train_preds, train_label_means, train_label_sd)
train_preds = np.exp(train_preds)

# make scatter plots
cn.plot_preds_labels(train_preds, denormalized_train_labels, param_names = param_names, prefix='train', plot_dir=plot_dir, title='Train predictions')

# scatter plot test prediction to truth
normalized_test_preds = mymodel.predict([test_data_tensor, test_stats_tensor])

# reversing normalization
denormalized_test_labels = cn.denormalize(norm_test_labels, train_label_means, train_label_sd)
denormalized_test_labels = np.exp(denormalized_test_labels)
test_preds = cn.denormalize(normalized_test_preds, train_label_means, train_label_sd)
test_preds = np.exp(test_preds)

# summarize results
cn.plot_preds_labels(test_preds[0:1000,:], denormalized_test_labels[0:1000,:], param_names=param_names, prefix='test', plot_dir=plot_dir, title='Test predictions')

# SAVE MODEL to FILE
all_means = train_label_means #np.append(train_label_means, train_aux_priors_means)
all_sd = train_label_sd #np.append(train_label_sd, train_aux_priors_sd)
with open(model_csv_fn, 'w') as file:
    the_writer = csv.writer(file)
    the_writer.writerow(np.append( 'mean_sd', param_names ))
    the_writer.writerow(np.append( 'mean', all_means))
    the_writer.writerow(np.append( 'sd', all_sd))

mymodel.save(model_sav_fn)


# merge pdfs
merger = PdfMerger()
files = os.listdir(plot_dir)
files.sort()
for f in files:
    if '.pdf' in f:
        merger.append(plot_dir + '/' + f)

merger.write(plot_dir + '/all_results.pdf')



#import visualkeras
#model_viz_fn = plot_dir + '/' + model_prefix + '.model_viz.pdf'
#visualkeras.layered_view(mymodel, legend=True, draw_volume=False, spacing=30, to_file=model_viz_fn) # without custom font

model_arch_fn = plot_dir + '/' + model_prefix + '.model_architecture.pdf'
#plot_model.plot_model(mymodel, to_file=model_arch_fn, show_shapes=True, show_layer_names=False, rankdir='TB', expand_nested=False, style=0, color=True, dpi=96)
tf.keras.utils.plot_model(mymodel, to_file=model_arch_fn, show_shapes=True, show_layer_names=True)
#mymodel.weights
#mymodel.trainable_variables
