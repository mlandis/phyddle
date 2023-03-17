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
import seaborn as sb

from scipy.stats import kde
from keras import *

# analysis settings
num_test = 5 #1000
num_validation = 10 #5000
num_epochs = 20
batch_size = 16

# IO
train_prefix = 'sim'
train_model = 'geosse_v2'
model_dir = '../model/' + train_model
train_dir = model_dir + '/data/train'

model_prefix = train_prefix + '_batchsize' + str(batch_size) + '_numepoch' + str(num_epochs)
model_csv_fn = model_dir + '/' + model_prefix + '.csv' 
model_sav_fn = model_dir + '/' + model_prefix + '.hdf5'
train_data_fn = train_dir + '/' + train_prefix + '.data.csv'
train_labels_fn = train_dir + '/' + train_prefix + '.labels.csv'

# load data
full_data,full_labels = cn.load_input(train_data_fn, train_labels_fn)
param_names = full_labels[0,:]
full_labels = full_labels[1:,:].astype('float64')



# data dimensions
max_tips = 501 # get width of input tensor
num_chars = 3
num_sample = full_data.shape[0]
num_params = full_labels.shape[1]

# not sure if we need this...
#num_tips = cn.get_num_tips(full_data)
#subsample_prop = full_data[:,(max_tips-1)*7] # why times 7?? I think this has to do with spare tree sampling
#mu = full_data[:,(max_tips-3)*7] # not sure

# subsample_prop : find subsample proportion stats from cblv file for epi phylogeo
# use max_tips to split cblv input tensor from summary stats tensor

# take logs of labels (rates)
# for variance stabilization for heteroskedastic (variance grows with mean)
full_labels = np.log(full_labels)

# randomize data to ensure iid of batches
# do not want to use exact same datapoints when iteratively improving
# training/validation accuracy
randomized_idx = np.random.permutation(full_data.shape[0])
full_data = full_data[randomized_idx,:]
full_labels = full_labels[randomized_idx,:]

# reshape full_data
full_data.shape = (num_sample,-1,2+num_chars)

# create input subsets
train_idx      = np.arange( num_test+num_validation, num_sample )
validation_idx = np.arange( num_test, num_test+num_validation )
test_idx       = np.arange( 0, num_test )

# create & normalize label tensors
labels = full_labels[:,:] # 2nd column was 5:12 previously to exclude ancestral stem location

# (option for diff schemes) try normalizing against 0 to 1
norm_train_labels, train_label_means, train_label_sd = cn.normalize( labels[train_idx,:] )
norm_validation_labels = cn.normalize(labels[validation_idx,:], (train_label_means, train_label_sd))
norm_test_labels = cn.normalize(labels[test_idx,:], (train_label_means, train_label_sd))

# potentially create new tensor that pulls info from both data and labels (Ammon advises put only in data)

# create data tensors

#full_treeLocation_tensor, full_prior_tensor = cn.create_data_tensors(data = full_data[0:num_sample,:], max_tips = max_tips, cblv_contains_mu_rho = False)
#train_treeLocation_tensor, validation_treeLocation_tensor, test_treeLocation_tensor = cn.create_train_val_test_tensors(full_treeLocation_tensor, num_validation, num_test)
#train_prior_tensor, validation_prior_tensor,  test_prior_tensor = cn.create_train_val_test_tensors(full_prior_tensor, num_validation, num_test)
train_treeLocation_tensor = full_data[train_idx,:]
validation_treeLocation_tensor = full_data[validation_idx,:]
test_treeLocation_tensor = full_data[test_idx,:]

# print(train_treeLocation_tensor.shape, train_prior_tensor.shape)
# print(test_treeLocation_tensor.shape, test_prior_tensor.shape)
# print(validation_treeLocation_tensor.shape, validation_prior_tensor.shape)

# Build CNN
input_treeLocation_tensor = Input(shape = train_treeLocation_tensor.shape[1:3])

# double-check this stuff
# 64 patterns you expect to see, width of 3, stride (skip-size) of 1, padding zeroes so all windows are 'same'
w_input = layers.Conv1D(64, 3, strides = 1, activation = 'relu', padding = 'same')(input_treeLocation_tensor)

# convolutional layers
w_conv = layers.Conv1D(64, 5, activation = 'relu', padding = 'same')(w_input)
#w = layers.MaxPooling1D(pool_size = 3, stride = 1)(w)
w_conv = layers.Conv1D(96, 5, activation = 'relu', padding = 'same')(w_conv)
w_conv = layers.Conv1D(128, 5, activation = 'relu', padding = 'same')(w_conv)
w_conv = layers.Conv1D(256, 7, activation = 'relu', padding = 'same')(w_conv)
w_conv_global_avg = layers.GlobalAveragePooling1D(name = 'w_conv_global_avg')(w_conv)

# stride layers
w_stride = layers.Conv1D(64, 7, strides = 3, activation = 'relu', padding = 'same')(input_treeLocation_tensor)
w_stride = layers.Conv1D(96, 9, strides = 6, activation = 'relu', padding = 'same')(w_stride)
w_stride_global_avg = layers.GlobalAveragePooling1D(name = 'w_stride_global_avg')(w_stride)

# tree + geolocation dilated
w_dilated = layers.Conv1D(32, 3, dilation_rate = 2, activation = 'relu', padding = "same")(input_treeLocation_tensor)
w_dilated = layers.Conv1D(64, 5, dilation_rate = 4, activation = 'relu', padding = "same")(w_dilated)
w_dilated = layers.Conv1D(128, 7, dilation_rate = 8, activation = 'relu', padding = "same")(w_dilated)
w_dilated_global_avg = layers.GlobalAveragePooling1D(name = 'w_dilated_global_avg')(w_dilated)

# # prior known parameters and data statistics
# input_priors_tensor = Input(shape = train_prior_tensor.shape[1:3])
# priors = layers.Flatten()(input_priors_tensor)
# priors = layers.Dense(32, activation = 'relu', kernel_initializer = 'VarianceScaling', name = 'prior1')(priors)

# concatenate all above -> deep fully connected network
concatenated_wxyz = layers.Concatenate(axis = 1, name = 'all_concatenated')([w_stride_global_avg,
                                                                             w_conv_global_avg,
                                                                             w_dilated_global_avg])
                                                                             #priors])

# VarianceScaling for kernel initializer (look up?? )
wxyz = layers.Dense(256, activation = 'relu', kernel_initializer = 'VarianceScaling')(concatenated_wxyz)
wxyz = layers.Dense(128, activation = 'relu', kernel_initializer = 'VarianceScaling')(wxyz)
wxyz = layers.Dense(64, activation = 'relu', kernel_initializer = 'VarianceScaling')(wxyz)
wxyz = layers.Dense(32, activation = 'relu', kernel_initializer = 'VarianceScaling')(wxyz)

output_params = layers.Dense(num_params, activation = 'linear', name = "params")(wxyz)

my_loss = "mse"

# instantiate MODEL
mymodel = Model(inputs = [input_treeLocation_tensor], #, input_priors_tensor], 
                outputs = output_params)

mymodel.compile(optimizer = 'adam', 
                loss = my_loss, 
                metrics = ['mae', 'acc', 'mape'])

history = mymodel.fit([train_treeLocation_tensor], #, train_prior_tensor], 
                      norm_train_labels,
                      epochs = num_epochs, batch_size = batch_size, 
                      validation_data = ([validation_treeLocation_tensor], norm_validation_labels))


# make history plots
cn.make_history_plot(history)

mymodel.evaluate([test_treeLocation_tensor], norm_test_labels)


# scatter plot training prediction to truth
normalized_train_preds = mymodel.predict([train_treeLocation_tensor[0:1000,:,:]]) #, 
                                          #train_prior_tensor[0:1000,:,:]])

# reverse normalization
denormalized_train_labels = cn.denormalize(norm_train_labels[0:1000,:], train_label_means, train_label_sd)
denormalized_train_labels = np.exp(denormalized_train_labels)
train_preds = cn.denormalize(normalized_train_preds, train_label_means, train_label_sd)
train_preds = np.exp(train_preds)

# make scatter plots
cn.plot_preds_labels(train_preds, denormalized_train_labels, 
                     param_names = param_names)

# scatter plot test prediction to truth
normalized_test_preds = mymodel.predict([test_treeLocation_tensor]) #, test_prior_tensor])

# reversing normalization
denormalized_test_labels = cn.denormalize(norm_test_labels, train_label_means, train_label_sd)
denormalized_test_labels = np.exp(denormalized_test_labels)
test_preds = cn.denormalize(normalized_test_preds, train_label_means, train_label_sd)
test_preds = np.exp(test_preds)

# summarize results
cn.plot_preds_labels(test_preds[0:1000,:], denormalized_test_labels[0:1000,:], param_names = param_names)

# SAVE MODEL to FILE
all_means = train_label_means #np.append(train_label_means, train_aux_priors_means)
all_sd = train_label_sd #np.append(train_label_sd, train_aux_priors_sd)
with open(model_csv_fn, 'w') as file:
    the_writer = csv.writer(file)
    the_writer.writerow(np.append( 'mean_sd', param_names ))
    the_writer.writerow(np.append( 'mean', all_means))
    the_writer.writerow(np.append( 'sd', all_sd))

mymodel.save(model_sav_fn)
