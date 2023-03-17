#!/usr/bin/env python
# coding: utf-8


import os, shutil
from keras import models
from keras import layers
from keras import losses

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as sp
from sklearn import metrics



# loss functions
def myLoss(y_true, y_pred):
    power = 3
    power_loss = tf.math.abs(y_true - y_pred)**power
    return tf.reduce_mean(power_loss, axis=-1)


def summarize_categorical_performance(y_true, y_pred):
    accuracy = np.max(y_pred * y_true[:,:5], axis = 1)
    auc = metrics.roc_auc_score(y_true[:,:5], y_pred)
    
    ### eps set due to 3 sig digits rounding in get_root_state_probs.sh script. 
    # set to midpoint between 0 and 0.001
    cross_entropy = metrics.log_loss(y_true[:,:5], y_pred, eps = 5e-4) 
    
    return accuracy, auc, cross_entropy

    
def tip_freq_accuracy(treeLocation_tensor, labels, num_locs = 5):

    tip_loc_counts = np.zeros((treeLocation_tensor.shape[0], num_locs))
    tip_loc_distro = np.zeros((treeLocation_tensor.shape[0], num_locs))
    accuracy_tipfreq = np.zeros((treeLocation_tensor.shape[0]))

    for i in range(0, treeLocation_tensor.shape[0]):    
        tip_loc_counts[i,:] = sum(treeLocation_tensor[i,:,2:2+num_locs])
        tip_loc_distro[i,:] = tip_loc_counts[i,:] / sum(tip_loc_counts[i,:])
        accuracy_tipfreq[i] = sum(tip_loc_distro[i,:] * labels[i,:5])
        
    return accuracy_tipfreq, tip_loc_distro

    
def get_num_tips(tree_data_tensor):
    # tree size
    num_sample = tree_data_tensor.shape[0]
    tree_data_tensor = tree_data_tensor.reshape((num_sample, 502, 7), order = 'C')
    num_tips = []
    for i in range(tree_data_tensor.shape[0]):
        num_tips.append(len(np.where(tree_data_tensor[i,:,0] > 0)[0]))
    num_tips = np.asarray(num_tips)
    
    return np.array(num_tips)


def normalize_01(data, min_max = None):
    if(type(min_max) == type(None)):
        max_value = data.max(axis = 0)
        min_value = data.min(axis = 0)
        difference = max_value - min_value
        difference[np.where(difference <= 0)] = 1
        return (max_value - data)/difference, min_value, max_value
    else:
        min_value = min_max[0]
        max_value = min_max[1]
        difference = max_value - min_value
        difference[np.where(difference <= 0)] = 1
        return (max_value - data)/difference

    
    
def normalize(data, m_sd = None):
    if(type(m_sd) == type(None )):
        m = data.mean(axis = 0)
        sd = data.std(axis = 0)
        sd[np.where(sd == 0)] = 1
        return (data - m)/sd, m, sd
    else:
        m_sd[1][np.where(m_sd[1] == 0)] = 1
        return (data - m_sd[0])/m_sd[1]
        
    

    
def denormalize(data, train_mean, train_sd, log_labels = False):
    return data * train_sd + train_mean


def denormalize_01(data, train_min, train_max):
    return train_max - data * (train_max - train_min)

        
def create_data_tensors2(data, mu, subsample_prop,
                            tmrca, mean_bl, num_tips, num_locs, max_tips,
                           cblv_contains_mu_rho = True):
    
    num_sample = data.shape[0]
    
    # reshape data tensor    
    full_data_tensor = data.reshape((num_sample, max_tips, num_locs + 2), order = 'C')

    # create tree/location tensor
    if(cblv_contains_mu_rho):
        full_treeLocation_tensor = full_data_tensor[:,:max_tips-3,:]
    else:
        full_treeLocation_tensor = full_data_tensor
        
    
    # create prior tensor
    subsample_prop = np.repeat(subsample_prop, 2)
    subsample_prop = subsample_prop.reshape((num_sample, 1, 2))
    mu = np.repeat(mu , 2)
    mu = mu.reshape((num_sample , 1, 2))
    num_tips = np.repeat(num_tips, 2)
    num_tips = num_tips.reshape((num_sample, 1, 2))
    tmrca = np.repeat(tmrca, 2)
    tmrca = tmrca.reshape((num_sample, 1, 2))
    mean_bl = np.repeat(mean_bl, 2)
    mean_bl = mean_bl.reshape((num_sample, 1, 2))
    
    full_prior_tensor = np.concatenate((mu, subsample_prop, num_tips, tmrca, mean_bl), axis = 1)
    
    return full_treeLocation_tensor, full_prior_tensor

def create_train_val_test_tensors(full_tensor, num_validation, num_test):
    # training tensors
    train_tensor = full_tensor[num_test + num_validation:,:,:]

    # validation tensors
    validation_tensor = full_tensor[num_test:num_test + num_validation,:,:]

    # testing tensors
    test_tensor = full_tensor[:num_test,:,:]

    return train_tensor, validation_tensor, test_tensor



#######################
# PLotting functions ##
#######################

def plot_root_pred_examples(labels, preds, phylo_post, tip_loc_distro, num_plots = 10, num_locs = 5):
    cats = np.arange(num_locs)
    barwidth = 0.2
    randidx = np.random.permutation(labels.shape[0]-1)[0:num_plots]
    for i in randidx:
        print(i)
        plt.figure(figsize=(num_locs+2, 1))
        plt.bar(cats + barwidth, preds[i,:], barwidth, label = "prediction")
        plt.bar(cats, labels[i,:5], barwidth, label = "truth")
        plt.bar(cats + 2 * barwidth, phylo_post[i,:], barwidth, label = "phylo", color = "red")
        plt.bar(cats + 3 * barwidth, tip_loc_distro[i,:], barwidth, label = 'tip frequency')
        plt.show()
    plt.close()
    

def root_summary_plots(cnn_root_accuracy, phylo_root_accuracy, accuracy_tipfreq):

    plt.hist(cnn_root_accuracy, bins = 20, range = [0,1], color = 'blue')
    plt.xlabel('CNN and Phylo accuracy')
    plt.hist(phylo_root_accuracy, bins = 20, range = [0,1], alpha = 0.5, color = 'red')
    plt.legend(['CNN', 'Phylo'])
    plt.show()

    plt.hist(phylo_root_accuracy - cnn_root_accuracy, bins = 20)
    plt.axline((0,0), slope = 100000, color = 'red', alpha=0.75)
    plt.xlabel("phylo_accuracy - cnn accuracy")
    plt.show()

    plt.hist(accuracy_tipfreq, bins = 20, range = [0,1])
    plt.xlabel('Tip frequency accuracy')
    plt.show()

    plt.scatter(cnn_root_accuracy, phylo_root_accuracy)
    plt.xlabel("CNN accuracy")
    plt.ylabel("phylo accuracy")
    plt.axline((np.min(cnn_root_accuracy),np.min(phylo_root_accuracy)), slope = 1, color = 'red', alpha=0.75)
    plt.show()

    
def plot_preds_labels(preds, labels, param_names = ["R0", "sample rate", "migration rate"], axis_labels = ["prediction", "truth"]):
    for i in range(0, len(param_names)):
        plt.scatter(preds[:,i], labels[:,i], alpha =0.25)
        plt.xlabel(param_names[i] + " " +  axis_labels[0])
        plt.ylabel(param_names[i] + " " +  axis_labels[1])
        plt.axline((np.min(labels[:,i]),np.min(labels[:,i])), slope = 1, color = 'red', alpha=0.75)
        plt.show()
        
def plot_overlaid_scatter(sample_1, sample_2, reference_sample, 
                          sample_names = ['CNN', 'phylo'],
                          param_names = ["R0", "sample rate", "migration rate"], 
                          axis_labels = ["estimate", "truth"]):
    dot_colors = ['blue', 'red']
    for i in range(0, sample_1.shape[1]):
        minimum = np.min([sample_1[:,i], reference_sample[:,i]])
        plt.scatter(sample_1[:,i], reference_sample[:,i], alpha =0.75, color = dot_colors[0])
        plt.scatter(sample_2[:,i], reference_sample[:,i], alpha =0.75, color = dot_colors[1])
        plt.xlabel(param_names[i] + " " + axis_labels[0])
        plt.ylabel(param_names[i] + " " + axis_labels[1])
        plt.legend(sample_names)
        plt.axline((minimum, minimum), slope = 1, color = 'red', alpha=0.75)
        plt.show()
        

def make_history_plot(history):
    epochs =range(1, len(history.history['loss']) + 1)
    num_metrics = int(len(history.history.keys()) / 2)
    key_length = len(history.history.keys())
    val_keys = list(history.history.keys())[0:num_metrics]
    train_keys = list(history.history.keys())[num_metrics:key_length]
    for i in range(0,num_metrics):
        plt.plot(epochs, history.history[train_keys[i]], 'bo', label = i)
        plt.plot(epochs, history.history[val_keys[i]], 'b', label = 'Validation mae')
        plt.title('Training and val ' + train_keys[i])
        plt.xlabel('Epochs')
        plt.ylabel(train_keys[i])
        plt.legend()
        plt.show()     
    

def plot_convlayer_weights(model, layer_num):
    layer_num = layer_num
    print(model.layers[layer_num].get_config())
    print(model.layers[layer_num].get_weights()[0].shape)
    layer_biases = model.layers[layer_num].get_weights()[1]
    layer_weights = model.layers[layer_num].get_weights()[0]
    for j in range(0, layer_weights.shape[2]):
        filter_num = j
        print(filter_num)
        for k in range(0,layer_weights.shape[1]):    
            plt.hlines(0,0,layer_weights.shape[0]-1, linestyle='dashed', color = "black")
            plt.plot(layer_weights[:,k,filter_num], color=np.random.rand(3,))
            plt.vlines(0,-0.5,0.5, color = "white")
        plt.show()
    
    
    
    
def plot_denselayer_weights(model, layer_num):
    layer_num = layer_num
    print(model.layers[layer_num].get_config())
    print(model.layers[layer_num].get_weights()[0].shape)
    layer_biases = model.layers[layer_num].get_weights()[1]
    layer_weights = model.layers[layer_num].get_weights()[0]
    for j in range(0, layer_weights.shape[1]):
        filter_num = j
        print(filter_num)
        plt.hlines(0,0,layer_weights.shape[0]-1, linestyle='dashed', color = "black")
        plt.plot(layer_weights[:,filter_num], color=np.random.rand(3,))
        plt.vlines(0,-0.5,0.5, color = "white")
        # set to true for first dense layer after concatenation
        if(False):
            plt.vlines([w_global_avg.shape[1], 
                        w_global_avg.shape[1] + w_dilated_global_avg.shape[1]],-0.5,0.5)
        plt.show()



def qq_plot(sample_1, sample_2, num_quantiles=100, axlabels=['sample 1', 'sample 2']):
    plt.scatter(np.quantile(sample_1, np.arange(0,1,1/num_quantiles)), 
           np.quantile(sample_2, np.arange(0,1,1/num_quantiles)))
    plt.axline((np.mean(sample_1), np.mean(sample_2)), slope = 1, color = "red")
    plt.xlabel(axlabels[0])
    plt.ylabel(axlabels[1])
    plt.show()

    

def make_experiment_density_plots(ref_pred_ape, ref_phylo_ape, 
                         misspec_pred_ape,  misspec_phylo_ape, 
                         baseline_ape, 
                            xlabel = ["R0", "sample rate", "migration rate"],
                           plot_legend = ['random', 'CNN', 'CNN misspec', 'Phylo', 'Phylo misspec']):
    # ref for cnn and phylo, then misspecified for cnn and phylo
    
    colors = ['g', 'b', 'b', 'r', 'r']
    line_styles = [':','-','--', '-','--']
    
    # make density plots
    for i in range(0, ref_pred_ape.shape[1]):
        xlim_low = np.min(np.concatenate([np.log(baseline_ape[:,i]),
          np.log(ref_pred_ape[:,i]),
          np.log(misspec_pred_ape[:,i]), 
          np.log(ref_phylo_ape[:,i]),
          np.log(misspec_phylo_ape[:,i])]))
        xlim_high = np.max(np.concatenate([np.log(baseline_ape[:,i]),
          np.log(ref_pred_ape[:,i]),
          np.log(misspec_pred_ape[:,i]), 
          np.log(ref_phylo_ape[:,i]),
          np.log(misspec_phylo_ape[:,i])]))
        df = pd.DataFrame([np.log(baseline_ape[:,i]),
          np.log(ref_pred_ape[:,i]),
          np.log(misspec_pred_ape[:,i]), 
          np.log(ref_phylo_ape[:,i]),
          np.log(misspec_phylo_ape[:,i])])
        df.transpose().plot(kind = 'density',
                           style = line_styles,
                           color = colors,
                           xlim = [xlim_low-1, xlim_high+1])
        plt.xlabel(xlabel[i] + " log abs. % error ")
        plt.legend(plot_legend)
        plt.show()

        # make boxplots
        box = plt.boxplot([ref_pred_ape[:,i],
          misspec_pred_ape[:,i], 
          ref_phylo_ape[:,i],
          misspec_phylo_ape[:,i]],
                   labels = ['CNN true', 'CNN misspec', 
                            'phylo true', 'phylo misspec'], 
                          showfliers = False, widths = 0.9, patch_artist = True)
        plt.axline((0.5,0), slope = 0, color = "red")
        plt.ylabel('percent error (APE)')
        plt.title(xlabel[i])
        for box, color in zip(box['boxes'], colors[1:]):
            box.set_edgecolor(color)
            box.set_facecolor('w')
        plt.show()
        
        # make histograms        
        plt.hist((misspec_pred_ape[:,i]) - (misspec_phylo_ape[:,i]), bins = 20)
        plt.axline((0,0), slope = 1000000, color = "red")
        plt.xlabel('cnn APE - phylo APE')
        plt.title(xlabel[i])
        plt.show()

        # print summary stats


def load_input( data_fn, label_fn ):
    data = pd.read_csv(data_fn, header=None, on_bad_lines='skip').to_numpy()
    labels = pd.read_csv(label_fn, header=None, on_bad_lines='skip').to_numpy()
    return data,labels
