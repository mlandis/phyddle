#!/usr/bin/env python
"""
network
========
Defines classes for neural networks, loss functions, and datasets using
PyTorch.

Authors:   Michael Landis and Ammon Thompson
Copyright: (c) 2022-2023, Michael Landis and Ammon Thompson
License:   MIT
"""

# standard imports
#   none

# external imports
import numpy as np
import torch
import torch.nn.functional as func
from torch import nn

# phyddle imports
# from phyddle import utilities as util

##################################################


class Dataset(torch.utils.data.Dataset):
    """
    Dataset class for training. It is used by torch.utils.data.DataLoader to
    generate training batches for the training loop. Training examples include
    phylogenetic-state tensors, auxiliary data tensors, and labels.
    """
    # Constructor
    def __init__(self, phy_data, aux_data, labels_real, labels_cat):
        self.phy_data    = np.transpose(phy_data, axes=[0,2,1]).astype('float32')
        self.aux_data    = aux_data.astype('float32')
        self.labels_real = labels_real.astype('float32')
        self.labels_cat  = labels_cat.astype('int')
        self.len         = self.labels_real.shape[0]

    # Getting the dataq
    def __getitem__(self, index):
        return (self.phy_data[index], self.aux_data[index],
                self.labels_real[index], self.labels_cat[index])
    
    # Getting length of the data
    def __len__(self):
        return self.len

##################################################


class ParameterEstimationNetwork(nn.Module):
    """
    Parameter estimation neural network. This class defines the network
    structure, activation functions, and forward pass behavior for of input
    to predict labels.
    
    Args:
            args (dict): Contains phyddle settings.
    """
    def __init__(self, phy_dat_width, phy_dat_height, aux_dat_width,
                 lbl_width, param_cat, args):
        
        # initialize base class
        super(ParameterEstimationNetwork, self).__init__()

        # width for key input/output
        self.phy_dat_width  = phy_dat_width
        self.phy_dat_height = phy_dat_height
        self.aux_dat_width  = aux_dat_width
        self.lbl_width      = lbl_width
        self.param_cat      = param_cat
        self.param_cat_size = dict()
        
        self.has_param_real = self.lbl_width > 0
        self.has_param_cat  = len(self.param_cat) > 0

        # collect args
        self.phy_std_out_size       = list(args['phy_channel_plain'])
        self.phy_std_kernel_size    = list(args['phy_kernel_plain'])
        self.phy_stride_out_size    = list(args['phy_channel_stride'])
        self.phy_stride_kernel_size = list(args['phy_kernel_stride'])
        self.phy_stride_stride_size = list(args['phy_stride_stride'])
        self.phy_dilate_out_size    = list(args['phy_channel_dilate'])
        self.phy_dilate_kernel_size = list(args['phy_kernel_dilate'])
        self.phy_dilate_dilate_size = list(args['phy_dilate_dilate'])
        self.aux_out_size           = list(args['aux_channel'])
        self.lbl_channel            = list(args['lbl_channel'])

        # standard convolution and pooling layers for CPV+S
        self.phy_std_in_size = [ self.phy_dat_width ] + self.phy_std_out_size[:-1]
        assert len(self.phy_std_out_size) == len(self.phy_std_kernel_size)

        # stride convolution and pooling layers for CPV+S
        self.phy_stride_in_size = [ self.phy_dat_width ] + self.phy_stride_out_size[:-1]
        assert len(self.phy_stride_out_size) == len(self.phy_stride_kernel_size)
        assert len(self.phy_stride_out_size) == len(self.phy_stride_stride_size)
        
        # dilate convolution and pooling layers for CPV+S
        self.phy_dilate_in_size = [ self.phy_dat_width ] + self.phy_dilate_out_size[:-1]
        assert len(self.phy_dilate_out_size) == len(self.phy_dilate_kernel_size)
        assert len(self.phy_dilate_out_size) == len(self.phy_dilate_dilate_size)
        
        # dense feed-forward layers for aux. data
        self.aux_in_size = [ self.aux_dat_width ] + self.aux_out_size[:-1]
        
        # concatenation layer size (used in build_network
        self.concat_size = self.phy_std_out_size[-1] + \
                           self.phy_stride_out_size[-1] + \
                           self.phy_dilate_out_size[-1] + \
                           self.aux_out_size[-1]
        
        # dense layers for output predictions
        self.label_real_out_size = self.lbl_channel + [ self.lbl_width ]
        self.label_in_size = [ self.concat_size ] + self.label_real_out_size[:-1]
        self.label_cat_out_size = dict()
        for k,v in self.param_cat.items():
            self.label_cat_out_size[k] = self.lbl_channel + [ int(v) ]
        
        # build network
        # standard convolution and pooling layers for CPV+S
        self.phy_std = nn.ModuleList([])
        for i in range(len(self.phy_std_out_size)):
            c_in  = self.phy_std_in_size[i]
            c_out = self.phy_std_out_size[i]
            k     = self.phy_std_kernel_size[i]
            self.phy_std.append(nn.Conv1d(in_channels=c_in,
                                          out_channels=c_out,
                                          kernel_size=k,
                                          padding='same'))
        self.phy_std.append(nn.AdaptiveAvgPool1d(1))

        # stride convolution and pooling layers for CPV+S
        self.phy_stride = nn.ModuleList([])
        for i in range(len(self.phy_stride_out_size)):
            c_in  = self.phy_stride_in_size[i]
            c_out = self.phy_stride_out_size[i]
            k     = self.phy_stride_kernel_size[i]
            s     = self.phy_stride_stride_size[i]
            self.phy_stride.append(nn.Conv1d(in_channels=c_in,
                                             out_channels=c_out,
                                             kernel_size=k,
                                             stride=s))
        self.phy_stride.append(nn.AdaptiveAvgPool1d(1))

        # dilate convolution and pooling layers for CPV+S
        self.phy_dilate = nn.ModuleList([])
        for i in range(len(self.phy_dilate_out_size)):
            c_in  = self.phy_dilate_in_size[i]
            c_out = self.phy_dilate_out_size[i]
            k     = self.phy_dilate_kernel_size[i]
            d     = self.phy_dilate_dilate_size[i]
            self.phy_dilate.append(nn.Conv1d(in_channels=c_in,
                                             out_channels=c_out,
                                             kernel_size=k,
                                             dilation=d, padding='same'))
        self.phy_dilate.append(nn.AdaptiveAvgPool1d(1))

        # dense feed-forward layers for aux. data
        self.aux_ffnn = nn.ModuleList([])
        for i in range(len(self.aux_out_size)):
            c_in  = self.aux_in_size[i]
            c_out = self.aux_out_size[i]
            self.aux_ffnn.append(nn.Linear(c_in, c_out))

        if self.has_param_real:
            # dense layers for point/bound estimates
            self.point_ffnn = nn.ModuleList([])
            self.lower_ffnn = nn.ModuleList([])
            self.upper_ffnn = nn.ModuleList([])
            for i in range(len(self.label_real_out_size)):
                c_in  = self.label_in_size[i]
                c_out = self.label_real_out_size[i]
                self.point_ffnn.append(nn.Linear(c_in, c_out))
                self.lower_ffnn.append(nn.Linear(c_in, c_out))
                self.upper_ffnn.append(nn.Linear(c_in, c_out))

        if self.has_param_cat:
            # dense layers for categorical predictions
            self.categ_ffnn = dict()
            for k,v in self.param_cat.items():
                k_str = f'{k}_categ_ffnn'
                k_mod_list = nn.ModuleList([])
                for i in range(len(self.label_real_out_size)):
                    c_in  = self.label_in_size[i]
                    c_out = self.label_cat_out_size[k][i]
                    k_mod_list.append(nn.Linear(c_in, c_out))
                setattr(self, k_str, k_mod_list)
                
        # initialize weights for layers
        self._initialize_weights()

        return

    def _initialize_weights(self):
        """Initializes weights for network."""
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity='relu')
                torch.nn.init.constant_(m.bias, 0)
            if isinstance(m, torch.nn.Conv1d):
                torch.nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity='relu')
                torch.nn.init.constant_(m.bias, 0)
        return

    def forward(self, phy_dat, aux_dat):
        """Forward-pass function of input through network to output labels."""
        
        # Phylogenetic Tensor forwarding
        num_sample = phy_dat.shape[0]

        # MJL: Does this need to be set? Seems like no.
        # phy_dat.requires_grad = True
        # aux_dat.requires_grad = True

        # standard conv + pool layers
        x_std = phy_dat
        for i in range(len(self.phy_std)-1):
            x_std = func.relu(self.phy_std[i](x_std))
        x_std = self.phy_std[-1](x_std)
        
        # stride conv + pool layers
        x_stride = phy_dat
        for i in range(len(self.phy_stride)-1 ):
            x_stride = func.relu(self.phy_stride[i](x_stride))
        x_stride = self.phy_stride[-1](x_stride)
        
        # dilation conv + pool layers
        x_dilate = phy_dat
        for i in range(len(self.phy_dilate)-1):
            x_dilate = func.relu(self.phy_dilate[i](x_dilate))
        x_dilate = self.phy_dilate[-1](x_dilate)
        
        # dense aux. dat layers
        x_aux = aux_dat
        for i in range(len(self.aux_ffnn)):
            x_aux = func.relu(self.aux_ffnn[i](x_aux))
        x_aux = x_aux.unsqueeze(dim=2)

        # Concatenate phylo and aux layers
        x_concat = torch.cat((x_std, x_stride, x_dilate, x_aux), dim=1).squeeze()
        
        if self.has_param_real:
            # Point estimate path
            x_point = x_concat
            for i in range(len(self.point_ffnn)-1):
                x_point = func.relu(self.point_ffnn[i](x_point))
            x_point = self.point_ffnn[-1](x_point)
    
            # Lower quantile path
            x_lower = x_concat
            for i in range(len(self.lower_ffnn)-1):
                x_lower = func.relu(self.lower_ffnn[i](x_lower))
            x_lower = self.lower_ffnn[-1](x_lower)
    
            # Upper quantile path
            x_upper = x_concat
            for i in range(len(self.upper_ffnn)-1):
                x_upper = func.relu(self.upper_ffnn[i](x_upper))
            x_upper = self.upper_ffnn[-1](x_upper)
        else:
            x_point = torch.empty((num_sample,0))
            x_lower = torch.empty((num_sample,0))
            x_upper = torch.empty((num_sample,0))

        x_categ = dict()
        if self.has_param_cat:
            # Categorical paths
            for k,v in self.param_cat.items():
                # take initial input from x_concat
                if k not in x_categ:
                    x_categ[k] = x_concat
                k_str = f'{k}_categ_ffnn'
                k_mod_list = getattr(self, k_str)
                for i in range(len(k_mod_list)-1):
                    x_categ[k] = func.relu(k_mod_list[i](x_categ[k]))
                # x_categ[k] = func.softmax(k_mod_list[-1](x_categ[k]), dim=1)
                x_categ[k] = k_mod_list[-1](x_categ[k])
                setattr(self, k_str, k_mod_list)
        
        # return loss
        return x_point, x_lower, x_upper, x_categ

##################################################


class QuantileLoss(nn.Module):
    """
    Quantile loss function. This function uses an asymmetric quantile
    (or "pinball") loss function.
    
    https://medium.com/the-artificial-impostor/quantile-regression-part-2-6fdbc26b2629
    """
    def __init__(self, alpha):
        """Defines quantile loss function to predict interval that captures
        alpha-% of output predictions."""
        super(QuantileLoss, self).__init__()
        self.alpha = alpha
        return

    def forward(self, predictions, targets):
        """Simple quantile loss function for prediction intervals."""
        err = targets - predictions
        return torch.mean(torch.max(self.alpha*err, (self.alpha-1)*err))


##################################################

class CrossEntropyLoss(nn.Module):
    """
    Quantile loss function. This function uses an asymmetric quantile
    (or "pinball") loss function.
    
    https://medium.com/the-artificial-impostor/quantile-regression-part-2-6fdbc26b2629
    """
    def __init__(self):
        """Defines quantile loss function to predict interval that captures
        alpha-% of output predictions."""
        super(CrossEntropyLoss, self).__init__()
        return

    def forward(self, predictions, targets):
        """Simple quantile loss function for prediction intervals."""
        
        loss_list = []
        loss_func = torch.nn.CrossEntropyLoss()
        
        # assumes that order of entries in predictions
        # matches order of entries in targets; could be unsafe
        for i,(k,v) in enumerate(predictions.items()):
            loss_list.append(loss_func(v, targets[:,i]))
            
        return torch.sum(torch.stack(loss_list))

