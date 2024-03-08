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
    def __init__(self, phy_data, aux_data, labels):
        self.phy_data = np.transpose(phy_data, axes=[0,2,1]).astype('float32')
        self.aux_data = aux_data.astype('float32')
        self.labels   = labels.astype('float32')
        self.len = self.labels.shape[0]

    # Getting the dataq
    def __getitem__(self, index):
        return self.phy_data[index], self.aux_data[index], self.labels[index]
    
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
                 lbl_width, args):
        
        # initialize base class
        super(ParameterEstimationNetwork, self).__init__()

        # width for key input/output
        self.phy_dat_width  = phy_dat_width
        self.phy_dat_height = phy_dat_height
        self.aux_dat_width  = aux_dat_width
        self.lbl_width      = lbl_width

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
        self.label_out_size = self.lbl_channel + [ self.lbl_width ]
        self.label_in_size = [ self.concat_size ] + self.label_out_size
        
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

        # dense layers for point estimates
        self.point_ffnn = nn.ModuleList([])
        self.lower_ffnn = nn.ModuleList([])
        self.upper_ffnn = nn.ModuleList([])
        for i in range(len(self.label_out_size)):
            c_in  = self.label_in_size[i]
            c_out = self.label_out_size[i]
            self.point_ffnn.append(nn.Linear(c_in, c_out))
            self.lower_ffnn.append(nn.Linear(c_in, c_out))
            self.upper_ffnn.append(nn.Linear(c_in, c_out))

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
        phy_dat = phy_dat.float()
        aux_dat = aux_dat.float()

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
        x_cat = torch.cat((x_std, x_stride, x_dilate, x_aux), dim=1).squeeze()
        
        # Point estimate path
        x_point = x_cat
        for i in range(len(self.point_ffnn)-1):
            x_point = func.relu(self.point_ffnn[i](x_point))
        x_point = self.point_ffnn[-1](x_point)

        # Lower quantile path
        x_lower = x_cat
        for i in range(len(self.lower_ffnn)-1):
            x_lower = func.relu(self.lower_ffnn[i](x_lower))
        x_lower = self.lower_ffnn[-1](x_lower)

        # Upper quantile path
        x_upper = x_cat
        for i in range(len(self.upper_ffnn)-1):
            x_upper = func.relu(self.upper_ffnn[i](x_upper))
        x_upper = self.upper_ffnn[-1](x_upper)
        
        # return loss
        return x_point, x_lower, x_upper
    

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
