#!/usr/bin/env python
"""
network
========
Defines classes for neural networks, loss functions, and datasets using PyTorch. 

Authors:   Michael Landis and Ammon Thompson
Copyright: (c) 2022-2023, Michael Landis and Ammon Thompson
License:   MIT
"""

# standard imports
#   none

# external imports
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

# phyddle imports
from phyddle import utilities as util

#-------------------------#

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

#-------------------------#
    
class ParameterEstimationNetwork(nn.Module):
    """
    Parameter estimation neural network. This class defines the network
    structure, activation functions, and forward pass behavior for of input
    to predict labels.
    """
    def __init__(self, phy_dat_width, phy_dat_height, aux_dat_width, lbl_width, args):    
        
        # initialize base class
        super(ParameterEstimationNetwork, self).__init__()

        # set basic arguments
        self.set_args(args)

        # input/output
        self.phy_dat_width  = phy_dat_width
        self.phy_dat_height = phy_dat_height
        self.aux_dat_width  = aux_dat_width
        self.lbl_width      = lbl_width

        # compute layer settigns
        self.make_layer_settings()

        # build network
        self.build_network()

        return

    def set_args(self, args):
        """Assigns phyddle settings as Network attributes.

        Args:
            args (dict): Contains phyddle settings.

        """
        self.args = args
        step_args = util.make_step_args('T', args)
        for k,v in step_args.items():
            setattr(self, k, v)

        return

    def make_layer_settings(self):
        """
        Assigns neural network layer settings.
        """

        # standard convolution and pooling layers for CPV+S
        self.phy_std_out_size       = self.phy_channel_plain # [64, 96, 128]
        self.phy_std_kernel_size    = self.phy_kernel_plain # [3, 5, 7]
        self.phy_std_in_size        = [ self.phy_dat_width ] + self.phy_std_out_size[:-1]
        assert(len(self.phy_std_out_size) == len(self.phy_std_kernel_size))

        # stride convolution and pooling layers for CPV+S
        self.phy_stride_out_size    = self.phy_channel_stride # [64, 96]
        self.phy_stride_kernel_size = self.phy_kernel_stride # [7, 9]
        self.phy_stride_stride_size = self.phy_stride_stride # [3, 6]
        self.phy_stride_in_size     = [ self.phy_dat_width ] + self.phy_stride_out_size[:-1]
        # for i in range(len(self.phy_stride_out_size)):
        #     phy_stride_output_size = int(np.ceil(phy_stride_output_size / phy_stride_size[i]))
        assert(len(self.phy_stride_out_size) == len(self.phy_stride_kernel_size))
        assert(len(self.phy_stride_out_size) == len(self.phy_stride_stride_size))
        
        # dilate convolution and pooling layers for CPV+S
        self.phy_dilate_out_size    = self.phy_channel_dilate # [32, 64]
        self.phy_dilate_kernel_size = self.phy_kernel_dilate # [3, 5]
        self.phy_dilate_dilate_size = self.phy_dilate_dilate # [2, 4]
        self.phy_dilate_in_size     = [ self.phy_dat_width ] + self.phy_dilate_out_size[:-1]
        assert(len(self.phy_dilate_out_size) == len(self.phy_dilate_kernel_size))
        assert(len(self.phy_dilate_out_size) == len(self.phy_dilate_dilate_size))

        # dense feed-forward layers for aux. data
        self.aux_out_size           = self.aux_channel # [128, 64, 32]
        self.aux_in_size            = [ self.aux_dat_width ] + self.aux_out_size[:-1]

        # concat layer size??
        self.concat_size = self.phy_std_out_size[-1] + \
                           self.phy_stride_out_size[-1] + \
                           self.phy_dilate_out_size[-1] + \
                           self.aux_out_size[-1]

        # dense layers for output predictions
        # self.label_out_size         = [128, 64, 32] + [self.lbl_width]
        self.label_out_size         = self.lbl_channel + [ self.lbl_width ]
        self.label_in_size          = [ self.concat_size ] + self.label_out_size

        return

    def build_network(self):

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
        
        #concat_size = phy_std_channels[-1] + phy_stride_channels[-1] + phy_dilate_channels[-1] + aux_channels[-1]
        #print(phy_std_channels[-1], phy_stride_channels[-1], phy_dilate_channels[-1], aux_channels[-1])
        #print('concat size', concat_size)

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

        #self.layers = self._get_layers()
        self._initialize_weights()

        # done
        return


    def _initialize_weights(self):
        '''Initializes weights for network.'''
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity='relu')
                torch.nn.init.constant_(m.bias, 0)
            if isinstance(m, torch.nn.Conv1d):
                torch.nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity='relu')
                torch.nn.init.constant_(m.bias, 0)
        return

    def forward(self, phy_dat, aux_dat):
        '''Forward-pass function of input through network to output labels.'''
        
        # Phylogenetic Tensor forwarding
        phy_dat = phy_dat.float()
        aux_dat = aux_dat.float()

        # MJL: Does this need to be set? Seems like no.
        # phy_dat.requires_grad = True
        # aux_dat.requires_grad = True

        # standard conv + pool layers
        x_std = phy_dat
        for i in range(len(self.phy_std)-1):
            x_std = F.relu(self.phy_std[i](x_std))
        x_std = self.phy_std[-1](x_std)
        
        # stride conv + pool layers
        x_stride = phy_dat
        for i in range(len(self.phy_stride)-1 ):
            x_stride = F.relu(self.phy_stride[i](x_stride))
        x_stride = self.phy_stride[-1](x_stride)
        
        # dilation conv + pool layers
        x_dilate = phy_dat
        for i in range(len(self.phy_dilate)-1):
            x_dilate = F.relu(self.phy_dilate[i](x_dilate))
        x_dilate = self.phy_dilate[-1](x_dilate)
        
        # dense aux. dat layers
        x_aux = aux_dat
        for i in range(len(self.aux_ffnn)):
            x_aux = F.relu(self.aux_ffnn[i](x_aux))
        x_aux = x_aux.unsqueeze(dim=2)

        # Concatenate phylo and aux layers
        x_cat = torch.cat((x_std, x_stride, x_dilate, x_aux), dim=1).squeeze()
        
        # Point estimate path
        x_point = x_cat
        for i in range(len(self.point_ffnn)-1):
            x_point = F.relu(self.point_ffnn[i](x_point))
        x_point = self.point_ffnn[-1](x_point)

        # Lower quantile path
        x_lower = x_cat
        for i in range(len(self.lower_ffnn)-1):
            x_lower = F.relu(self.lower_ffnn[i](x_lower))
        x_lower = self.lower_ffnn[-1](x_lower)

        # Upper quantile path
        x_upper = x_cat
        for i in range(len(self.upper_ffnn)-1):
            x_upper = F.relu(self.upper_ffnn[i](x_upper))
        x_upper = self.upper_ffnn[-1](x_upper)
        
        # return loss
        return (x_point, x_lower, x_upper)
    

class QuantileLoss(nn.Module):
    """
    Quantile loss function. This function uses an asymmetric quantile
    (or "pinball") loss function.
    
    https://medium.com/the-artificial-impostor/quantile-regression-part-2-6fdbc26b2629
    """
    def __init__(self, alpha):
        '''Defines quantile loss function to predict interval that captures
        alpha-% of output predictions.'''
        super(QuantileLoss, self).__init__()
        self.alpha = alpha
        return

    def forward(self, predictions, targets):
        '''Simple quantile loss function for prediction intervals.'''
        err = targets - predictions
        return torch.mean(torch.max(self.alpha*err, (self.alpha-1)*err))


# ------------#
