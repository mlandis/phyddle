#!/usr/bin/env python
"""
train
=====
Defines classes and methods for the Training step, which builds and trains a
network using the tensor data from the Formatting step.

Authors:   Michael Landis and Ammon Thompson
Copyright: (c) 2022-2023, Michael Landis and Ammon Thompson
License:   MIT
"""

# standard imports
# import json
import os

# external imports
import h5py
import numpy as np
import pandas as pd
# from tensorflow.keras import Input
# from tensorflow.keras import Model
# from tensorflow.keras import layers
# from tensorflow.keras import backend as K
# from tensorflow.keras import callbacks
from multiprocessing import cpu_count
# import copy
from tqdm import tqdm


# phyddle imports
from phyddle import utilities as util

# torch
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
#from neuralforecast.losses.pytorch import QuantileLoss
import torch.nn.functional as F


#-------------------------#




class PhyddleDataset(Dataset):    
    # Constructor
    def __init__(self, phy_data, aux_data, labels): #, phy_dat_shape):
        
        # self.labels     = pd.read_csv(labels_fn, sep=',').to_numpy(dtype='float32')
        # self.phy_data   = pd.read_csv(phy_dat_fn, sep=',', header=None).to_numpy(dtype='float32')
        # self.aux_data   = pd.read_csv(aux_dat_fn, sep=',').to_numpy(dtype='float32')
        #self.labels = 
        # print('PhyddleDataset')
        self.phy_data = np.transpose(phy_data, axes=[0,2,1]).astype('float32')
        self.aux_data = aux_data.astype('float32')
        self.labels   = labels.astype('float32')
        # print(type(phy_data))
        # print(phy_data.shape)
        # #print(np.transpose(phy_data, [0,2,1]))
        # print(type(aux_data))
        # print(aux_data.shape)

        # self.phy_data = self.phy_data
        # self.aux_data = 'float32'
        # self.labels.dtype   = 'float32'

        self.len = self.labels.shape[0]
        #self.phy_data.shape = (self.len, phy_dat_shape[0], phy_dat_shape[1])
    

    # Getting the dataq
    def __getitem__(self, index):    
        return self.phy_data[index], self.aux_data[index], self.labels[index]
    
    # Getting length of the data
    def __len__(self):
        return self.len




class NeuralNetwork(nn.Module):
    def __init__(self, phy_dat_width, phy_dat_height, aux_dat_width, lbl_width):    
        # initialize base class
        super(NeuralNetwork, self).__init__()

        # print('phy_dat_width = ', phy_dat_width)
        # print('phy_dat_height = ', phy_dat_height)
        # print('aux_dat_width = ', aux_dat_width)
        # print('lbl_width = ', lbl_width)

        # Standard convolution layers
        phy_std_channels = [64, 96, 128]
        self.phy_conv_std1 = nn.Conv1d(in_channels=phy_dat_width,       out_channels=phy_std_channels[0], kernel_size=3, padding='same')
        self.phy_drop_std1 = nn.Dropout(0.1, inplace=False)
        self.phy_conv_std2 = nn.Conv1d(in_channels=phy_std_channels[0], out_channels=phy_std_channels[1], kernel_size=5, padding='same')
        self.phy_drop_std2 = nn.Dropout(0.3, inplace=False)
        self.phy_conv_std3 = nn.Conv1d(in_channels=phy_std_channels[1], out_channels=phy_std_channels[2], kernel_size=7, padding='same')
        self.phy_drop_std3 = nn.Dropout(0.5, inplace=False)
        #self.phy_pool_std = nn.AvgPool1d(kernel_size=phy_dat_height)
        self.phy_pool_std = nn.AdaptiveAvgPool1d(1)
        
        # Stride convolution layers
        phy_stride_channels    = [64, 96]
        phy_stride_size        = [3, 6]
        phy_stride_kernel_size = [7, 9]
        phy_stride_pad_size    = [3, 4]
        phy_stride_output_size = phy_dat_height
        for i in range(len(phy_stride_channels)):
            #print( phy_stride_output_size / phy_stride_size[i] )
            phy_stride_output_size = int(np.ceil(phy_stride_output_size / phy_stride_size[i]))

        #print('phy stride output', phy_stride_output_size)

        self.phy_conv_stride1 = nn.Conv1d(in_channels=phy_dat_width,          out_channels=phy_stride_channels[0], kernel_size=phy_stride_kernel_size[0], stride=phy_stride_size[0])#, padding=phy_stride_pad_size[0])
        self.phy_drop_stride1 = nn.Dropout(0.1, inplace=False)
        self.phy_conv_stride2 = nn.Conv1d(in_channels=phy_stride_channels[0], out_channels=phy_stride_channels[1], kernel_size=phy_stride_kernel_size[1], stride=phy_stride_size[1])#, padding=phy_stride_pad_size[1]) #, padding='same')
        self.phy_drop_stride2 = nn.Dropout(0.5, inplace=False)
        #print('phy stride weight size',self.phy_conv_stride2.weight.size())
        #self.phy_pool_stride  = nn.AvgPool1d(kernel_size=phy_stride_output_size)
        self.phy_pool_stride = nn.AdaptiveAvgPool1d(1)
        
        # Dilated convolution layers
        phy_dilate_channels = [32, 64]
        self.phy_conv_dilate1 = nn.Conv1d(in_channels=phy_dat_width,          out_channels=phy_dilate_channels[0], kernel_size=3, dilation=2, padding='same')
        self.phy_drop_dilate1 = nn.Dropout(0.1, inplace=False)
        self.phy_conv_dilate2 = nn.Conv1d(in_channels=phy_dilate_channels[0], out_channels=phy_dilate_channels[1], kernel_size=5, dilation=4, padding='same')
        self.phy_drop_dilate2 = nn.Dropout(0.5, inplace=False)
        #self.phy_pool_dilate  = nn.AvgPool1d(kernel_size=phy_dat_height)
        self.phy_pool_dilate = nn.AdaptiveAvgPool1d(1)

        # Auxiliary Data layers
        aux_channels = [128, 64, 32]
        #aux_channels = [64, 48, 32]
        self.aux_ffnn1 = nn.Linear(aux_dat_width,   aux_channels[0])
        self.aux_dropout1 = torch.nn.Dropout(p=0.5, inplace=False)
        self.aux_ffnn2 = nn.Linear(aux_channels[0], aux_channels[1])
        self.aux_dropout2 = torch.nn.Dropout(p=0.5, inplace=False)
        self.aux_ffnn3 = nn.Linear(aux_channels[1], aux_channels[2])
        self.aux_dropout3 = torch.nn.Dropout(p=0.5, inplace=False)

        concat_size = phy_std_channels[-1] + phy_stride_channels[-1] + phy_dilate_channels[-1] + aux_channels[-1]
        #print(phy_std_channels[-1], phy_stride_channels[-1], phy_dilate_channels[-1], aux_channels[-1])
        #print('concat size', concat_size)

        # Label Value layers
        lbl_channels = [128, 64, 32]
        #lbl_channels = [64, 48, 32]
        self.point_ffnn1 = nn.Linear(concat_size,     lbl_channels[0])
        self.point_dropout1 = torch.nn.Dropout(p=0.5, inplace=False)
        self.point_ffnn2 = nn.Linear(lbl_channels[0], lbl_channels[1])
        self.point_dropout2 = torch.nn.Dropout(p=0.5, inplace=False)
        self.point_ffnn3 = nn.Linear(lbl_channels[1], lbl_channels[2])
        self.point_dropout3 = torch.nn.Dropout(p=0.5, inplace=False)
        self.point_ffnn4 = nn.Linear(lbl_channels[2], lbl_width)

        # Label Lower layers
        self.lower_ffnn1 = nn.Linear(concat_size,     lbl_channels[0])
        self.lower_dropout1 = torch.nn.Dropout(p=0.5, inplace=False)
        self.lower_ffnn2 = nn.Linear(lbl_channels[0], lbl_channels[1])
        self.lower_dropout2 = torch.nn.Dropout(p=0.5, inplace=False)
        self.lower_ffnn3 = nn.Linear(lbl_channels[1], lbl_channels[2])
        self.lower_dropout3 = torch.nn.Dropout(p=0.5, inplace=False)
        self.lower_ffnn4 = nn.Linear(lbl_channels[2], lbl_width)
        
        # Label Upper layers
        self.upper_ffnn1 = nn.Linear(concat_size,     lbl_channels[0])
        self.upper_dropout1 = torch.nn.Dropout(p=0.5, inplace=False)
        self.upper_ffnn2 = nn.Linear(lbl_channels[0], lbl_channels[1])
        self.upper_dropout2 = torch.nn.Dropout(p=0.5, inplace=False)
        self.upper_ffnn3 = nn.Linear(lbl_channels[1], lbl_channels[2])
        self.upper_dropout3 = torch.nn.Dropout(p=0.5, inplace=False)
        self.upper_ffnn4 = nn.Linear(lbl_channels[2], lbl_width)

        #self.layers = self._get_layers()
        self._initialize_weights()

    def _initialize_weights(self):
        s = 0
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
            #     # This bit loads weights from numpy arrays
            #     # The loaded weights have identical forward passes as in TF!
            #     # m.weight = torch.nn.Parameter(torch.from_numpy(np.load(str(s) + '_trained.npy')).T)
            #     # s +=1
            #     # m.bias = torch.nn.Parameter(torch.from_numpy(np.load(str(s) + '_trained.npy')))
            #     # s +=1
                #print('init Linear')
                #torch.nn.init.xavier_uniform_(m.weight, gain=5/3)
                torch.nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity='relu')
                torch.nn.init.constant_(m.bias, 0)
                # This saves the weight init
                # np.save(str(s) + '_torch.npy', m.weight.detach().numpy())
                # s+=1
                # np.save(str(s) + '_torch.npy', m.bias.detach().numpy())
                # s+=1
            if isinstance(m, torch.nn.Conv1d):
                #print('init Conv1d')
            #     # This bit loads weights from numpy arrays
            #     # The loaded weights have identical forward passes as in TF!
            #     # m.weight = torch.nn.Parameter(torch.from_numpy(np.load(str(s) + '_trained.npy')).T)
            #     # s +=1
            #     # m.bias = torch.nn.Parameter(torch.from_numpy(np.load(str(s) + '_trained.npy')))
            #     # s +=1
                #torch.nn.init.xavier_uniform_(m.weight, gain=5/3)
                torch.nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity='relu')
                torch.nn.init.constant_(m.bias, 0)
                
    def forward(self, phy_dat, aux_dat):
        
        # Phylogenetic Tensor forwarding
        phy_dat = phy_dat.float()
        aux_dat = aux_dat.float()

        # MJL does this need to be set?
        # phy_dat.requires_grad = True
        # aux_dat.requires_grad = True

        # Standard convolutions
        x_std = F.relu(self.phy_conv_std1(phy_dat))
        #x_std = self.phy_drop_std1(x_std)
        x_std = F.relu(self.phy_conv_std2(x_std))
        #x_std = self.phy_drop_std2(x_std)
        x_std = F.relu(self.phy_conv_std3(x_std))
        #x_std = self.phy_drop_std3(x_std)
        x_std = self.phy_pool_std(x_std)
        #print('std_pool',x_std.shape)
        
        # Stride convolutions
        # for i in range(num_phy_stride):
        #     if i == 0:
        #         x_stride = F.relu(self.phy_conv_stride[i](phy_dat))
        #     else:
        #         x_stride = F.relu(self.phy_conv_stride[i](x_stride))
        x_stride = F.relu(self.phy_conv_stride1(phy_dat))
        #x_stride = self.phy_drop_stride1(x_stride)
        x_stride = F.relu(self.phy_conv_stride2(x_stride))
        #x_stride = self.phy_drop_stride2(x_stride)
        x_stride = self.phy_pool_stride(x_stride)
        #print('stride_pool',x_stride.shape)
        
        # # Dilated convolutions
        x_dilated = F.relu(self.phy_conv_dilate1(phy_dat))
        #x_dilated = self.phy_drop_dilate1(x_dilated)
        x_dilated = F.relu(self.phy_conv_dilate2(x_dilated))
        #x_dilated = self.phy_drop_dilate2(x_dilated)
        x_dilated = self.phy_pool_dilate(x_dilated)
        #print(x_dilated.shape)

        # # Auxiliary Data Tensor forwarding
        x_aux_ffnn = F.relu(self.aux_ffnn1(aux_dat))
        #x_aux_ffnn = self.aux_dropout1(x_aux_ffnn)
        x_aux_ffnn = F.relu(self.aux_ffnn2(x_aux_ffnn))
        #x_aux_ffnn = self.aux_dropout2(x_aux_ffnn)
        x_aux_ffnn = F.relu(self.aux_ffnn3(x_aux_ffnn))
        #x_aux_ffnn = self.aux_dropout3(x_aux_ffnn)
        #print(x_aux_ffnn.shape)

        # Concatenate phylo and aux layers
        x_cat = torch.cat((x_std, x_stride, x_dilated, x_aux_ffnn.unsqueeze(dim=2)), dim=1).squeeze()
        #print('cat', x_cat.shape)
        
        # Point estimate path
        x_point_est = F.relu(self.point_ffnn1(x_cat))
        #x_point_est = self.point_dropout1(x_point_est)
        x_point_est = F.relu(self.point_ffnn2(x_point_est))
        #x_point_est = self.point_dropout2(x_point_est)
        x_point_est = F.relu(self.point_ffnn3(x_point_est))
        #x_point_est = self.point_dropout3(x_point_est)
        x_point_est = self.point_ffnn4(x_point_est)

        # # Lower quantile path
        x_lower_quantile = F.relu(self.lower_ffnn1(x_cat))
        #x_lower_quantile = self.lower_dropout1(x_lower_quantile)
        x_lower_quantile = F.relu(self.lower_ffnn2(x_lower_quantile))
        #x_lower_quantile = self.lower_dropout2(x_lower_quantile)
        x_lower_quantile = F.relu(self.lower_ffnn3(x_lower_quantile))
        #x_lower_quantile = self.lower_dropout3(x_lower_quantile)
        x_lower_quantile = self.lower_ffnn4(x_lower_quantile)

        # # Upper quantile path
        x_upper_quantile = F.relu(self.upper_ffnn1(x_cat))
        #x_upper_quantile = self.lower_dropout1(x_upper_quantile)
        x_upper_quantile = F.relu(self.upper_ffnn2(x_upper_quantile))
        #x_upper_quantile = self.lower_dropout2(x_upper_quantile)
        x_upper_quantile = F.relu(self.upper_ffnn3(x_upper_quantile))
        #x_upper_quantile = self.lower_dropout3(x_upper_quantile)
        x_upper_quantile = self.upper_ffnn4(x_upper_quantile)

        # https://medium.com/the-artificial-impostor/quantile-regression-part-2-6fdbc26b2629
        # return loss
        return (x_point_est, x_lower_quantile, x_upper_quantile)
        


class QuantileLoss(nn.Module):
    def __init__(self, alpha):
        super(QuantileLoss, self).__init__()
        self.alpha = alpha

    def forward(self, predictions, targets):
        # err = y_true - y_pred
        # return K.mean(K.maximum(alpha*err, (alpha-1)*err), axis=-1)
        err = targets - predictions
        # max_val = torch.max(self.alpha*err, (self.alpha-1)*err)
        # print(max_val)
        # print(max_val.shape)
        # mean_val = torch.mean(max_val) #, dim=-1)
        # print(mean_val)
        # print(mean_val.shape)
        return torch.mean(torch.max(self.alpha*err, (self.alpha-1)*err))
        #return mean_val


#------------------------------------------------------------------------------#

def load(args):
    """Load a Trainer object.

    This function creates an instance of the Trainer class, initialized using
    phyddle settings stored in args (dict).

    Args:
        args (dict): Contains phyddle settings.

    """

    # load object
    trn_objective = args['trn_objective']
    if trn_objective == 'param_est':
        return CnnTrainer(args)
    elif trn_objective == 'model_test':
        raise NotImplementedError
    else:
        return NotImplementedError

#------------------------------------------------------------------------------#

class Trainer:
    """
    Class for training neural networks with CPV+S and auxiliary data tensors
    tensors from the Format step. Results from Trainer objects are used in the
    Estimate and Plot steps.
    """

    def __init__(self, args):
        """Initializes a new Trainer object.

        Args:
            args (dict): Contains phyddle settings.

        """
        # initialize with phyddle settings
        self.set_args(args)
        # construct filepaths
        self.prepare_filepaths()
        # set CPUs
        if self.num_proc <= 0:
            self.num_proc = cpu_count() + self.num_proc
        # get size of CPV+S tensors
        self.num_tree_row = util.get_num_tree_row(self.tree_encode,
                                                  self.brlen_encode)
        self.num_char_row = util.get_num_char_row(self.char_encode,
                                                  self.num_char,
                                                  self.num_states)
        self.num_data_row = self.num_tree_row + self.num_char_row
        # create logger to track runtime info
        self.logger = util.Logger(args)
        # torch devic
        self.TORCH_DEVICE_STR = (    
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        self.TORCH_DEVICE = torch.device(self.TORCH_DEVICE_STR)
        # done
        return
    
    def set_args(self, args):
        """Assigns phyddle settings as Trainer attributes.

        Args:
            args (dict): Contains phyddle settings.

        """
        self.args = args
        step_args = util.make_step_args('T', args)
        for k,v in step_args.items():
            setattr(self, k, v)

        # special case
        self.kernel_init       = 'glorot_uniform'

        return
    
    def prepare_filepaths(self):
        """Prepare filepaths for the project.

        This script generates all the filepaths for input and output based off
        of Trainer attributes.

        """
        # main directories
        self.fmt_proj_dir = f'{self.fmt_dir}/{self.fmt_proj}'
        self.trn_proj_dir = f'{self.trn_dir}/{self.trn_proj}'

        # input prefix
        input_prefix      = f'{self.fmt_proj_dir}/train.nt{self.tree_width}'
        network_prefix    = f'network_nt{self.tree_width}'
        output_prefix     = f'{self.trn_proj_dir}/{network_prefix}'

        # input dataset filenames for csv or hdf5
        self.input_phy_data_fn      = f'{input_prefix}.phy_data.csv'
        self.input_aux_data_fn      = f'{input_prefix}.aux_data.csv'
        self.input_labels_fn        = f'{input_prefix}.labels.csv'
        self.input_hdf5_fn          = f'{input_prefix}.hdf5'

        # output network model info
        self.model_arch_fn          = f'{output_prefix}.trained_model.pkl'
        self.model_weights_fn       = f'{output_prefix}.train_weights.hdf5'
        self.model_history_fn       = f'{output_prefix}.train_history.csv'
        self.model_cpi_fn           = f'{output_prefix}.cpi_adjustments.csv'

        # output scaling terms
        self.train_labels_norm_fn   = f'{output_prefix}.train_label_norm.csv'
        self.train_aux_data_norm_fn = f'{output_prefix}.train_aux_data_norm.csv'
        
        # output training labels
        self.train_label_true_fn          = f'{output_prefix}.train_true.labels.csv'
        self.train_label_est_calib_fn     = f'{output_prefix}.train_est.labels.csv'
        self.train_label_est_nocalib_fn   = f'{output_prefix}.train_label_est_nocalib.csv'
        
        return

    def run(self):
        """Builds and trains the network.

        This method loads all training examples, builds the network, trains the
        network, collects results, then saves results to file.

        """
        verbose = self.verbose

        # print header
        util.print_step_header('trn', [self.fmt_proj_dir], self.trn_proj_dir, verbose)
        # prepare workspace
        os.makedirs(self.trn_proj_dir, exist_ok=True)

        # start time
        start_time,start_time_str = util.get_time()
        util.print_str(f'▪ Start time of {start_time_str}', verbose)

        # perform run tasks
        util.print_str('▪ Loading input', verbose)
        self.load_input()

        util.print_str('▪ Building network', verbose)
        self.build_network()

        util.print_str('▪ Training network', verbose)
        self.train()

        util.print_str('▪ Processing results', verbose)
        self.make_results()

        util.print_str('▪ Saving results', verbose)
        self.save_results()

        # end time
        end_time,end_time_str = util.get_time()
        run_time = util.get_time_diff(start_time, end_time)
        # util.print_str(f'▪ End time:     {end_time_str}', verbose)
        util.print_str(f'▪ End time of {end_time_str} (+{run_time})', verbose)

        util.print_str('▪ ... done!', verbose)
        return
    
    def load_input(self):
        """Loads the input data for the network."""
        raise NotImplementedError

    def build_network(self):
        """Builds the network architecture."""
        raise NotImplementedError

    def train(self):
        """Trains the network using the loaded input data."""
        raise NotImplementedError

    def make_results(self):
        """Generates the results using the trained network."""
        raise NotImplementedError

    def save_results(self):
        """Saves the generated results to a file or storage."""
        raise NotImplementedError

################################################################################

class CnnTrainer(Trainer):
    """
    Class for Convolutional Neural Network (CNN) Trainer.
    """
    def __init__(self, args):
        """
        Initializes a new CnnTrainer object.

        Args:
            args (dict): Contains phyddle settings.

        """
        # initialize base class
        super().__init__(args)
        return
    
    # splits input into training, test, validation, and calibration
    def split_tensor_idx(self, num_sample):
        """
        Split tensor into parts.

        This function splits the indexes for training examples into training,
        validation, and calibration sets. 

        Args:
            num_sample (int): The total number of samples in the dataset.
            combine_test_val (bool): Combine the test and validation indices.

        Returns:
            train_idx (numpy.ndarray): The indices for the training subset.
            val_idx (numpy.ndarray): The indices for the validation subset.
            calib_idx (numpy.ndarray): The indices for the calibration subset.

        """

        # get number of training, validation, and calibration datapoints
        num_calib = int(np.floor(num_sample * self.prop_cal)) 
        num_val   = int(np.floor(num_sample * self.prop_val))
        num_train = num_sample - (num_val + num_calib)
        assert(num_train > 0)

        # create input subsets
        train_idx = np.arange(num_train, dtype='int')
        val_idx   = np.arange(num_val,   dtype='int') + num_train
        calib_idx = np.arange(num_calib,  dtype='int') + num_train + num_val

        # return
        return train_idx, val_idx, calib_idx
    
    def validate_tensor_idx(self, train_idx, val_idx, calib_idx):
        """
        Validates input tensors.

        Checks that training, validation, and calibration input tensors are
        each non-empty.

        Args:
            train_idx (list): Training example indices.
            val_idx (list): Validation example indices.
            calib_idx (list): Calibration example indices.

        Returns:
            ValueError if any of the datasets are empty, otherwise returns None.

        """

        msg = ''
        if len(train_idx) == 0:
            msg = 'Training dataset is empty: len(train_idx) == 0'
        elif len(val_idx) == 0:
            msg = 'Validation dataset is empty: len(val_idx) == 0'
        elif len(calib_idx) == 0:
            msg = 'Calibration dataset is empty: len(calib_idx) == 0'           
        if msg != '':
            self.logger.write_log('trn', msg)
            raise ValueError(msg)

        return

    def load_input(self):
        """Load input data for the model.

        This function loads input data based on the specified tensor format
        (csv or hdf5). It performs necessary preprocessing steps such as reading
        data from files, reshaping tensors, normalizing summary stats and
        labels, randomizing data, and splitting the dataset into training,
        validation, test, and calibration parts.

        """
        # read phy. data, aux. data, and labels
        if self.tensor_format == 'csv':
            full_phy_data = pd.read_csv(self.input_phy_data_fn, header=None,
                                        on_bad_lines='skip').to_numpy()
            full_aux_data = pd.read_csv(self.input_aux_data_fn, header=None,
                                        on_bad_lines='skip').to_numpy()
            full_labels   = pd.read_csv(self.input_labels_fn, header=None,
                                        on_bad_lines='skip').to_numpy()
            self.aux_data_names = full_aux_data[0,:]
            self.label_names    = full_labels[0,:]
            full_aux_data       = full_aux_data[1:,:].astype('float64')
            full_labels         = full_labels[1:,:].astype('float64')   

        elif self.tensor_format == 'hdf5':
            hdf5_file = h5py.File(self.input_hdf5_fn, 'r')
            self.aux_data_names = [ s.decode() for s in hdf5_file['aux_data_names'][0,:] ]
            self.label_names    = [ s.decode() for s in hdf5_file['label_names'][0,:] ]
            full_phy_data       = pd.DataFrame(hdf5_file['phy_data']).to_numpy()
            full_aux_data       = pd.DataFrame(hdf5_file['aux_data']).to_numpy()
            full_labels         = pd.DataFrame(hdf5_file['labels']).to_numpy()
            hdf5_file.close()

        # data dimensions
        num_sample = full_phy_data.shape[0]
        self.num_params = full_labels.shape[1]
        self.num_stats = full_aux_data.shape[1]

        # logs of labels (rates) for variance stabilization against
        # heteroskedasticity (variance grows with mean)
        full_labels = np.log(full_labels + self.log_offset)
        full_aux_data = np.log(full_aux_data + self.log_offset)

        # shuffle datasets
        randomized_idx = np.random.permutation(full_phy_data.shape[0])
        full_phy_data  = full_phy_data[randomized_idx,:]
        full_aux_data  = full_aux_data[randomized_idx,:]
        full_labels    = full_labels[randomized_idx,:]

        # reshape phylogenetic tensor data based on CPV+S
        full_phy_data.shape = (num_sample, -1, self.num_data_row)

        # split dataset into training, test, validation, and calibration parts
        train_idx, val_idx, calib_idx = self.split_tensor_idx(num_sample)
        self.validate_tensor_idx(train_idx, val_idx, calib_idx)
        
        # merge test and validation datasets
        # if self.combine_test_val:
            # test_idx = np.concatenate([, val_idx])
            # val_idx = test_idx

        # save original training input
        self.train_label_true = np.exp(full_labels[train_idx,:]) - self.log_offset

        # normalize auxiliary data
        self.norm_train_aux_data, train_aux_data_means, train_aux_data_sd = util.normalize(full_aux_data[train_idx,:])
        self.train_aux_data_mean_sd = (train_aux_data_means, train_aux_data_sd)
        self.norm_val_aux_data = util.normalize(full_aux_data[val_idx,:],
                                                self.train_aux_data_mean_sd)
        self.norm_calib_aux_data = util.normalize(full_aux_data[calib_idx,:],
                                                  self.train_aux_data_mean_sd)

        # normalize labels
        self.norm_train_labels, train_label_means, train_label_sd = util.normalize(full_labels[train_idx,:])
        self.train_labels_mean_sd = (train_label_means, train_label_sd)
        self.norm_val_labels     = util.normalize(full_labels[val_idx,:],
                                                  self.train_labels_mean_sd)
        self.norm_calib_labels   = util.normalize(full_labels[calib_idx,:],
                                                  self.train_labels_mean_sd)

        # create phylogenetic data tensors
        self.train_phy_data_tensor = full_phy_data[train_idx,:]
        self.val_phy_data_tensor   = full_phy_data[val_idx,:]
        self.calib_phy_data_tensor = full_phy_data[calib_idx,:]

        # create auxiliary data tensors (with scaling)
        self.train_aux_data_tensor = self.norm_train_aux_data
        self.val_aux_data_tensor   = self.norm_val_aux_data
        self.calib_aux_data_tensor = self.norm_calib_aux_data

        # torch datasets
        self.train_dataset = PhyddleDataset(self.train_phy_data_tensor,
                                            self.norm_train_aux_data,
                                            self.norm_train_labels)
        self.calib_dataset = PhyddleDataset(self.calib_phy_data_tensor,
                                            self.norm_calib_aux_data,
                                            self.norm_calib_labels)
        self.val_dataset = PhyddleDataset(self.val_phy_data_tensor,
                                          self.norm_val_aux_data,
                                          self.norm_val_labels)

        return
    
#------------------------------------------------------------------------------#

    def build_network(self):
        
        # torch multiprocessing, eventually need to get working with cuda
        torch.set_num_threads(self.num_proc)

        # build model architecture
        self.model = NeuralNetwork(phy_dat_width=self.num_data_row,
                              phy_dat_height=self.tree_width,
                              aux_dat_width=self.num_stats,
                              lbl_width=self.num_params)
        
        self.model.phy_dat_shape = (self.num_data_row, self.tree_width)
        self.model.aux_dat_shape = (self.num_stats,)

        #print(self.model)
        #model.to(TORCH_DEVICE)

        return
#------------------------------------------------------------------------------#


    def train(self):
        """Trains the neural network model.

        This function compiles the network model to prepare it for training. 
        Perform training by compiling the model with appropriate loss functions
        and metrics, and then fitting the model to the training data. Training
        produces a history dictionary, which is saved.

        Returns:
            None

        """
        # dataset
        trainloader = DataLoader(dataset=self.train_dataset,
                                 batch_size=self.trn_batch_size)

        # loss functions
        loss_value_func = torch.nn.MSELoss()
        q_width = self.cpi_coverage
        q_tail = (1.0 - q_width) / 2
        q_lower = q_tail
        q_upper = 1.0 - q_tail
        loss_lower_func = QuantileLoss(alpha=q_lower)
        loss_upper_func = QuantileLoss(alpha=q_upper)

        # optimizer
        optimizer = torch.optim.Adam(self.model.parameters())
        # optimizer = torch.optim.Adam(self.model.parameters(),
        #                              lr=0.001,
        #                              weight_decay = 0.002)
        # optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001,
        #                               betas=(0.9, 0.999), eps=1e-08,
        #                               weight_decay=0.01, amsgrad=False)

        # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
        #                                             step_size = 50, 
        #                                             gamma = 0.1)

        # test one iteration of training
        #phy_dat, aux_dat, lbls = list(trainloader)[0]
        #lbls_hat = self.model(phy_dat, aux_dat)
        val_phy_dat = torch.Tensor(self.val_dataset.phy_data)
        val_aux_dat = torch.Tensor(self.val_dataset.aux_data)
        val_lbls    = torch.Tensor(self.val_dataset.labels)
        num_batches = int(np.ceil(self.train_dataset.phy_data.shape[0] / self.trn_batch_size))

        history_col_names = ['epoch', 'dataset', 'metric', 'value']
        self.train_history = pd.DataFrame(columns=history_col_names)

        # res = [ self.sim_one(idx) for idx in tqdm(self.rep_idx,
        #                                               total=len(self.rep_idx),
        #                                               desc='Simulating',
        #                                               smoothing=0) ]
        # training
        prev_trn_loss_combined = None
        prev_val_loss_combined = None
        for i in range(self.num_epochs):
            
            #print('-----')
            trn_loss_value = 0.
            trn_loss_lower = 0.
            trn_loss_upper = 0.
            trn_loss_combined = 0.
            trn_mse_value = 0.
            trn_mape_value = 0.
            trn_mae_value = 0.

            # num_batches = 0

            # for idx in tqdm(self.rep_idx, total=len(self.rep_idx), desc='Training', smoothing=0) 
            train_msg = f'Training epoch {i+1} of {self.num_epochs}'
            for j, (phy_dat, aux_dat, lbls) in tqdm(enumerate(trainloader),
                                                    total=num_batches,
                                                    desc=train_msg,
                                                    smoothing=0):
                
                # short cut batches for training
                # if j > 1:
                #     break
                
                # send labels to device
                lbls.to(self.TORCH_DEVICE)
                #phy_dat.to(self.TORCH_DEVICE)
                #aux_dat.to(self.TORCH_DEVICE)
                
                # reset gradients for tensors
                optimizer.zero_grad()
                
                # forward pass of training data to estimate labels
                lbls_hat = self.model(phy_dat, aux_dat)

                # calculating the loss between original and predicted data points
                loss_value     = loss_value_func(lbls_hat[0], lbls)
                loss_lower     = loss_lower_func(lbls_hat[1], lbls)
                loss_upper     = loss_upper_func(lbls_hat[2], lbls)
                loss_combined  = loss_value + loss_lower + loss_upper

                # collect history stats
                trn_loss_value += loss_value.item() / num_batches
                trn_loss_lower += loss_lower.item() / num_batches
                trn_loss_upper += loss_upper.item() / num_batches
                trn_loss_combined += loss_combined.item() / num_batches
                # print(sum_loss_combined)
                trn_mse_value  += ( torch.mean((lbls - lbls_hat[0])**2) ).item() / num_batches
                trn_mae_value  += ( torch.mean(torch.abs(lbls - lbls_hat[0])) ).item() / num_batches
                trn_mape_value += ( torch.mean(torch.abs((lbls - lbls_hat[0]) / lbls)) ).item() / num_batches
                # num_batches    += 1
                
                # backward pass to update gradients
                loss_combined.backward()

                # update network parameters
                optimizer.step()
                #lr_scheduler.step()
                
                # print(f'  batch {j+1} of {len(trainloader)}')
                # print('   true       = ', lbls[0,:])
                # print('   est_value  = ', lbls_hat[0][0])
                # print('   est_lower  = ', lbls_hat[1][0])
                # print('   est_upper  = ', lbls_hat[2][0])
                # print('   loss_value = ', loss_value)
                # print('   loss_lower = ', loss_lower)
                # print('   loss_upper = ', loss_upper)
                # print('   loss       = ', loss)
                # print('   ---')

            self.train_history.loc[len(self.train_history.index)] = [i, 'train', 'loss_value',     trn_loss_value]
            self.train_history.loc[len(self.train_history.index)] = [i, 'train', 'loss_lower',     trn_loss_lower]
            self.train_history.loc[len(self.train_history.index)] = [i, 'train', 'loss_upper',     trn_loss_upper]
            self.train_history.loc[len(self.train_history.index)] = [i, 'train', 'loss_combined',  trn_loss_combined]
            self.train_history.loc[len(self.train_history.index)] = [i, 'train', 'mse_value',      trn_mse_value]
            self.train_history.loc[len(self.train_history.index)] = [i, 'train', 'mae_value',      trn_mae_value]
            self.train_history.loc[len(self.train_history.index)] = [i, 'train', 'mape_value',     trn_mape_value]

            # forward pass of validation to estimate labels
            val_lbls_hat       = self.model(val_phy_dat, val_aux_dat)

            # collect validation metrics
            val_loss_value     = loss_value_func(val_lbls_hat[0], val_lbls).item()
            val_loss_lower     = loss_lower_func(val_lbls_hat[1], val_lbls).item()
            val_loss_upper     = loss_upper_func(val_lbls_hat[2], val_lbls).item()
            val_loss_combined  = val_loss_value + val_loss_lower + val_loss_upper
            val_mse_value      = ( torch.mean((val_lbls - val_lbls_hat[0])**2) ).item()
            val_mae_value      = ( torch.mean(torch.abs(val_lbls - val_lbls_hat[0])) ).item()
            val_mape_value     = ( torch.mean(torch.abs((val_lbls - val_lbls_hat[0]) / val_lbls)) ).item()
            self.train_history.loc[len(self.train_history.index)] = [i, 'validation', 'loss_value',     val_loss_value]
            self.train_history.loc[len(self.train_history.index)] = [i, 'validation', 'loss_lower',     val_loss_lower]
            self.train_history.loc[len(self.train_history.index)] = [i, 'validation', 'loss_upper',     val_loss_upper]
            self.train_history.loc[len(self.train_history.index)] = [i, 'validation', 'loss_combined',  val_loss_combined ]
            self.train_history.loc[len(self.train_history.index)] = [i, 'validation', 'mse_value',      val_mse_value]
            self.train_history.loc[len(self.train_history.index)] = [i, 'validation', 'mae_value',      val_mae_value]
            self.train_history.loc[len(self.train_history.index)] = [i, 'validation', 'mape_value',     val_mape_value]

            trn_loss_str = f'    Train        --  loss: {"{0:.4f}".format(trn_loss_combined)}'
            val_loss_str = f'    Validation   --  loss: {"{0:.4f}".format(val_loss_combined)}'
            
            if i > 0:
                diff_trn_loss = trn_loss_combined - prev_trn_loss_combined
                diff_val_loss = val_loss_combined - prev_val_loss_combined
                rat_trn_loss  = 100 * round(trn_loss_combined / prev_trn_loss_combined - 1.0, ndigits=3)
                rat_val_loss  = 100 * round(val_loss_combined / prev_val_loss_combined - 1.0, ndigits=3)
                # diff_trn_loss_str = str(np.round(diff_trn_loss, decimals=4)).rjust(5, ' ')
                # diff_val_loss_str = str(np.round(diff_val_loss, decimals=4)).rjust(5, ' ')
                # rat_trn_loss_str  = str(np.round(rat_trn_loss, decimals=1)).rjust(5, ' ')
                # rat_val_loss_str  = str(np.round(rat_val_loss, decimals=1)).rjust(5, ' ')
                diff_trn_loss_str = '{0:+.4f}'.format(diff_trn_loss)
                diff_val_loss_str = '{0:+.4f}'.format(diff_val_loss)
                rat_trn_loss_str  = '{0:+.2f}'.format(rat_trn_loss).rjust(4, ' ')
                rat_val_loss_str  = '{0:+.2f}'.format(rat_val_loss).rjust(4, ' ')
                
                # if trn_loss_combined > prev_trn_loss_combined:
                #     diff_trn_loss_str = '+' + diff_trn_loss_str
                #     rat_trn_loss_str  = '+' + rat_trn_loss_str
                # if val_loss_combined > prev_val_loss_combined:
                #     diff_val_loss_str = '+' + diff_val_loss_str
                #     rat_val_loss_str  = '+' + rat_val_loss_str
            
                trn_loss_str += f'  abs: {diff_trn_loss_str}  rel: {rat_trn_loss_str}%'
                val_loss_str += f'  abs: {diff_val_loss_str}  rel: {rat_val_loss_str}%'

            prev_trn_loss_combined = trn_loss_combined
            prev_val_loss_combined = val_loss_combined

            print(trn_loss_str)
            print(val_loss_str)
            print('')

        # print(self.train_history)
        return


    def make_results(self):
        """Makes all results from the Train step.

        This function undoes all the transformation and rescaling for the 
        input and output datasets.

        """
        
        # training label estimates
        #norm_train_label_est = self.mymodel.predict([self.train_phy_data_tensor, self.train_aux_data_tensor])
        trainloader = DataLoader(dataset=self.train_dataset, batch_size = 1000)
                                 #batch_size=len(self.train_dataset))
        train_phy_dat, train_aux_dat, train_lbl = next(iter(trainloader))
        norm_train_label_est = self.model(train_phy_dat, train_aux_dat)
        
        # we want an array of 3 outputs [point, lower, upper], N examples, K parameters
        #norm_train_label_est = np.array(norm_train_label_est)
        norm_train_label_est = torch.stack(norm_train_label_est).detach().numpy()
        self.train_label_est = util.denormalize(norm_train_label_est,
                                                self.train_labels_mean_sd,
                                                exp=True) - self.log_offset
 
        # calibration label estimates + CPI adjustment
        # norm_calib_label_est = self.mymodel.predict([self.calib_phy_data_tensor,
        #                                              self.calib_aux_data_tensor])
        calibloader = DataLoader(dataset=self.calib_dataset, batch_size = self.calib_phy_data_tensor.shape[0])
        calib_phy_dat, calib_aux_dat, calib_lbl = next(iter(calibloader))
        norm_calib_label_est = self.model(calib_phy_dat, calib_aux_dat)
                                 #batch_size=len(self.train_dataset))
        norm_calib_label_est = torch.stack(norm_calib_label_est).detach().numpy()
        #norm_calib_label_est = np.array(norm_calib_label_est)
        norm_calib_est_quantiles = norm_calib_label_est[1:,:,:]
        self.cpi_adjustments = self.get_CQR_constant(norm_calib_est_quantiles,
                                                     self.norm_calib_labels,
                                                     inner_quantile=self.cpi_coverage,
                                                     asymmetric=self.cpi_asymmetric)
        self.cpi_adjustments = np.array(self.cpi_adjustments).reshape((2,-1))
    
        # training predictions with calibrated CQR CIs
        norm_train_label_est_calib = norm_train_label_est
        norm_train_label_est_calib[1,:,:] = norm_train_label_est_calib[1,:,:] - self.cpi_adjustments[0,:]
        norm_train_label_est_calib[2,:,:] = norm_train_label_est_calib[2,:,:] + self.cpi_adjustments[1,:]
        self.train_label_est_calib = util.denormalize(norm_train_label_est_calib,
                                                      self.train_labels_mean_sd,
                                                      exp=True) - self.log_offset

        return


    def save_results(self):
        """Save training results.

        Saves all results from training procedure. Saved results include the
        trained network, the normalization parameters for training/calibration,
        CPI adjustment terms, and the training history.

        """
        max_idx = 1000
        
        # save model to file
        torch.save(self.model, self.model_arch_fn)

        # save json history from running MASTER
        self.train_history.to_csv(self.model_history_fn, index=False, sep=',')


        # save aux_data names, means, sd for new test dataset normalization
        df_aux_data = pd.DataFrame([self.aux_data_names,
                                    self.train_aux_data_mean_sd[0],
                                    self.train_aux_data_mean_sd[1]]).T
        df_aux_data.columns = ['name', 'mean', 'sd']
        df_aux_data.to_csv(self.train_aux_data_norm_fn, index=False, sep=',')
 
        # save label names, means, sd for new test dataset normalization
        df_labels = pd.DataFrame([self.label_names,
                                  self.train_labels_mean_sd[0],
                                  self.train_labels_mean_sd[1]]).T
        df_labels.columns = ['name', 'mean', 'sd']
        df_labels.to_csv(self.train_labels_norm_fn, index=False, sep=',')

        # save train/test scatterplot results (Value, Lower, Upper)
        df_train_label_est_nocalib = util.make_param_VLU_mtx(self.train_label_est[0:max_idx,:], self.label_names )
        df_train_label_est_calib   = util.make_param_VLU_mtx(self.train_label_est_calib[0:max_idx,:], self.label_names )
        
        # save train/test labels
        df_train_label_true = pd.DataFrame(self.train_label_true[0:max_idx,:], columns=self.label_names )
        
        # save CPI intervals
        df_cpi_intervals = pd.DataFrame( self.cpi_adjustments, columns=self.label_names )

        # convert to csv and save
        df_train_label_est_nocalib.to_csv(self.train_label_est_nocalib_fn, index=False, sep=',')
        df_train_label_est_calib.to_csv(self.train_label_est_calib_fn, index=False, sep=',')
        df_train_label_true.to_csv(self.train_label_true_fn, index=False, sep=',')
        df_cpi_intervals.to_csv(self.model_cpi_fn, index=False, sep=',')

        return

#------------------------------------------------------------------------------#
        
    def get_CQR_constant(self, ests, true, inner_quantile=0.95, asymmetric = True):
        """Computes the conformalized quantile regression (CQR) constants.
        
        This function computes symmetric or asymmetric CQR constants for the
        specified inner-quantile range.

        Notes:
            # ests axis 0 is the lower and upper quants,
            # axis 1 is the replicates, and axis 2 is the params

        Arguments:
            ests (array-like): The input data.
            true (array-like): The target data.
            q_lower (function): The lower quantile function.
            q_upper (function): The upper quantile function.

        Returns:
            array-like: The conformity scores.

        """
        
        # compute non-comformity scores
        Q = np.empty((2, ests.shape[2]))
        
        for i in range(ests.shape[2]):
            if asymmetric:
                # Asymmetric non-comformity score
                lower_s = np.array(true[:,i] - ests[0][:,i])
                upper_s = np.array(true[:,i] - ests[1][:,i])
                lower_p = (1 - inner_quantile)/2 * (1 + 1/ests.shape[1])
                upper_p = (1 + inner_quantile)/2 * (1 + 1/ests.shape[1])
                if lower_p < 0.:
                    self.logger.write_log('trn',
                                          'get_CQR_constant: lower_p >= 0.')
                    lower_p = 0.
                if upper_p > 1.:
                    self.logger.write_log('trn',
                                          'get_CQR_constant: upper_p <= 1.')
                    upper_p = 1.
                lower_q = np.quantile(lower_s, lower_p)
                upper_q = np.quantile(upper_s, upper_p)
            else:
                # Symmetric non-comformity score
                s = np.amax(np.array((ests[0][:,i]-true[:,i], true[:,i]-ests[1][:,i])), axis=0)
                # get adjustment constant: 1 - alpha/2's quintile of non-comformity scores
                symm_p = inner_quantile * (1 + 1/ests.shape[1])
                if symm_p < 0.:
                    self.logger.write_log('trn',
                                          'get_CQR_constant: symm_p >= 0.')
                    symm_p = 0.
                elif symm_p > 1.:
                    self.logger.write_log('trn',
                                          'get_CQR_constant: symm_p <= 1.')
                    symm_p = 1.                    
                lower_q = np.quantile(s, symm_p)
                upper_q = lower_q
                #Q[:,i] = np.array([lower_q, upper_q])

            Q[:,i] = np.array([lower_q, upper_q])
                                
        return Q
    
#------------------------------------------------------------------------------#

    
    # def train2(self):
    #     """Trains the neural network model.

    #     This function compiles the network model to prepare it for training. 
    #     Perform training by compiling the model with appropriate loss functions
    #     and metrics, and then fitting the model to the training data. Training
    #     produces a history dictionary, which is saved.

    #     Returns:
    #         None

    #     """
    #     # gather loss functions
    #     my_loss = [self.loss,
    #                self.pinball_loss_q_0_025,
    #                self.pinball_loss_q_0_975]

    #     # compile model        
    #     self.mymodel.compile(optimizer=self.optimizer, 
    #                          loss=my_loss,
    #                          metrics=self.metrics)
        
    #     # early stopping
    #     # es = callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)
    #     #es = callbacks.EarlyStopping(monitor='val_loss', mode='max', min_delta=0.01)

    #     # run training
    #     self.history = self.mymodel.fit(\
    #         verbose = 2,
    #         x = [self.train_phy_data_tensor,
    #              self.train_aux_data_tensor], 
    #         y = self.norm_train_labels,
    #         epochs = self.num_epochs,
    #         batch_size = self.trn_batch_size,
    #         # callbacks = [es], 
    #         validation_data = ([self.val_phy_data_tensor,
    #                             self.val_aux_data_tensor],
    #                             self.norm_val_labels))
        
    #     # store training history
    #     self.history_dict = self.history.history
    #     # done
    #     return


    # def pinball_loss(self, y_true, y_pred, alpha):
    #     """Calculate the pinball loss.

    #     This function calculates the pinball Loss, which measures the
    #     difference between the true target values (`y_true`) and the predicted
    #     target values (`y_pred`) for a quantile regression model. Used for
    #     quantile regression.

    #     The Pinball Loss is calculated using the following formula:
    #         mean(maximum(alpha * err, (alpha - 1) * err))

    #     Arguments:
    #         y_true (numpy.ndarray): Array of true target values.
    #         y_pred (numpy.ndarray): Array of predicted target values.
    #         alpha (float): Quantile level.

    #     Returns:
    #         float: Value of pinball loss

    #     """

    #     err = y_true - y_pred
    #     return K.mean(K.maximum(alpha*err, (alpha-1)*err), axis=-1)

    
    # def pinball_loss_q_0_025(self, y_true, y_pred):
    #     """Pinball loss for lower 95% quantile"""
    #     return self.pinball_loss(y_true, y_pred, alpha=0.025)
    
    # def pinball_loss_q_0_975(self, y_true, y_pred):
    #     """Pinball loss for upper 95% quantile"""
    #     return self.pinball_loss(y_true, y_pred, alpha=0.975)
    
    # def pinball_loss_q_0_05(self, y_true, y_pred):
    #     """Pinball loss for lower 90% quantile"""
    #     return self.pinball_loss(y_true, y_pred, alpha=0.05)
    
    # def pinball_loss_q_0_95(self, y_true, y_pred):
    #     """Pinball loss for upper 90% quantile"""
    #     return self.pinball_loss(y_true, y_pred, alpha=0.95)
    
    # def pinball_loss_q_0_10(self, y_true, y_pred):
    #     """Pinball loss for lower 80% quantile"""
    #     return self.pinball_loss(y_true, y_pred, alpha=0.10)
    
    # def pinball_loss_q_0_90(self, y_true, y_pred):
    #     """Pinball loss for upper 80% quantile"""
    #     return self.pinball_loss(y_true, y_pred, alpha=0.90)
    
    # def pinball_loss_q_0_15(self, y_true, y_pred):
    #     """Pinball loss for lower 70% quantile"""
    #     return self.pinball_loss(y_true, y_pred, alpha=0.15)
    
    # def pinball_loss_q_0_85(self, y_true, y_pred):
    #     """Pinball loss for upper 70% quantile"""
    #     return self.pinball_loss(y_true, y_pred, alpha=0.85)

    # def get_pinball_loss_fns(self, coverage):
    #     """Gets correct pinball loss functions.

    #     The CnnTrainer class currently implements lower and upper quantiles for
    #     70%, 80%, 90%, 95%.

    #     Arguments:
    #         coverage (float): The desired coverage level.

    #     Returns:
    #         tuple: Two pinball loss functions for the specified coverage level.

    #     """

    #     if coverage == 0.95:
    #         return self.pinball_loss_q_0_025, self.pinball_loss_q_0_975
    #     elif coverage == 0.90:
    #         return self.pinball_loss_q_0_05, self.pinball_loss_q_0_95
    #     elif coverage == 0.80:
    #         return self.pinball_loss_q_0_10, self.pinball_loss_q_0_90
    #     elif coverage == 0.70:
    #         return self.pinball_loss_q_0_15, self.pinball_loss_q_0_85
    #     else:
    #         raise NotImplementedError


    # def save_results2(self):
    #     """Save training results.

    #     Saves all results from training procedure. Saved results include the
    #     trained network, the normalization parameters for training/calibration,
    #     CPI adjustment terms, and the training history.

    #     """
    #     max_idx = 1000
        
    #     # save model to file
    #     #self.mymodel.save('my_model.keras')
    #     self.mymodel.save(self.model_arch_fn)

    #     # save weights to file
    #     # self.mymodel.save_weights(self.model_weights_fn)

    #     # save json history from running MASTER
    #     json.dump(self.history_dict, open(self.model_history_fn, 'w'))


    #     # save aux_data names, means, sd for new test dataset normalization
    #     df_aux_data = pd.DataFrame([self.aux_data_names,
    #                                 self.train_aux_data_mean_sd[0],
    #                                 self.train_aux_data_mean_sd[1]]).T
    #     df_aux_data.columns = ['name', 'mean', 'sd']
    #     df_aux_data.to_csv(self.train_aux_data_norm_fn, index=False, sep=',')
 
    #     # save label names, means, sd for new test dataset normalization
    #     df_labels = pd.DataFrame([self.label_names,
    #                               self.train_labels_mean_sd[0],
    #                               self.train_labels_mean_sd[1]]).T
    #     df_labels.columns = ['name', 'mean', 'sd']
    #     df_labels.to_csv(self.train_labels_norm_fn, index=False, sep=',')

    #     # save train/test scatterplot results (Value, Lower, Upper)
    #     df_train_label_est_nocalib = util.make_param_VLU_mtx(self.train_label_est[0:max_idx,:], self.label_names )
    #     df_train_label_est_calib   = util.make_param_VLU_mtx(self.train_label_est_calib[0:max_idx,:], self.label_names )
        
    #     # save train/test labels
    #     df_train_label_true = pd.DataFrame(self.train_label_true[0:max_idx,:], columns=self.label_names )
        
    #     # save CPI intervals
    #     df_cpi_intervals = pd.DataFrame( self.cpi_adjustments, columns=self.label_names )

    #     # convert to csv and save
    #     df_train_label_est_nocalib.to_csv(self.train_label_est_nocalib_fn, index=False, sep=',')
    #     df_train_label_est_calib.to_csv(self.train_label_est_calib_fn, index=False, sep=',')
    #     df_train_label_true.to_csv(self.train_label_true_fn, index=False, sep=',')
    #     df_cpi_intervals.to_csv(self.model_cpi_fn, index=False, sep=',')

    #     return



        
    # def make_results2(self):
    #     """Makes all results from the Train step.

    #     This function undoes all the transformation and rescaling for the 
    #     input and output datasets.

    #     """
        
    #     # training label estimates
    #     norm_train_label_est = self.mymodel.predict([self.train_phy_data_tensor, self.train_aux_data_tensor])
    #     norm_train_label_est = np.array(norm_train_label_est)
    #     self.train_label_est = util.denormalize(norm_train_label_est,
    #                                             self.train_labels_mean_sd,
    #                                             exp=True) - self.log_offset
 
    #     # calibration label estimates + CPI adjustment
    #     norm_calib_label_est = self.mymodel.predict([self.calib_phy_data_tensor,
    #                                                  self.calib_aux_data_tensor])
    #     norm_calib_label_est = np.array(norm_calib_label_est)
    #     norm_calib_est_quantiles = norm_calib_label_est[1:,:,:]
    #     self.cpi_adjustments = self.get_CQR_constant(norm_calib_est_quantiles,
    #                                                  self.norm_calib_labels,
    #                                                  inner_quantile=self.cpi_coverage,
    #                                                  asymmetric=self.cpi_asymmetric)
    #     self.cpi_adjustments = np.array(self.cpi_adjustments).reshape((2,-1))
    
    #     # training predictions with calibrated CQR CIs
    #     norm_train_label_est_calib = norm_train_label_est
    #     norm_train_label_est_calib[1,:,:] = norm_train_label_est_calib[1,:,:] - self.cpi_adjustments[0,:]
    #     norm_train_label_est_calib[2,:,:] = norm_train_label_est_calib[2,:,:] + self.cpi_adjustments[1,:]
    #     self.train_label_est_calib = util.denormalize(norm_train_label_est_calib,
    #                                                   self.train_labels_mean_sd,
    #                                                   exp=True) - self.log_offset

    #     return


    

    # def build_network2(self):
    #     """Builds the network architecture.

    #     This function constructs the network architecture by assembling the
    #     input layers, phylo. data layers, aux. data layers, and output layers.
    #     It then instantiates the model using the assembled layers.

    #     Simplified network architecture:
        
    #                           ,--> Conv1D-normal + Pool --. 
    #     Phylo. Data Tensor --+---> Conv1D-stride + Pool ---\\                         ,--> Point estimate
    #                           '--> Conv1D-dilate + Pool ----+--> Concat + Output(s)--+---> Lower quantile
    #                                                        /                          '--> Upper quantile
    #     Aux. Data Tensor   ------> Dense -----------------'

    #     Returns:
    #         None
    #     """

    #     #input layers
    #     input_layers    = self.build_network_input_layers()
    #     phylo_layers    = self.build_network_phylo_layers(input_layers['phylo'])
    #     aux_layers      = self.build_network_aux_layers(input_layers['aux'])
    #     output_layers   = self.build_network_output_layers(phylo_layers, aux_layers)
    
    #     # instantiate model
    #     self.mymodel = Model(inputs = [input_layers['phylo'],
    #                                    input_layers['aux']], 
    #                          outputs = output_layers)
        
    # def build_network_input_layers(self):
    #     """Build the input layers for the network.

    #     Returns:
    #         dict: A dictionary containing the input layers for phylogenetic
    #               state data and auxiliary data tensors.

    #     """

    #     input_phylo_tensor = Input(shape=self.train_phy_data_tensor.shape[1:3], name='input_phylo')
    #     input_aux_tensor   = Input(shape=self.train_aux_data_tensor.shape[1:2], name='input_aux')

    #     return {'phylo': input_phylo_tensor, 'aux': input_aux_tensor }

    
    # def build_network_aux_layers(self, input_aux_tensor):
    #     """Build the auxiliary data layers for the network.

    #     This function assumes a densely connected feed-forward neural network
    #     design for the layers. This is later concatenated with the CNN arms.

    #     Args:
    #         input_aux_tensor: The input layer for the auxiliary data input.

    #     Returns:
    #         list: A list of auxiliary data FFNN layers.

    #     """

    #     w_aux_ffnn = layers.Dense(128, activation = 'relu', kernel_initializer = self.kernel_init, name='ff_aux1')(input_aux_tensor)
    #     w_aux_ffnn = layers.Dense(64, activation = 'relu', kernel_initializer = self.kernel_init, name='ff_aux2')(w_aux_ffnn)
    #     w_aux_ffnn = layers.Dense(32, activation = 'relu', kernel_initializer = self.kernel_init, name='ff_aux3')(w_aux_ffnn)
        
    #     return [ w_aux_ffnn ]

    # def build_network_phylo_layers(self, input_data_tensor):
    #     """Build the phylogenetic state data layers for the network.

    #     This function assumes a convolutional neural network design, composed
    #     of three 1D-convolution + pool layer seqeuences ("arms"). The three
    #     arms vary in terms of node count, number of layers, width, stride,
    #     and dilation. This is later concatenated with the FFNN arm.

    #     Args:
    #         input_data_tensor: The input data tensor.

    #     Returns:
    #         list: A list of phylo layers.

    #     """

    #     # convolutional layers
    #     # e.g. you expect to see 64 patterns, width of 3,
    #     # stride (skip-size) of 1, padding zeroes so all windows are 'same'
    #     w_conv = layers.Conv1D(64, 3, activation = 'relu', padding = 'same', name='conv_std1')(input_data_tensor)
    #     w_conv = layers.Conv1D(96, 5, activation = 'relu', padding = 'same', name='conv_std2')(w_conv)
    #     w_conv = layers.Conv1D(128, 7, activation = 'relu', padding = 'same', name='conv_std3')(w_conv)
    #     w_conv_gavg = layers.GlobalAveragePooling1D(name='pool_std')(w_conv)

    #     # stride layers (skip sizes during slide)
    #     w_stride = layers.Conv1D(64, 7, strides = 3, activation = 'relu',padding = 'same', name='conv_stride1')(input_data_tensor)
    #     w_stride = layers.Conv1D(96, 9, strides = 6, activation = 'relu', padding = 'same', name='conv_stride2')(w_stride)
    #     w_stride_gavg = layers.GlobalAveragePooling1D(name = 'pool_stride')(w_stride)

    #     # dilation layers (spacing among grid elements)
    #     w_dilated = layers.Conv1D(32, 3, dilation_rate = 2, activation = 'relu', padding = 'same', name='conv_dilate1')(input_data_tensor)
    #     w_dilated = layers.Conv1D(64, 5, dilation_rate = 4, activation = 'relu', padding = 'same', name='conv_dilate2')(w_dilated)
    #     w_dilated_gavg = layers.GlobalAveragePooling1D(name = 'pool_dilate')(w_dilated)

    #     return [ w_conv_gavg, w_stride_gavg, w_dilated_gavg ]

    # def build_network_output_layers(self, phylo_layers, aux_layers):
    #     """Build the output layers for the network.

    #     This function concatenates the output from the CNN and FFNN arms of the
    #     network, and then constructs three new FFNN arms that lead towards
    #     output layers for estimating parameter (label) values and upper and
    #     lower CPI bounds.

    #     Args:
    #         phylo_layers: The phylo layers.
    #         aux_layers: The auxiliary layers.

    #     Returns:
    #         list: A list of output layers.

    #     """

    #     # combine phylo and aux layers lists
    #     all_layers = phylo_layers + aux_layers

    #     # concatenate all above -> deep fully connected network
    #     w_concat = layers.Concatenate(axis = 1, name = 'concat_out')(all_layers)

    #     # point estimate for parameters
    #     w_point_est = layers.Dense(128, activation = 'relu',kernel_initializer = self.kernel_init, name='ff_value1')(w_concat)
    #     w_point_est = layers.Dense( 64, activation = 'relu', kernel_initializer = self.kernel_init, name='ff_value2')(w_point_est)
    #     w_point_est = layers.Dense( 32, activation = 'relu', kernel_initializer = self.kernel_init, name='ff_value3')(w_point_est)
    #     w_point_est = layers.Dense(self.num_params, activation = 'linear', name = 'param_value')(w_point_est)

    #     # lower quantile for parameters
    #     w_lower_quantile = layers.Dense(128, activation = 'relu', kernel_initializer = self.kernel_init, name='ff_lower1')(w_concat)
    #     w_lower_quantile = layers.Dense( 64, activation = 'relu', kernel_initializer = self.kernel_init, name='ff_lower2')(w_lower_quantile)
    #     w_lower_quantile = layers.Dense( 32, activation = 'relu', kernel_initializer = self.kernel_init, name='ff_lower3')(w_lower_quantile)
    #     w_lower_quantile = layers.Dense(self.num_params, activation = 'linear', name = 'param_lower')(w_lower_quantile)

    #     # upper quantile for parameters
    #     w_upper_quantile = layers.Dense(128, activation = 'relu', kernel_initializer = self.kernel_init, name='ff_upper1')(w_concat)
    #     w_upper_quantile = layers.Dense( 64, activation = 'relu', kernel_initializer = self.kernel_init, name='ff_upper2')(w_upper_quantile)
    #     w_upper_quantile = layers.Dense( 32, activation = 'relu', kernel_initializer = self.kernel_init, name='ff_upper3')(w_upper_quantile)
    #     w_upper_quantile = layers.Dense(self.num_params, activation = 'linear', name = 'param_upper')(w_upper_quantile)

    #     return [ w_point_est, w_lower_quantile, w_upper_quantile ]
