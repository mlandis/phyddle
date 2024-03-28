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
import os

# external imports
import h5py
import numpy as np
import pandas as pd
import torch
from multiprocessing import cpu_count
from tqdm import tqdm

# phyddle imports
from phyddle import utilities as util
from phyddle import network


##################################################

def load(args):
    """Load a Trainer object.

    This function creates an instance of the Trainer class, initialized using
    phyddle settings stored in args (dict).

    Args:
        args (dict): Contains phyddle settings.

    """

    # load object
    train_method = 'default'
    if train_method == 'default':
        return CnnTrainer(args)
    else:
        return NotImplementedError
    
##################################################


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
        
        # args
        self.args                   = args

        # filesystem
        self.fmt_prefix             = str(args['fmt_prefix'])
        self.trn_prefix             = str(args['trn_prefix'])
        self.fmt_dir                = str(args['fmt_dir'])
        self.trn_dir                = str(args['trn_dir'])
        self.log_dir                = str(args['log_dir'])
        
        # analysis settings
        self.verbose            = bool(args['verbose'])
        self.num_proc           = int(args['num_proc'])
        self.use_parallel       = bool(args['use_parallel'])
        
        # dataset dimensions
        self.num_char           = int(args['num_char'])
        self.num_states         = int(args['num_states'])
        self.tree_width         = int(args['tree_width'])
        
        # dataset processing
        self.tree_encode        = str(args['tree_encode'])
        self.char_encode        = str(args['char_encode'])
        self.brlen_encode       = str(args['brlen_encode'])
        self.char_format        = str(args['char_format'])
        self.tensor_format      = str(args['tensor_format'])
        self.param_est          = dict(args['param_est'])
        self.param_data         = dict(args['param_data'])
        self.prop_test          = float(args['prop_test'])
        self.log_offset         = float(args['log_offset'])
        self.save_phyenc_csv    = bool(args['save_phyenc_csv'])
        
        # train settings
        self.prop_cal           = float(args['prop_cal'])
        self.prop_val           = float(args['prop_val'])
        self.num_epochs         = int(args['num_epochs'])
        self.trn_batch_size     = int(args['trn_batch_size'])
        self.cpi_coverage       = float(args['cpi_coverage'])
        self.cpi_asymmetric     = bool(args['cpi_asymmetric'])
        self.loss_real          = str(args['loss_real'])
        self.use_cuda           = bool(args['use_cuda'])
        self.num_early_stop     = int(args['num_early_stop'])

        # initialized later
        self.phy_tensors        = dict()   # init with encode_all()
        self.train_dataset      = None     # init with load_input()
        self.val_dataset        = None     # init with load_input()
        self.calib_dataset      = None     # init with load_input()
        
        # set CPUs
        if self.num_proc <= 0:
            self.num_proc = cpu_count() + self.num_proc
        if self.num_proc <= 0:
            self.num_proc = 1

        # get size of CPV+S tensors
        self.num_tree_col = util.get_num_tree_col(self.tree_encode,
                                                  self.brlen_encode)
        self.num_char_col = util.get_num_char_col(self.char_encode,
                                                  self.num_char,
                                                  self.num_states)
        self.num_data_col = self.num_tree_col + self.num_char_col

        # create logger to track runtime info
        self.logger = util.Logger(args)
        
        # set torch device
        # NOTE: need to test against cuda
        self.TORCH_DEVICE_STR = (
            "cuda"
            if torch.cuda.is_available() and self.use_cuda
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        self.TORCH_DEVICE = torch.device(self.TORCH_DEVICE_STR)
        
        # done
        return

    def run(self):
        """Builds and trains the network.

        This method loads all training examples, builds the network, trains the
        network, collects results, then saves results to file.

        """
        verbose = self.verbose

        # print header
        util.print_step_header('trn', [self.fmt_dir], self.trn_dir,
                               [self.fmt_prefix], self.trn_prefix, verbose)
        
        # prepare workspace
        os.makedirs(self.trn_dir, exist_ok=True)

        # start time
        start_time,start_time_str = util.get_time()
        util.print_str(f'▪ Start time of {start_time_str}', verbose)

        # perform run tasks
        util.print_str('▪ Loading input:', verbose)
        self.load_input()
        num_rjust = len(str(len(self.train_dataset)))
        util.print_str(f'  ▪ ' + str(len(self.train_dataset)).rjust(num_rjust) + ' training examples', verbose)
        util.print_str(f'  ▪ ' + str(len(self.calib_dataset)).rjust(num_rjust) + ' calibration examples', verbose)
        util.print_str(f'  ▪ ' + str(len(self.val_dataset)).rjust(num_rjust) + ' validation examples', verbose)

        util.print_str('▪ Training targets:', verbose)
        num_ljust = max([len(k) for k in self.param_est.keys()])
        for k,v in self.param_est.items():
            util.print_str(f'  ▪ {k.ljust(num_ljust)}  [type: {v}]', verbose)


        util.print_str('▪ Building network', verbose)
        self.build_network()

        util.print_str('▪ Training network', verbose)
        device_info = ''
        if self.TORCH_DEVICE_STR == 'cuda':
            device_info = '  ▪ using CUDA + GPU'
            device_info += ' [device: ' + torch.cuda.get_device_properties(0).name + ']'
        elif self.TORCH_DEVICE_STR == 'cpu':
            num_cpu = os.cpu_count()
            device_info = '  ▪ using CPUs [num: ' + str(num_cpu) + ']'
        if device_info != '':
            util.print_str(device_info, verbose)
        self.train()

        util.print_str('▪ Processing results', verbose)
        self.make_results()

        util.print_str('▪ Saving results', verbose)
        self.save_results()

        # end time
        end_time,end_time_str = util.get_time()
        run_time = util.get_time_diff(start_time, end_time)
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
        
        self.aux_data_names = list()    # init with load_input()
        self.label_names    = list()    # init with load_input()
        self.num_aux_data   = int()     # init with load_input()
        self.param_cat_names = list()   # init with load_input()
        self.param_real_names = list()  # init with load_input()
        self.num_param_real = int()     # init with load_input()
        self.num_param_cat  = int()     # init with load_input()
        self.param_cat      = dict()    # init with load_input()

        # todo: revisit and simplify, provide types
        self.train_dataset = None       # init with load_input()
        self.val_dataset   = None       # init with load_input()
        self.calib_dataset = None       # init with load_input()
        self.model = None               # init with build_network()
        self.train_label_real_est = None
        self.train_label_real_true = None
        self.train_label_cat_est = None
        self.train_label_cat_true = None
        self.train_label_real_est_calib = None
        self.calib_phy_data_tensor = None
        self.train_history = None       # init with train()
        self.train_label_true = None    # init with load_input()
        self.train_aux_data_mean_sd = (0,0)
        self.train_labels_real_mean_sd = (0,0)
        self.cpi_adjustments = np.array([0,0])
        self.norm_calib_labels_real = None
        self.has_label_cat = False
        self.has_label_real = False
        
        return
    
    # splits input into training, test, validation, and calibration
    def split_tensor_idx(self, num_sample):
        """
        Split tensor into parts.

        This function splits the indexes for training examples into training,
        validation, and calibration sets.

        Args:
            num_sample (int): The total number of samples in the dataset.

        Returns:
            train_idx (numpy.ndarray): The indices for the training subset.
            val_idx (numpy.ndarray): The indices for the validation subset.
            calib_idx (numpy.ndarray): The indices for the calibration subset.

        """

        # get number of training, validation, and calibration datapoints
        num_calib = int(np.floor(num_sample * self.prop_cal))
        num_val   = int(np.floor(num_sample * self.prop_val))
        num_train = num_sample - (num_val + num_calib)
        assert num_train > 0

        # create input subsets
        train_idx = np.arange(num_train, dtype='int')
        val_idx   = np.arange(num_val, dtype='int') + num_train
        calib_idx = np.arange(num_calib, dtype='int') + num_train + num_val

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

        # input dataset filenames for csv or hdf5
        path_prefix = f'{self.fmt_dir}/{self.fmt_prefix}.train'
        input_phy_data_fn = f'{path_prefix}.phy_data.csv'
        input_aux_data_fn = f'{path_prefix}.aux_data.csv'
        input_labels_fn = f'{path_prefix}.labels.csv'
        input_hdf5_fn = f'{path_prefix}.hdf5'
        
        # read phy. data, aux. data, and labels
        full_phy_data = None
        full_aux_data = None
        full_labels   = None
        if self.tensor_format == 'csv':
            full_phy_data = pd.read_csv(input_phy_data_fn, header=None,
                                        on_bad_lines='skip').to_numpy()
            full_aux_data = pd.read_csv(input_aux_data_fn, header=None,
                                        on_bad_lines='skip').to_numpy()
            full_labels   = pd.read_csv(input_labels_fn, header=None,
                                        on_bad_lines='skip').to_numpy()
            self.aux_data_names = full_aux_data[0,:]
            self.label_names    = full_labels[0,:]
            full_aux_data       = full_aux_data[1:,:].astype('float64')
            full_labels         = full_labels[1:,:].astype('float64')

        elif self.tensor_format == 'hdf5':
            hdf5_file = h5py.File(input_hdf5_fn, 'r')
            self.aux_data_names = [ s.decode() for s in hdf5_file['aux_data_names'][0,:] ]
            self.label_names    = [ s.decode() for s in hdf5_file['label_names'][0,:] ]
            full_phy_data       = pd.DataFrame(hdf5_file['phy_data']).to_numpy()
            full_aux_data       = pd.DataFrame(hdf5_file['aux_data']).to_numpy()
            full_labels         = pd.DataFrame(hdf5_file['labels']).to_numpy()
            hdf5_file.close()
            
        # separate labels for categorical param_est targets
        full_labels_real, full_labels_cat = self.separate_labels(full_labels)
        
        # data dimensions
        num_sample             = full_phy_data.shape[0]
        self.num_param_real    = full_labels_real.shape[1]
        self.num_param_cat     = full_labels_cat.shape[1]
        self.num_aux_data      = full_aux_data.shape[1]
        
        # shuffle datasets
        randomized_idx     = np.random.permutation(full_phy_data.shape[0])
        full_phy_data      = full_phy_data[randomized_idx,:]
        full_aux_data      = full_aux_data[randomized_idx,:]
        full_labels_real   = full_labels_real[randomized_idx,:]
        full_labels_cat    = full_labels_cat[randomized_idx,:]

        # reshape phylogenetic tensor data based on CPV+S
        full_phy_data.shape = (num_sample, -1, self.num_data_col)

        # split dataset into training, test, validation, and calibration parts
        train_idx, val_idx, calib_idx = self.split_tensor_idx(num_sample)
        self.validate_tensor_idx(train_idx, val_idx, calib_idx)
        
        # save original training input
        self.train_label_true = full_labels[train_idx,:]

        # normalize auxiliary data
        norm_train_aux_data, train_aux_data_means, train_aux_data_sd = util.normalize(full_aux_data[train_idx,:])
        self.train_aux_data_mean_sd = (train_aux_data_means, train_aux_data_sd)
        norm_val_aux_data = util.normalize(full_aux_data[val_idx,:],
                                           self.train_aux_data_mean_sd)
        norm_calib_aux_data = util.normalize(full_aux_data[calib_idx,:],
                                             self.train_aux_data_mean_sd)

        # normalize labels
        norm_train_labels_real, train_labels_real_means, train_labels_real_sd = util.normalize(full_labels_real[train_idx,:])
        self.train_labels_real_mean_sd = (train_labels_real_means, train_labels_real_sd)
        norm_val_labels_real = util.normalize(full_labels_real[val_idx,:],
                                              self.train_labels_real_mean_sd)
        self.norm_calib_labels_real = util.normalize(full_labels_real[calib_idx,:],
                                                     self.train_labels_real_mean_sd)

        # create phylogenetic data tensors
        train_phy_data_tensor = full_phy_data[train_idx,:,:]
        val_phy_data_tensor = full_phy_data[val_idx,:,:]
        self.calib_phy_data_tensor = full_phy_data[calib_idx,:,:]

        # create categorical label tensors
        train_labels_cat = full_labels_cat[train_idx,:]
        val_labels_cat = full_labels_cat[val_idx,:]
        calib_labels_cat = full_labels_cat[calib_idx,:]
        
        # torch datasets
        self.train_dataset = network.Dataset(train_phy_data_tensor,
                                             norm_train_aux_data,
                                             norm_train_labels_real,
                                             train_labels_cat)
        self.calib_dataset = network.Dataset(self.calib_phy_data_tensor,
                                             norm_calib_aux_data,
                                             self.norm_calib_labels_real,
                                             calib_labels_cat)
        self.val_dataset   = network.Dataset(val_phy_data_tensor,
                                             norm_val_aux_data,
                                             norm_val_labels_real,
                                             val_labels_cat)

        return
    
##################################################

    def separate_labels(self, labels):
        """Separates labels for categorical param_est targets.
        
        This function separates labels into real and categorical subsets
        based on the param_est dictionary.
        
        Args:
            labels (numpy.ndarray): The input labels.
            
        Returns:
            labels_real (numpy.ndarray): The real-valued labels.
            labels_cat (numpy.ndarray): The categorical labels.
        
        """

        idx_real = list()
        idx_cat = list()
        
        for k,v in self.param_est.items():
            if v == 'cat':
                self.has_label_cat = True
                idx = self.label_names.index(k)
                unique_cats, encoded_cats = np.unique(labels[:,idx],
                                                      return_inverse=True)
                self.param_cat[k] = len(unique_cats)
                labels[:,idx] = encoded_cats
                idx_cat.append( idx )
                self.param_cat_names.append(k)
                
            elif v == 'real':
                self.has_label_real = True
                idx_real.append( self.label_names.index(k) )
                self.param_real_names.append(k)
                
        # get data subsets
        labels_real = labels[:,idx_real].copy()
        labels_cat = labels[:,idx_cat].copy()

        # done
        return labels_real, labels_cat

##################################################

    def build_network(self):
        
        # torch multiprocessing, eventually need to get working with cuda
        torch.set_num_threads(self.num_proc)

        # build model architecture
        self.model = network.ParameterEstimationNetwork(phy_dat_width=self.num_data_col,
                                                        phy_dat_height=self.tree_width,
                                                        aux_dat_width=self.num_aux_data,
                                                        lbl_width=self.num_param_real,
                                                        param_cat=self.param_cat,
                                                        args=self.args)
        
        self.model.phy_dat_shape = (self.num_data_col, self.tree_width)
        self.model.aux_dat_shape = (self.num_aux_data,)

        # print(self.model)
        self.model.to(self.TORCH_DEVICE)

        return


##################################################

    def make_loss_real_func(self):
        """Makes loss function for real-valued labels.

        This function makes a loss function for real-valued labels based on the
        specified loss function in the phyddle settings.

        Returns:
            loss_func (torch.nn.Module): The loss function for real-valued labels.

        """
        if self.loss_real == 'mse':
            loss_func = torch.nn.MSELoss()
        elif self.loss_real == 'mae':
            loss_func = torch.nn.L1Loss()
        else:
            raise ValueError(f'Unknown loss function: {self.loss_real}')
        
        return loss_func

    def train(self):
        """Trains the neural network model.

        This function compiles the network model to prepare it for training.
        Perform training by compiling the model with appropriate loss functions
        and metrics, and then fitting the model to the training data. Training
        produces a history dictionary, which is saved.

        Returns:
            None

        """
        # training dataset
        train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset,
                                                   batch_size=self.trn_batch_size)
        num_batches = int(np.ceil(self.train_dataset.phy_data.shape[0] / self.trn_batch_size))

        # validation dataset
        val_phy_dat  = torch.Tensor(self.val_dataset.phy_data).to(self.TORCH_DEVICE)
        val_aux_dat  = torch.Tensor(self.val_dataset.aux_data).to(self.TORCH_DEVICE)
        val_lbl_real = torch.Tensor(self.val_dataset.labels_real).to(self.TORCH_DEVICE)
        val_lbl_cat  = torch.LongTensor(self.val_dataset.labels_cat).to(self.TORCH_DEVICE)
        val_bad_count = 0

        # model device
        # self.model.to(self.TORCH_DEVICE)

        # loss functions
        q_width = self.cpi_coverage
        q_tail  = (1.0 - q_width) / 2
        q_lower = q_tail
        q_upper = 1.0 - q_tail
        loss_value_func = self.make_loss_real_func()
        loss_lower_func = network.QuantileLoss(alpha=q_lower)
        loss_upper_func = network.QuantileLoss(alpha=q_upper)
        loss_categ_func = network.CrossEntropyLoss()

        # optimizer
        optimizer = torch.optim.Adam(self.model.parameters())
        # optimizer = torch.optim.Adam(self.model.parameters(),
        #                              lr=0.001,
        #                              weight_decay = 0.002)
        # optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001,
        #                               betas=(0.9, 0.999), eps=1e-08,
        #                               weight_decay=0.01, ams_grad=False)

        # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
        #                                             step_size = 50,
        #                                             gamma = 0.1)
        
        # TODO: simplify training logging!!
        # gather training history
        history_col_names = ['epoch', 'dataset', 'metric', 'value']
        self.train_history = pd.DataFrame(columns=history_col_names)

        # training
        metric_names = ['loss_lower', 'loss_upper', 'loss_value',
                        'loss_combined', 'mse_value', 'mae_value', 'mape_value']
        prev_trn_loss_combined = None
        prev_val_loss_combined = None
        for i in range(self.num_epochs):
            # print('-----')
            trn_loss_value = 0.
            trn_loss_lower = 0.
            trn_loss_upper = 0.
            trn_loss_combined = 0.
            trn_mse_value = 0.
            trn_mape_value = 0.
            trn_mae_value = 0.

            train_msg = f'Training epoch {i+1} of {self.num_epochs}'
            for j, (phy_dat, aux_dat, lbl_real, lbl_cat) in tqdm(enumerate(train_loader),
                                                                 total=num_batches,
                                                                 desc=train_msg,
                                                                 smoothing=0):
                
                # short cut batches for training
                # if j > 1:
                #     break
                
                # send labels to device
                phy_dat = phy_dat.to(self.TORCH_DEVICE)
                aux_dat = aux_dat.to(self.TORCH_DEVICE)
                lbl_real = lbl_real.to(self.TORCH_DEVICE)
                lbl_cat = lbl_cat.to(self.TORCH_DEVICE)
                
                # reset gradients for tensors
                optimizer.zero_grad()
                
                # forward pass of training data to estimate labels
                lbls_hat = self.model(phy_dat, aux_dat)

                # calculating the loss between original and predicted data points
                loss_list = list()
                if self.has_label_real:
                    loss_value = loss_value_func(lbls_hat[0], lbl_real)
                    loss_lower = loss_lower_func(lbls_hat[1], lbl_real)
                    loss_upper = loss_upper_func(lbls_hat[2], lbl_real)
                    loss_list += [ loss_value, loss_lower, loss_upper ]
                if self.has_label_cat:
                    loss_categ = loss_categ_func(lbls_hat[3], lbl_cat)
                    loss_list += [ loss_categ ]
                loss_combined = torch.stack(loss_list).sum()

                # collect history stats
                if self.has_label_real:
                    trn_loss_value    += loss_value.item() / num_batches
                    trn_loss_lower    += loss_lower.item() / num_batches
                    trn_loss_upper    += loss_upper.item() / num_batches
                    trn_mse_value     += ( torch.mean((lbl_real - lbls_hat[0])**2) ).item() / num_batches
                    trn_mae_value     += ( torch.mean(torch.abs(lbl_real - lbls_hat[0])) ).item() / num_batches
                    trn_mape_value    += ( torch.mean(torch.abs((lbl_real - lbls_hat[0]) / lbl_real)) ).item() / num_batches
                trn_loss_combined += loss_combined.item() / num_batches
                
                # backward pass to update gradients
                loss_combined.backward()

                # update network parameters
                optimizer.step()
                # lr_scheduler.step()
            
            train_metric_vals = [ trn_loss_lower, trn_loss_upper, trn_loss_value,
                                  trn_loss_combined, trn_mse_value,
                                  trn_mae_value, trn_mape_value ]

            # forward pass of validation to estimate labels
            val_lbls_hat       = self.model(val_phy_dat, val_aux_dat)
            
            # collect validation metrics
            val_loss_list = list()
            val_loss_value = 0.
            val_loss_lower = 0.
            val_loss_upper = 0.
            if self.has_label_real:
                val_loss_value = loss_value_func(val_lbls_hat[0], val_lbl_real).item()
                val_loss_lower = loss_lower_func(val_lbls_hat[1], val_lbl_real).item()
                val_loss_upper = loss_upper_func(val_lbls_hat[2], val_lbl_real).item()
                val_loss_list += [ val_loss_value, val_loss_lower, val_loss_upper ]
            # val_loss_combined  = val_loss_value + val_loss_lower + val_loss_upper
            if self.has_label_cat:
                val_loss_categ = loss_categ_func(val_lbls_hat[3], val_lbl_cat).item()
                val_loss_list += [ val_loss_categ ]
            
            val_loss_combined = sum(val_loss_list)
                
            val_mse_value = 0.
            val_mae_value = 0.
            val_mape_value = 0.
            if self.has_label_real:
                val_mse_value      = ( torch.mean((val_lbl_real - val_lbls_hat[0])**2) ).item()
                val_mae_value      = ( torch.mean(torch.abs(val_lbl_real - val_lbls_hat[0])) ).item()
                val_mape_value     = ( torch.mean(torch.abs((val_lbl_real - val_lbls_hat[0]) / val_lbl_real)) ).item()
                
            val_metric_vals = [ val_loss_value, val_loss_lower, val_loss_upper,
                                val_loss_combined, val_mse_value, val_mae_value,
                                val_mape_value ]

            # raw training metrics for epoch
            trn_loss_str = f'    Train        --   loss: {"{0:.4f}".format(trn_loss_combined)}'
            val_loss_str = f'    Validation   --   loss: {"{0:.4f}".format(val_loss_combined)}'
            
            # changes in training metrics between epochs
            if i > 0:
                diff_trn_loss = trn_loss_combined - prev_trn_loss_combined
                diff_val_loss = val_loss_combined - prev_val_loss_combined
                rat_trn_loss  = 100 * round(trn_loss_combined / prev_trn_loss_combined - 1.0, ndigits=4)
                rat_val_loss  = 100 * round(val_loss_combined / prev_val_loss_combined - 1.0, ndigits=4)
                
                diff_trn_loss_str = '{0:+.4f}'.format(diff_trn_loss)
                diff_val_loss_str = '{0:+.4f}'.format(diff_val_loss)
                rat_trn_loss_str  = '{0:+.2f}'.format(rat_trn_loss).rjust(4, ' ')
                rat_val_loss_str  = '{0:+.2f}'.format(rat_val_loss).rjust(4, ' ')
            
                trn_color = 31 if diff_trn_loss >= 0 else 32  # green or red
                val_color = 31 if diff_val_loss >= 0 else 32  # green or red
                trn_loss_change_str  = f'  abs: {util.phyddle_str(diff_trn_loss_str, style=0, color=trn_color)}'
                trn_loss_change_str += f'  rel: {util.phyddle_str(rat_trn_loss_str, style=0, color=trn_color)}%'
                val_loss_change_str  = f'  abs: {util.phyddle_str(diff_val_loss_str, style=0, color=val_color)}'
                val_loss_change_str += f'  rel: {util.phyddle_str(rat_val_loss_str, style=0, color=val_color)}%'

                trn_loss_str += trn_loss_change_str
                val_loss_str += val_loss_change_str

                if diff_val_loss >= 0:
                    val_bad_count += 1
                else:
                    val_bad_count = 0

                
            prev_trn_loss_combined = trn_loss_combined
            prev_val_loss_combined = val_loss_combined

            # display training metric progress
            print(trn_loss_str)
            print(val_loss_str)
            print('')

            # update train history log
            self.update_train_history(i, metric_names, train_metric_vals, 'train')
            self.update_train_history(i, metric_names, val_metric_vals, 'validation')

            # early stopping
            if val_bad_count >= self.num_early_stop:
                print(f'Early stop: validation loss increased for num_early_stop={self.num_early_stop} consecutive epochs')
                break

        # print(self.train_history)

        return
    
    def update_train_history(self, epoch, metric_names, metric_vals, dataset_name='train',):
        """Updates train history dataframe.
        
        This function appends new rows to the train history dataframe.
        
        Args:
            epoch (int): current epoch
            metric_names (list): names for metrics to be logged
            metric_vals (list): values for metrics to be logged
            dataset_name (str): name of dataset that is logged (e.g. train or validation)

        """

        assert len(metric_names) == len(metric_vals)
        
        for i,(j,k) in enumerate(zip(metric_names, metric_vals)):
            self.train_history.loc[len(self.train_history.index)] = [ epoch, dataset_name, j, k ]
        
        return

    def perform_cpi_calibration(self):
        """Performs CPI calibration.

        This function performs CPI calibration to estimate the CPI adjustment
        terms for the training dataset.

        """

        # temporarily switch to CPU
        self.model.to('cpu')

        # make initial CPI estimates
        num_calib_examples = self.calib_phy_data_tensor.shape[0]
        calib_loader = torch.utils.data.DataLoader(dataset=self.calib_dataset,
                                                   batch_size=num_calib_examples)
        calib_batch = next(iter(calib_loader))
        calib_phy_dat, calib_aux_dat = calib_batch[0], calib_batch[1]
        calib_phy_dat = calib_phy_dat.to('cpu')
        calib_aux_dat = calib_aux_dat.to('cpu')
        
        # get calib estimates
        calib_label_est = self.model(calib_phy_dat, calib_aux_dat)

        # make CPI adjustments
        norm_calib_label_real_est = torch.stack(calib_label_est[0:3]).cpu().detach().numpy()
        norm_calib_real_est_quantiles = norm_calib_label_real_est[1:,:,:]
        self.cpi_adjustments = self.get_cqr_constant(norm_calib_real_est_quantiles,
                                                     self.norm_calib_labels_real,
                                                     inner_quantile=self.cpi_coverage,
                                                     asymmetric=self.cpi_asymmetric)
        self.cpi_adjustments = np.array(self.cpi_adjustments).reshape((2,-1))

        # restore device
        self.model.to(self.TORCH_DEVICE)

        # done
        return
    
    def make_results(self):
        """Makes all results from the Train step.

        This function undoes all the transformation and rescaling for the
        input and output datasets.

        """
        
        # get uncalibrated estimates
        # training label estimates
        num_train_examples = 1000
        train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset,
                                                   batch_size=num_train_examples)
        train_batch = next(iter(train_loader))
        train_phy_dat, train_aux_dat, train_labels_real, train_labels_cat = train_batch
        train_phy_dat = train_phy_dat.to(self.TORCH_DEVICE)
        train_aux_dat = train_aux_dat.to(self.TORCH_DEVICE)
        train_labels_real = train_labels_real.to(self.TORCH_DEVICE)
        train_labels_cat = train_labels_cat.to(self.TORCH_DEVICE)

        # get train estimates
        label_est = self.model(train_phy_dat, train_aux_dat)
        
        # real vs. cat estimates
        labels_real_est = label_est[0:3]
        labels_real_est = torch.stack(labels_real_est).cpu().detach().numpy()
        labels_cat_est  = label_est[3]

        if self.has_label_real:
            train_labels_real = train_labels_real.cpu().detach().numpy()
            self.train_label_real_true = train_labels_real
            
            # uncalibrated training estimates of real labels
            self.train_label_real_est = util.denormalize(labels_real_est.copy(),
                                                         self.train_labels_real_mean_sd)

            # generate calibration factors
            self.perform_cpi_calibration()

            # calibrate original estimates
            labels_real_est_calib = labels_real_est.copy()
            labels_real_est_calib[1,:,:] = labels_real_est_calib[1,:,:] - self.cpi_adjustments[0,:]
            labels_real_est_calib[2,:,:] = labels_real_est_calib[2,:,:] + self.cpi_adjustments[1,:]
            
            # denormalize calibrated estimates
            self.train_label_real_est_calib = labels_real_est_calib

        # reformat categorical estimates, if they exist
        if self.has_label_cat:
            self.train_label_cat_true = train_labels_cat.cpu().detach().numpy().astype('int')
            self.train_label_cat_est = self.format_label_cat(labels_cat_est)
        
        return

    def format_label_cat(self, x):
        """Formats categorical labels.

        Formats categorical labels for training and validation datasets.

        """
        
        df_list = list()
        for k,v in x.items():
            v = torch.softmax(v, dim=1).cpu().detach().numpy()
            col_names = [ f'{k}_{i}' for i in range(v.shape[1]) ]
            df = pd.DataFrame(v, columns=col_names)
            df_list.append(df)
            
        return pd.concat(df_list, axis=1)

    def save_results(self):
        """Save training results.

        Saves all results from training procedure. Saved results include the
        trained network, the normalization parameters for training/calibration,
        CPI adjustment terms, and the training history.

        """
        max_idx = 1000

        path_prefix = f'{self.trn_dir}/{self.trn_prefix}'
        
        # output network model info
        model_arch_fn                = f'{path_prefix}.trained_model.pkl'
        model_history_fn             = f'{path_prefix}.train_history.csv'
        model_cpi_fn                 = f'{path_prefix}.cpi_adjustments.csv'
        # model_weights_fn           = f'{path_prefix}.train_weights.hdf5'
        
        # output scaling terms
        train_labels_real_norm_fn    = f'{path_prefix}.train_norm.labels_real.csv'
        train_aux_data_norm_fn       = f'{path_prefix}.train_norm.aux_data.csv'

        # output training labels
        train_label_real_true_fn     = f'{path_prefix}.train_true.labels_real.csv'
        train_label_real_est_fn      = f'{path_prefix}.train_est.labels_real.csv'
        train_label_est_nocalib_fn   = f'{path_prefix}.train_est.labels_real_nocalib.csv'
        train_label_cat_true_fn      = f'{path_prefix}.train_true.labels_cat.csv'
        train_label_cat_est_fn       = f'{path_prefix}.train_est.labels_cat.csv'
        
        # save model to file
        torch.save(self.model, model_arch_fn)

        # save json history from running MASTER
        self.train_history.to_csv(model_history_fn, index=False, sep=',',
                                  float_format=util.PANDAS_FLOAT_FMT_STR)

        # save aux_data names, means, sd for new test dataset normalization
        df_aux_data = pd.DataFrame({'name':self.aux_data_names,
                                    'mean':self.train_aux_data_mean_sd[0],
                                    'sd':self.train_aux_data_mean_sd[1]})
        df_aux_data.to_csv(train_aux_data_norm_fn, index=False, sep=',',
                           float_format=util.PANDAS_FLOAT_FMT_STR)
 
        if self.has_label_real:
            # save label names, means, sd for new test dataset normalization
            df_labels = pd.DataFrame({'name':self.param_real_names,
                                      'mean':self.train_labels_real_mean_sd[0],
                                      'sd':self.train_labels_real_mean_sd[1]})
            df_labels.to_csv(train_labels_real_norm_fn, index=False, sep=',',
                             float_format=util.PANDAS_FLOAT_FMT_STR)
    
            # save CPI intervals
            df_cpi_intervals = pd.DataFrame(self.cpi_adjustments,
                                            columns=self.param_real_names)
            df_cpi_intervals.to_csv(model_cpi_fn,
                                    index=False, sep=',',
                                    float_format=util.PANDAS_FLOAT_FMT_STR)
            
            # downsample all true training labels
            df_train_label_true = pd.DataFrame(self.train_label_real_true[0:max_idx,:],
                                               columns=self.param_real_names )
            
            # save true values for train real labels
            df_train_label_real_true = df_train_label_true[self.param_real_names]
            df_train_label_real_true.to_csv(train_label_real_true_fn,
                                            index=False, sep=',',
                                            float_format=util.PANDAS_FLOAT_FMT_STR)
            
            # save train real label estimates
            df_train_label_real_est_nocalib = util.make_param_VLU_mtx(self.train_label_real_est[0:max_idx,:],
                                                                      self.param_real_names )
            df_train_label_real_est_calib   = util.make_param_VLU_mtx(self.train_label_real_est_calib[0:max_idx,:],
                                                                      self.param_real_names )
    
            # convert to csv and save
            df_train_label_real_est_nocalib.to_csv(train_label_est_nocalib_fn,
                                                   index=False, sep=',',
                                                   float_format=util.PANDAS_FLOAT_FMT_STR)
            df_train_label_real_est_calib.to_csv(train_label_real_est_fn,
                                                 index=False, sep=',',
                                                 float_format=util.PANDAS_FLOAT_FMT_STR)
    
        if self.has_label_cat:
            # save true values for train categ. labels
            df_train_label_cat_true = pd.DataFrame(self.train_label_cat_true[0:max_idx,:],
                                                   columns=self.param_cat_names )
            df_train_label_cat_true.to_csv(train_label_cat_true_fn,
                                           index=False, sep=',')
    
            # save train categorical label estimates
            self.train_label_cat_est.to_csv(train_label_cat_est_fn,
                                            index=False, sep=',',
                                            float_format=util.PANDAS_FLOAT_FMT_STR)

        return

##################################################
        
    def get_cqr_constant(self, ests, true, inner_quantile=0.95, asymmetric=True):
        """Computes the conformalized quantile regression (CQR) constants.
        
        This function computes symmetric or asymmetric CQR constants for the
        specified inner-quantile range.

        Notes:
            # ests axis 0 is the lower and upper quants,
            # axis 1 is the replicates, and axis 2 is the params

        Arguments:
            ests (array-like): The input data.
            true (array-like): The target data.
            inner_quantile (float): The inner quantile range.
            asymmetric (bool): If True, computes asymmetric CQR constants.

        Returns:
            array-like: The conformity scores.

        """
        
        # compute non-comformity scores
        q_score = np.empty((2, ests.shape[2]))
        
        for i in range(ests.shape[2]):
            if asymmetric:
                # Asymmetric non-comformity score
                lower_s = np.array(true[:,i] - ests[0][:,i])
                upper_s = np.array(true[:,i] - ests[1][:,i])
                lower_p = (1 - inner_quantile)/2 * (1 + 1/ests.shape[1])
                upper_p = (1 + inner_quantile)/2 * (1 + 1/ests.shape[1])
                if lower_p < 0.:
                    self.logger.write_log('trn',
                                          'get_cqr_constant: lower_p >= 0.')
                    lower_p = 0.
                if upper_p > 1.:
                    self.logger.write_log('trn',
                                          'get_cqr_constant: upper_p <= 1.')
                    upper_p = 1.
                lower_q = np.quantile(lower_s, lower_p)
                upper_q = np.quantile(upper_s, upper_p)
            else:
                # Symmetric non-comformity score
                s = np.amax(np.array((ests[0][:,i]-true[:,i], true[:,i]-ests[1][:,i])), axis=0)
                # get adjustment constant: 1 - alpha/2's quantile of non-comformity scores
                symm_p = inner_quantile * (1 + 1/ests.shape[1])
                if symm_p < 0.:
                    self.logger.write_log('trn',
                                          'get_cqr_constant: symm_p >= 0.')
                    symm_p = 0.
                elif symm_p > 1.:
                    self.logger.write_log('trn',
                                          'get_cqr_constant: symm_p <= 1.')
                    symm_p = 1.
                lower_q = np.quantile(s, symm_p)
                upper_q = lower_q
                # Q[:,i] = np.array([lower_q, upper_q])

            q_score[:,i] = np.array([lower_q, upper_q])
                                
        return q_score
    
##################################################
