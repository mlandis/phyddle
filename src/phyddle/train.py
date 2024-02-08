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
        num_sample        = full_phy_data.shape[0]
        self.num_params   = full_labels.shape[1]
        self.num_aux_data = full_aux_data.shape[1]

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
        # TODO: probably not, but does this need to be (see below) ... ?
        # full_phy_data.shape = (num_sample, self.num_data_col, -1)
        full_phy_data.shape = (num_sample, -1, self.num_data_col)

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
        self.norm_val_aux_data      = util.normalize(full_aux_data[val_idx,:],
                                                     self.train_aux_data_mean_sd)
        self.norm_calib_aux_data    = util.normalize(full_aux_data[calib_idx,:],
                                                     self.train_aux_data_mean_sd)

        # normalize labels
        self.norm_train_labels, train_label_means, train_label_sd = util.normalize(full_labels[train_idx,:])
        self.train_labels_mean_sd = (train_label_means, train_label_sd)
        self.norm_val_labels      = util.normalize(full_labels[val_idx,:],
                                                   self.train_labels_mean_sd)
        self.norm_calib_labels    = util.normalize(full_labels[calib_idx,:],
                                                   self.train_labels_mean_sd)

        # create phylogenetic data tensors
        self.train_phy_data_tensor = full_phy_data[train_idx,:,:]
        self.val_phy_data_tensor   = full_phy_data[val_idx,:,:]
        self.calib_phy_data_tensor = full_phy_data[calib_idx,:,:]

        # create auxiliary data tensors (with scaling)
        self.train_aux_data_tensor = self.norm_train_aux_data
        self.val_aux_data_tensor   = self.norm_val_aux_data
        self.calib_aux_data_tensor = self.norm_calib_aux_data

        # torch datasets
        self.train_dataset = network.Dataset(self.train_phy_data_tensor,
                                             self.norm_train_aux_data,
                                             self.norm_train_labels)
        self.calib_dataset = network.Dataset(self.calib_phy_data_tensor,
                                             self.norm_calib_aux_data,
                                             self.norm_calib_labels)
        self.val_dataset   = network.Dataset(self.val_phy_data_tensor,
                                             self.norm_val_aux_data,
                                             self.norm_val_labels)

        return
    
#------------------------------------------------------------------------------#

    def build_network(self):
        
        # torch multiprocessing, eventually need to get working with cuda
        torch.set_num_threads(self.num_proc)

        # build model architecture
        self.model = network.ParameterEstimationNetwork(phy_dat_width=self.num_data_col,
                                                        phy_dat_height=self.tree_width,
                                                        aux_dat_width=self.num_aux_data,
                                                        lbl_width=self.num_params,
                                                        args=self.args)
        
        self.model.phy_dat_shape = (self.num_data_col, self.tree_width)
        self.model.aux_dat_shape = (self.num_aux_data,)

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
        # training dataset
        trainloader = torch.utils.data.DataLoader(dataset = self.train_dataset,
                                                  batch_size = self.trn_batch_size)
        num_batches = int(np.ceil(self.train_dataset.phy_data.shape[0] / self.trn_batch_size))

        # validation dataset
        val_phy_dat = torch.Tensor(self.val_dataset.phy_data)
        val_aux_dat = torch.Tensor(self.val_dataset.aux_data)
        val_lbls    = torch.Tensor(self.val_dataset.labels)

        # loss functions
        q_width = self.cpi_coverage
        q_tail  = (1.0 - q_width) / 2
        q_lower = q_tail
        q_upper = 1.0 - q_tail
        loss_value_func = torch.nn.MSELoss()
        loss_lower_func = network.QuantileLoss(alpha=q_lower)
        loss_upper_func = network.QuantileLoss(alpha=q_upper)

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
            
            #print('-----')
            trn_loss_value = 0.
            trn_loss_lower = 0.
            trn_loss_upper = 0.
            trn_loss_combined = 0.
            trn_mse_value = 0.
            trn_mape_value = 0.
            trn_mae_value = 0.

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
                trn_loss_value    += loss_value.item() / num_batches
                trn_loss_lower    += loss_lower.item() / num_batches
                trn_loss_upper    += loss_upper.item() / num_batches
                trn_loss_combined += loss_combined.item() / num_batches
                trn_mse_value     += ( torch.mean((lbls - lbls_hat[0])**2) ).item() / num_batches
                trn_mae_value     += ( torch.mean(torch.abs(lbls - lbls_hat[0])) ).item() / num_batches
                trn_mape_value    += ( torch.mean(torch.abs((lbls - lbls_hat[0]) / lbls)) ).item() / num_batches
                
                # backward pass to update gradients
                loss_combined.backward()

                # update network parameters
                optimizer.step()
                #lr_scheduler.step()
            
            train_metric_vals = [ trn_loss_lower, trn_loss_upper, trn_loss_value,
                                  trn_loss_combined, trn_mse_value,
                                  trn_mae_value, trn_mape_value ]

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
            val_metric_vals = [ val_loss_value, val_loss_lower, val_loss_upper,
                                val_loss_combined, val_mse_value, val_mae_value,
                                val_mape_value ]

            # self.train_history.loc[len(self.train_history.index)] = [i, 'train', 'loss_lower',     trn_loss_lower]
            # self.train_history.loc[len(self.train_history.index)] = [i, 'train', 'loss_upper',     trn_loss_upper]
            # self.train_history.loc[len(self.train_history.index)] = [i, 'train', 'loss_value',     trn_loss_value]
            # self.train_history.loc[len(self.train_history.index)] = [i, 'train', 'loss_combined',  trn_loss_combined]
            # self.train_history.loc[len(self.train_history.index)] = [i, 'train', 'mse_value',      trn_mse_value]
            # self.train_history.loc[len(self.train_history.index)] = [i, 'train', 'mae_value',      trn_mae_value]
            # self.train_history.loc[len(self.train_history.index)] = [i, 'train', 'mape_value',     trn_mape_value]            
            # self.train_history.loc[len(self.train_history.index)] = [i, 'validation', 'loss_value',     val_loss_value]
            # self.train_history.loc[len(self.train_history.index)] = [i, 'validation', 'loss_lower',     val_loss_lower]
            # self.train_history.loc[len(self.train_history.index)] = [i, 'validation', 'loss_upper',     val_loss_upper]
            # self.train_history.loc[len(self.train_history.index)] = [i, 'validation', 'loss_combined',  val_loss_combined ]
            # self.train_history.loc[len(self.train_history.index)] = [i, 'validation', 'mse_value',      val_mse_value]
            # self.train_history.loc[len(self.train_history.index)] = [i, 'validation', 'mae_value',      val_mae_value]
            # self.train_history.loc[len(self.train_history.index)] = [i, 'validation', 'mape_value',     val_mape_value]

            # raw training metrics for epoch
            trn_loss_str = f'    Train        --   loss: {"{0:.4f}".format(trn_loss_combined)}'
            val_loss_str = f'    Validation   --   loss: {"{0:.4f}".format(val_loss_combined)}'
            
            # changes in training metrics between epochs
            if i > 0:
                diff_trn_loss = trn_loss_combined - prev_trn_loss_combined
                diff_val_loss = val_loss_combined - prev_val_loss_combined
                rat_trn_loss  = 100 * round(trn_loss_combined / prev_trn_loss_combined - 1.0, ndigits=3)
                rat_val_loss  = 100 * round(val_loss_combined / prev_val_loss_combined - 1.0, ndigits=3)
                
                diff_trn_loss_str = '{0:+.4f}'.format(diff_trn_loss)
                diff_val_loss_str = '{0:+.4f}'.format(diff_val_loss)
                rat_trn_loss_str  = '{0:+.2f}'.format(rat_trn_loss).rjust(4, ' ')
                rat_val_loss_str  = '{0:+.2f}'.format(rat_val_loss).rjust(4, ' ')
            
                trn_loss_str += f'  abs: {diff_trn_loss_str}  rel: {rat_trn_loss_str}%'
                val_loss_str += f'  abs: {diff_val_loss_str}  rel: {rat_val_loss_str}%'

            prev_trn_loss_combined = trn_loss_combined
            prev_val_loss_combined = val_loss_combined

            # display training metric progress
            print(trn_loss_str)
            print(val_loss_str)
            print('')

            # update train history log
            self.update_train_history(metric_names, train_metric_vals, 'train')
            self.update_train_history(metric_names, val_metric_vals, 'validation')

        # print(self.train_history)
        return

    def update_train_history(self, metric_names, metric_vals, dataset_name='train',):
        """Updates train history dataframe.
        
        This function appends new rows to the train history dataframe.
        
        Args:
            metric_names (list): names for metrics to be logged
            metric_vals (list): values for metrics to be logged
            dataset_name (str): name of dataset that is logged (e.g. train or validation)

        """

        assert(len(metric_names) == len(metric_vals))
        
        for i,(j,k) in enumerate(zip(metric_names, metric_vals)):
            self.train_history.loc[len(self.train_history.index)] = [ i, dataset_name, j, k ]
        
        return

    def make_results(self):
        """Makes all results from the Train step.

        This function undoes all the transformation and rescaling for the 
        input and output datasets.

        """
        
        # training label estimates
        trainloader = torch.utils.data.DataLoader(dataset=self.train_dataset,
                                                  batch_size = 1000)
        train_phy_dat, train_aux_dat, train_lbl = next(iter(trainloader))
        norm_train_label_est = self.model(train_phy_dat, train_aux_dat)
        
        # we want an array of 3 outputs [point, lower, upper], N examples, K parameters
        norm_train_label_est = torch.stack(norm_train_label_est).detach().numpy()
        self.train_label_est = util.denormalize(norm_train_label_est,
                                                self.train_labels_mean_sd,
                                                exp=True) - self.log_offset
 
        # make initial CPI estimates
        calibloader = torch.utils.data.DataLoader(dataset = self.calib_dataset,
                                                  batch_size = self.calib_phy_data_tensor.shape[0])
        calib_phy_dat, calib_aux_dat, calib_lbl = next(iter(calibloader))
        norm_calib_label_est = self.model(calib_phy_dat, calib_aux_dat)

        # make CPI adjustments
        norm_calib_label_est = torch.stack(norm_calib_label_est).detach().numpy()
        norm_calib_est_quantiles = norm_calib_label_est[1:,:,:]
        self.cpi_adjustments = self.get_CQR_constant(norm_calib_est_quantiles,
                                                     self.norm_calib_labels,
                                                     inner_quantile=self.cpi_coverage,
                                                     asymmetric=self.cpi_asymmetric)
        self.cpi_adjustments = np.array(self.cpi_adjustments).reshape((2,-1))
    
        # make final CPI estimates
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
        
        # format str
        float_fmt_str = '%.4e' #'{{:0.{:d}e}}'.format(util.OUTPUT_PRECISION)

        # save model to file
        torch.save(self.model, self.model_arch_fn)

        # save json history from running MASTER
        self.train_history.to_csv(self.model_history_fn, index=False, sep=',', float_format=util.PANDAS_FLOAT_FMT_STR)


        # save aux_data names, means, sd for new test dataset normalization
        df_aux_data = pd.DataFrame({'name':self.aux_data_names,
                                    'mean':self.train_aux_data_mean_sd[0],
                                    'sd':self.train_aux_data_mean_sd[1]})
        #df_aux_data.columns = ['name', 'mean', 'sd']
        df_aux_data.to_csv(self.train_aux_data_norm_fn, index=False, sep=',', float_format=util.PANDAS_FLOAT_FMT_STR)
 
        # save label names, means, sd for new test dataset normalization
        df_labels = pd.DataFrame({'name':self.label_names,
                                  'mean':self.train_labels_mean_sd[0],
                                  'sd':self.train_labels_mean_sd[1]})
        #df_labels.columns = ['name', 'mean', 'sd']
        df_labels.to_csv(self.train_labels_norm_fn, index=False, sep=',', float_format=util.PANDAS_FLOAT_FMT_STR)

        # save train/test scatterplot results (Value, Lower, Upper)
        df_train_label_est_nocalib = util.make_param_VLU_mtx(self.train_label_est[0:max_idx,:], self.label_names )
        df_train_label_est_calib   = util.make_param_VLU_mtx(self.train_label_est_calib[0:max_idx,:], self.label_names )
        
        # save train/test labels
        df_train_label_true = pd.DataFrame(self.train_label_true[0:max_idx,:], columns=self.label_names )
        
        # save CPI intervals
        df_cpi_intervals = pd.DataFrame( self.cpi_adjustments, columns=self.label_names )

        # convert to csv and save
        df_train_label_est_nocalib.to_csv(self.train_label_est_nocalib_fn, index=False, sep=',', float_format=util.PANDAS_FLOAT_FMT_STR)
        df_train_label_est_calib.to_csv(self.train_label_est_calib_fn, index=False, sep=',', float_format=util.PANDAS_FLOAT_FMT_STR)
        df_train_label_true.to_csv(self.train_label_true_fn, index=False, sep=',', float_format=util.PANDAS_FLOAT_FMT_STR)
        df_cpi_intervals.to_csv(self.model_cpi_fn, index=False, sep=',', float_format=util.PANDAS_FLOAT_FMT_STR)

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
