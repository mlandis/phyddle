#!/usr/bin/env python
"""
Learning
========
Defines classes and methods for the Learning step, which builds and trains a
network using the tensor data from the Formatting step.

Authors:   Michael Landis and Ammon Thompson
Copyright: (c) 2022-2023, Michael Landis and Ammon Thompson
License:   MIT
"""


# standard imports
import json
import os

# external imports
import h5py
import numpy as np
import pandas as pd
from keras import Input,Model
from keras import layers
from keras import backend as K

# phyddle imports
from phyddle import utilities

#-----------------------------------------------------------------------------------------------------------------#

def load(args):
    """Load the learner based on the given arguments.

    Args: args (dict): A dictionary containing the arguments.

    Returns: CnnLearner: An instance of the CnnLearner class if the learn_method argument is set to 'param_est'. None: If the learn_method argument is set to any value other than 'param_est'. NotImplementedError: If the learn_method argument is set to 'model_test', as this method is not yet implemented.

    """
    learn_method = args['learn_method']
    if learn_method == 'param_est':
        return CnnLearner(args)
    elif learn_method == 'model_test':
        raise NotImplementedError
    else:
        return None

#-----------------------------------------------------------------------------------------------------------------#

class Learner:
    """
    This class represents a Learner object that is used for training and testing models.

    Args:
        args (dict): A dictionary containing various arguments and settings for the Learner.

    """

    def __init__(self, args):
        """
        Initializes a new Learner object.

        Args:
            args (dict): A dictionary containing various arguments and settings for the Learner.
        """
        self.set_args(args)
        self.prepare_files()
        self.logger = utilities.Logger(args)
        return
    
    def set_args(self, args):
        """
        Sets the arguments for the Learner.

        Args:
            args (dict): A dictionary containing various arguments and settings for the Learner.
        """
        self.args              = args
        self.num_test          = 0
        self.num_validation    = 0
        self.prop_test         = 0
        self.prop_test         = 0
        self.proj              = args['proj']
        self.verbose           = args['verbose']
        self.tree_width        = args['tree_width']
        self.tree_type         = args['tree_type']
        self.tree_encode_type  = args['tree_encode_type']
        self.char_encode_type  = args['char_encode_type']
        self.tensor_format     = args['tensor_format']
        self.num_char          = args['num_char']
        self.num_states        = args['num_states']
        self.fmt_dir           = args['fmt_dir']
        self.plt_dir           = args['plt_dir']
        self.net_dir           = args['lrn_dir']
        self.learn_method      = args['learn_method']
        if self.learn_method == 'model_test':
            self.test_models = args['test_models']
        self.batch_size        = args['batch_size']
        self.num_epochs        = args['num_epochs']    
        if 'num_test' in args and 'num_validation' in args:
            self.num_test          = args['num_test']
            self.num_validation    = args['num_validation']
            self.num_calibration   = args['num_calibration']
        elif 'prop_test' in args and 'prop_validation' in args:
            self.prop_test         = args['prop_test']
            self.prop_validation   = args['prop_validation']
            self.prop_calibration  = args['prop_calibration']
        self.cpi_coverage          = args['cpi_coverage']
        self.cpi_asymmetric          = args['cpi_asymmetric']
        self.loss              = args['loss']
        self.optimizer         = args['optimizer']
        self.metrics           = args['metrics']
        self.kernel_init       = 'glorot_uniform'

        self.num_tree_row = utilities.get_num_tree_row(self.tree_type, self.tree_encode_type)
        self.num_char_row = utilities.get_num_char_row(self.char_encode_type, self.num_char, self.num_states)
        self.num_data_row = self.num_tree_row + self.num_char_row

        return
    
    def prepare_files(self):

        """Prepare files for the project.

        This function prepares the necessary directories and filenames for the
        project. It creates the main directories for input, plots, and network
        files. It also creates new job directories within the plot and network
        directories if they do not already exist.

        The function then initializes various filenames based on the given
        parameters. These filenames include the model prefix, model CSV file,
        model save file, normalized training label and summary statistics files,
        training history file, CPI adjustments file, and various prediction
        and label files for training and testing data. It also initializes
        filenames for input statistics, labels, and data files based on the
        tree width and tree type.

        Returns: None """

        # main directories
        self.input_dir   = self.fmt_dir + '/' + self.proj
        self.plot_dir    = self.plt_dir + '/' + self.proj
        self.network_dir = self.net_dir + '/' + self.proj

        # create new job directories
        os.makedirs(self.plot_dir, exist_ok=True)
        os.makedirs(self.network_dir, exist_ok=True)

        # main job filenames
        self.model_prefix       = f'sim_batchsize{self.batch_size}_numepoch{self.num_epochs}_nt{self.tree_width}'
        self.model_csv_fn       = f'{self.network_dir}/{self.model_prefix}.csv'
        self.model_sav_fn       = f'{self.network_dir}/{self.model_prefix}.hdf5'
        self.model_trn_lbl_norm_fn  = f'{self.network_dir}/{self.model_prefix}.train_label_norm.csv'
        self.model_trn_ss_norm_fn   = f'{self.network_dir}/{self.model_prefix}.train_summ_stat_norm.csv'
        self.model_hist_fn      = f'{self.network_dir}/{self.model_prefix}.train_history.json'
        self.model_cpi_fn       = f'{self.network_dir}/{self.model_prefix}.cpi_adjustments.csv'
        self.train_pred_calib_fn      = f'{self.network_dir}/{self.model_prefix}.train_pred.csv'
        self.test_pred_calib_fn       = f'{self.network_dir}/{self.model_prefix}.test_pred.csv'
        self.train_pred_nocalib_fn    = f'{self.network_dir}/{self.model_prefix}.train_pred_nocalib.csv'
        self.test_pred_nocalib_fn     = f'{self.network_dir}/{self.model_prefix}.test_pred_nocalib.csv'
        self.train_labels_fn    = f'{self.network_dir}/{self.model_prefix}.train_labels.csv'
        self.test_labels_fn     = f'{self.network_dir}/{self.model_prefix}.test_labels.csv'
        self.input_stats_fn     = f'{self.input_dir}/sim.nt{self.tree_width}.summ_stat.csv'
        self.input_labels_fn    = f'{self.input_dir}/sim.nt{self.tree_width}.labels.csv'
        if self.tree_type == 'extant':
            self.input_data_fn  = f'{self.input_dir}/sim.nt{self.tree_width}.cdvs.data.csv'
        elif self.tree_type == 'serial':
            self.input_data_fn  = f'{self.input_dir}/sim.nt{self.tree_width}.cblvs.data.csv' 
        self.input_hdf5_fn      = f'{self.input_dir}/sim.nt{self.tree_width}.hdf5'

        return

    def run(self):
        """
        Runs the entire phyddle process.

        This function executes the following steps:
        1. Prints information about the project and network directories if the verbose mode is enabled.
        2. Loads the input data.
        3. Builds the neural network.
        4. Trains the network.
        5. Processes the results.
        6. Saves the results.
        7. Prints "Done!" if the verbose mode is enabled.
        """
        if self.verbose: print(utilities.phyddle_info('lrn', self.proj, [self.fmt_dir], self.net_dir))

        if self.verbose: print(utilities.phyddle_str('▪ loading input ...'))
        self.load_input()
       
        if self.verbose: print(utilities.phyddle_str('▪ building network ...'))
        self.build_network()

        if self.verbose: print(utilities.phyddle_str('▪ training network ...'))
        self.train()

        if self.verbose: print(utilities.phyddle_str('▪ processing results ...'))
        self.make_results()

        if self.verbose: print(utilities.phyddle_str('▪ saving results ...'))
        self.save_results()

        if self.verbose: print(utilities.phyddle_str('▪ done!'))
    
    def load_input(self):
        """
        Loads the input data for the network.

        Raises:
            NotImplementedError: If the method is not implemented by the subclass.
        """
        raise NotImplementedError


    def build_network(self):
        """
        Builds the network architecture.

        Raises:
            NotImplementedError: If the method is not implemented by the subclass.
        """
        raise NotImplementedError


    def train(self):
        """
        Trains the network using the loaded input data.

        Raises:
            NotImplementedError: If the method is not implemented by the subclass.
        """
        raise NotImplementedError


    def make_results(self):
        """
        Generates the results using the trained network.

        Raises:
            NotImplementedError: If the method is not implemented by the subclass.
        """
        raise NotImplementedError


    def save_results(self):
        """
        Saves the generated results to a file or storage.

        Raises:
            NotImplementedError: If the method is not implemented by the subclass.
        """
        raise NotImplementedError

#-----------------------------------------------------------------------------------------------------------------#

class CnnLearner(Learner):
    """A class representing a CNN learner.

    Args:
        args (any): The arguments to initialize the CNN learner.

    Attributes:
        args (any): The arguments passed to the constructor.

    Inherits:
        Learner: Inherits from the Learner class.
    """
    def __init__(self, args):
        super().__init__(args)
        return
    
    # splits input into training, test, validation, and calibration
    def split_tensor_idx(self, num_sample):
        """
        Splits the input into training, test, validation, and calibration subsets.

        Args:
            num_sample (int): The total number of samples in the dataset.

        Returns:
            train_idx (numpy.ndarray): The indices for the training subset.
            val_idx (numpy.ndarray): The indices for the validation subset.
            test_idx (numpy.ndarray): The indices for the test subset.
            calib_idx (numpy.ndarray): The indices for the calibration subset.
        """

        # split dataset into training, test, and validation parts
        if self.num_test != 0 and self.num_validation != 0 and self.num_calibration != 0:
            num_val   = self.num_validation
            num_test  = self.num_test
            num_calib = self.num_calib

        elif self.prop_test != 0 and self.prop_validation != 0 and self.prop_calibration != 0:
            num_calib = int(np.floor(num_sample * self.prop_calibration)) 
            num_val   = int(np.floor(num_sample * self.prop_validation))
            num_test  = int(np.floor(num_sample * self.prop_test))
        
        # all unclaimed datapoints are used for training
        num_train = num_sample - (num_val + num_test + num_calib)
        assert(num_train > 0)
        # if num_train < 0:
        #     raise ValueError

        # create input subsets
        train_idx = np.arange(num_train, dtype='int')
        val_idx   = np.arange(num_val,   dtype='int') + num_train
        test_idx  = np.arange(num_test,  dtype='int') + num_train + num_val
        calib_idx = np.arange(num_calib, dtype='int') + num_train + num_val + num_test

        # maybe save these to file?
        return train_idx, val_idx, test_idx, calib_idx
    
    def validate_tensor_idx(self, train_idx, val_idx, test_idx, calib_idx):
        """
        Validates the sizes of the training, validation, test, and calibration datasets.

        Args:
            train_idx (list): The index of the training dataset.
            val_idx (list): The index of the validation dataset.
            test_idx (list): The index of the test dataset.
            calib_idx (list): The index of the calibration dataset.

        Returns:
            ValueError if any of the datasets are empty. Otherwise, returns None.
        """
        msg = ''
        if len(train_idx) == 0:
            msg = 'Training dataset is empty: len(train_idx) == 0'
        elif len(val_idx) == 0:
            msg = 'Validation dataset is empty: len(val_idx) == 0'
        elif len(test_idx) == 0:
            msg = 'Test dataset is empty: len(test_idx) == 0'
        elif len(calib_idx) == 0:
            msg = 'Calibration dataset is empty: len(calib_idx) == 0'
                
        if msg != '':
            self.logger.write_log('lrn', msg)
            raise ValueError(msg)

        return

    def load_input(self):

        """ Load input data for the model.

        This function loads input data based on the specified tensor format
        (csv or hdf5). It performs necessary preprocessing steps such as reading
        data from files, reshaping tensors, normalizing summary stats and
        labels, randomizing data, and splitting the dataset into training,
        validation, test, and calibration parts.

        Parameters:
            self: The instance of the class calling this function.

        Returns: None

        Raises: ValueError: If the tensor format is 'csv', indicating that the csv
            tensors need to be reshaped.
        """
        if self.tensor_format == 'csv':
            # read data
            full_data   = pd.read_csv(self.input_data_fn, header=None, on_bad_lines='skip').to_numpy()
            full_stats  = pd.read_csv(self.input_stats_fn, header=None, on_bad_lines='skip').to_numpy()
            full_labels = pd.read_csv(self.input_labels_fn, header=None, on_bad_lines='skip').to_numpy()
            # get summary stats
            self.stat_names = full_stats[0,:]
            full_stats = full_stats[1:,:].astype('float64')
            # get parameters and labels
            self.param_names = full_labels[0,:]
            full_labels = full_labels[1:,:].astype('float64')   
            #raise ValueError('csv tensors needs to be reshaped, oops!')

        elif self.tensor_format == 'hdf5':
            hdf5_file = h5py.File(self.input_hdf5_fn, 'r')
            self.stat_names = [ s.decode() for s in hdf5_file['summ_stat_names'][0,:] ]
            self.param_names = [ s.decode() for s in hdf5_file['label_names'][0,:] ]
            full_data = pd.DataFrame( hdf5_file['data'] ).to_numpy()
            #full_data = pd.DataFrame( hdf5_file['data'] ).to_numpy()
            full_stats = pd.DataFrame( hdf5_file['summ_stat'] ).to_numpy()
            full_labels = pd.DataFrame( hdf5_file['labels'] ).to_numpy()
            hdf5_file.close()

        # data dimensions
        num_sample = full_data.shape[0]
        self.num_params = full_labels.shape[1]
        self.num_stats = full_stats.shape[1]

        # take logs of labels (rates)
        # for variance stabilization for heteroskedastic (variance grows with mean)
        full_labels = np.log(full_labels)

        # randomize data to ensure iid of batches
        # do not want to use exact same datapoints when iteratively improving
        # training/validation accuracy
        # ... could move randomization into split_tensor_idx(), but it's slightly easier
        # ... to debug randomization issues before splitting
        randomized_idx = np.random.permutation(full_data.shape[0])
        #print(full_data.shape)
        full_data      = full_data[randomized_idx,:]
        full_stats     = full_stats[randomized_idx,:]
        full_labels    = full_labels[randomized_idx,:]

        # reshape phylogenetic tensor data based on CBLVS/CDVS format
        full_data.shape = ( num_sample, -1, (self.num_tree_row+self.num_char_row) )

        # split dataset into training, test, validation, and calibration parts
        train_idx, val_idx, test_idx, calib_idx = self.split_tensor_idx(num_sample)
        self.validate_tensor_idx(train_idx, val_idx, test_idx, calib_idx)

        # normalize summary stats
        self.denormalized_train_stats = full_stats[train_idx,:]
        self.norm_train_stats, self.train_stats_means, self.train_stats_sd = utilities.normalize( full_stats[train_idx,:] )
        self.norm_val_stats   = utilities.normalize(full_stats[val_idx,:], (self.train_stats_means, self.train_stats_sd))
        self.norm_test_stats  = utilities.normalize(full_stats[test_idx,:], (self.train_stats_means, self.train_stats_sd))
        self.norm_calib_stats = utilities.normalize(full_stats[calib_idx,:], (self.train_stats_means, self.train_stats_sd))

        # (option for diff schemes) try normalizing against 0 to 1
        self.denormalized_train_labels = full_labels[train_idx,:]
        self.norm_train_labels, self.train_label_means, self.train_label_sd = utilities.normalize( full_labels[train_idx,:] )
        self.norm_val_labels   = utilities.normalize(full_labels[val_idx,:], (self.train_label_means, self.train_label_sd))
        self.norm_test_labels  = utilities.normalize(full_labels[test_idx,:], (self.train_label_means, self.train_label_sd))
        self.norm_calib_labels = utilities.normalize(full_labels[calib_idx,:], (self.train_label_means, self.train_label_sd))

        # create data tensors
        self.train_data_tensor = full_data[train_idx,:]
        self.val_data_tensor   = full_data[val_idx,:]
        self.test_data_tensor  = full_data[test_idx,:]
        self.calib_data_tensor = full_data[calib_idx,:]

        # summary stats
        self.train_stats_tensor = self.norm_train_stats
        self.val_stats_tensor   = self.norm_val_stats
        self.test_stats_tensor  = self.norm_test_stats
        self.calib_stats_tensor = self.norm_calib_stats

        return
    
    def build_network(self):
        """Builds the network architecture.

        This function constructs the network architecture by assembling the input layers, phylo layers, aux layers, and output layers. It then instantiates the model using the assembled layers.

        
        Simplified network architecture:
        
                              ,--> Conv1D-normal + Pool --. 
        Phylo. Data Tensor --+---> Conv1D-stride + Pool ---\                          ,--> Point estimate
                              '--> Conv1D-dilate + Pool ----+--> Concat + Output(s)--+---> Lower quantile
                                                           /                          '--> Upper quantile
        Aux. Data Tensor   ------> Dense -----------------'
        

        Returns:
            None

        Raises:
            None
        """

        #input layers
        input_layers    = self.build_network_input_layers()
        phylo_layers    = self.build_network_phylo_layers(input_layers['phylo'])
        aux_layers      = self.build_network_aux_layers(input_layers['aux'])
        output_layers   = self.build_network_output_layers(phylo_layers, aux_layers)
    
        # instantiate model
        self.mymodel = Model(inputs = [input_layers['phylo'], input_layers['aux']], 
                        outputs = output_layers)
        
    def build_network_input_layers(self):
        """
        Build the input layers for the network.

        Returns:
            dict: A dictionary containing the input layers for phylo and aux data.
        """
        input_phylo_tensor = Input(shape = self.train_data_tensor.shape[1:3],  name='input_phylo')
        input_aux_tensor   = Input(shape = self.train_stats_tensor.shape[1:2], name='input_aux')

        return {'phylo': input_phylo_tensor, 'aux': input_aux_tensor }

    
    def build_network_aux_layers(self, input_aux_tensor):
        """
        Build the auxiliary layers for the network.

        Args:
            input_aux_tensor: The input tensor for the auxiliary layers.

        Returns:
            list: A list of auxiliary layers.
        """
        w_aux_ffnn = layers.Dense(128, activation = 'relu', kernel_initializer = self.kernel_init, name='ff_aux1')(input_aux_tensor)
        w_aux_ffnn = layers.Dense( 64, activation = 'relu', kernel_initializer = self.kernel_init, name='ff_aux2')(w_aux_ffnn)
        w_aux_ffnn = layers.Dense( 32, activation = 'relu', kernel_initializer = self.kernel_init, name='ff_aux3')(w_aux_ffnn)
        
        return [ w_aux_ffnn ]

    def build_network_phylo_layers(self, input_data_tensor):
        """
        Build the phylo layers for the network.

        Args:
            input_data_tensor: The input data tensor.

        Returns:
            list: A list of phylo layers.
        """
        # convolutional layers
        # e.g. you expect to see 64 patterns, width of 3, stride (skip-size) of 1, padding zeroes so all windows are 'same'
        w_conv = layers.Conv1D(64, 3, activation = 'relu', padding = 'same', name='conv_std1')(input_data_tensor)
        w_conv = layers.Conv1D(96, 5, activation = 'relu', padding = 'same', name='conv_std2')(w_conv)
        w_conv = layers.Conv1D(128, 7, activation = 'relu', padding = 'same', name='conv_std3')(w_conv)
        w_conv_global_avg = layers.GlobalAveragePooling1D(name = 'pool_std')(w_conv)

        # stride layers (skip sizes during slide)
        w_stride = layers.Conv1D(64, 7, strides = 3, activation = 'relu', padding = 'same', name='conv_stride1')(input_data_tensor)
        w_stride = layers.Conv1D(96, 9, strides = 6, activation = 'relu', padding = 'same', name='conv_stride2')(w_stride)
        w_stride_global_avg = layers.GlobalAveragePooling1D(name = 'pool_stride')(w_stride)

        # dilation layers (spacing among grid elements)
        w_dilated = layers.Conv1D(32, 3, dilation_rate = 2, activation = 'relu', padding = 'same', name='conv_dilate1')(input_data_tensor)
        w_dilated = layers.Conv1D(64, 5, dilation_rate = 4, activation = 'relu', padding = 'same', name='conv_dilate2')(w_dilated)
        w_dilated_global_avg = layers.GlobalAveragePooling1D(name = 'pool_dilate')(w_dilated)

        return [ w_conv_global_avg, w_stride_global_avg, w_dilated_global_avg ]
    

    def build_network_output_layers(self, phylo_layers, aux_layers):
        """
        Build the output layers for the network.

        Args:
            phylo_layers: The phylo layers.
            aux_layers: The auxiliary layers.

        Returns:
            list: A list of output layers.
        """
        # combine phylo and aux layers lists
        all_layers = phylo_layers + aux_layers

        # concatenate all above -> deep fully connected network
        w_concat = layers.Concatenate(axis = 1, name = 'concat_out')(all_layers)

        # point estimate for parameters
        w_point_est = layers.Dense(128, activation = 'relu', kernel_initializer = self.kernel_init, name='ff_value1')(w_concat)
        w_point_est = layers.Dense( 64, activation = 'relu', kernel_initializer = self.kernel_init, name='ff_value2')(w_point_est)
        w_point_est = layers.Dense( 32, activation = 'relu', kernel_initializer = self.kernel_init, name='ff_value3')(w_point_est)
        w_point_est = layers.Dense(self.num_params, activation = 'linear', name = 'param_value')(w_point_est)

        # lower quantile for parameters
        w_lower_quantile = layers.Dense(128, activation = 'relu', kernel_initializer = self.kernel_init, name='ff_lower1')(w_concat)
        w_lower_quantile = layers.Dense( 64, activation = 'relu', kernel_initializer = self.kernel_init, name='ff_lower2')(w_lower_quantile)
        w_lower_quantile = layers.Dense( 32, activation = 'relu', kernel_initializer = self.kernel_init, name='ff_lower3')(w_lower_quantile)
        w_lower_quantile = layers.Dense(self.num_params, activation = 'linear', name = 'param_lower')(w_lower_quantile)

        # upper quantile for parameters
        w_upper_quantile = layers.Dense(128, activation = 'relu', kernel_initializer = self.kernel_init, name='ff_upper1')(w_concat)
        w_upper_quantile = layers.Dense( 64, activation = 'relu', kernel_initializer = self.kernel_init, name='ff_upper2')(w_upper_quantile)
        w_upper_quantile = layers.Dense( 32, activation = 'relu', kernel_initializer = self.kernel_init, name='ff_upper3')(w_upper_quantile)
        w_upper_quantile = layers.Dense(self.num_params, activation = 'linear', name = 'param_upper')(w_upper_quantile)

        return [ w_point_est, w_lower_quantile, w_upper_quantile ]


    def train(self):
        """
        Trains the model.

        Perform training by compiling the model with appropriate loss functions and metrics,
        and then fitting the model to the training data.

        Returns:
            None
        """
        lower_quantile_loss,upper_quantile_loss = self.get_pinball_loss_fns(self.cpi_coverage)
        
        self.pinball_loss_q_0_025, self.pinball_loss_q_0_975
        my_loss = [ self.loss, self.pinball_loss_q_0_025, self.pinball_loss_q_0_975 ]

        # compile model        
        self.mymodel.compile(optimizer = self.optimizer, 
                        loss = my_loss,
                        metrics = self.metrics)
     
        # run learning
        self.history = self.mymodel.fit(\
            verbose = 2,
            x = [self.train_data_tensor, self.train_stats_tensor], 
            y = self.norm_train_labels,
            epochs = self.num_epochs,
            batch_size = self.batch_size, 
            validation_data = ([self.val_data_tensor, self.val_stats_tensor], self.norm_val_labels))
        
        # store learning history
        self.history_dict = self.history.history

        
        return

    def make_results(self):

        # evaluate ???
        print(self.test_data_tensor.shape)
        print(self.test_stats_tensor.shape)
        self.mymodel.evaluate([self.test_data_tensor, self.test_stats_tensor], self.norm_test_labels)

        # scatter of pred vs true for training data
        self.normalized_train_preds       = self.mymodel.predict([self.train_data_tensor, self.train_stats_tensor])

        self.normalized_train_preds       = np.array(self.normalized_train_preds)
        self.denormalized_train_preds     = utilities.denormalize(self.normalized_train_preds, self.train_label_means, self.train_label_sd)
        self.denormalized_train_preds     = np.exp(self.denormalized_train_preds)
        self.denormalized_train_labels    = utilities.denormalize(self.norm_train_labels, self.train_label_means, self.train_label_sd)
        self.denormalized_train_labels    = np.exp(self.denormalized_train_labels)

        # scatter of pred vs true for test data
        self.normalized_test_preds        = self.mymodel.predict([self.test_data_tensor, self.test_stats_tensor])

        self.normalized_test_preds        = np.array(self.normalized_test_preds)
        self.denormalized_test_preds      = utilities.denormalize(self.normalized_test_preds, self.train_label_means, self.train_label_sd)
        self.denormalized_test_preds      = np.exp(self.denormalized_test_preds)
        self.denormalized_test_labels     = utilities.denormalize(self.norm_test_labels, self.train_label_means, self.train_label_sd)
        self.denormalized_test_labels     = np.exp(self.denormalized_test_labels)
        
         # scatter of pred vs true for test data
        self.normalized_calib_preds       = self.mymodel.predict([self.calib_data_tensor, self.calib_stats_tensor])
        self.normalized_calib_preds       = np.array(self.normalized_calib_preds)

        # drop 0th column containing point estimate predictions for calibration dataset
        norm_calib_pred_quantiles         = self.normalized_calib_preds[1:,:,:]
        self.cpi_adjustments              = self.get_CQR_constant(norm_calib_pred_quantiles, self.norm_calib_labels, inner_quantile=self.cpi_coverage, asymmetric=self.cpi_asymmetric)
        self.cpi_adjustments              = np.array( self.cpi_adjustments ).reshape((2,-1))


        # training predictions with calibrated CQR CIs
        self.denorm_train_preds_calib        = self.normalized_train_preds
        self.denorm_train_preds_calib[1,:,:] = self.denorm_train_preds_calib[1,:,:] - self.cpi_adjustments[0,:]
        self.denorm_train_preds_calib[2,:,:] = self.denorm_train_preds_calib[2,:,:] + self.cpi_adjustments[1,:]
        self.denorm_train_preds_calib        = utilities.denormalize(self.denorm_train_preds_calib, self.train_label_means, self.train_label_sd)
        self.denorm_train_preds_calib        = np.exp(self.denorm_train_preds_calib)

        # test predictions with calibrated CQR CIs
        self.denorm_test_preds_calib        = self.normalized_test_preds
        self.denorm_test_preds_calib[1,:,:] = self.denorm_test_preds_calib[1,:,:] - self.cpi_adjustments[0,:]
        self.denorm_test_preds_calib[2,:,:] = self.denorm_test_preds_calib[2,:,:] + self.cpi_adjustments[1,:]
        self.denorm_test_preds_calib        = utilities.denormalize(self.denorm_test_preds_calib, self.train_label_means, self.train_label_sd)
        self.denorm_test_preds_calib        = np.exp(self.denorm_test_preds_calib)

        return
    
    def save_results(self):
        """
        Saves all results from training procedure. Saved results include the
        trained network, the normalization parameters for training/calibration,
        CPI adjustment terms, and the training history.

        Returns:
            None
        """
        max_idx = 1000
        
        # save model to file
        self.mymodel.save(self.model_sav_fn)

        # save summ_stat names, means, sd for new test dataset normalization
        df_stats = pd.DataFrame( [self.stat_names, self.train_stats_means, self.train_stats_sd] ).T # columns=['name', 'mean', 'sd'] )
        df_stats.columns = ['name', 'mean', 'sd']
        df_stats.to_csv(self.model_trn_ss_norm_fn, index=False, sep=',')
 
        # save label names, means, sd for new test dataset normalization
        df_labels = pd.DataFrame( [self.param_names, self.train_label_means, self.train_label_sd] ).T # columns=['name', 'mean', 'sd'] )
        df_labels.columns = ['name', 'mean', 'sd']
        df_labels.to_csv(self.model_trn_lbl_norm_fn, index=False, sep=',')

        # save train prediction scatter data
        df_train_pred_nocalib   = utilities.make_param_VLU_mtx(self.denormalized_train_preds[0:max_idx,:], self.param_names )
        df_test_pred_nocalib    = utilities.make_param_VLU_mtx(self.denormalized_test_preds[0:max_idx,:], self.param_names )
        df_train_pred_calib     = utilities.make_param_VLU_mtx(self.denorm_train_preds_calib[0:max_idx,:], self.param_names )
        df_test_pred_calib      = utilities.make_param_VLU_mtx(self.denorm_test_preds_calib[0:max_idx,:], self.param_names )
        #df_train_pred   = pd.DataFrame( self.denormalized_train_preds[0:max_idx,:], columns=param_pred_names )
        #df_test_pred    = pd.DataFrame( self.denormalized_test_preds[0:max_idx,:], columns=param_pred_names )

        df_train_labels  = pd.DataFrame( self.denormalized_train_labels[0:max_idx,:], columns=self.param_names )
        df_test_labels   = pd.DataFrame( self.denormalized_test_labels[0:max_idx,:], columns=self.param_names )
        df_cpi_intervals = pd.DataFrame( self.cpi_adjustments, columns=self.param_names )

        df_train_pred_nocalib.to_csv(self.train_pred_nocalib_fn, index=False, sep=',')
        df_test_pred_nocalib.to_csv(self.test_pred_nocalib_fn, index=False, sep=',')
        df_train_pred_calib.to_csv(self.train_pred_calib_fn, index=False, sep=',')
        df_test_pred_calib.to_csv(self.test_pred_calib_fn, index=False, sep=',')
        df_train_labels.to_csv(self.train_labels_fn, index=False, sep=',')
        df_test_labels.to_csv(self.test_labels_fn, index=False, sep=',')
        df_cpi_intervals.to_csv(self.model_cpi_fn, index=False, sep=',')
        
        #, self.denormalized_train_labels[0:1000,:] ], )
        json.dump(self.history_dict, open(self.model_hist_fn, 'w'))

        return


    def pinball_loss(self, y_true, y_pred, alpha):
        """
        Calculate the Pinball Loss for quantile regression.

        This function calculates the Pinball Loss, which measures the difference between
        the true target values (`y_true`) and the predicted target values (`y_pred`) for
        a quantile regression model.

        The Pinball Loss is calculated using the following formula:
            mean(maximum(alpha * err, (alpha - 1) * err))

        Parameters:
            y_true (numpy.ndarray): Array of true target values.
            y_pred (numpy.ndarray): Array of predicted target values.
            alpha (float): Quantile level.

        Returns:
            float: The calculated Pinball Loss.

        """
        err = y_true - y_pred
        return K.mean(K.maximum(alpha*err, (alpha-1)*err), axis=-1)

    # lower 95% quantile
    def pinball_loss_q_0_025(self, y_true, y_pred):
        return self.pinball_loss(y_true, y_pred, alpha=0.025)
    # upper 95% quantile
    def pinball_loss_q_0_975(self, y_true, y_pred):
        return self.pinball_loss(y_true, y_pred, alpha=0.975)
    # lower 90% quantile
    def pinball_loss_q_0_05(self, y_true, y_pred):
        return self.pinball_loss(y_true, y_pred, alpha=0.05)
    # upper 90% quantile
    def pinball_loss_q_0_95(self, y_true, y_pred):
        return self.pinball_loss(y_true, y_pred, alpha=0.95)
    # lower 80% quantile
    def pinball_loss_q_0_10(self, y_true, y_pred):
        return self.pinball_loss(y_true, y_pred, alpha=0.10)
    # upper 80% quantile
    def pinball_loss_q_0_90(self, y_true, y_pred):
        return self.pinball_loss(y_true, y_pred, alpha=0.90)
    # lower 70% quantile
    def pinball_loss_q_0_15(self, y_true, y_pred):
        return self.pinball_loss(y_true, y_pred, alpha=0.15)
    # lower 80% quantile
    def pinball_loss_q_0_85(self, y_true, y_pred):
        return self.pinball_loss(y_true, y_pred, alpha=0.85)

    def get_pinball_loss_fns(self, coverage):
        """Returns the pinball loss functions corresponding to a given coverage level.

        Args:
            coverage (float): The desired coverage level.

        Returns:
            tuple: A tuple of two pinball loss functions corresponding to the specified coverage level.

        Raises:
            NotImplementedError: If the specified coverage level is not supported.

        Example:
            pinball_loss_q_0_025, pinball_loss_q_0_975 = get_pinball_loss_fns(0.95)

        Note:
            Supported coverage levels and their corresponding pinball loss functions:
            - 0.95: pinball_loss_q_0_025, pinball_loss_q_0_975
            - 0.90: pinball_loss_q_0_05, pinball_loss_q_0_95
            - 0.80: pinball_loss_q_0_10, pinball_loss_q_0_90
            - 0.70: pinball_loss_q_0_15, pinball_loss_q_0_85
        """
        if coverage == 0.95:
            return self.pinball_loss_q_0_025, self.pinball_loss_q_0_975
        elif coverage == 0.90:
            return self.pinball_loss_q_0_05, self.pinball_loss_q_0_95
        elif coverage == 0.80:
            return self.pinball_loss_q_0_10, self.pinball_loss_q_0_90
        elif coverage == 0.70:
            return self.pinball_loss_q_0_15, self.pinball_loss_q_0_85
        else:
            raise NotImplementedError
        return

    # computes the distance y_i is inside/outside the lower(x_i) and upper(x_i) quantiles
    # there are three cases to consider:
    #   1. y_i is under the lower bound: max-value will be q_lower(x_i) - y_i & positive
    #   2. y_i is over the upper bound:  max-value will be y_i - q_upper(x_i) & positive
    #   3. y_i is between the bounds:    max-value will be the difference between y_i and the closest bound & negative
    def compute_conformity_scores(self, x, y, q_lower, q_upper):
        """
        Computes the conformity scores based on the input parameters.

        Args:
            x (array-like): The input data.
            y (array-like): The target data.
            q_lower (function): The lower quantile function.
            q_upper (function): The upper quantile function.

        Returns:
            array-like: The conformity scores.

        """
        return np.max( q_lower(x)-y, y-q_upper(x) )

    def get_CQR_constant(self, preds, true, inner_quantile=0.95, asymmetric = True):
        """
        Computes the conformity scores based on the input parameters.

        Args:
            x (array-like): The input data.
            y (array-like): The target data.
            q_lower (function): The lower quantile function.
            q_upper (function): The upper quantile function.

        Returns:
            array-like: The conformity scores.

        """
        #preds axis 0 is the lower and upper quants, axis 1 is the replicates, and axis 2 is the params
        # compute non-comformity scores
        Q = np.empty((2, preds.shape[2]))
        
        for i in range(preds.shape[2]):
            if asymmetric:
                # Asymmetric non-comformity score
                lower_s = np.array(true[:,i] - preds[0][:,i])
                upper_s = np.array(true[:,i] - preds[1][:,i])
                lower_p = (1 - inner_quantile)/2 * (1 + 1/preds.shape[1])
                upper_p = (1 + inner_quantile)/2 * (1 + 1/preds.shape[1])
                if lower_p < 0.:
                    self.logger.write_log('lrn', 'get_CQR_constant: lower_p >= 0.')
                    lower_p = 0.
                if upper_p > 1.:
                    self.logger.write_log('lrn', 'get_CQR_constant: upper_p <= 1.')
                    upper_p = 1.
                lower_q = np.quantile(lower_s, lower_p)
                upper_q = np.quantile(upper_s, upper_p)
                # get (lower_q adjustment, upper_q adjustment)
            else:
                # Symmetric non-comformity score
                s = np.amax(np.array((preds[0][:,i] - true[:,i], true[:,i] - preds[1][:,i])), axis=0)
                # get adjustment constant: 1 - alpha/2's quintile of non-comformity scores
                #Q = np.append(Q, np.quantile(s, inner_quantile * (1 + 1/preds.shape[1])))
                symm_p = inner_quantile * (1 + 1/preds.shape[1])
                if symm_p < 0.:
                    self.logger.write_log('lrn', 'get_CQR_constant: symm_p >= 0.')
                    symm_p = 0.
                elif symm_p > 1.:
                    self.logger.write_log('lrn', 'get_CQR_constant: symm_p <= 1.')
                    symm_p = 1.                    
                lower_q = np.quantile(s, symm_p)
                upper_q = lower_q
                #Q[:,i] = np.array([lower_q, upper_q])

            Q[:,i] = np.array([lower_q, upper_q])
                                
        return Q


# #-----------------------------------------------------------------------------------------------------------------#
  
# class ParamEstLearner(CnnLearner):
#     def __init__(self, args):
#         super().__init__(args)
#         return
    

# #-----------------------------------------------------------------------------------------------------------------#
  
# class ModelTestLearner(CnnLearner):
#     def __init__(self, args):
#         super().__init__(args)
#         return
    

# #-----------------------------------------------------------------------------------------------------------------#
  
# class AncStateLearner(CnnLearner):
#     def __init__(self, args):
#         super().__init__(args)
#         return

