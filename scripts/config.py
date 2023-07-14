#==============================================================================#
# Default phyddle config file                                                  #
#==============================================================================#

# external import
import scipy.stats
import scipy as sp

# helper variables
num_char = 3
num_states = 2

args = {
    
    #-------------------------------#
    # Project organization          #
    #-------------------------------#
    'proj'           : 'my_project',        # directory name for pipeline project
    'step'           : 'all',                         # step(s) to run? all, sim, fmt, lrn, prd, plt
    'sim_dir'        : '../workspace/raw_data',       # directory for simulated data
    'fmt_dir'        : '../workspace/tensor_data',    # directory for tensor-formatted data
    'net_dir'        : '../workspace/network',        # directory for trained network
    'plt_dir'        : '../workspace/plot',           # directory for plotted figures
    'pred_dir'       : '../workspace/predict',        # directory for predictions on new data
    'pred_prefix'    : 'new.1',             # prefix for new dataset to predict
    
    #-------------------------------#
    # Multiprocessing               #
    #-------------------------------#
    'use_parallel'   : True,                # use multiprocessing to speed up jobs?
    'num_proc'       : -2,                  # how many CPUs to use (-2 means all but 2)
    
    #-------------------------------#
    # Model Configuration           #
    #-------------------------------#
    'model_type'         : 'geosse',        # model type defines general states and events
    'model_variant'      : 'equal_rates',   # model variant defines rate assignments
    'num_char'           : num_char,        # number of evolutionary characters
    'num_states'         : num_states,      # number of states per character
    'rv_fn'              : {                # distributions for model parameters
        'w': sp.stats.expon.rvs,
        'e': sp.stats.expon.rvs,
        'd': sp.stats.expon.rvs,
        'b': sp.stats.expon.rvs
    },
    'rv_arg'             : {                # loc/scale/shape for model parameter dists
        'w': { 'scale' : 0.2 },
        'e': { 'scale' : 0.1 },
        'd': { 'scale' : 0.1 },
        'b': { 'scale' : 0.5 }
    },

    #-------------------------------#
    # Simulating Step settings      #
    #-------------------------------#
    'sim_method'        : 'master',         # command, master, [phylojunction], ...
    'sim_command'       : 'beast',          # exact command string, argument is output file prefix
    'sim_logging'       : 'verbose',        # verbose, compressed, or clean
    'start_idx'         : 0,                # first simulation replicate index
    'end_idx'           : 1000,             # last simulation replicate index
    'sample_population' : ['S'],            # name of population to sample
    'stop_time'         : 10,               # time to stop simulation
    'min_num_taxa'      : 10,               # min number of taxa for valid sim
    'max_num_taxa'      : 500,              # max number of taxa for valid sim

    #-------------------------------#
    # Formatting Step settings      #
    #-------------------------------#
    'tree_type'         : 'extant',         # use model with serial or extant tree
    'tree_width_cats'   : [ 200, 500 ],     # tree width categories for phylo-state tensors
    'tree_encode_type'  : 'height_brlen',   # how to encode phylo brlen? height_only or height_brlen
    'char_encode_type'  : 'integer',        # how to encode discrete states? one_hot or integer
    'param_pred'        : [                 # model parameters to predict (labels)
        'w_0', 'e_0', 'd_0_1', 'b_0_1'
    ],
    'param_data'        : [],               # model parameters that are known (aux. data)
    'tensor_format'     : 'hdf5',           # save as compressed HDF5 or raw csv
    'save_phyenc_csv'   : False,            # save intermediate phylo-state vectors to file

    #-------------------------------#
    # Learning Step settings        #
    #-------------------------------#
    'learn_method'      : 'param_est',      # what is the learning task? param_est or model_test
    'tree_width'        : 500,              # tree width category used to train network
    'num_epochs'        : 20,               # number of training intervals (epochs)
    'prop_test'         : 0.05,             # proportion of sims in test dataset
    'prop_validation'   : 0.05,             # proportion of sims in validation dataset
    'prop_calibration'  : 0.20,             # proportion of sims in CPI calibration dataset 
    'cpi_coverage'      : 0.95,             # coverage level for CPIs
    'batch_size'        : 128,              # number of samples in each training batch
    'loss'              : 'mse',            # loss function for learning
    'optimizer'         : 'adam',           # optimizer for network weight/bias parameters
    'metrics'           : ['mae', 'acc'],   # recorded training metrics

    #-------------------------------#
    # Plotting Step settings        #
    #-------------------------------#
    'plot_train_color'      : 'blue',       # plot color for training data
    'plot_test_color'       : 'purple',     # plot color for test data
    'plot_val_color'        : 'red',        # plot color for validation data
    'plot_aux_color'        : 'green',      # plot color for input auxiliary data
    'plot_label_color'      : 'orange',     # plot color for labels (params)
    'plot_pred_color'       : 'black'       # plot color for predictions

    #-------------------------------#
    # Predicting Step settings      #
    #-------------------------------#
    # prediction already handled by previously defined settings
    # no prediction-specific settings currently implemented
}

