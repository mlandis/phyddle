#==============================================================================#
# Default phyddle config file                                                  #
#==============================================================================#

# external import
import scipy as sp

# helper variables
num_char = 3

args = {
    
    #-------------------------------#
    # Project organization          #
    #-------------------------------#
    'proj'           : 'sirm',              # directory name for pipeline project
    'step'           : 'all',               # step(s) to run? all, sim, fmt, lrn, prd, plt
    'verbose'        : True,                #
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
    'model_type'         : 'sirm',          # model type defines general states and events
    'model_variant'      : 'equal_rates',   # model variant defines rate assignments
    'num_char'           : num_char,        # number of evolutionary characters
    'rv_fn'              : {                # distributions for model parameters
        'R0'             : sp.stats.uniform.rvs,
        'recovery'       : sp.stats.uniform.rvs,
        'sampling'       : sp.stats.uniform.rvs,
        'migration'      : sp.stats.uniform.rvs,
        'S0'             : sp.stats.uniform.rvs
    },
    'rv_arg'             : {                # loc/scale/shape for model parameter dists
        'R0'             : { 'loc' : 1.,    'scale' : 9.    },
        'recovery'       : { 'loc' : 0.01,  'scale' : 0.09  },
        'sampling'       : { 'loc' : 0.1,   'scale' : 0.9   },
        'migration'      : { 'loc' : 0.1,   'scale' : 0.9   },
        'S0'             : { 'loc' : 1000., 'scale' : 4000. }
    },

    #-------------------------------#
    # Simulating Step settings      #
    #-------------------------------#
    'sim_method'        : 'master',         # command, master, [phylojunction], ...
    'sim_command'       : 'beast',          # exact command string, argument is output file prefix
    'sim_logging'       : 'clean',          # verbose, compressed, or clean
    'start_idx'         : 0,                # first simulation replicate index
    'end_idx'           : 1000,             # last simulation replicate index
    'sample_population' : ['S'],            # name of population to sample
    'stop_time'         : 10,               # time to stop simulation
    'min_num_taxa'      : 10,               # min number of taxa for valid sim
    'max_num_taxa'      : 500,              # max number of taxa for valid sim

    #-------------------------------#
    # Formatting Step settings      #
    #-------------------------------#
    'tree_type'         : 'serial',         # use model with serial or extant tree
    'tree_width_cat'    : [ 200, 500 ],     # tree size classes for phylo-state tensors
    'tree_encode_type'  : 'height_brlen',   # how to encode phylo brlen? height_only or height_brlen
    'char_encode_type'  : 'one_hot',        # how to encode discrete states? one_hot or integer
    'param_pred'        : [                 # model parameters to predict (labels)
        'R0_0', 'sampling_0', 'migration_0_1'
    ],
    'param_data'        : [                 # model parameters that are known (aux. data)
        'recovery_0', 'S0_0'
    ],
    'tensor_format'     : 'hdf5',           # save as compressed HDF5 or raw csv
    'save_phyenc_csv'   : False,            # save intermediate phylo-state vectors to file

    #-------------------------------#
    # Learning Step settings        #
    #-------------------------------#
    'learn_method'      : 'param_est',      # what is the learning task? param_est or model_test
    'tree_width'        : 500,              # tree size class used to train network
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

