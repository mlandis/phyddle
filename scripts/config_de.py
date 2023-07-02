#==============================================================================#
# Default phyddle config file                                                  #
#==============================================================================#

# external import
import scipy as sp

# helper variables
num_char = 4

args = {
    
    #-------------------------------#
    # Project organization          #
    #-------------------------------#
    'proj'           : 'geosse_de_n4',      # directory name for pipeline project
    'sim_dir'        : '../raw_data',       # directory for simulated data
    'fmt_dir'        : '../tensor_data',    # directory for tensor-formatted data
    'net_dir'        : '../network',        # directory for trained network
    'plt_dir'        : '../plot',           # directory for plotted figures
    'pred_dir'       : '../predict',        # directory for predictions on new data
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
    'model_variant'      : 'density_effect',   # model variant defines rate assignments
    'num_char'           : num_char,        # number of evolutionary characters
    'rv_fn'              : {                # distributions for model parameters
        'w': sp.stats.expon.rvs,
        'e': sp.stats.expon.rvs,
        'd': sp.stats.expon.rvs,
        'b': sp.stats.expon.rvs,
        'ed': sp.stats.expon.rvs
    },
    'rv_arg'             : {                # loc/scale/shape for model parameter dists
        'w': { 'scale' : 1.0 },
        'e': { 'scale' : 0.5 },
        'd': { 'scale' : 0.4 },
        'b': { 'scale' : 1.0 },
        'ed': { 'scale' : 0.2 }
    },

    #-------------------------------#
    # Simulating Step settings      #
    #-------------------------------#
    'sim_logging'       : 'compress',       # verbose, compressed, or clean
    'start_idx'         : 0,                # first simulation replicate index
    'end_idx'           : 1000,             # last simulation replicate index
    'sample_population' : ['S'],            # name of population to sample
    'stop_time'         : 100,              # time to stop simulation
    'min_num_taxa'      : 50,               # min number of taxa for valid sim
    'max_num_taxa'      : 400,              # max number of taxa for valid sim

    #-------------------------------#
    # Formatting Step settings      #
    #-------------------------------#
    'tree_type'         : 'extant',         # use model with serial or extant tree
    'tree_width_cats'   : [ 250 ],          # tree width categories for phylo-state tensors
    'param_pred'        : [                 # model parameters to predict (labels)
        'w_0', 'e_0', 'd_0_1', 'b_0_1', 'ed_0'
    ],
    'param_data'        : [],               # model parameters that are known (aux. data)
    'tensor_format'     : 'hdf5',           # save as compressed HDF5 or raw csv
    'save_phyenc_csv'   : False,            # save intermediate phylo-state vectors to file

    #-------------------------------#
    # Learning Step settings        #
    #-------------------------------#
    'tree_width'        : 500,              # tree width category used to train network
    'num_epochs'        : 15,               # number of training intervals (epochs)
    'prop_test'         : 0.05,             # proportion of sims in test dataset
    'prop_validation'   : 0.05,             # proportion of sims in validation dataset
    'prop_calibration'  : 0.20,             # proportion of sims in CPI calibration dataset 
    'cpi_coverage'      : 0.95,             # coverage level for CPIs
    'batch_size'        : 256,              # number of samples in each training batch
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

