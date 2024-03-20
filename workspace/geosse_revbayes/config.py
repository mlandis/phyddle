#==============================================================================#
# Default phyddle config file                                                  #
#==============================================================================#

args = {
    
    #-------------------------------#
    # Project organization          #
    #-------------------------------#
    'step'           : 'SFTEP',                  # steps to run? all, sim, fmt, trn, est, plt
    'verbose'        : 'T',
    'dir'            : './',
    'prefix'         : 'out',

    #-------------------------------#
    # Multiprocessing               #
    #-------------------------------#
    'use_parallel'   : 'T',                 # use multiprocessing to speed up jobs?
    'use_cuda'       : 'T',
    'num_proc'       : -2,                  # how many CPUs to use (-2 means all but 2)
    
    #-------------------------------#
    # Simulate settings             #
    #-------------------------------#
    'sim_command'       : f'rb sim_geosse.Rev --args',   # exact command string, argument is output file prefix
    'sim_logging'       : 'verbose',        # verbose, compressed, or clean
    'start_idx'         : 0,                # first simulation replicate index
    'end_idx'           : 100,              # last simulation replicate index
    'sim_batch_size'    : 100,               # number of simulations per batch

    #-------------------------------#
    # Format settings               #
    #-------------------------------#
    'encode_all_sim'    : 'T',              # encode all simulated datasets
    'num_char'          : 1,                # number of evolutionary characters
    'num_states'        : 7,                # number of states per discrete character
    'min_num_taxa'      : 10,
    'max_num_taxa'      : 500,
    'tree_width'        : 500,              # tree width category used to train network
    'tree_encode'       : 'extant',         # use model with serial or extant tree
    'brlen_encode'      : 'height_brlen',   # how to encode phylo brlen? height_only or height_brlen
    'char_encode'       : 'one_hot',        # how to encode discrete states? one_hot or integer 
    'param_est'        : {                  # model parameters to predict (labels)
        'log10_rho_w':'real',
        'log10_rho_d':'real',
        'log10_rho_e':'real',
        'log10_rho_b':'real'
    },
    'param_data'        : {                 # model parameters that are known (aux. data)
        'sample_frac':'real'
    },               
    'tensor_format'     : 'hdf5',           # save as compressed HDF5 or raw csv
    'char_format'       : 'nexus',          # expect character data is in nexus or csv format
    'save_phyenc_csv'   : 'F',              # save intermediate phylo-state vectors to file

    #-------------------------------#
    # Train settings                #
    #-------------------------------#
    'num_epochs'        : 20,               # number of training intervals (epochs)
    'prop_test'         : 0.05,             # proportion of sims in test dataset
    'prop_val'          : 0.05,             # proportion of sims in validation dataset
    'prop_cal'          : 0.20,             # proportion of sims in CPI calibration dataset 
    'cpi_coverage'      : 0.95,             # coverage level for CPIs
    'cpi_asymmetric'    : 'T',              # two-sided (True) or one-sided (False) CPI adjustments
    'trn_batch_size'    : 1024,             # number of samples in each training batch
    'loss'              : 'mse',            # loss function for learning
    'optimizer'         : 'adam',           # optimizer for network weight/bias parameters
    'log_offset'        : 1.0,

    #-------------------------------#
    # Estimate settings             #
    #-------------------------------#

    #-------------------------------#
    # Plot settings                 #
    #-------------------------------#
    'plot_train_color'      : 'blue',       # plot color for training data
    'plot_test_color'       : 'purple',     # plot color for test data
    'plot_val_color'        : 'red',        # plot color for validation data
    'plot_aux_color'        : 'green',      # plot color for input auxiliary data
    'plot_label_color'      : 'orange',     # plot color for labels (params)
    'plot_est_color'        : 'black',      # plot color for predictions

}

