#==============================================================================#
# GeoSSE phyddle config file                                                   #
#==============================================================================#


# helper variables

args = {
    
    #-------------------------------#
    # Project organization          #
    #-------------------------------#
    'step'    : 'SFTEP',                    # step(s) to run
    'verbose' : 'True',                     # print verbose phyddle output?
    'dir'     : './',                       # project directory
    'prefix'  : 'out',                      # project prefix
    
    #-------------------------------#
    # Multiprocessing               #
    #-------------------------------#
    'use_parallel'   : 'True',              # use multiprocessing to speed up jobs?
    'use_cuda'       : 'True',              # use CUDA for Train
    'num_proc'       : -2,                  # how many CPUs to use (-2 means all but 2)
    
    #-------------------------------#
    # Simulate Step settings        #
    #-------------------------------#
    'sim_command'       : 'python3 sim_geosse.py', # exact command string, argument is output file prefix
    'sim_logging'       : 'verbose',        # verbose, compressed, or clean
    'start_idx'         : 0,                # first simulation replicate index
    'end_idx'           : 1000,             # last simulation replicate index
    'sim_batch_size'    : 1,                # num replicates per sim job

    #-------------------------------#
    # Format Step settings          #
    #-------------------------------#
    'num_char'          : 3,                # number of evolutionary characters
    'num_states'        : 2,                # number of states per character
    'min_num_taxa'      : 10,               # min number of taxa for valid sim
    'max_num_taxa'      : 500,              # max number of taxa for valid sim
    'tree_encode'       : 'extant',         # use model with serial or extant tree
    'char_format'       : 'nexus',
    'brlen_encode'      : 'height_brlen',   # how to encode phylo brlen? height_only or height_brlen
    'char_encode'       : 'integer',        # how to encode discrete states? one_hot or integer
    'param_est'         : {                 # model parameters to estimate (labels)
                           'log10_w_0':'real',
                           'log10_e_0':'real',
                           'log10_d_0_1':'real',
                           'log10_b_0_1':'real'
                           },
    'param_data'        : { },               # model parameters that are known (aux. data)
    'tensor_format'     : 'hdf5',           # save as compressed HDF5 or raw csv
    'save_phyenc_csv'   : 'F',              # save intermediate phylo-state vectors to file

    #-------------------------------#
    # Train Step settings           #
    #-------------------------------#
    'tree_width'        : 200,              # tree width category used to train network
    'num_epochs'        : 20,               # number of training intervals (epochs)
    'prop_test'         : 0.05,             # proportion of sims in test dataset
    'prop_val'          : 0.05,             # proportion of sims in validation dataset
    'prop_cal'          : 0.20,             # proportion of sims in CPI calibration dataset 
    'cpi_coverage'      : 0.95,             # coverage level for CPIs
    'cpi_asymmetric'    : 'T',              # upper/lower (True) or symmetric (False) CPI adjustments
    'trn_batch_size'    : 128,              # number of samples in each training batch
    'optimizer'         : 'adam',           # optimizer for network weight/bias parameters

    #-------------------------------#
    # Estimate Step settings        #
    #-------------------------------#

    #-------------------------------#
    # Plot Step settings            #
    #-------------------------------#
    'plot_train_color'      : 'blue',       # plot color for training data
    'plot_test_color'       : 'purple',     # plot color for test data
    'plot_val_color'        : 'red',        # plot color for validation data
    'plot_aux_color'        : 'green',      # plot color for input auxiliary data
    'plot_label_color'      : 'orange',     # plot color for labels (params)
    'plot_est_color'        : 'black'       # plot color for estimated data/values

}

