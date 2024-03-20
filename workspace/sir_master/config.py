#==============================================================================#
# MASTER phyddle config file                                                   #
#==============================================================================#


# helper variables

args = {
    
    #-------------------------------#
    # Project organization          #
    #-------------------------------#
    'step'    : 'SFTEP',                   # step(s) to run
    'verbose' : 'T',                       # print verbose phyddle output?
    'dir'     : './',                      # default base projet directory
    'prefix'  : 'out',                     # default step filename prefix
    
    #-------------------------------#
    # Multiprocessing               #
    #-------------------------------#
    'use_parallel'   : 'T',                # use multiprocessing to speed up jobs?
    'use_cuda'       : 'T',                # use CUDA for Train
    'num_proc'       : -2,                 # how many CPUs to use (-2 means all but 2)
    
    #-------------------------------#
    # Simulate Step settings        #
    #-------------------------------#
    'sim_command'       : 'python3 sim_sir.py', # exact command string, argument is output file prefix
    'sim_logging'       : 'verbose',        # verbose, compressed, or clean
    'start_idx'         : 0,                # first simulation replicate index
    'end_idx'           : 1000,             # last simulation replicate index
    'sim_batch_size'    : 1,                # num replicates per sim job

    #-------------------------------#
    # Format Step settings          #
    #-------------------------------#
    'num_char'          : 1,                # number of evolutionary characters
    'num_states'        : 1,                # number of states per character
    'min_num_taxa'      : 10,               # min number of taxa for valid sim
    'max_num_taxa'      : 1000,             # max number of taxa for valid sim
    'prop_test'         : 0.05,             # proportion of sims in test dataset
    'tree_encode'       : 'serial',         # use model with serial or extant tree
    'char_format'       : 'nexus',
    'tree_width'        : 500,              # tree width categories for phylo-state tensors
    'brlen_encode'      : 'height_brlen',   # how to encode phylo brlen? height_only or height_brlen
    'char_encode'       : 'one_hot',        # how to encode discrete states? one_hot or integer
    'param_est'         : {                 # model parameters to estimate (labels)
                           'log10_R0_0':'real',
                           'log10_Sample_0':'real',
                           'log10_Infect_0':'real'
                          },
    'param_data'        : {                 # model parameters that are known (aux. data)
                           'log10_Recover_0':'real',
                           'log10_S0_0_0':'real'
                          },
    'tensor_format'     : 'hdf5',           # save as compressed HDF5 or raw csv
    'save_phyenc_csv'   : 'F',            # save intermediate phylo-state vectors to file
    'log_offset'        : 1.0,

    #-------------------------------#
    # Train Step settings           #
    #-------------------------------#
    'num_epochs'        : 20,               # number of training intervals (epochs)
    'prop_val'          : 0.05,             # proportion of sims in validation dataset
    'prop_cal'          : 0.20,             # proportion of sims in CPI calibration dataset 
    'combine_test_val'  : 'F',            # combine test and validation data?
    'cpi_coverage'      : 0.95,             # coverage level for CPIs
    'cpi_asymmetric'    : 'T',             # upper/lower ('T') or symmetric ('F') CPI adjustments
    'batch_size'        : 128,              # number of samples in each training batch
    'loss'              : 'mse',            # loss function for learning
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

