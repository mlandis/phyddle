#==============================================================================#
# Config:       Default phyddle config file                                    #
# Authors:      Michael Landis and Ammon Thompson                              #
# Date:         230804                                                         #
# Description:  Simple birth-death and equal-rates CTMC model in R using ape   #
#==============================================================================#


args = {

    #-------------------------------#
    # Project organization          #
    #-------------------------------#
    'step'    : 'SFTEP',                   # Step(s) to run
    'verbose' : 'T',                       # print verbose phyddle output?
    'prefix'  : 'out',
    'dir'     : './',
    'output_precision'   : 12,             # Number of digits (precision) for numbers in output files

    #-------------------------------#
    # Multiprocessing               #
    #-------------------------------#
    'use_parallel'   : 'T',                 # use multiprocessing to speed up jobs?
    'num_proc'       : -2,                  # how many CPUs to use (-2 means all but 2)

    #-------------------------------#
    # Simulate Step settings        #
    #-------------------------------#
    'sim_command'       : f'Rscript sim_levy.R', # exact command string, argument is output file prefix
    'sim_logging'       : 'verbose',             # verbose, compressed, or clean
    'start_idx'         : 0,                     # first simulation replicate index
    'end_idx'           : 100000,                # last simulation replicate index
    'sim_batch_size'    : 100,

    #-------------------------------#
    # Format Step settings          #
    #-------------------------------#
    'encode_all_sim'    : 'T',
    'num_char'          : 1,                # number of evolutionary characters
    'num_states'        : 2,                # number of states per character
    'min_num_taxa'      : 2,                # min number of taxa for valid sim
    'max_num_taxa'      : 500,              # max number of taxa for valid sim
    'tree_width'        : 300,              # tree width category used to train network
    'tree_encode'       : 'extant',         # use model with serial or extant tree
    'brlen_encode'      : 'height_brlen',   # how to encode phylo brlen? height_only or height_brlen
    'char_encode'       : 'numeric',        # how to encode discrete states? one_hot or integer
    'param_est'         : {                 # model parameters to predict (labels)
    #                       'log10_process_var':'num',
    #                       'log10_process_kurt':'num',
                           # 'log10_sigma_tip':'num',
                           # 'trait_orig':'num',
                           'model_type':'cat'
                          },
    'param_data'        : {                 # model parameters that are known (aux. data)
                           'sample_frac':'num',
    #                       'trait_min':'num',
    #                       'trait_max':'num'
                          },
    'tensor_format'     : 'hdf5',           # save as compressed HDF5 or raw csv
    'char_format'       : 'csv',
    'save_phyenc_csv'   : 'F',              # save intermediate phylo-state vectors to file

    #-------------------------------#
    # Train Step settings           #
    #-------------------------------#
    'num_epochs'        : 200,              # number of training intervals (epochs)
    'num_early_stop'    : 3,
    'prop_test'         : 0.05,             # proportion of sims in test dataset
    'prop_val'          : 0.05,             # proportion of sims in validation dataset
    'prop_cal'          : 0.20,             # proportion of sims in CPI calibration dataset
    'cpi_coverage'      : 0.80,             # coverage level for CPIs
    'cpi_asymmetric'    : 'T',              # upper/lower ('T') or symmetric ('F') CPI adjustments
    'trn_batch_size'    : 2048,             # number of samples in each training batch
    'loss_numerical'    : 'mae',            # loss function for learning
    'optimizer'         : 'adam',           # optimizer for network weight/bias parameters
    'log_offset'        : 1.0,

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
    'plot_emp_color'        : 'black',      # plot color for estimated data/values
}
