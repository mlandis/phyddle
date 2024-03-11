#==============================================================================#
# Config:       Default phyddle config file                                    #
# Authors:      Michael Landis and Ammon Thompson                              #
# Date:         230804                                                         #
# Description:  Simple birth-death and equal-rates CTMC model in R using ape   #
#==============================================================================#

work_dir = './workspace/bisse_r'

args = {

    #-------------------------------#
    # Project organization          #
    #-------------------------------#
    'step'    : 'SFTEP',                   # Step(s) to run
    'verbose' : 'T',                       # print verbose phyddle output?
  	'sim_prefix'  : 'out',          # Prefix for raw simulated data
  	'emp_prefix'  : 'out',          # Prefix for raw empirical data
  	'fmt_prefix'  : 'out',          # Prefix for tensor-formatted data
  	'trn_prefix'  : 'out',          # Prefix for trained networks and training output
  	'est_prefix'  : 'out',          # Prefix for new datasets and estimates
  	'plt_prefix'  : 'out',          # Prefix for plotted results
    'sim_dir'     : f'{work_dir}/simulate',    # Directory for simulated data
    'emp_dir'     : f'{work_dir}/empirical',   # Directory for empirical data
    'fmt_dir'     : f'{work_dir}/format',      # Directory for tensor-formatted data
    'trn_dir'     : f'{work_dir}/train',       # Directory for trained network
    'plt_dir'     : f'{work_dir}/plot',        # Directory for plotted figures
    'est_dir'     : f'{work_dir}/estimate',    # Directory for predictions on new data
    'log_dir'     : f'{work_dir}/log',         # Directory for analysis logs
    'output_precision'   : 12,             # Number of digits (precision) for numbers in output files

    #-------------------------------#
    # Multiprocessing               #
    #-------------------------------#
    'use_parallel'   : 'T',                 # use multiprocessing to speed up jobs?
    'num_proc'       : -2,                  # how many CPUs to use (-2 means all but 2)

    #-------------------------------#
    # Simulate Step settings        #
    #-------------------------------#
    'sim_command'       : f'Rscript {work_dir}/sim_bisse.R', # exact command string, argument is output file prefix
    'sim_logging'       : 'verbose',        # verbose, compressed, or clean
    'start_idx'         : 0,                # first simulation replicate index
    'end_idx'           : 1000,             # last simulation replicate index
    'sim_batch_size'    : 10,

    #-------------------------------#
    # Format Step settings          #
    #-------------------------------#
    'encode_all_sim'    : 'T',
    'num_char'          : 1,                # number of evolutionary characters
    'num_states'        : 2,                # number of states per character
    'min_num_taxa'      : 10,               # min number of taxa for valid sim
    'max_num_taxa'      : 500,              # max number of taxa for valid sim
    'tree_width'        : 500,              # tree width category used to train network
    'tree_encode'       : 'extant',         # use model with serial or extant tree
    'brlen_encode'      : 'height_brlen',   # how to encode phylo brlen? height_only or height_brlen
    'char_encode'       : 'integer',        # how to encode discrete states? one_hot or integer
    'param_est'         : {                 # model parameters to predict (labels)
                           'ln_birth_1'     : 'real',
                           'ln_birth_2'     : 'real',
                           'ln_death'       : 'real',
                           'ln_state_rate'  : 'real',
                           #'model_type'     : 'cat',
                          },
    'param_data'        : {                 # model parameters that are known (aux. data)
                           'ln_sample_frac' : 'real'
                          },
    'tensor_format'     : 'hdf5',           # save as compressed HDF5 or raw csv
    'char_format'       : 'csv',
    'save_phyenc_csv'   : 'F',              # save intermediate phylo-state vectors to file

    #-------------------------------#
    # Train Step settings           #
    #-------------------------------#
    'trn_objective'     : 'param_est',      # what is the learning task? param_est or model_test
    'num_epochs'        : 10,               # number of training intervals (epochs)
    'prop_test'         : 0.05,             # proportion of sims in test dataset
    'prop_val'          : 0.05,             # proportion of sims in validation dataset
    'prop_cal'          : 0.20,             # proportion of sims in CPI calibration dataset
    'combine_test_val'  : 'T',
    'cpi_coverage'      : 0.95,             # coverage level for CPIs
    'cpi_asymmetric'    : 'T',              # upper/lower ('T') or symmetric ('F') CPI adjustments
    'batch_size'        : 1024,             # number of samples in each training batch
    'loss'              : 'mse',            # loss function for learning (real-valued) targets
    'optimizer'         : 'adam',           # optimizer for network weight/bias parameters
    'metrics'           : ['mae', 'acc'],   # recorded training metrics
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
    'plot_est_color'        : 'black',      # plot color for estimated data/values
    'plot_scatter_log'      : 'T',          # Use log values for scatter plots when possible?
    'plot_contour_log'      : 'T',          # Use log values for scatter plots when possible?
    'plot_density_log'      : 'T'           # Use log values for scatter plots when possible?
}
