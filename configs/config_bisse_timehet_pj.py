#==============================================================================#
# Config:       Default phyddle config file                                    #
# Authors:      Michael Landis and Ammon Thompson                              #
# Date:         230804                                                         #
# Description:  Simple birth-death and equal-rates CTMC model in R using ape   #
#==============================================================================#

work_dir = "./workspace/bisse_timehet_pj"

args = {

    #-------------------------------#
    # Project organization          #
    #-------------------------------#
    'step'    : 'SFTEP',                    # step(s) to run
    'verbose' : 'T',                        # print verbose phyddle output?
    'sim_dir' : f'{work_dir}/simulate',    # directory for simulated data
    'fmt_dir' : f'{work_dir}/format',      # directory for tensor-formatted data
    'trn_dir' : f'{work_dir}/train',       # directory for trained network
    'plt_dir' : f'{work_dir}/plot',        # directory for plotted figures
    'est_dir' : f'{work_dir}/estimate',    # directory for predictions on new data
    'log_dir' : f'{work_dir}/log',         # directory for analysis logs
    'output_precision'   : 12,                    # Number of digits (precision) for numbers in output files

    #-------------------------------#
    # Multiprocessing               #
    #-------------------------------#
    'use_parallel'   : 'T',                 # use multiprocessing to speed up jobs?
    'num_proc'       : -2,                  # how many CPUs to use (-2 means all but 2)

    #-------------------------------#
    # Simulate Step settings        #
    #-------------------------------#
    'sim_command'       : 'python3 ./scripts/sim/pj/sim_bisse_timehet.py ./scripts/sim/pj/bisse_timehet.pj trs', # exact command string, argument is output file prefix
    'sim_logging'       : 'verbose',        # verbose, compressed, or clean
    'start_idx'         : 0,                # first simulation replicate index
    'end_idx'           : 1000,             # last simulation replicate index
    'sim_batch_size'    : 1,

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
    'param_est'         : [                 # model parameters to predict (labels)
        'birth_rate0_t0', 'birth_rate0_t1'
    ],
    'param_data'        : [                 # model parameters that are known (aux. data)
    ],
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
    'loss'              : 'mse',            # loss function for learning
    'optimizer'         : 'adam',           # optimizer for network weight/bias parameters
    'metrics'           : ['mae', 'acc'],   # recorded training metrics
    'log_offset'        : 1.0,

    #-------------------------------#
    # Estimate Step settings        #
    #-------------------------------#
    'est_prefix'     : 'new.0',             # prefix for new dataset to predict

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