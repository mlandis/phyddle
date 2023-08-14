#==============================================================================#
# MASTER phyddle config file                                                   #
#==============================================================================#


# helper variables

args = {
    
    #-------------------------------#
    # Project organization          #
    #-------------------------------#
    'proj'    : 'MASTER_example',           # project name(s)
    'step'    : 'SFTEP',                    # step(s) to run
    'verbose' : 'True',                     # print verbose phyddle output?
    'sim_dir' : '../workspace/simulate',    # directory for simulated data
    'fmt_dir' : '../workspace/format',      # directory for tensor-formatted data
    'trn_dir' : '../workspace/train',       # directory for trained network
    'plt_dir' : '../workspace/plot',        # directory for plotted figures
    'est_dir' : '../workspace/estimate',    # directory for predictions on new data
    'log_dir' : '../workspace/log',         # directory for analysis logs
    
    #-------------------------------#
    # Multiprocessing               #
    #-------------------------------#
    'use_parallel'   : 'True',              # use multiprocessing to speed up jobs?
    'num_proc'       : -2,                  # how many CPUs to use (-2 means all but 2)
    
    #-------------------------------#
    # Simulate Step settings        #
    #-------------------------------#
    'sim_command'       : './sim/MASTER/sim_one.py', # exact command string, argument is output file prefix
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
    'param_est'         : [                 # model parameters to estimate (labels)
        'w_0', 'e_0', 'd_0_1', 'b_0_1'
    ],
    'param_data'        : [],               # model parameters that are known (aux. data)
    'tensor_format'     : 'hdf5',           # save as compressed HDF5 or raw csv
    'save_phyenc_csv'   : 'F',              # save intermediate phylo-state vectors to file

    #-------------------------------#
    # Train Step settings           #
    #-------------------------------#
    'trn_objective'     : 'param_est',      # what is the learning task? param_est or model_test
    'tree_width'        : 200,              # tree width category used to train network
    'num_epochs'        : 20,               # number of training intervals (epochs)
    'prop_test'         : 0.05,             # proportion of sims in test dataset
    'prop_val'          : 0.05,             # proportion of sims in validation dataset
    'prop_cal'          : 0.20,             # proportion of sims in CPI calibration dataset 
    'cpi_coverage'      : 0.95,             # coverage level for CPIs
    'cpi_asymmetric'    : 'T',              # upper/lower (True) or symmetric (False) CPI adjustments
    'trn_batch_size'    : 128,              # number of samples in each training batch
    'loss'              : 'mse',            # loss function for learning
    'optimizer'         : 'adam',           # optimizer for network weight/bias parameters
    'metrics'           : ['mae', 'acc'],   # recorded training metrics

    #-------------------------------#
    # Estimate Step settings        #
    #-------------------------------#
    'est_prefix'     : 'new.1',             # prefix for new dataset to predict

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

