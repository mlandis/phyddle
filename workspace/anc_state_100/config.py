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
    'dir'     : './',

    #-------------------------------#
    # Simulate Step settings        #
    #-------------------------------#
    'sim_command'       : 'Rscript sim_varyTree.R', # exact command string, argument is output file prefix

    'start_idx'          : 1,                         # Start index for simulated training replicates
    #'end_idx'            : 1,                      # End index for simulated training replicates
    'end_idx'            : 50000,                      # End index for simulated training replicates
    'sim_batch_size'     : 500,                        # Number of replicates per simulation command

    #'use_cuda'           : 'F',
    #-------------------------------#
    # Format Step settings          #
    #-------------------------------#
    'num_char'          : 1,                # number of evolutionary characters
    'num_states'        : 2,                # number of states per character
    'tree_width'        : 100,
    'tree_encode'       : 'extant',         # use model with serial or extant tree
    'brlen_encode'      : 'height_brlen',   # how to encode phylo brlen? height_only or height_brlen
    'char_encode'       : 'integer',        # how to encode discrete states? one_hot or integer
    'param_est'         : {                 # model parameters to predict (labels)
                            'anc_state_1'     : 'cat',
 # If you change the values in format for estimate, the true values change but not the inferred values in terms of what variables are printed out
                          },
    'asr_est'            : 'T',
    #'param_data'        : {                 # model parameters that are known (aux. data)
    #                        'logit_sample_frac' : 'num'
    #                      },
    'char_format'       : 'csv',
    'min_num_taxa'       : 10,                   # Minimum number of taxa allowed when formatting
    'max_num_taxa'       : 100,                  # Maximum number of taxa allowed when formatting

    #-------------------------------#
    # Train                         #
    #-------------------------------#
    #'trn_batch_size'     : 100,                 # Training batch sizes

    #'phy_kernel_stride'  : [7, 8],               # Kernel sizes for stride convolutional layers
                                                 #     for phylogenetic state input
    #'use_parallel'       : 'F',            # Use parallelization? (recommended)
}
