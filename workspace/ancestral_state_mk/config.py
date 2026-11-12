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
    'sim_command'       : 'Rscript sim_tree.R', # exact command string, argument is output file prefix

    'start_idx'          : 1,                         # Start index for simulated training replicates
    'end_idx'            : 50000,                      # End index for simulated training replicates
    'sim_batch_size'     : 2500,                        # Number of replicates per simulation command

    #-------------------------------#
    # Format Step settings          #
    #-------------------------------#
    'num_char'          : 1,                # number of evolutionary characters
    'num_states'        : 2,                # number of states per character
    'tree_width'        : 50,
    'tree_encode'       : 'extant',         # use model with serial or extant tree
    'brlen_encode'      : 'height_brlen',   # how to encode phylo brlen? height_only or height_brlen
    'char_encode'       : 'integer',        # how to encode discrete states? one_hot or integer
    'asr_est'            : 'T',
    'char_format'       : 'csv',
    'min_num_taxa'       : 10,                   # Minimum number of taxa allowed when formatting
    'max_num_taxa'       : 50,                  # Maximum number of taxa allowed when formatting
    'shuffle_test'       : 'F', 

}
