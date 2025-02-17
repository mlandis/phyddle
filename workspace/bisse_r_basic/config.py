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
    'sim_command'       : 'Rscript sim_bisse.R', # exact command string, argument is output file prefix

    #-------------------------------#
    # Format Step settings          #
    #-------------------------------#
    'num_char'          : 1,                # number of evolutionary characters
    'num_states'        : 2,                # number of states per character
    'tree_encode'       : 'extant',         # use model with serial or extant tree
    'brlen_encode'      : 'height_brlen',   # how to encode phylo brlen? height_only or height_brlen
    'char_encode'       : 'integer',        # how to encode discrete states? one_hot or integer
    'param_est'         : {                 # model parameters to predict (labels)
                            'log_birth_1'     : 'num',
                            'log_birth_2'     : 'num',
                            'log_death'       : 'num',
                            'log_state_rate'  : 'num',
                            'model_type'      : 'cat',
                            'start_state'     : 'cat',
                          },
    'param_data'        : {                 # model parameters that are known (aux. data)
                            'logit_sample_frac' : 'num'
                          },
    'char_format'       : 'csv',

}
