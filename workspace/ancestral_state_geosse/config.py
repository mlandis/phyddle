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
    'prefix'  : 'sim',                     # Prefix for output for all setps
    'dir'     : './',

    #-------------------------------#
    # Multiprocessing               #
    #-------------------------------#
    'use_parallel'   : 'T',                 # use multiprocessing to speed up jobs?
    'use_cuda'       : 'T',

    #-------------------------------#
    # Simulate Step settings        #
    #-------------------------------#
    'sim_command'       : f'Rscript simple_simulate.R', # exact command string, argument is output file prefix
    'sim_logging'       : 'verbose',        # verbose, compressed, or clean
    'start_idx'         : 1,                # first simulation replicate index
    'end_idx'           : 250000,             # last simulation replicate index
    'sim_batch_size'    : 5000,

    #-------------------------------#
    # Format Step settings          #
    #-------------------------------#
    'encode_all_sim'    : 'T',
    'num_char'          : 2,                # number of evolutionary characters
    'num_states'        : 2,                # number of states per character
    'min_num_taxa'      : 52,               # min number of taxa for valid sim
    'max_num_taxa'      : 52,              # max number of taxa for valid sim

    'rel_extant_age_tol': 1E-8,
    'tree_width'        : 52,              # tree width category used to train network
    'tree_encode'       : 'extant',         # use model with serial or extant tree
    'brlen_encode'      : 'height_brlen',   # how to encode phylo brlen? height_only or height_brlen
    'char_encode'       : 'integer',        # how to encode discrete states? one_hot or integer
    'asr_est'           : 'T', 
    'asr_rotate'        : {3 : 4,
                           4 : 3,
                           5 : 6,
                           6 : 5,
                           7 : 8,
                           8 : 7 },

    'tensor_format'     : 'hdf5',           # save as compressed HDF5 or raw csv
    'char_format'       : 'csv',
    'save_phyenc_csv'   : 'F',              # save intermediate phylo-state vectors to file
    'shuffle_test'       : 'F', 

    #-------------------------------#
    # Train Step settings           #
    #-------------------------------#
    #'num_epochs'        : 200,              # number of training intervals (epochs)
    'cpi_coverage'      : 0.80,             # coverage level for CPIs
    'cpi_asymmetric'    : 'T',              # upper/lower ('T') or symmetric ('F') CPI adjustments
    'trn_batch_size'    : 2048,             # number of samples in each training batch
    'loss_numerical'    : 'mae',            # loss function for learning (real-valued) targets
    'optimizer'         : 'adam',           # optimizer for network weight/bias parameters
    'prop_test' : .05, 
    'prop_val' : .05, 
    'num_epochs' : 500, 
    'num_early_stop' : 25, 

    #-------------------------------#
    # Estimate Step settings        #
    #-------------------------------#

    'asr_map_tip_states'    : {0 : [1, 0],
                               1 : [0, 1],
                               2 : [1, 1]},

    'asr_map_triplet_states' : { 0: (0, 0, 0),       # A  -> A ,  A
                                 1: (1, 1, 1),       # B  -> B ,  B
                                 2: (2, 0, 1),       # AB -> A ,  B
                                 3: (2, 1, 0),       # AB -> B ,  A
                                 4: (2, 2, 0),       # AB -> AB,  A
                                 5: (2, 0, 2),       # AB -> A , AB
                                 6: (2, 2, 1),       # AB -> AB,  B
                                 7: (2, 1, 2)},      # AB -> B', AB
    'asr_rb_nexus' : 'T'
 }
