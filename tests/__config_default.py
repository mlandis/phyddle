#==============================================================================#
# Default phyddle config file                                                  #
#==============================================================================#

args = {
  #-------------------------------#
  # Basic                         #
  #-------------------------------#
  'cfg'                : 'config.py',          # Config file name               
  'proj'               : 'my_project',         # Project name(s) for pipeline step(s)
  'step'               : 'SFTEP',              # Pipeline step(s) defined with (S)imulate, (F)ormat, (T)rain, (E)stimate, (P)lot, or (A)ll
  'verbose'            : True,                 # Verbose output to screen?      
  'force'              : None,                 # Arguments override config file settings
  'make_cfg'           : None,                 # Write default config file to 'config_default.py'?'

  #-------------------------------#
  # Analysis                      #
  #-------------------------------#
  'use_parallel'       : True,                 # Use parallelization? (recommended)
  'num_proc'           : -2,                   # Number of cores for multiprocessing (-N for all but N)

  #-------------------------------#
  # Workspace                     #
  #-------------------------------#
  'sim_dir'            : '../workspace/simulate',  # Directory for raw simulated data
  'fmt_dir'            : '../workspace/format',   # Directory for tensor-formatted simulated data
  'trn_dir'            : '../workspace/train',    # Directory for trained networks and training output
  'est_dir'            : '../workspace/estimate',  # Directory for new datasets and estimates
  'plt_dir'            : '../workspace/plot',     # Directory for plotted results
  'log_dir'            : '../workspace/log',      # Directory for logs of analysis metadata

  #-------------------------------#
  # Simulate                      #
  #-------------------------------#
  'sim_command'        : None,                 # Simulation command to run single job (see documentation)
  'sim_logging'        : 'clean',              # Simulation logging style       
  'start_idx'          : 0,                    # Start replicate index for simulated training dataset
  'end_idx'            : 1000,                 # End replicate index for simulated training dataset
  'sim_more'           : 0,                    # Add more simulations with auto-generated indices
  'sim_batch_size'     : 1,                    # Number of replicates per simulation command

  #-------------------------------#
  # Format                        #
  #-------------------------------#
  'encode_all_sim'     : True,                 # Encode all simulated replicates into tensor?
  'num_char'           : None,                 # Number of characters           
  'num_states'         : None,                 # Number of states per character 
  'min_num_taxa'       : 10,                   # Minimum number of taxa allowed when formatting
  'max_num_taxa'       : 500,                  # Maximum number of taxa allowed when formatting
  'downsample_taxa'    : 'uniform',            # Downsampling strategy taxon count
  'tree_width'         : 500,                  # Width of phylo-state tensor    
  'tree_encode'        : 'extant',             # Encoding strategy for tree     
  'brlen_encode'       : 'height_brlen',       # Encoding strategy for branch lengths
  'char_encode'        : 'one_hot',            # Encoding strategy for character data
  'param_est'          : None,                 # Model parameters to estimate   
  'param_data'         : None,                 # Model parameters treated as data
  'char_format'        : 'nexus',              # File format for character data 
  'tensor_format'      : 'hdf5',               # File format for training example tensors
  'save_phyenc_csv'    : False,                # Save encoded phylogenetic tensor encoding to csv?

  #-------------------------------#
  # Train                         #
  #-------------------------------#
  'trn_objective'      : 'param_est',          # Objective of training procedure
  'num_epochs'         : 20,                   # Number of training epochs      
  'trn_batch_size'     : 128,                  # Training batch sizes           
  'prop_test'          : 0.05,                 # Proportion of data used as test examples (assess trained network performance)
  'prop_val'           : 0.05,                 # Proportion of data used as validation examples (diagnose network overtraining)
  'prop_cal'           : 0.2,                  # Proportion of data used as calibration examples (calibrate CPIs)
  'combine_test_val'   : True,                 # Combine test and validation datasets when assessing network fit?
  'cpi_coverage'       : 0.95,                 # Expected coverage percent for calibrated prediction intervals (CPIs)
  'cpi_asymmetric'     : True,                 # Use asymmetric (True) or symmetric (False) adjustments for CPIs?
  'loss'               : 'mse',                # Loss function for optimization 
  'optimizer'          : 'adam',               # Method used for optimizing neural network
  'metrics'            : ['mae', 'acc'],       # Recorded training metrics      

  #-------------------------------#
  # Estimate                      #
  #-------------------------------#
  'est_prefix'         : None,                 # Predict results for this dataset

  #-------------------------------#
  # Plot                          #
  #-------------------------------#
  'plot_train_color'   : 'blue',               # Plotting color for training data elements
  'plot_label_color'   : 'purple',             # Plotting color for training label elements
  'plot_test_color'    : 'red',                # Plotting color for test data elements
  'plot_val_color'     : 'green',              # Plotting color for validation data elements
  'plot_aux_color'     : 'orange',             # Plotting color for auxiliary data elements
  'plot_est_color'     : 'black',              # Plotting color for new estimation elements

}
