#==============================================================================#
# Default phyddle config file                                                  #
#==============================================================================#

args = {
  #-------------------------------#
  # Basic                         #
  #-------------------------------#
  'cfg'                : 'config.py',          # Config file name               
  'step'               : 'SFTEP',              # Pipeline step(s) defined with (S)imulate, (F)ormat, (T)rain, (E)stimate, (P)lot, or (A)ll
  'verbose'            : 'T',                  # Verbose output to screen?      
  'make_cfg'           : None,                 # Write default config file to '__config_default.py'?
  'output_precision'   : 16,                   # Number of digits (precision) for numbers in output files

  #-------------------------------#
  # Analysis                      #
  #-------------------------------#
  'use_parallel'       : 'T',                  # Use parallelization? (recommended)
  'num_proc'           : -2,                   # Number of cores for multiprocessing (-N for all but N)

  #-------------------------------#
  # Workspace                     #
  #-------------------------------#
  'sim_dir'            : './my_project/simulate',  # Directory for raw simulated data
  'fmt_dir'            : './my_project/format',   # Directory for tensor-formatted simulated data
  'trn_dir'            : './my_project/train',    # Directory for trained networks and training output
  'est_dir'            : './my_project/estimate',  # Directory for new datasets and estimates
  'plt_dir'            : './my_project/plot',     # Directory for plotted results
  'log_dir'            : './my_project/log',      # Directory for logs of analysis metadata

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
  'encode_all_sim'     : 'T',                  # Encode all simulated replicates into tensor?
  'num_char'           : None,                 # Number of characters           
  'num_states'         : None,                 # Number of states per character 
  'min_num_taxa'       : 10,                   # Minimum number of taxa allowed when formatting
  'max_num_taxa'       : 1000,                 # Maximum number of taxa allowed when formatting
  'downsample_taxa'    : 'uniform',            # Downsampling strategy taxon count
  'tree_width'         : 500,                  # Width of phylo-state tensor    
  'tree_encode'        : 'extant',             # Encoding strategy for tree     
  'brlen_encode'       : 'height_brlen',       # Encoding strategy for branch lengths
  'char_encode'        : 'one_hot',            # Encoding strategy for character data
  'param_est'          : ['my_rate'],          # Model parameters to estimate   
  'param_data'         : ['my_stat'],          # Model parameters treated as data
  'char_format'        : 'nexus',              # File format for character data 
  'tensor_format'      : 'hdf5',               # File format for training example tensors
  'save_phyenc_csv'    : 'F',                  # Save encoded phylogenetic tensor encoding to csv?

  #-------------------------------#
  # Train                         #
  #-------------------------------#
  'trn_objective'      : 'param_est',          # Objective of training procedure
  'num_epochs'         : 20,                   # Number of training epochs      
  'trn_batch_size'     : 512,                  # Training batch sizes           
  'prop_test'          : 0.05,                 # Proportion of data used as test examples (assess trained network performance)
  'prop_val'           : 0.05,                 # Proportion of data used as validation examples (diagnose network overtraining)
  'prop_cal'           : 0.2,                  # Proportion of data used as calibration examples (calibrate CPIs)
  'cpi_coverage'       : 0.95,                 # Expected coverage percent for calibrated prediction intervals (CPIs)
  'cpi_asymmetric'     : 'T',                  # Use asymmetric (True) or symmetric (False) adjustments for CPIs?
  'loss'               : 'mse',                # Loss function for optimization 
  'optimizer'          : 'adam',               # Method used for optimizing neural network
  'metrics'            : ['mae', 'acc'],       # Recorded training metrics      
  'log_offset'         : 1.0,                  # Offset size c when taking ln(x+c) for potentially zero-valued variables
  'phy_channel_plain'  : [64, 96, 128],        # Output channel sizes for plain convolutional layers for phylogenetic state input
  'phy_channel_stride' : [64, 96],             # Output channel sizes for stride convolutional layers for phylogenetic state input
  'phy_channel_dilate' : [32, 64],             # Output channel sizes for dilate convolutional layers for phylogenetic state input
  'aux_channel'        : [128, 64, 32],        # Output channel sizes for dense layers for auxiliary data input
  'lbl_channel'        : [128, 64, 32],        # Output channel sizes for dense layers for label outputs
  'phy_kernel_plain'   : [3, 5, 7],            # Kernel sizes for plain convolutional layers for phylogenetic state input
  'phy_kernel_stride'  : [7, 9],               # Kernel sizes for stride convolutional layers for phylogenetic state input
  'phy_kernel_dilate'  : [3, 5],               # Kernel sizes for dilate convolutional layers for phylogenetic state input
  'phy_stride_stride'  : [3, 6],               # Stride sizes for stride convolutional layers for phylogenetic state input
  'phy_dilate_dilate'  : [3, 5],               # Dilation sizes for dilate convolutional layers for phylogenetic state input

  #-------------------------------#
  # Estimate                      #
  #-------------------------------#
  'est_prefix'         : None,                 # Predict results for this dataset

  #-------------------------------#
  # Plot                          #
  #-------------------------------#
  'plot_train_color'   : 'blue',               # Plotting color for training data elements
  'plot_label_color'   : 'orange',             # Plotting color for training label elements
  'plot_test_color'    : 'purple',             # Plotting color for test data elements
  'plot_val_color'     : 'red',                # Plotting color for validation data elements
  'plot_aux_color'     : 'green',              # Plotting color for auxiliary data elements
  'plot_est_color'     : 'black',              # Plotting color for new estimation elements
  'plot_scatter_log'   : 'T',                  # Use log values for scatter plots when possible?
  'plot_contour_log'   : 'T',                  # Use log values for contour plots when possible?
  'plot_density_log'   : 'T',                  # Use log values for density plots when possible?

}