#==============================================================================#
# Default phyddle config file                                                  #
#==============================================================================#


args = {
  #-------------------------------#
  # Basic                         #
  #-------------------------------#
  'step'               : 'SFTEP',              # Pipeline step(s) defined with (S)imulate, (F)ormat, (T)rain, (E)stimate, (P)lot, or (A)ll
  'verbose'            : 'T',                  # Verbose output to screen?      
  'output_precision'   : 16,                   # Number of digits (precision) for numbers in output files
  'dir'                : './',
  'prefix'             : 'out',

  #-------------------------------#
  # Analysis                      #
  #-------------------------------#
  'use_parallel'       : 'T',                  # Use parallelization? (recommended)
  'use_cuda'           : 'T',                  # Use CUDA for Train?
  'num_proc'           : -2,                   # Number of cores for multiprocessing (-N for all but N)

  #-------------------------------#
  # Simulate                      #
  #-------------------------------#
  'sim_command'        : 'python3 sim_bdk.py', # Simulation command to run single job (see documentation)
  'sim_logging'        : 'clean',              # Simulation logging style       
  'start_idx'          : 0,                    # Start replicate index for simulated training dataset
  'end_idx'            : 1000,                 # End replicate index for simulated training dataset
  'sim_batch_size'     : 1,                    # Number of replicates per simulation command

  #-------------------------------#
  # Format                        #
  #-------------------------------#
  'encode_all_sim'     : 'T',                  # Encode all simulated replicates into tensor?
  'num_char'           : 1,                    # Number of characters           
  'num_states'         : 1,                    # Number of states per character 
  'min_num_taxa'       : 10,                   # Minimum number of taxa allowed when formatting
  'max_num_taxa'       : 1000,                 # Maximum number of taxa allowed when formatting
  'downsample_taxa'    : 'uniform',            # Downsampling strategy taxon count
  'tree_width'         : 200,                  # Width of phylo-state tensor    
  'tree_encode'        : 'extant',             # Encoding strategy for tree     
  'brlen_encode'       : 'height_brlen',       # Encoding strategy for branch lengths
  'char_encode'        : 'integer',            # Encoding strategy for character data
  'param_est'          : {
      'BirthConst_0':'real',
      'DeathConst_0':'real',
      'DeathDensity_0':'real'
    },          # Model parameters to estimate   
  'param_data'         : { },          # Model parameters treated as data
  'char_format'        : 'nexus',              # File format for character data 
  'tensor_format'      : 'hdf5',               # File format for training example tensors
  'save_phyenc_csv'    : 'F',                  # Save encoded phylogenetic tensor encoding to csv?

  #-------------------------------#
  # Train                         #
  #-------------------------------#
  'num_epochs'         : 20,                   # Number of training epochs      
  'trn_batch_size'     : 512,                  # Training batch sizes           
  'prop_test'          : 0.05,                 # Proportion of data used as test examples (assess trained network performance)
  'prop_val'           : 0.05,                 # Proportion of data used as validation examples (diagnose network overtraining)
  'prop_cal'           : 0.2,                  # Proportion of data used as calibration examples (calibrate CPIs)
  'cpi_coverage'       : 0.95,                 # Expected coverage percent for calibrated prediction intervals (CPIs)
  'cpi_asymmetric'     : 'T',                  # Use asymmetric (True) or symmetric (False) adjustments for CPIs?
  'loss'               : 'mse',                # Loss function for optimization 
  'optimizer'          : 'adam',               # Method used for optimizing neural network
  'phy_channel_plain'  : [32, 64],        # Output channel sizes for plain convolutional layers for phylogenetic state input
  'phy_channel_stride' : [32, 64],             # Output channel sizes for stride convolutional layers for phylogenetic state input
  'phy_channel_dilate' : [32, 64],             # Output channel sizes for dilate convolutional layers for phylogenetic state input
  'aux_channel'        : [64, 32],        # Output channel sizes for dense layers for auxiliary data input
  'lbl_channel'        : [64, 32],        # Output channel sizes for dense layers for label outputs
  'phy_kernel_plain'   : [3, 5],            # Kernel sizes for plain convolutional layers for phylogenetic state input
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

}
