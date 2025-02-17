#====================================================================#
# Default phyddle config file                                        #
#====================================================================#

args = {
  #-------------------------------#
  # Workspace                     #
  #-------------------------------#
  'dir'                : '.',  # Parent directory for all step directories unless step directory given
  'sim_dir'            : None,                  # Directory for raw simulated data
  'emp_dir'            : None,                  # Directory for raw empirical data
  'fmt_dir'            : None,                  # Directory for tensor-formatted data
  'trn_dir'            : None,                  # Directory for trained networks and training output
  'est_dir'            : None,                  # Directory for new datasets and estimates
  'plt_dir'            : None,                  # Directory for plotted results 
  'log_dir'            : None,                  # Directory for logs of analysis metadata
  'prefix'             : None,                 # Prefix for all output unless step prefix given
  'sim_prefix'         : None,                  # Prefix for raw simulated data 
  'emp_prefix'         : None,                  # Prefix for raw empirical data 
  'fmt_prefix'         : None,                  # Prefix for tensor-formatted data
  'trn_prefix'         : None,                  # Prefix for trained networks and training output
  'est_prefix'         : None,                  # Prefix for estimate results   
  'plt_prefix'         : None,                  # Prefix for plotted results    

  #-------------------------------#
  # Analysis                      #
  #-------------------------------#
  'use_parallel'       : 'T',                  # Use parallelization? (recommended)
  'use_cuda'           : 'F',                  # Use CUDA parallelization? (recommended; requires Nvidia GPU)
  'num_proc'           : -1,                   # Number of cores for multiprocessing (-N for all but N)
  'no_emp'             : False,                # Disable Format/Estimate steps for empirical data?
  'no_sim'             : False,                # Disable Format/Estimate steps for simulated data?

  #-------------------------------#
  # Simulate                      #
  #-------------------------------#
  'sim_command'        : 'python3 ./sim_sirm_exp.py',   # Simulation command to run single job (see documentation)
  'sim_logging'        : 'clean',              # Simulation logging style       
  'start_idx'          : 0,                    # Start replicate index for simulated training dataset
  'end_idx'            : 1000,                 # End replicate index for simulated training dataset
  'sim_more'           : 0,                    # Add more simulations with auto-generated indices
  'sim_batch_size'     : 1,                    # Number of replicates per simulation command

  #-------------------------------#
  # Format                        #
  #-------------------------------#
  'encode_all_sim'     : 'T',                  # Encode all simulated replicates into tensor?
  'num_char'           : 5,                 # Number of characters           
  'num_states'         : 2,                 # Number of states per character 
  'min_num_taxa'       : 20,                   # Minimum number of taxa allowed when formatting
  'max_num_taxa'       : 500,                 # Maximum number of taxa allowed when formatting
  'downsample_taxa'    : 'uniform',            # Downsampling strategy taxon count
  'tree_width'         : 500,                  # Width of phylo-state tensor    
  'tree_encode'        : 'serial',             # Encoding strategy for tree     
  'brlen_encode'       : 'height_brlen',       # Encoding strategy for branch lengths
  'char_encode'        : 'integer',            # Encoding strategy for character data
  'param_est'          : {
          "log_R0_0" : 'num',
          "log_Sample_0" : "num",
          "log_Migrate_0_0" : "num"
          },                   # Model parameters and variables to estimate

  'param_data'         : {
          'log_Recover_0' : "num",
          'proportion_sample_in_tree_0' : "num"
          },                   # Model parameters and variables treated as data

  'char_format'        : 'nexus',              # File format for character data 
  'tensor_format'      : 'hdf5',               # File format for training example tensors
  'save_phyenc_csv'    : 'F',                  # Save encoded phylogenetic tensor encoding to csv?

  #-------------------------------#
  # Train                         #
  #-------------------------------#
  'num_epochs'         : 200,                  # Number of training epochs      
  'trn_batch_size'     : 512,                  # Training batch sizes           
  'prop_test'          : 0.05,                # Proportion of data used as test examples (assess trained network performance)
  'prop_val'           : 0.05,                 # Proportion of data used as validation examples (diagnose network overtraining)
  'prop_cal'           : 0.05,                 # Proportion of data used as calibration examples (calibrate CPIs)
  'cpi_coverage'       : 0.95,                 # Expected coverage percent for calibrated prediction intervals (CPIs)
  'cpi_asymmetric'     : 'T',                  # Use asymmetric (True) or symmetric (False) adjustments for CPIs?
  'loss_numerical'     : 'mse',                # Loss function for num value estimates
  'optimizer'          : 'adam',               # Method used for optimizing neural network
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
  # none currently

  #-------------------------------#
  # Plot                          #
  #-------------------------------#
  'plot_train_color'   : 'blue',               # Plotting color for training data elements
  'plot_test_color'    : 'purple',             # Plotting color for test data elements
  'plot_val_color'     : 'red',                # Plotting color for validation data elements
  'plot_label_color'   : 'orange',             # Plotting color for label elements
  'plot_aux_color'     : 'green',              # Plotting color for auxiliary data elements
  'plot_emp_color'     : 'black',              # Plotting color for empirical elements
  'plot_num_scatter'   : 50,                   # Number of examples in scatter plot
  'plot_min_emp'       : 10,                   # Minimum number of empirical datasets to plot densities
  'plot_num_emp'       : 5,                    # Number of empirical results to plot

  #-------------------------------#
  # Basic                         #
  #-------------------------------#
  'cfg'                : 'config.py',          # Config file name               
  'step'               : 'SFTEP',              # Pipeline step(s) defined with (S)imulate, (F)ormat, (T)rain, (E)stimate, (P)lot, or (A)ll
  'verbose'            : 'T',                  # Verbose output to screen?      
  'output_precision'   : 16,                   # Number of digits (precision) for numbers in output files

}
