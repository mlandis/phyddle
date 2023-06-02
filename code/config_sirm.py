
import scipy as sp

#####################
# PIPELINE SETTINGS #
#####################
my_all_args = { 'job_name' : 'sirm1' }
num_char = 3

# infection rate (beta) constant should be proportional to number of pairwise combinations
# (i.e. number of susceptibles) during exponential growth phase

# R0 is defined as beta/gammma

# so choose gamma such that R0 tends to be > 1

##################
# MODEL SETTINGS #
##################
my_mdl_args = {
    'model_type'    : 'sirm',
    'model_variant' : 'equal_rates',
    'num_char' : num_char,
    'rv_fn' : {
        'R0'        : sp.stats.uniform.rvs,
        'recovery'  : sp.stats.uniform.rvs,
        'sampling'  : sp.stats.uniform.rvs,
        'migration' : sp.stats.uniform.rvs,
        'S0'        : sp.stats.uniform.rvs },
    'rv_arg' : {
        'R0'        : { 'loc': 1., 'scale' : 9. },
        'recovery'  : { 'loc': 0.01, 'scale' : 0.09 },
        'sampling'  : { 'loc': 0.10, 'scale' : 0.90 },
        'migration' : { 'loc': 0.10, 'scale' : 0.90 },
        'S0'        : { 'loc': 1000., 'scale': 4000. }
    }
}
my_all_args = my_all_args | my_mdl_args

#######################
# SIMULATION SETTINGS #
#######################
my_sim_args = {
    'sim_dir'           : '../raw_data',
    'sim_logging'       : 'clean',
    'start_idx'         : 0,
    'end_idx'           : 100,
    'tree_sizes'        : [ 200, 500 ],
    'stop_time'         : 10,
    'use_parallel'      : True,
    'num_proc'          : -2,
    'sample_population' : ['S'],
    'stop_floor_sizes'  : 0,
    'stop_ceil_sizes'   : 450   # MASTER seems to generate too many taxa?
} #| my_all_args
my_all_args = my_all_args | my_sim_args

####################
# ENCODER SETTINGS #
####################
my_enc_args = {

}
my_all_args = my_all_args | my_enc_args

###################
# FORMAT SETTINGS #
###################
my_fmt_args = {
    'fmt_dir'     : '../tensor_data',
    'sim_dir'     : '../raw_data',
    'tree_type'   : 'serial',
    'param_pred'  : [ 'R0_0', 'sampling_0', 'migration_0_1' ],
    'param_data'  : [ 'recovery_0', 'S0_0' ],
    'tensor_format' : 'csv'
}#| my_all_args
my_all_args = my_all_args | my_fmt_args

#####################
# LEARNING SETTINGS #
#####################
my_lrn_args = { 
    #'fmt_dir'        : '../tensor_data',
    'net_dir'        : '../network',
    #'plt_dir'        : '../plot',
    'tree_size'      : 500,
   #'tree_type'      : 'serial',
    #'num_char'       : NUM_LOC,
    'num_epochs'     : 20,
    'prop_test'       : 0.05,
    'prop_validation' : 0.05,
    'batch_size'     : 128,
    'loss'           : 'mse',
    'optimizer'      : 'adam',
    'metrics'        : ['mae', 'acc', 'mape']
} #| my_all_args
my_all_args = my_all_args | my_lrn_args



#####################
# PLOTTING SETTINGS #
#####################
my_plt_args = {
    'plt_dir'        : '../plot',
    'network_prefix' : 'sim_batchsize128_numepoch20_nt500'
}
my_all_args = my_all_args | my_plt_args


#######################
# PREDICTING SETTINGS #
#######################

my_prd_args = {
    'pred_dir'    : '../raw_data/my_job_new',
    'pred_prefix' : 'sim.0'
}
my_all_args = my_all_args | my_prd_args
