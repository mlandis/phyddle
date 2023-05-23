
import scipy as sp

#####################
# PIPELINE SETTINGS #
#####################
my_all_args = { 'job_name' : 'test_sirm' }
NUM_LOC = 3

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
    'num_locations' : NUM_LOC,
    'rv_fn' : {
        'R0'        : sp.stats.uniform.rvs,
        'recovery'  : sp.stats.expon.rvs,
        'sampling'  : sp.stats.expon.rvs,
        'migration' : sp.stats.expon.rvs,
        'S0'        : sp.stats.uniform.rvs },
    'rv_arg' : {
        'R0'        : { 'loc': 1., 'scale' : 9. },
        'recovery'  : { 'loc': 0., 'scale' : 1. },
        'sampling'  : { 'loc': 0., 'scale' : 1./10. },
        'migration' : { 'loc': 0., 'scale' : 1./10. },
        'S0'        : { 'loc': 1000., 'scale': 9000. }
    }
}

#######################
# SIMULATION SETTINGS #
#######################
my_sim_args = {
    'sim_dir'           : '../raw_data',
    'start_idx'         : 0,
    'end_idx'           : 100,
    'tree_sizes'        : [ 200, 500 ],
    'use_parallel'      : True,
    'num_proc'          : -2,
    'sample_population' : ['S'],
    'stop_floor_sizes'  : 0,
    'stop_ceil_sizes'   : 450   # MASTER seems to generate too many taxa?
} | my_all_args

###################
# FORMAT SETTINGS #
###################
my_fmt_args = {
    'fmt_dir'     : '../tensor_data',
    'sim_dir'     : '../raw_data',
    'param_pred'  : [ 'R0_0', 'sampling_0', 'migration_0_1' ],
    'param_data'  : [ 'recovery_0', 'S0_0', 'S0_1', 'S0_2' ]
} | my_all_args

#####################
# LEARNING SETTINGS #
#####################
my_lrn_args = { 
    'fmt_dir'        : '../tensor_data',
    'net_dir'        : '../network',
    'plt_dir'        : '../plot',
    'tree_size'      : 200,
    'tree_type'      : 'serial',
    'num_char'       : NUM_LOC,
    'num_epochs'     : 20,
    'prop_test'       : 0.05,
    'prop_validation' : 0.05,
    'batch_size'     : 32,
    'loss'           : 'mse',
    'optimizer'      : 'adam',
    'metrics'        : ['mae', 'acc', 'mape']
} | my_all_args
