
import scipy as sp

#####################
# PIPELINE SETTINGS #
#####################
args = { 'proj' : 'sirm1' }
num_char = 3

##################
# MODEL SETTINGS #
##################
mdl_args = {
    'model_type'    : 'sirm',
    'model_variant' : 'equal_rates',
    'num_char'      : num_char,
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
args = args | mdl_args

#######################
# SIMULATION SETTINGS #
#######################
sim_args = {
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
}
args = args | sim_args

###################
# FORMAT SETTINGS #
###################
fmt_args = {
    'fmt_dir'     : '../tensor_data',
    'sim_dir'     : '../raw_data',
    'tree_type'   : 'serial',
    'param_pred'  : [ 'R0_0', 'sampling_0', 'migration_0_1' ],
    'param_data'  : [ 'recovery_0', 'S0_0' ],
    'tensor_format' : 'hdf5'
}
args = args | fmt_args

#####################
# LEARNING SETTINGS #
#####################
lrn_args = { 
    'net_dir'          : '../network',
    'tree_size'        : 500,
    'num_epochs'       : 20,
    'prop_test'        : 0.05,
    'prop_validation'  : 0.05,
    'prop_calibration' : 0.20,
    'alpha_CQRI'       : 0.95,
    'batch_size'       : 128,
    'loss'             : 'mse',
    'optimizer'        : 'adam',
    'metrics'          : ['mae', 'acc']
}
args = args | lrn_args



#####################
# PLOTTING SETTINGS #
#####################
plt_args = {
    'plt_dir'        : '../plot',
    'network_prefix' : 'sim_batchsize128_numepoch20_nt500'
}
args = args | plt_args


#######################
# PREDICTING SETTINGS #
#######################

prd_args = {
    'pred_dir'    : '../raw_data/my_job_new',
    'pred_prefix' : 'sim.0'
}
args = args | prd_args
