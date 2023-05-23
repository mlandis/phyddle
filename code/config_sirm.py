
import scipy as sp

#####################
# PIPELINE SETTINGS #
#####################
my_all_args = { 'job_name' : 'test_sirm' }
NUM_LOC = 3

##################
# MODEL SETTINGS #
##################
my_mdl_args = {
    'model_type'    : 'sirm',
    'model_variant' : 'equal_rates',
    'num_locations' : NUM_LOC,
    'rv_fn' : {
        's': sp.stats.expon.rvs,
        'i': sp.stats.expon.rvs,
        'r': sp.stats.expon.rvs,
        'm': sp.stats.expon.rvs,
        'n0': sp.stats.gamma.rvs },
    'rv_arg' : {
        's': { 'loc': 0.0, 'scale' : 1.0 },
        'i': { 'loc': 1.0, 'scale' : 0.5 },
        'r': { 'loc': 0.0, 'scale' : 1.0 },
        'm': { 'loc': 0.0, 'scale' : 1.0 },
        'n0': { 'a':0.5, 'scale':1000000 }
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
    'stop_ceil_sizes'   : 450                # MASTER seems to generate too many taxa?
} | my_all_args

###################
# FORMAT SETTINGS #
###################
my_fmt_args = {
    'fmt_dir'     : '../tensor_data',
    'sim_dir'     : '../raw_data',
    'param_pred'  : [ 's_0', 'i_0', 'm_0_1' ],
    'param_data'  : [ 'r_0' ]
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
