
import scipy as sp

#####################
# PIPELINE SETTINGS #
#####################
my_all_args = { 'job_name' : 'test_sirm' }

##################
# MODEL SETTINGS #
##################
my_mdl_args = {
    'model_type'    : 'sirm',
    'model_variant' : 'equal_rates',
    'num_locations' : 3,
    'rv_fn' : {
        's': sp.stats.expon.rvs,
        'i': sp.stats.expon.rvs,
        'r': sp.stats.expon.rvs,
        'm': sp.stats.expon.rvs,
        'n0': sp.stats.gamma.rvs },
    'rv_arg' : {
        's': { 'scale' : 1.0 },
        'i': { 'scale' : 0.5 },
        'r': { 'scale' : 1.0 },
        'm': { 'scale' : 3.0 },
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
    'stop_ceil_sizes'   : 300                # MASTER seems to generate too many taxa?
} | my_all_args


# define tensor-formatting settings
my_fmt_args = {
    'fmt_dir' : '../tensor_data',
    'sim_dir' : '../raw_data'
} | my_all_args

# define learning settings
my_lrn_args = { 
    'fmt_dir'        : '../tensor_data',
    'net_dir'        : '../network',
    'plt_dir'        : '../plot',
    'tree_size'      : 500,
    'tree_type'      : 'extant',
    'predict_idx'    : [ 0, 3, 6, 18 ],
    'num_epochs'     : 20,
    'num_test'       : 5,
    'num_validation' : 5,
    'batch_size'     : 32,
    'loss'           : 'mse',
    'optimizer'      : 'adam',
    'metrics'        : ['mae', 'acc', 'mape']
} | my_all_args
