
import scipy as sp

#####################
# PIPELINE SETTINGS #
#####################
my_all_args = { 'job_name' : 'my_job' }

##################
# MODEL SETTINGS #
##################
my_mdl_args = {
    'model_type'    : 'geosse',
    'model_variant' : 'equal_rates',
    'num_locations' : 3,
    'rv_fn' : {
        'w': sp.stats.expon.rvs,
        'e': sp.stats.expon.rvs,
        'd': sp.stats.expon.rvs,
        'b': sp.stats.expon.rvs },
    'rv_arg' : {
        'w': { 'scale' : 1.0 },
        'e': { 'scale' : 0.5 },
        'd': { 'scale' : 1.0 },
        'b': { 'scale' : 3.0 }
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
    'num_proc'          : 12,                # set to max - 2
    #'start_sizes'       : {},                # move setting into model spec
    #'start_state'       : { 'S' : 0 },       # move setting into model spec
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
    'num_test'       : 20,
    'num_validation' : 20,
    'batch_size'     : 32,
    'loss'           : 'mse',
    'optimizer'      : 'adam',
    'metrics'        : ['mae', 'acc', 'mape']
} | my_all_args
