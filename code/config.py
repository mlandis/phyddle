import scipy as sp

num_locations = 3

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
    'num_locations' : num_locations,
    'rv_fn' : {
        'w': sp.stats.expon.rvs,
        'e': sp.stats.expon.rvs,
        'd': sp.stats.expon.rvs,
        'b': sp.stats.expon.rvs },
    'rv_arg' : {
        'w': { 'scale' : 0.2 },
        'e': { 'scale' : 0.1 },
        'd': { 'scale' : 0.1 },
        'b': { 'scale' : 0.5 }
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
    'stop_ceil_sizes'   : 400                # MASTER seems to generate too many taxa?
} | my_all_args


# define tensor-formatting settings
my_fmt_args = {
    'fmt_dir' : '../tensor_data',
    'sim_dir' : '../raw_data',
    'tree_type'  : 'extant',
    'param_pred' : ['w_0', 'e_0', 'd_0_1', 'b_0_1'],
    'param_data' : []
} | my_all_args

# define learning settings
my_lrn_args = { 
    'fmt_dir'        : '../tensor_data',
    'net_dir'        : '../network',
    'plt_dir'        : '../plot',
    'tree_size'      : 200,
    'tree_type'      : 'extant',
    'num_char'       : num_locations,
    'num_epochs'     : 20,
    'prop_test'       : 0.05,
    'prop_validation' : 0.05,
    'batch_size'     : 128,
    'loss'           : 'mse',
    'optimizer'      : 'adam',
    'metrics'        : ['mae', 'acc', 'mape']
} | my_all_args
