
import scipy as sp

#####################
# PIPELINE SETTINGS #
#####################
args = { 'job_name' : 'test_cont' }

##################
# MODEL SETTINGS #
##################
mdl_args = {
    'model_type'    : 'cont_trait',
    'model_variant' : 'BM_iid',
    'num_traits'    : 3,
    'num_bins'      : 20,
    'rv_fn' : { 
        'mu'   : sp.stats.norm.rvs,
        'sigma': sp.stats.expon.rvs },
    'rv_arg' : {
        'mu'   : { 'loc': 0.0, 'scale' : 1.0 },
        'sigma': { 'scale' : 1.0 }
    }
}

#######################
# SIMULATION SETTINGS #
#######################
sim_args = {
    'sim_dir'           : '../raw_data',
    'start_idx'         : 0,
    'end_idx'           : 100,
    'tree_sizes'        : [ 200, 500 ],
    'use_parallel'      : True,
    'num_proc'          : -2,
    'sample_population' : ['S'],
    'stop_floor_sizes'  : 0,
    'stop_ceil_sizes'   : 300                # MASTER seems to generate too many taxa?
}
args = args | sim_args

###################
# FORMAT SETTINGS #
###################
fmt_args = {
    'fmt_dir'     : '../tensor_data',
    'sim_dir'     : '../raw_data',
    'param_pred'  : [ 's_0', 'i_0', 'm_0_1' ],
    'param_data'  : [ 'r_0' ]
}
args = args | fmt_args

#####################
# LEARNING SETTINGS #
#####################
lrn_args = { 
    'fmt_dir'        : '../tensor_data',
    'net_dir'        : '../network',
    'plt_dir'        : '../plot',
    'tree_size'      : 200,
    'tree_type'      : 'serial',
    'num_epochs'     : 20,
    'num_test'       : 200,
    'num_validation' : 200,
    'batch_size'     : 32,
    'loss'           : 'mse',
    'optimizer'      : 'adam',
    'metrics'        : ['mae', 'acc', 'mape']
}
args = args | lrn_args
